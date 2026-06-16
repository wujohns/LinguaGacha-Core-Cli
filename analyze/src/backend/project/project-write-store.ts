import type { ApiJsonValue } from "../api/api-types";
import { ProjectDatabase } from "../database/database-operations";
import type { DatabaseJsonValue, DatabaseOperation } from "../database/database-types";
import { TASK_PROGRESS_STATUSES } from "../../domain/task";
import { count_analysis_glossary_candidates } from "../../shared/analysis-candidate";
import type { ProjectDataSection, ProjectWriteResult } from "../../shared/project-event";
import type {
  ProjectChangePayloadMode,
  ProjectChangeSectionPayload,
} from "../../shared/project-event";
import { get_section_revision } from "./project-data";
import {
  ProjectWriteCoordinator,
  type ProjectWriteChangeRequest,
  type ProjectWriteRevisionContext,
} from "./project-changes";
import type { ProjectChangePublisher } from "./project-changes";
import type { ProjectEventBus } from "./project-events";

type JsonRecord = Record<string, ApiJsonValue>;
type MutableJsonRecord = Record<string, ApiJsonValue>;

type RuntimeCommitRequest = {
  projectPath: string;
  expectedSectionRevisions?: ApiJsonValue;
  requireExpectedSectionRevisions: boolean;
  revisionSections: ProjectDataSection[];
  source: string;
  updatedSections: ProjectDataSection[];
  sections?: Partial<
    Record<ProjectDataSection, { payloadMode: ProjectChangePayloadMode; data?: ApiJsonValue }>
  >;
  buildOperations: (context: ProjectWriteRevisionContext) => DatabaseOperation[];
};

export type ProjectWriteSectionAck = {
  changed_item_ids: number[];
  section_revisions: MutableJsonRecord;
};

export class ProjectWriteStore {
  private readonly database: ProjectDatabase;
  private readonly write_coordinator: ProjectWriteCoordinator;

  public constructor(
    database: ProjectDatabase,
    project_event_bus: ProjectEventBus,
    project_change_publisher: ProjectChangePublisher | null,
  ) {
    this.database = database;
    this.write_coordinator = new ProjectWriteCoordinator(
      database,
      project_change_publisher,
      project_event_bus,
    );
  }

  public update_task_progress_meta(request: {
    projectPath: string;
    meta: MutableJsonRecord;
  }): void {
    this.database.execute_transaction([
      this.op("upsertMetaEntries", {
        projectPath: request.projectPath,
        meta: request.meta as unknown as DatabaseJsonValue,
      }),
    ]);
  }

  public async commit_analysis_artifacts(request: {
    projectPath: string;
    successCheckpoints: ApiJsonValue | undefined;
    errorCheckpoints: ApiJsonValue | undefined;
    glossaryEntries: ApiJsonValue | undefined;
    progressSnapshot: ApiJsonValue | undefined;
  }): Promise<MutableJsonRecord> {
    const project_path = request.projectPath;
    const success_checkpoints = this.normalize_checkpoint_rows(request.successCheckpoints);
    const error_checkpoints = this.normalize_error_checkpoint_rows(
      project_path,
      request.errorCheckpoints,
    );
    const glossary_entries = this.normalize_glossary_entries(request.glossaryEntries);
    const progress_snapshot = this.normalize_nullable_progress_snapshot(request.progressSnapshot);
    const meta = this.read_project_meta(project_path);
    const candidate_result = this.build_next_candidate_rows(
      project_path,
      glossary_entries,
      this.read_number(meta["analysis_candidate_count"], 0),
    );
    await this.commit_runtime_change({
      projectPath: project_path,
      requireExpectedSectionRevisions: false,
      revisionSections: ["analysis"],
      source: "analysis_batch_update",
      updatedSections: ["analysis"],
      sections: {
        analysis: {
          payloadMode: "canonical-delta",
          data: this.build_analysis_section_delta(
            progress_snapshot ?? this.normalize_object(meta["analysis_extras"]),
            candidate_result.count,
          ) as unknown as ApiJsonValue,
        },
      },
      buildOperations: (revision_context) => {
        const operations: DatabaseOperation[] = [];
        if (success_checkpoints.length > 0 || error_checkpoints.length > 0) {
          operations.push(
            this.op("upsertAnalysisItemCheckpoints", {
              projectPath: project_path,
              checkpoints: [
                ...success_checkpoints,
                ...error_checkpoints,
              ] as unknown as DatabaseJsonValue,
            }),
          );
        }
        if (candidate_result.rows.length > 0) {
          operations.push(
            this.op("upsertAnalysisCandidateAggregates", {
              projectPath: project_path,
              aggregates: candidate_result.rows as unknown as DatabaseJsonValue,
            }),
          );
        }
        operations.push(
          this.op("upsertMetaEntries", {
            projectPath: project_path,
            meta: {
              ...(progress_snapshot === null ? {} : { analysis_extras: progress_snapshot }),
              analysis_candidate_count: candidate_result.count,
            } as unknown as DatabaseJsonValue,
          }),
          ...this.write_coordinator.build_section_revision_operations(revision_context),
        );
        return operations;
      },
    });
    return {
      inserted_count: glossary_entries.length,
      analysis_candidate_count: candidate_result.count,
      section_revisions: this.build_section_revisions(project_path, ["analysis"]),
    };
  }

  public async reset_analysis_state(request: {
    projectPath: string;
    expectedSectionRevisions?: ApiJsonValue;
    requireExpectedSectionRevisions: boolean;
    source: string;
    analysisExtras: MutableJsonRecord;
    analysisCandidateCount?: number;
    sectionData?: MutableJsonRecord;
  }): Promise<ProjectWriteResult> {
    const meta: MutableJsonRecord = {
      analysis_extras: request.analysisExtras as unknown as ApiJsonValue,
    };
    if (request.analysisCandidateCount !== undefined) {
      meta["analysis_candidate_count"] = Math.max(0, Math.trunc(request.analysisCandidateCount));
    }
    return await this.commit_runtime_change({
      projectPath: request.projectPath,
      expectedSectionRevisions: request.expectedSectionRevisions,
      requireExpectedSectionRevisions: request.requireExpectedSectionRevisions,
      revisionSections: ["analysis"],
      source: request.source,
      updatedSections: ["analysis"],
      sections:
        request.sectionData === undefined
          ? undefined
          : {
              analysis: {
                payloadMode: "canonical-delta",
                data: request.sectionData as unknown as ApiJsonValue,
              },
            },
      buildOperations: (revision_context) => [
        this.op("upsertMetaEntries", {
          projectPath: request.projectPath,
          meta: meta as unknown as DatabaseJsonValue,
        }),
        this.op("deleteAnalysisItemCheckpoints", { projectPath: request.projectPath }),
        this.op("clearAnalysisCandidateAggregates", { projectPath: request.projectPath }),
        ...this.write_coordinator.build_section_revision_operations(revision_context),
      ],
    });
  }

  public async restore_failed_analysis_checkpoints_for_continue(request: {
    projectPath: string;
    checkpoints: ApiJsonValue | undefined;
  }): Promise<MutableJsonRecord> {
    const checkpoints = this.normalize_checkpoint_rows(request.checkpoints).map((checkpoint) => ({
      ...checkpoint,
      status: "NONE",
      error_count: 0,
    }));
    if (checkpoints.length === 0) {
      return { restored_count: 0 };
    }
    const meta = this.read_project_meta(request.projectPath);
    await this.commit_runtime_change({
      projectPath: request.projectPath,
      requireExpectedSectionRevisions: false,
      revisionSections: ["analysis"],
      source: "analysis_continue_restore",
      updatedSections: ["analysis"],
      sections: {
        analysis: {
          payloadMode: "canonical-delta",
          data: this.build_analysis_section_delta(
            this.normalize_object(meta["analysis_extras"]),
            this.read_number(meta["analysis_candidate_count"], 0),
          ) as unknown as ApiJsonValue,
        },
      },
      buildOperations: (revision_context) => [
        this.op("upsertAnalysisItemCheckpoints", {
          projectPath: request.projectPath,
          checkpoints: checkpoints as unknown as DatabaseJsonValue,
        }),
        ...this.write_coordinator.build_section_revision_operations(revision_context),
      ],
    });
    return {
      restored_count: checkpoints.length,
      section_revisions: this.build_section_revisions(request.projectPath, ["analysis"]),
    };
  }

  private async commit_runtime_change(request: RuntimeCommitRequest): Promise<ProjectWriteResult> {
    const revision_context = request.requireExpectedSectionRevisions
      ? this.write_coordinator.assert_expected_section_revisions(
          request.projectPath,
          request.expectedSectionRevisions,
          request.revisionSections,
        )
      : {
          project_path: request.projectPath,
          meta: this.read_project_meta(request.projectPath),
          sections: request.revisionSections,
        };
    const operations = request.buildOperations(revision_context);
    this.database.execute_transaction(operations);
    const change_request: ProjectWriteChangeRequest = {
      projectPath: request.projectPath,
      source: request.source,
      updatedSections: request.updatedSections,
      ...(request.sections === undefined
        ? {}
        : {
            sections: request.sections as Partial<
              Record<ProjectDataSection, ProjectChangeSectionPayload>
            >,
          }),
    };
    await this.write_coordinator.publish_app_events_for_committed_change(change_request);
    return this.write_coordinator.publish_project_data_change(change_request);
  }

  private build_next_candidate_rows(
    project_path: string,
    glossary_entries: MutableJsonRecord[],
    current_count: number,
  ): { rows: MutableJsonRecord[]; count: number } {
    const normalized_entries = glossary_entries.filter((entry) => {
      const src = String(entry["src"] ?? "").trim();
      const dst = String(entry["dst"] ?? "").trim();
      return src !== "" && dst !== "";
    });
    if (normalized_entries.length === 0) {
      return { rows: [], count: Math.max(0, current_count) };
    }
    const touched_srcs = [
      ...new Set(normalized_entries.map((entry) => String(entry["src"] ?? "").trim())),
    ];
    const aggregate = new Map<string, MutableJsonRecord>();
    for (const row of this.get_candidate_aggregate_by_srcs(project_path, touched_srcs)) {
      const src = String(row["src"] ?? "").trim();
      if (src !== "") {
        aggregate.set(src, {
          ...row,
          dst_votes: this.normalize_vote_map(row["dst_votes"]),
          info_votes: this.normalize_vote_map(row["info_votes"]),
        });
      }
    }
    const previous_touched_count = count_analysis_glossary_candidates(aggregate.values());
    const now = new Date().toISOString();
    for (const entry of normalized_entries) {
      const src = String(entry["src"] ?? "").trim();
      const dst = String(entry["dst"] ?? "").trim();
      if (src === "" || dst === "") {
        continue;
      }
      const current =
        aggregate.get(src) ??
        ({
          src,
          dst_votes: {},
          info_votes: {},
          observation_count: 0,
          first_seen_at: now,
          last_seen_at: now,
          case_sensitive: Boolean(entry["case_sensitive"] ?? false),
        } as MutableJsonRecord);
      const dst_votes = this.normalize_vote_map(current["dst_votes"]);
      const info_votes = this.normalize_vote_map(current["info_votes"]);
      const info = String(entry["info"] ?? "").trim();
      dst_votes[dst] = this.read_number(dst_votes[dst] as ApiJsonValue, 0) + 1;
      if (info !== "") {
        info_votes[info] = this.read_number(info_votes[info] as ApiJsonValue, 0) + 1;
      }
      current["dst_votes"] = dst_votes as unknown as ApiJsonValue;
      current["info_votes"] = info_votes as unknown as ApiJsonValue;
      current["observation_count"] = this.read_number(current["observation_count"], 0) + 1;
      current["last_seen_at"] = now;
      current["case_sensitive"] =
        Boolean(current["case_sensitive"]) || Boolean(entry["case_sensitive"]);
      aggregate.set(src, current);
    }
    const rows = [...aggregate.values()];
    const next_touched_count = count_analysis_glossary_candidates(rows);
    return {
      rows,
      count: Math.max(0, current_count - previous_touched_count + next_touched_count),
    };
  }

  private normalize_vote_map(value: ApiJsonValue | undefined): Record<string, number> {
    if (!this.is_record(value)) {
      return {};
    }
    const result: Record<string, number> = {};
    for (const [key, raw_votes] of Object.entries(value)) {
      const text = String(key).trim();
      const votes = this.read_number(raw_votes, 0);
      if (text !== "" && votes > 0) {
        result[text] = (result[text] ?? 0) + votes;
      }
    }
    return result;
  }

  private normalize_checkpoint_rows(value: ApiJsonValue | undefined): MutableJsonRecord[] {
    if (!Array.isArray(value)) {
      return [];
    }
    const rows: MutableJsonRecord[] = [];
    for (const raw_row of value) {
      if (!this.is_record(raw_row)) {
        continue;
      }
      const item_id = this.read_number(raw_row["item_id"], 0);
      const status = String(raw_row["status"] ?? "");
      if (item_id <= 0 || !(TASK_PROGRESS_STATUSES as readonly string[]).includes(status)) {
        continue;
      }
      rows.push({
        item_id,
        status,
        updated_at: String(raw_row["updated_at"] ?? new Date().toISOString()),
        error_count: this.read_number(raw_row["error_count"], 0),
      });
    }
    return rows;
  }

  private normalize_error_checkpoint_rows(
    project_path: string,
    value: ApiJsonValue | undefined,
  ): MutableJsonRecord[] {
    const existing = new Map<number, MutableJsonRecord>();
    for (const row of this.get_analysis_checkpoints(project_path)) {
      existing.set(this.read_number(row["item_id"], 0), row);
    }
    const now = new Date().toISOString();
    return this.normalize_checkpoint_rows(value).map((row) => {
      const item_id = this.read_number(row["item_id"], 0);
      const previous = existing.get(item_id);
      const previous_error_count =
        previous?.["status"] === "ERROR" ? this.read_number(previous["error_count"], 0) : 0;
      return {
        ...row,
        status: "ERROR",
        updated_at: now,
        error_count: previous_error_count + 1,
      };
    });
  }

  private normalize_glossary_entries(value: ApiJsonValue | undefined): MutableJsonRecord[] {
    if (!Array.isArray(value)) {
      return [];
    }
    const entries: MutableJsonRecord[] = [];
    const seen = new Set<string>();
    for (const raw_entry of value) {
      if (!this.is_record(raw_entry)) {
        continue;
      }
      const src = String(raw_entry["src"] ?? "").trim();
      const dst = String(raw_entry["dst"] ?? "").trim();
      const info = String(raw_entry["info"] ?? "").trim();
      const case_sensitive = Boolean(raw_entry["case_sensitive"] ?? false);
      const key = `${src}\u0000${dst}\u0000${info}\u0000${case_sensitive ? "1" : "0"}`;
      if (src === "" || dst === "" || seen.has(key)) {
        continue;
      }
      seen.add(key);
      entries.push({ src, dst, info, case_sensitive });
    }
    return entries;
  }

  private normalize_nullable_progress_snapshot(
    value: ApiJsonValue | undefined,
  ): MutableJsonRecord | null {
    if (!this.is_record(value)) {
      return null;
    }
    return this.normalize_progress_snapshot(value);
  }

  private normalize_progress_snapshot(value: JsonRecord): MutableJsonRecord {
    return {
      start_time: this.read_float(value["start_time"], 0),
      time: this.read_float(value["time"], 0),
      total_line: this.read_number(value["total_line"], 0),
      line: this.read_number(value["line"], 0),
      processed_line: this.read_number(value["processed_line"], 0),
      error_line: this.read_number(value["error_line"], 0),
      total_tokens: this.read_number(value["total_tokens"], 0),
      total_input_tokens: this.read_number(value["total_input_tokens"], 0),
      total_output_tokens: this.read_number(value["total_output_tokens"], 0),
    };
  }

  private build_analysis_section_delta(
    analysis_extras: MutableJsonRecord,
    candidate_count: number,
  ): MutableJsonRecord {
    const snapshot = this.normalize_progress_snapshot(analysis_extras);
    return {
      extras: snapshot,
      candidate_count: Math.max(0, Math.trunc(candidate_count)),
      status_summary: {
        total_line: this.read_number(snapshot["total_line"], 0),
        processed_line: this.read_number(snapshot["processed_line"], 0),
        error_line: this.read_number(snapshot["error_line"], 0),
        line: this.read_number(snapshot["line"], 0),
      },
    };
  }

  private get_analysis_checkpoints(project_path: string): MutableJsonRecord[] {
    const value = this.database.execute(
      this.op("getAnalysisItemCheckpoints", { projectPath: project_path }),
    );
    return Array.isArray(value)
      ? value.filter((row): row is JsonRecord => this.is_record(row)).map((row) => ({ ...row }))
      : [];
  }

  private get_candidate_aggregate_by_srcs(
    project_path: string,
    srcs: string[],
  ): MutableJsonRecord[] {
    const value = this.database.execute(
      this.op("getAnalysisCandidateAggregatesBySrcs", {
        projectPath: project_path,
        srcs: srcs as unknown as DatabaseJsonValue,
      }),
    );
    return Array.isArray(value)
      ? value.filter((row): row is JsonRecord => this.is_record(row)).map((row) => ({ ...row }))
      : [];
  }

  private build_section_revisions(
    project_path: string,
    sections: ProjectDataSection[],
  ): MutableJsonRecord {
    const meta = this.read_project_meta(project_path);
    const result: MutableJsonRecord = {};
    for (const section of sections) {
      result[section] = get_section_revision(meta, section);
    }
    return result;
  }

  private read_project_meta(project_path: string): MutableJsonRecord {
    return this.normalize_object(
      this.database.execute(this.op("getAllMeta", { projectPath: project_path })),
    );
  }

  private normalize_object(value: ApiJsonValue | undefined): MutableJsonRecord {
    return this.is_record(value) ? { ...value } : {};
  }

  private read_number(value: ApiJsonValue | undefined, fallback: number): number {
    const number_value = Number(value ?? fallback);
    return Number.isFinite(number_value) ? Math.trunc(number_value) : fallback;
  }

  private read_float(value: ApiJsonValue | undefined, fallback: number): number {
    const number_value = Number(value ?? fallback);
    return Number.isFinite(number_value) ? number_value : fallback;
  }

  private is_record(value: unknown): value is JsonRecord {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  private op(name: string, args: Record<string, DatabaseJsonValue>): DatabaseOperation {
    return { name, args };
  }
}
