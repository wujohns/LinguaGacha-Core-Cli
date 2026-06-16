import type { ApiJsonValue } from "../../api/api-types";
import type { CacheReadPort } from "../../cache/cache-types";
import { ProjectDatabase } from "../../database/database-operations";
import type { DatabaseJsonValue, DatabaseOperation } from "../../database/database-types";
import { ProjectWriteStore } from "../../project/project-write-store";
import { ProjectSessionState } from "../../project/project-session";
import type { JsonRecord, MutableJsonRecord } from "../run/task-run-types";
import type { TaskArtifact } from "../protocol/artifact";
import { QualityRuleSnapshotTool } from "../../../shared/quality/snapshot";
import { TASK_PROGRESS_STATUSES } from "../../../domain/task";
import * as AppErrors from "../../../shared/error";

export class ProjectTaskStore {
  private readonly database: ProjectDatabase;
  private readonly session_state: ProjectSessionState;
  private readonly cache: CacheReadPort;
  private readonly write_store: ProjectWriteStore;

  public constructor(
    database: ProjectDatabase,
    session_state: ProjectSessionState,
    _task_run_state: unknown,
    cache: CacheReadPort,
    write_store: ProjectWriteStore,
  ) {
    this.database = database;
    this.session_state = session_state;
    this.cache = cache;
    this.write_store = write_store;
  }

  public acquire_project_lease(owner: string): () => void {
    return this.database.acquire_project_lease(this.require_loaded_project_path(), owner);
  }

  public build_quality_snapshot(): ApiJsonValue {
    const state = this.session_state.snapshot();
    if (!state.loaded || state.projectPath === "") {
      return QualityRuleSnapshotTool.to_json(QualityRuleSnapshotTool.from_json({}));
    }
    return QualityRuleSnapshotTool.to_json(
      QualityRuleSnapshotTool.from_json({
        quality: this.cache.quality.readBlock(),
        prompts: this.cache.prompts.readBlock(),
      }),
    ) as unknown as ApiJsonValue;
  }

  public get_analysis_context(_request: JsonRecord): MutableJsonRecord {
    const project_path = this.require_loaded_project_path();
    return {
      items: this.cache.items.readItems() as unknown as ApiJsonValue,
      checkpoints: this.get_analysis_checkpoints(project_path) as unknown as ApiJsonValue,
      meta: this.get_all_meta(project_path),
    };
  }

  public async reset_analysis_progress(_request: JsonRecord): Promise<MutableJsonRecord> {
    const project_path = this.require_loaded_project_path();
    await this.write_store.reset_analysis_state({
      projectPath: project_path,
      requireExpectedSectionRevisions: false,
      source: "analysis_reset",
      analysisExtras: {},
      analysisCandidateCount: 0,
      sectionData: this.build_analysis_section_delta({}, 0),
    });
    return { accepted: true };
  }

  public async restore_failed_analysis_items_for_continue(): Promise<MutableJsonRecord> {
    const project_path = this.require_loaded_project_path();
    const failed_checkpoints = this.get_analysis_checkpoints(project_path).filter(
      (checkpoint) => String(checkpoint["status"] ?? "") === "ERROR",
    );
    if (failed_checkpoints.length === 0) {
      return { restored_count: 0 };
    }
    const now = new Date().toISOString();
    const restored_checkpoints = failed_checkpoints
      .map((checkpoint) => this.read_number(checkpoint["item_id"], 0))
      .filter((item_id) => item_id > 0)
      .map((item_id) => ({
        item_id,
        status: "NONE",
        updated_at: now,
        error_count: 0,
      }));
    if (restored_checkpoints.length === 0) {
      return { restored_count: 0 };
    }
    return await this.write_store.restore_failed_analysis_checkpoints_for_continue({
      projectPath: project_path,
      checkpoints: restored_checkpoints as unknown as ApiJsonValue,
    });
  }

  public update_analysis_progress(request: JsonRecord): MutableJsonRecord {
    const project_path = this.require_loaded_project_path();
    const snapshot = this.normalize_progress_snapshot(
      this.normalize_object(request["analysis_extras"]),
    );
    this.write_store.update_task_progress_meta({
      projectPath: project_path,
      meta: { analysis_extras: snapshot as unknown as ApiJsonValue },
    });
    const meta = this.get_all_meta(project_path);
    return {
      analysis_extras: snapshot,
      analysis_candidate_count: this.read_number(meta["analysis_candidate_count"], 0),
    };
  }

  public async commit_artifacts(request: JsonRecord): Promise<MutableJsonRecord> {
    const artifacts = this.normalize_artifacts(request["artifacts"]);
    const progress_snapshot = this.normalize_nullable_progress_snapshot(
      request["progress_snapshot"],
    );
    const checkpoints = artifacts.filter((artifact) => artifact.kind === "analysis_checkpoints");
    const candidates = artifacts.filter((artifact) => artifact.kind === "analysis_candidates");
    return await this.write_store.commit_analysis_artifacts({
      projectPath: this.require_loaded_project_path(),
      successCheckpoints: checkpoints.flatMap((artifact) =>
        this.normalize_checkpoint_rows(artifact.checkpoints),
      ) as unknown as ApiJsonValue,
      errorCheckpoints: [] as unknown as ApiJsonValue,
      glossaryEntries: candidates.flatMap((artifact) =>
        this.normalize_glossary_entries(artifact.entries),
      ) as unknown as ApiJsonValue,
      progressSnapshot: (progress_snapshot ?? {}) as unknown as ApiJsonValue,
    });
  }

  private get_analysis_checkpoints(project_path: string): MutableJsonRecord[] {
    const rows = this.database.execute(this.op("getAnalysisItemCheckpoints", { projectPath: project_path }));
    return this.normalize_record_list(rows);
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

  private normalize_artifacts(value: ApiJsonValue | undefined): TaskArtifact[] {
    if (!Array.isArray(value)) {
      return [];
    }
    const artifacts: TaskArtifact[] = [];
    for (const raw_artifact of value) {
      if (!this.is_record(raw_artifact)) {
        continue;
      }
      if (raw_artifact["kind"] === "analysis_checkpoints") {
        artifacts.push({
          kind: "analysis_checkpoints",
          checkpoints: this.normalize_checkpoint_rows(raw_artifact["checkpoints"]) as unknown as ApiJsonValue,
        });
      } else if (raw_artifact["kind"] === "analysis_candidates") {
        artifacts.push({
          kind: "analysis_candidates",
          entries: this.normalize_glossary_entries(raw_artifact["entries"]) as unknown as ApiJsonValue,
        });
      }
    }
    return artifacts;
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

  private get_all_meta(project_path: string): MutableJsonRecord {
    return this.normalize_object(
      this.database.execute(this.op("getAllMeta", { projectPath: project_path })),
    );
  }

  private require_loaded_project_path(): string {
    const state = this.session_state.snapshot();
    if (!state.loaded || state.projectPath === "") {
      throw new AppErrors.ProjectNotLoadedError();
    }
    return state.projectPath;
  }

  private normalize_record_list(value: ApiJsonValue | undefined): MutableJsonRecord[] {
    return Array.isArray(value)
      ? value
          .filter((item): item is JsonRecord => this.is_record(item))
          .map((item) => ({ ...item }))
      : [];
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
