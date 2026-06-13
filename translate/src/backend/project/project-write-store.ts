import type { ApiJsonValue } from "../api/api-types";
import { ProjectDatabase } from "../database/database-operations";
import type { DatabaseJsonValue, DatabaseOperation } from "../database/database-types";
import { Item, type ItemNameField, type ItemStatus } from "../../domain/item";
import type {
  ProjectChangeFilesPayload,
  ProjectChangeItemsPayload,
  ProjectChangePayloadMode,
  ProjectDataSection,
  ProjectWriteResult,
} from "../../shared/project-event";
import * as AppErrors from "../../shared/error";
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

export type TranslationItemPatch = {
  item_id: number; // 任务 artifact 和公开项目行的唯一主键
  patch: {
    dst?: string;
    name_dst?: ItemNameField;
    status?: ItemStatus;
    retry_count?: number;
  };
};

type RuntimeCommitRequest = {
  projectPath: string;
  expectedSectionRevisions?: ApiJsonValue;
  requireExpectedSectionRevisions: boolean;
  revisionSections: ProjectDataSection[];
  source: string;
  updatedSections: ProjectDataSection[];
  buildOperations: (context: ProjectWriteRevisionContext) => DatabaseOperation[];
  items?: Pick<
    ProjectChangeItemsPayload,
    "payloadMode" | "changedIds" | "deleteIds" | "fieldPatch"
  >;
  files?: Pick<ProjectChangeFilesPayload, "payloadMode" | "changedPaths" | "deletePaths">;
  sections?: Partial<
    Record<ProjectDataSection, { payloadMode: ProjectChangePayloadMode; data?: ApiJsonValue }>
  >;
  sectionModes?: Partial<Record<ProjectDataSection, ProjectChangePayloadMode>>;
};

type RuntimeCommitOptions = {
  publishPublic?: boolean;
};

/**
 * ProjectWriteSectionAck 是任务 artifact 写入后回传给 engine 的 revision 确认。
 */
export type ProjectWriteSectionAck = {
  changed_item_ids: number[];
  section_revisions: MutableJsonRecord;
};

/**
 * loaded project 运行态事实的唯一语义写入口。
 */
export class ProjectWriteStore {
  private readonly database: ProjectDatabase; // workflow 是项目事实的物理写入边界

  private readonly write_coordinator: ProjectWriteCoordinator; // coordinator 统一 revision guard 与 committed event 发布

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

  /**
   * 普通翻译 artifact 只按 item_id 局部更新译文字段。
   */
  public async apply_translation_item_patches(request: {
    projectPath: string;
    items: ApiJsonValue | undefined;
    translationExtras: MutableJsonRecord;
  }): Promise<ProjectWriteSectionAck> {
    return await this.apply_task_item_patches({
      projectPath: request.projectPath,
      items: request.items,
      translationExtras: request.translationExtras,
      source: "translation_batch_update",
      updatedSections: ["items"],
    });
  }

  /**
   * 任务进度 meta 仍经由运行态写入口提交，避免任务层直接碰数据库 workflow。
   */
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

  /**
   * 项目设置镜像写入只发布内部 committed event，公开响应仍保持旧空变更语义。
   */
  public async apply_project_settings_meta(request: {
    projectPath: string;
    meta: MutableJsonRecord;
  }): Promise<ProjectWriteResult> {
    return await this.commit_runtime_change(
      {
        projectPath: request.projectPath,
        requireExpectedSectionRevisions: false,
        revisionSections: ["project"],
        source: "settings_alignment",
        updatedSections: ["project"],
        buildOperations: () => [
          this.op("upsertMetaEntries", {
            projectPath: request.projectPath,
            meta: request.meta as unknown as DatabaseJsonValue,
          }),
        ],
      },
      { publishPublic: false },
    );
  }

  /**
   * 质量规则条目和 meta 统一走 quality 运行态写入口。
   */
  public async save_quality_rules(request: {
    projectPath: string;
    expectedSectionRevisions: ApiJsonValue | undefined;
    source: string;
    rule?:
      | {
          databaseType: string;
          entries: JsonRecord[];
        }
      | undefined;
    metaEntries?: MutableJsonRecord;
    revisionKey: string;
  }): Promise<ProjectWriteResult> {
    return await this.commit_runtime_change({
      projectPath: request.projectPath,
      expectedSectionRevisions: request.expectedSectionRevisions,
      requireExpectedSectionRevisions: true,
      revisionSections: ["quality"],
      source: request.source,
      updatedSections: ["quality"],
      buildOperations: (revision_context) => {
        const operations: DatabaseOperation[] = [];
        if (request.rule !== undefined) {
          operations.push(
            this.op("setRules", {
              projectPath: request.projectPath,
              ruleType: request.rule.databaseType,
              rules: request.rule.entries as unknown as DatabaseJsonValue,
            }),
          );
        }
        for (const [key, value] of Object.entries(request.metaEntries ?? {})) {
          operations.push(this.op("setMeta", { projectPath: request.projectPath, key, value }));
        }
        operations.push(
          this.op("setMeta", {
            projectPath: request.projectPath,
            key: request.revisionKey,
            value: get_section_revision(revision_context.meta, "quality") + 1,
          }),
        );
        return operations;
      },
    });
  }

  /**
   * 工程提示词写入由 prompts section 独立提交。
   */
  public async save_quality_prompt(request: {
    projectPath: string;
    expectedSectionRevisions: ApiJsonValue | undefined;
    promptRuleType: string;
    text: string;
    revisionKey: string;
    enabledMetaKey?: string;
    enabled?: boolean;
  }): Promise<ProjectWriteResult> {
    return await this.commit_runtime_change({
      projectPath: request.projectPath,
      expectedSectionRevisions: request.expectedSectionRevisions,
      requireExpectedSectionRevisions: true,
      revisionSections: ["prompts"],
      source: "quality_prompt_save",
      updatedSections: ["prompts"],
      buildOperations: (revision_context) => {
        const operations: DatabaseOperation[] = [
          this.op("setRuleText", {
            projectPath: request.projectPath,
            ruleType: request.promptRuleType,
            text: request.text,
          }),
          this.op("setMeta", {
            projectPath: request.projectPath,
            key: request.revisionKey,
            value: get_section_revision(revision_context.meta, "prompts") + 1,
          }),
        ];
        if (request.enabledMetaKey !== undefined && request.enabled !== undefined) {
          operations.push(
            this.op("setMeta", {
              projectPath: request.projectPath,
              key: request.enabledMetaKey,
              value: request.enabled,
            }),
          );
        }
        return operations;
      },
    });
  }

  /**
   * 任务 artifact item patch 共享同一写入链路和进度 meta 更新。
   */
  private async apply_task_item_patches(request: {
    projectPath: string;
    items: ApiJsonValue | undefined;
    translationExtras: MutableJsonRecord;
    source: string;
    updatedSections: ProjectDataSection[];
  }): Promise<ProjectWriteSectionAck> {
    const patches = this.normalize_translation_item_patches(request.items);
    this.assert_patch_targets_exist(request.projectPath, patches);
    const changed_item_ids = patches.map((patch) => patch.item_id);
    await this.commit_runtime_change({
      projectPath: request.projectPath,
      requireExpectedSectionRevisions: false,
      revisionSections: request.updatedSections,
      source: request.source,
      updatedSections: request.updatedSections,
      items: {
        payloadMode: "canonical-delta",
        changedIds: changed_item_ids,
      },
      buildOperations: (revision_context) => [
        this.op("patchItemTranslationFields", {
          projectPath: request.projectPath,
          patches: this.to_database_translation_patches(patches),
        }),
        this.op("upsertMetaEntries", {
          projectPath: request.projectPath,
          meta: {
            translation_extras: request.translationExtras as unknown as ApiJsonValue,
          } as unknown as DatabaseJsonValue,
        }),
        ...this.write_coordinator.build_section_revision_operations(revision_context),
      ],
    });
    return {
      changed_item_ids,
      section_revisions: this.build_section_revisions(request.projectPath, request.updatedSections),
    };
  }

  private async commit_runtime_change(
    request: RuntimeCommitRequest,
    options: RuntimeCommitOptions = {},
  ): Promise<ProjectWriteResult> {
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
      ...(request.items === undefined ? {} : { items: request.items }),
      ...(request.files === undefined ? {} : { files: request.files }),
      ...(request.sections === undefined ? {} : { sections: request.sections }),
      ...(request.sectionModes === undefined ? {} : { sectionModes: request.sectionModes }),
    };
    await this.write_coordinator.publish_app_events_for_committed_change(change_request);
    if (options.publishPublic === false) {
      return this.empty_project_write_result();
    }
    return this.write_coordinator.publish_project_data_change(change_request);
  }

  /**
   * 复用写入协调器的空结果，保持无变化写入响应形状一致。
   */
  private empty_project_write_result(): ProjectWriteResult {
    return this.write_coordinator.empty_project_write_result();
  }

  private normalize_translation_item_patches(
    value: ApiJsonValue | undefined,
  ): TranslationItemPatch[] {
    if (!Array.isArray(value) || value.length === 0) {
      throw new AppErrors.InternalInvariantError({
        diagnostic_context: { reason: "empty_translation_item_patch" },
      });
    }
    const patches: TranslationItemPatch[] = [];
    const seen = new Set<number>();
    for (const raw_item of value) {
      if (!this.is_record(raw_item)) {
        throw new AppErrors.InternalInvariantError({
          diagnostic_context: { reason: "invalid_translation_item_patch" },
        });
      }
      const item_id = this.read_positive_item_id(
        raw_item["item_id"],
        "invalid_translation_item_id",
      );
      if (seen.has(item_id)) {
        throw new AppErrors.InternalInvariantError({
          diagnostic_context: { reason: "duplicate_translation_item_patch", item_id },
        });
      }
      seen.add(item_id);
      const patch: TranslationItemPatch["patch"] = {};
      if (Object.prototype.hasOwnProperty.call(raw_item, "dst")) {
        if (typeof raw_item["dst"] !== "string") {
          throw new AppErrors.InternalInvariantError({
            diagnostic_context: { reason: "invalid_translation_dst", item_id },
          });
        }
        patch.dst = raw_item["dst"];
      }
      if (Object.prototype.hasOwnProperty.call(raw_item, "name_dst")) {
        patch.name_dst = Item.normalize_name_field(raw_item["name_dst"]);
      }
      if (Object.prototype.hasOwnProperty.call(raw_item, "status")) {
        patch.status = Item.normalize_status(raw_item["status"]);
      }
      if (Object.prototype.hasOwnProperty.call(raw_item, "retry_count")) {
        patch.retry_count = this.read_non_negative_integer_or_throw(
          raw_item["retry_count"],
          "invalid_translation_retry_count",
          item_id,
        );
      }
      if (Object.keys(patch).length === 0) {
        throw new AppErrors.InternalInvariantError({
          diagnostic_context: { reason: "empty_translation_item_patch", item_id },
        });
      }
      patches.push({ item_id, patch });
    }
    return patches;
  }

  private assert_patch_targets_exist(project_path: string, patches: TranslationItemPatch[]): void {
    const rows = this.database.execute(
      this.op("getItemWriteFactsByIds", {
        projectPath: project_path,
        itemIds: patches.map((patch) => patch.item_id) as unknown as DatabaseJsonValue,
      }),
    );
    const existing_ids = new Set<number>();
    if (Array.isArray(rows)) {
      for (const row of rows) {
        if (this.is_record(row)) {
          const item_id = this.read_number(row["id"], 0);
          if (item_id > 0) {
            existing_ids.add(item_id);
          }
        }
      }
    }
    for (const patch of patches) {
      if (!existing_ids.has(patch.item_id)) {
        throw new AppErrors.InternalInvariantError({
          diagnostic_context: {
            reason: "translation_patch_item_not_found",
            item_id: patch.item_id,
          },
        });
      }
    }
  }

  private to_database_translation_patches(patches: TranslationItemPatch[]): DatabaseJsonValue {
    return patches.map((patch) => ({
      id: patch.item_id,
      patch: patch.patch as unknown as DatabaseJsonValue,
    })) as unknown as DatabaseJsonValue;
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

  private read_positive_item_id(value: ApiJsonValue | undefined, reason: string): number {
    const item_id = this.read_number(value, 0);
    if (!Number.isInteger(item_id) || item_id <= 0) {
      throw new AppErrors.InternalInvariantError({
        diagnostic_context: { reason },
      });
    }
    return item_id;
  }

  private read_non_negative_integer_or_throw(
    value: ApiJsonValue | undefined,
    reason: string,
    item_id: number,
  ): number {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      throw new AppErrors.InternalInvariantError({
        diagnostic_context: { reason, item_id },
      });
    }
    return Math.trunc(value);
  }

  private read_number(value: ApiJsonValue | undefined, fallback: number): number {
    const number_value = Number(value ?? fallback);
    return Number.isFinite(number_value) ? Math.trunc(number_value) : fallback;
  }

  private is_record(value: unknown): value is JsonRecord {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  private op(name: string, args: Record<string, DatabaseJsonValue>): DatabaseOperation {
    return { name, args };
  }
}
