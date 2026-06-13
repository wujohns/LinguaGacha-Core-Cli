import type { ApiJsonValue } from "../../api/api-types";
import type { CacheReadPort } from "../../cache/cache-types";
import { ProjectDatabase } from "../../database/database-operations";
import type { DatabaseJsonValue, DatabaseOperation } from "../../database/database-types";
import { ProjectWriteStore } from "../../project/project-write-store";
import { ProjectSessionState } from "../../project/project-session";
import type { JsonRecord, MutableJsonRecord } from "../run/task-run-types";
import type { TaskArtifact } from "../protocol/artifact";
import { QualityRuleSnapshotTool } from "../../../shared/quality/snapshot";
import * as AppErrors from "../../../shared/error";

/**
 * 项目任务存储端口，是 TaskEngine 读写项目任务事实的唯一内部入口
 */
export class ProjectTaskStore {
  private readonly database: ProjectDatabase; // 任务写库也必须经由 ProjectDatabase workflow

  private readonly session_state: ProjectSessionState; // 当前 loaded 工程是任务读写唯一目标

  private readonly cache: CacheReadPort; // 任务启动热读 items / quality / prompts，写库仍只走 ProjectDatabase

  private readonly write_store: ProjectWriteStore; // 任务提交只表达 TaskArtifact 语义，事务与事件由 ProjectWriteStore 统一完成

  /**
   * ProjectTaskStore 只组合现有 TS 权威，不自行持有长期项目缓存
   */
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

  /**
   * 任务启动前读取当前工程上下文，不依赖旧会话缓存
   */
  public get_project_context(_request: JsonRecord): MutableJsonRecord {
    const state = this.session_state.snapshot();
    return {
      loaded: state.loaded,
      project_path: state.projectPath,
      meta: state.loaded ? this.get_all_meta(state.projectPath) : {},
    };
  }

  /**
   * 后台任务长流程显式保留当前工程连接，结束后释放让 .lg 回到单文件稳定态
   */
  public acquire_project_lease(owner: string): () => void {
    return this.database.acquire_project_lease(this.require_loaded_project_path(), owner);
  }

  /**
   * 任务启动时从 `.lg` 读取质量规则和提示词快照，渲染进程缓存不再作为后端任务输入
   */
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

  /**
   * 翻译任务读取条目快照；RESET 只在任务内归零，不把重置写回数据库
   */
  public get_translation_items(request: JsonRecord): MutableJsonRecord {
    const project_path = this.require_loaded_project_path();
    const mode = String(request["mode"] ?? "NEW");
    const items = this.cache.items.readItems().map((item) => {
      if (mode !== "RESET") {
        return item;
      }
      return {
        ...item,
        dst: "",
        status: "NONE",
        retry_count: 0,
      };
    });
    return {
      items: items as unknown as ApiJsonValue,
      meta: this.get_all_meta(project_path),
    };
  }

  /**
   * CLI continue 启动前把上一轮失败项恢复为干净待处理状态。
   */
  public async restore_failed_translation_items_for_continue(): Promise<MutableJsonRecord> {
    const project_path = this.require_loaded_project_path();
    const items = this.cache.items.readItems();
    const failed_items = items.filter((item) => String(item["status"] ?? "") === "ERROR");
    if (failed_items.length === 0) {
      return { restored_count: 0 };
    }
    const restored_ids = failed_items
      .map((item) => this.read_number(item["item_id"] ?? item["id"], 0))
      .filter((item_id) => item_id > 0);
    if (restored_ids.length === 0) {
      return { restored_count: 0 };
    }
    const restored_id_set = new Set(restored_ids);
    const next_items = items.map((item) => {
      const item_id = this.read_number(item["item_id"] ?? item["id"], 0);
      if (!restored_id_set.has(item_id)) {
        return item;
      }
      return {
        ...item,
        dst: "",
        name_dst: null,
        status: "NONE",
        retry_count: 0,
      };
    });
    const extras = {
      ...this.normalize_progress_snapshot(this.normalize_object(this.get_all_meta(project_path)["translation_extras"])),
      ...this.build_translation_progress_from_items(next_items),
    };
    const ack = await this.write_store.apply_translation_item_patches({
      projectPath: project_path,
      items: restored_ids.map((item_id) => ({
        item_id,
        dst: "",
        name_dst: null,
        status: "NONE",
        retry_count: 0,
      })) as unknown as ApiJsonValue,
      translationExtras: extras,
    });
    return {
      restored_count: restored_ids.length,
      restored_item_ids: restored_ids as unknown as ApiJsonValue,
      section_revisions: ack.section_revisions,
    };
  }

  /**
   * artifact 是项目任务事实唯一写入口的公开提交协议，调用方不再接触数据库 operation 形状
   */
  public async commit_artifacts(request: JsonRecord): Promise<MutableJsonRecord> {
    const artifacts = this.normalize_artifacts(request["artifacts"]);
    const progress_snapshot = this.normalize_nullable_progress_snapshot(
      request["progress_snapshot"],
    );
    return await this.commit_translation_artifacts(artifacts, progress_snapshot);
  }

  /**
   * item_updates artifact 同事务写入 items 与 translation_extras。
   */
  private async commit_translation_artifacts(
    artifacts: TaskArtifact[],
    progress_snapshot: MutableJsonRecord | null,
  ): Promise<MutableJsonRecord> {
    const item_updates = artifacts.find((artifact) => artifact.kind === "item_updates");
    if (item_updates === undefined) {
      if (progress_snapshot !== null) {
        return this.update_translation_progress({
          translation_extras: progress_snapshot as unknown as ApiJsonValue,
        });
      }
      return { accepted: true };
    }
    const request = {
      items: item_updates.items,
      translation_extras: (progress_snapshot ?? {}) as unknown as ApiJsonValue,
    };
    return await this.commit_item_updates_batch(request);
  }

  /**
   * 翻译批次提交同事务写入 items 和 translation_extras，再发布后端权威行级增量
   */
  private async commit_item_updates_batch(request: JsonRecord): Promise<MutableJsonRecord> {
    const project_path = this.require_loaded_project_path();
    const extras = this.normalize_object(request["translation_extras"]);
    const ack = await this.write_store.apply_translation_item_patches({
      projectPath: project_path,
      items: request["items"],
      translationExtras: extras,
    });
    return {
      changed_item_ids: ack.changed_item_ids as unknown as ApiJsonValue,
      section_revisions: ack.section_revisions,
    };
  }

  /**
   * 翻译收尾只持久化进度 extras，避免无变更批次仍触发 item patch
   */
  public update_translation_progress(request: JsonRecord): MutableJsonRecord {
    const project_path = this.require_loaded_project_path();
    const extras = this.normalize_object(request["translation_extras"]);
    this.write_store.update_task_progress_meta({
      projectPath: project_path,
      meta: { translation_extras: extras as unknown as ApiJsonValue },
    });
    return { accepted: true };
  }

  private build_translation_progress_from_items(items: MutableJsonRecord[]): MutableJsonRecord {
    let total_line = 0;
    let processed_line = 0;
    let error_line = 0;
    for (const item of items) {
      const status = String(item["status"] ?? "");
      if (status === "NONE" || status === "PROCESSED" || status === "ERROR") {
        total_line += 1;
      }
      if (status === "PROCESSED") {
        processed_line += 1;
      }
      if (status === "ERROR") {
        error_line += 1;
      }
    }
    return {
      total_line,
      processed_line,
      error_line,
      line: processed_line + error_line,
    };
  }

  /**
   * 当前 loaded 工程是所有任务数据 API 的唯一目标
   */
  private require_loaded_project_path(): string {
    const state = this.session_state.snapshot();
    if (!state.loaded || state.projectPath === "") {
      throw new AppErrors.ProjectNotLoadedError();
    }
    return state.projectPath;
  }

  /**
   * 分析提交允许不带进度快照，null 表示只提交 checkpoint / 候选
   */
  private normalize_nullable_progress_snapshot(
    value: ApiJsonValue | undefined,
  ): MutableJsonRecord | null {
    if (!this.is_record(value)) {
      return null;
    }
    return this.normalize_progress_snapshot(value);
  }

  /**
   * 任务进度只接受旧快照固定字段，缺失和坏值统一归零
   */
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

  /**
   * meta 快照统一转成普通对象，避免 undefined 泄漏到内部 JSON
   */
  private get_all_meta(project_path: string): MutableJsonRecord {
    return this.normalize_object(
      this.database.execute(this.op("getAllMeta", { projectPath: project_path })),
    );
  }

  /**
   * work unit 提交的 item payload 必须先收窄为普通对象数组
   */
  private normalize_items(value: ApiJsonValue | undefined): MutableJsonRecord[] {
    return Array.isArray(value)
      ? value
          .filter((item): item is JsonRecord => this.is_record(item))
          .map((item) => ({ ...item }))
      : [];
  }

  /**
   * artifact 在 JSON 边界只接受已知 kind，坏载荷直接丢弃避免写错项目事实
   */
  private normalize_artifacts(value: ApiJsonValue | undefined): TaskArtifact[] {
    if (!Array.isArray(value)) {
      return [];
    }
    const artifacts: TaskArtifact[] = [];
    for (const raw_artifact of value) {
      if (!this.is_record(raw_artifact)) {
        continue;
      }
      if (raw_artifact["kind"] === "item_updates") {
        artifacts.push({
          kind: "item_updates",
          source: "translation",
          items: this.normalize_items(raw_artifact["items"]) as unknown as ApiJsonValue,
        });
      }
    }
    return artifacts;
  }

  /**
   * JSON 普通对象归一集中处理，数组不能被当作 record
   */
  private normalize_object(value: ApiJsonValue | undefined): MutableJsonRecord {
    return this.is_record(value) ? { ...value } : {};
  }

  /**
   * 整数读取用于行号、token 和计数字段，坏值回退到调用方默认值
   */
  private read_number(value: ApiJsonValue | undefined, fallback: number): number {
    const number_value = Number(value ?? fallback);
    return Number.isFinite(number_value) ? Math.trunc(number_value) : fallback;
  }

  /**
   * 浮点读取用于耗时字段，避免任务时间被错误截断
   */
  private read_float(value: ApiJsonValue | undefined, fallback: number): number {
    const number_value = Number(value ?? fallback);
    return Number.isFinite(number_value) ? number_value : fallback;
  }

  /**
   * 类型守卫集中收窄 JSON record，减少调用点重复判断
   */
  private is_record(value: unknown): value is JsonRecord {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  /**
   * database operation 统一构造，保证任务层不散落操作对象形状
   */
  private op(name: string, args: Record<string, DatabaseJsonValue>): DatabaseOperation {
    return { name, args };
  }
}
