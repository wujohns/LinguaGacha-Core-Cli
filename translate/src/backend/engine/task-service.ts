import type { ApiJsonValue } from "../api/api-types";
import type { ProjectOperationGate } from "../project/project-gate";
import { normalize_project_expected_section_revisions } from "../project/project-changes";
import { TaskEngine } from "../engine/core/engine";
import { TaskRunPublisher } from "../engine/run/task-run-publisher";
import { TaskSnapshotBuilder } from "../engine/run/task-snapshot-builder";
import { type JsonRecord, type MutableJsonRecord } from "../engine/run/task-run-types";
import type { StartTaskCommand, StopTaskCommand } from "../engine/protocol/task-command";
import { TranslationTaskDefinition } from "../engine/definitions/translation/translation-task-definition";
import * as AppErrors from "../../shared/error";
import {
  is_task_start_mode,
  is_task_type,
  type TaskStartMode,
  type TranslationScope,
} from "../../domain/task";

/**
 * 公开 `/api/tasks/*` 的任务服务，负责校验命令、调用 TaskEngine 并组装回执
 */
export class TaskService {
  private readonly task_engine: TaskEngine; // 后台任务生命周期、调度和停止的唯一执行权威

  private readonly snapshot_builder: TaskSnapshotBuilder; // 公开任务快照唯一组装口径，启动回执也复用它

  private readonly task_run_publisher: TaskRunPublisher; // 启动乐观态与失败回滚的唯一出口

  private readonly project_operation_gate: ProjectOperationGate; // 统一判断任务启动是否会撞上 busy 或结构性写入

  private readonly translation_definition = new TranslationTaskDefinition(); // CLI 只保留 translation 任务定义

  /**
   * 注入任务命令依赖，保持公开协议、运行态桥和配置读取边界可测试
   */
  public constructor(
    task_engine: TaskEngine,
    snapshot_builder: TaskSnapshotBuilder,
    task_run_publisher: TaskRunPublisher,
    project_operation_gate: ProjectOperationGate,
    _session_state: unknown,
  ) {
    this.task_engine = task_engine;
    this.snapshot_builder = snapshot_builder;
    this.task_run_publisher = task_run_publisher;
    this.project_operation_gate = project_operation_gate;
  }

  /**
   * 启动任务；公开层只做 JSON 收窄、revision 校验、模型检查和 Engine 命令转交
   */
  public async start_task(request: JsonRecord): Promise<MutableJsonRecord> {
    const command = this.normalize_start_command(request);
    const previous_state = this.task_run_publisher.snapshot_state();
    // assert_task_start_allowed 与 begin_task 之间不能插入 await，保证通过 gate 后立即写入 busy。
    this.project_operation_gate.assert_task_start_allowed();
    await this.task_run_publisher.begin_task(command.task_type, command.scope);
    try {
      await this.task_engine.start(command);
    } catch (error) {
      await this.task_run_publisher.restore(previous_state);
      throw error;
    }
    return {
      accepted: true,
      task: (await this.snapshot_builder.build_task_snapshot({
        task_type: command.task_type,
      })) as unknown as ApiJsonValue,
    };
  }

  /**
   * 停止任务；回包必须读取当前真实 snapshot，避免异步终态覆盖旧 stopping。
   */
  public async stop_task(request: JsonRecord): Promise<MutableJsonRecord> {
    const command = this.normalize_stop_command(request);
    const previous_state = this.task_run_publisher.snapshot_state();
    try {
      await this.task_engine.stop(command);
    } catch (error) {
      await this.task_run_publisher.restore(previous_state);
      throw error;
    }
    return {
      accepted: true,
      task: (await this.snapshot_builder.build_task_snapshot({
        task_type: command.task_type,
      })) as unknown as ApiJsonValue,
    };
  }

  /**
   * 显式读取任务快照；它是按需查询，不承担订阅职责
   */
  public async get_task_snapshot(request: JsonRecord): Promise<MutableJsonRecord> {
    return {
      task: (await this.snapshot_builder.build_task_snapshot(request)) as unknown as ApiJsonValue,
    };
  }

  /**
   * 任务启动必须声明所有被读取 section 的 revision，避免后台任务基于过期输入运行
   */
  private assert_expected_section_revisions(
    expected: Record<string, number> | null,
    sections: string[],
  ): void {
    if (expected === null) {
      throw new AppErrors.RequestValidationError();
    }
    for (const section of sections) {
      if (!(section in expected)) {
        throw new AppErrors.RequestValidationError({
          public_details: { section },
        });
      }
      this.assert_expected_revision(
        section,
        expected,
        this.snapshot_builder.get_section_revision(section),
      );
    }
  }

  /**
   * 单个 section revision 比对集中在这里，避免错误消息分支重复转换
   */
  private assert_expected_revision(
    section: string,
    expected: Record<string, number>,
    current_revision: number,
  ): void {
    const expected_revision = expected[section] ?? 0;
    if (current_revision !== expected_revision) {
      throw new AppErrors.RevisionConflictError({
        public_details: {
          current_revision,
          expected_revision,
          section,
        },
      });
    }
  }

  /**
   * expected_section_revisions 必须是对象；锁值只接受 JSON number 整数
   */
  private normalize_expected_section_revisions(
    value: ApiJsonValue | undefined,
  ): Record<string, number> | null {
    return normalize_project_expected_section_revisions(value);
  }

  /**
   * JSON record 收窄集中处理，保护数组和 null 不进入业务判断
   */
  private is_record(value: unknown): value is JsonRecord {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  /**
   * 统一 start 请求收窄为 Engine 命令，revision 依赖只由命令语义决定
   */
  private normalize_start_command(request: JsonRecord): StartTaskCommand {
    const task_type = this.require_task_type(request["task_type"]);
    const mode = this.normalize_mode(request["mode"]);
    const worker_count = this.normalize_optional_positive_integer(request["worker_count"]);
    const expected_section_revisions = this.normalize_expected_section_revisions(
      request["expected_section_revisions"],
    );
    const scope = this.normalize_translation_scope(request);
    const command: StartTaskCommand = {
      task_type,
      mode,
      scope,
      expected_section_revisions: expected_section_revisions ?? {},
      ...(worker_count === undefined ? {} : { worker_count }),
    };
    this.assert_expected_section_revisions(
      expected_section_revisions,
      this.translation_definition.revision_dependencies(command),
    );
    return this.translation_definition.normalize_command(command);
  }

  /**
   * stop 请求只允许指定 translation。
   */
  private normalize_stop_command(request: JsonRecord): StopTaskCommand {
    return { task_type: this.require_task_type(request["task_type"]) };
  }

  /**
   * task_type 是公开命令分发根，CLI 只接受 translation。
   */
  private require_task_type(value: ApiJsonValue | undefined): "translation" {
    if (is_task_type(value) && value === "translation") {
      return "translation";
    }
    throw new AppErrors.RequestValidationError();
  }

  /**
   * mode 在公开边界兼收大小写输入，进入 Engine 后固定为小写枚举
   */
  private normalize_mode(value: ApiJsonValue | undefined): TaskStartMode {
    const mode = String(value ?? "new").toLowerCase();
    if (!is_task_start_mode(mode)) {
      throw new AppErrors.RequestValidationError();
    }
    return mode;
  }

  private normalize_optional_positive_integer(value: ApiJsonValue | undefined): number | undefined {
    if (value === undefined || value === null) {
      return undefined;
    }
    const number_value = Number(value);
    if (!Number.isInteger(number_value) || number_value <= 0) {
      throw new AppErrors.RequestValidationError();
    }
    return number_value;
  }

  /**
   * CLI 只支持全量翻译 scope。
   */
  private normalize_translation_scope(request: JsonRecord): TranslationScope {
    const scope = this.is_record(request["scope"]) ? request["scope"] : {};
    const scope_kind = String(scope["kind"] ?? "all");
    if (scope_kind !== "all") {
      throw new AppErrors.RequestValidationError();
    }
    return { kind: "all" };
  }
}
