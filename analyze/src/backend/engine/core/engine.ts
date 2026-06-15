import crypto from "node:crypto";

import { resolve_active_model } from "../../model/model-config-resolver";
import type { ApiJsonValue } from "../../api/api-types";
import { TaskRunPublisher } from "../run/task-run-publisher";
import type { JsonRecord, MutableJsonRecord, TaskType } from "../run/task-run-types";
import { ProjectTaskStore } from "../store/project-task-store";
import { TaskArtifactCommitter } from "../store/task-artifact-committer";
import type { WorkUnitExecutor } from "../work-unit/work-unit-executor";
import type { StartTaskCommand, StopTaskCommand } from "../protocol/task-command";
import type { TaskStartMode } from "../../../domain/task";
import type { WorkUnitExecutionResult } from "../protocol/work-unit-result";
import { PromptBuilder } from "../work-unit/work-unit-prompt-builder";
import type {
  AnalysisWorkUnitResult,
  TaskEngineOptions,
  TaskProgressSnapshot,
  TaskRunHandle,
} from "./engine-options";
import type {
  AnalysisCommitEntry,
  AnalysisContext,
  TaskItemRecord,
} from "../planning/task-plan-types";
import { ModelKeyLeasePool } from "./model-key-lease-pool";
import { TaskPipeline } from "./pipeline-runner";
import { TaskProgressSnapshotTool } from "./progress-accumulator";
import { RunCoordinator } from "./run-coordinator";
import { TaskLogReplay } from "./log-replay";
import { is_task_skipped_item_status } from "../../../domain/task";
import { TextQualitySnapshotTool } from "../../../shared/text/text-types";
import * as AppErrors from "../../../shared/error";

const ANALYSIS_RETRY_LIMIT = 2;
const DEFAULT_ANALYSIS_WORKER_COUNT = 1;

interface TaskRunContext {
  config_snapshot: MutableJsonRecord;
  model: MutableJsonRecord;
}

export class TaskEngine {
  private readonly app_root: string;
  private readonly task_store: ProjectTaskStore;
  private readonly artifact_committer: TaskArtifactCommitter;
  private readonly task_run_publisher: TaskRunPublisher;
  private readonly executor_client: WorkUnitExecutor;
  private readonly task_planner: TaskEngineOptions["taskPlanner"];
  private readonly app_setting_service: TaskEngineOptions["AppSettingService"];
  private readonly run_coordinator: RunCoordinator;
  private readonly log_replay: TaskLogReplay;
  private readonly model_key_lease_pool = new ModelKeyLeasePool();
  private request_in_flight_count = 0;

  public constructor(options: TaskEngineOptions) {
    this.app_root = options.appRoot;
    this.task_store = options.taskStore;
    this.artifact_committer = new TaskArtifactCommitter(options.taskStore);
    this.task_run_publisher = options.taskRunPublisher;
    this.executor_client = options.executorClient;
    this.task_planner = options.taskPlanner;
    this.app_setting_service = options.AppSettingService;
    this.run_coordinator = new RunCoordinator(options.taskRunPublisher);
    this.log_replay = new TaskLogReplay(options.logManager);
  }

  public async start(command: StartTaskCommand): Promise<void> {
    const handle = this.run_coordinator.begin(command.task_type);
    void this.run_analysis(handle, command);
  }

  public async stop(command: StopTaskCommand): Promise<boolean> {
    return await this.run_coordinator.request_stop(command.task_type);
  }

  private async run_analysis(handle: TaskRunHandle, command: StartTaskCommand): Promise<void> {
    let final_status: "done" | "idle" | "error" = "done";
    let app_language: unknown = "ZH";
    let release_database_lease: (() => void) | null = null;
    const legacy_mode = this.to_legacy_mode(command.mode);
    try {
      await this.emit_status(handle.task_type, "running", true);
      release_database_lease = this.task_store.acquire_project_lease(
        `task:${handle.run_id}:analysis`,
      );
      const run_context = this.resolve_task_run_context();
      app_language = run_context.config_snapshot["app_language"];
      const quality_snapshot = this.task_store.build_quality_snapshot();
      await this.log_task_run_start("analysis", run_context, quality_snapshot, app_language);
      if (legacy_mode === "NEW" || legacy_mode === "RESET") {
        await this.task_store.reset_analysis_progress({});
      }
      const payload = this.task_store.get_analysis_context({});
      const all_items = this.normalize_record_list(payload["items"]);
      const checkpoints = this.normalize_record_list(payload["checkpoints"]);
      const meta = this.normalize_record(payload["meta"]);
      const contexts = await this.task_planner.build_analysis_contexts(
        all_items,
        checkpoints,
        run_context.model,
        handle.signal,
      );
      let progress = this.build_analysis_progress(legacy_mode, all_items, checkpoints, meta);
      await this.update_analysis_progress_if_current(handle, progress);
      await this.emit_progress(handle.task_type);
      const pipeline = new TaskPipeline<AnalysisContext, AnalysisCommitEntry>({
        worker_count: this.resolve_worker_count(command.worker_count),
        signal: handle.signal,
        execute: (context, signal) =>
          this.execute_analysis_context(handle, context, run_context, quality_snapshot, signal),
        commit: async (entries) => {
          progress = await this.commit_analysis_entries(handle, entries, progress);
        },
      });
      await pipeline.run(contexts);
      if (handle.signal.aborted) {
        final_status = "idle";
      }
      progress = TaskProgressSnapshotTool.with_elapsed(progress);
      await this.update_analysis_progress_if_current(handle, progress);
    } catch (error) {
      final_status = handle.signal.aborted ? "idle" : "error";
      if (!handle.signal.aborted) {
        this.log_replay.task_error("分析任务执行失败。", error);
      }
    } finally {
      this.log_replay.task_run_finish(final_status, app_language);
      await this.finish_run(handle, final_status);
      release_database_lease?.();
    }
  }

  private async execute_analysis_context(
    handle: TaskRunHandle,
    context: AnalysisContext,
    run_context: TaskRunContext,
    quality_snapshot: ApiJsonValue,
    signal: AbortSignal,
  ) {
    const result = await this.call_analysis_executor(handle, signal, () =>
      this.executor_client
        .execute_unit(
          {
            run_id: handle.run_id,
            unit_id: context.work_unit_id,
            kind: "analysis",
            model: this.model_key_lease_pool.lease_model(
              run_context.model,
            ) as unknown as ApiJsonValue,
            config_snapshot: run_context.config_snapshot as unknown as ApiJsonValue,
            quality_snapshot,
            payload: {
              file_path: context.file_path,
              items: context.items as unknown as ApiJsonValue,
            },
            diagnostics: {
              retry_count: context.retry_count,
            },
          },
          signal,
        )
        .then((unit_result) => this.to_analysis_work_unit_result(unit_result)),
    );
    this.log_replay.work_unit_logs(result.logs);
    return this.build_analysis_worker_result(context, result);
  }

  private async call_analysis_executor<T>(
    handle: TaskRunHandle,
    signal: AbortSignal,
    callback: () => Promise<T>,
  ): Promise<T> {
    if (signal.aborted) {
      throw signal.reason instanceof Error ? signal.reason : new Error("Analysis task stopped.");
    }
    await this.change_request_in_flight_count(handle.task_type, 1);
    try {
      return await callback();
    } finally {
      await this.change_request_in_flight_count(handle.task_type, -1);
    }
  }

  private to_analysis_work_unit_result(result: WorkUnitExecutionResult): AnalysisWorkUnitResult {
    if (result.kind !== "analysis" || result.output.kind !== "analysis") {
      throw new AppErrors.WorkerExecutionFailedError({
        diagnostic_context: {
          expected_kind: "analysis",
          result_kind: result.kind,
          output_kind: result.output.kind,
        },
      });
    }
    return {
      success: result.outcome === "success",
      stopped: result.outcome === "stopped",
      input_tokens: result.metrics.input_tokens,
      output_tokens: result.metrics.output_tokens,
      glossary_entries: this.normalize_record_list(result.output.glossary_entries),
      logs: result.logs,
    };
  }

  private build_analysis_worker_result(context: AnalysisContext, result: AnalysisWorkUnitResult) {
    if (result.stopped) {
      return { commit_entries: [], retry_contexts: [] };
    }
    if (result.success) {
      return {
        commit_entries: [
          {
            success_checkpoints: this.build_analysis_checkpoints(context, "PROCESSED"),
            error_checkpoints: [],
            glossary_entries: result.glossary_entries,
            input_tokens: result.input_tokens,
            output_tokens: result.output_tokens,
            processed_delta: context.items.length,
            error_delta: 0,
          },
        ],
        retry_contexts: [],
      };
    }
    if (context.retry_count < ANALYSIS_RETRY_LIMIT) {
      return {
        commit_entries: [
          {
            success_checkpoints: [],
            error_checkpoints: [],
            glossary_entries: [],
            input_tokens: result.input_tokens,
            output_tokens: result.output_tokens,
            processed_delta: 0,
            error_delta: 0,
          },
        ],
        retry_contexts: [
          { ...context, work_unit_id: crypto.randomUUID(), retry_count: context.retry_count + 1 },
        ],
      };
    }
    return {
      commit_entries: [
        {
          success_checkpoints: [],
          error_checkpoints: this.build_analysis_checkpoints(context, "ERROR"),
          glossary_entries: [],
          input_tokens: result.input_tokens,
          output_tokens: result.output_tokens,
          processed_delta: 0,
          error_delta: context.items.length,
        },
      ],
      retry_contexts: [],
    };
  }

  private async commit_analysis_entries(
    handle: TaskRunHandle,
    entries: AnalysisCommitEntry[],
    progress: TaskProgressSnapshot,
  ): Promise<TaskProgressSnapshot> {
    if (!this.run_coordinator.is_current(handle.run_id) || entries.length === 0) {
      return progress;
    }
    let next_progress = progress;
    for (const entry of entries) {
      next_progress = TaskProgressSnapshotTool.add_tokens(
        next_progress,
        entry.input_tokens,
        entry.output_tokens,
      );
      next_progress = TaskProgressSnapshotTool.with_counts(next_progress, {
        processed_line: next_progress.processed_line + entry.processed_delta,
        error_line: next_progress.error_line + entry.error_delta,
      });
    }
    next_progress = TaskProgressSnapshotTool.with_elapsed(next_progress);
    await this.artifact_committer.commit(
      "analysis",
      [
        {
          kind: "analysis_checkpoints",
          checkpoints: entries.flatMap((entry) => [
            ...entry.success_checkpoints,
            ...entry.error_checkpoints,
          ]) as unknown as ApiJsonValue,
        },
        {
          kind: "analysis_candidates",
          entries: entries.flatMap((entry) => entry.glossary_entries) as unknown as ApiJsonValue,
        },
      ],
      TaskProgressSnapshotTool.to_record(next_progress),
    );
    await this.emit_progress(handle.task_type);
    return next_progress;
  }

  private build_analysis_progress(
    mode: string,
    items: TaskItemRecord[],
    checkpoints: MutableJsonRecord[],
    meta: MutableJsonRecord,
  ): TaskProgressSnapshot {
    const checkpoint_status_by_id = this.build_checkpoint_status_map(checkpoints);
    const analyzable_items = items.filter((item) => this.is_analyzable_item(item));
    const processed_line = analyzable_items.filter(
      (item) => checkpoint_status_by_id.get(this.read_item_id(item)) === "PROCESSED",
    ).length;
    const error_line = analyzable_items.filter(
      (item) => checkpoint_status_by_id.get(this.read_item_id(item)) === "ERROR",
    ).length;
    const previous =
      mode === "CONTINUE"
        ? TaskProgressSnapshotTool.from_record(meta["analysis_extras"])
        : TaskProgressSnapshotTool.empty();
    return TaskProgressSnapshotTool.with_counts(
      {
        ...previous,
        start_time:
          mode === "CONTINUE" && previous.time > 0
            ? Date.now() / 1000 - previous.time
            : Date.now() / 1000,
      },
      { total_line: analyzable_items.length, processed_line, error_line },
    );
  }

  private async update_analysis_progress_if_current(
    handle: TaskRunHandle,
    progress: TaskProgressSnapshot,
  ): Promise<void> {
    if (!this.run_coordinator.is_current(handle.run_id)) {
      return;
    }
    this.task_store.update_analysis_progress({
      analysis_extras: TaskProgressSnapshotTool.to_record(progress) as unknown as ApiJsonValue,
    });
  }

  private async finish_run(
    handle: TaskRunHandle,
    status: "idle" | "done" | "error",
  ): Promise<void> {
    this.request_in_flight_count = 0;
    await this.run_coordinator.finish(handle, status);
  }

  private async emit_status(
    task_type: TaskType,
    status: "idle" | "requested" | "running" | "stopping" | "done" | "error",
    busy: boolean,
  ): Promise<void> {
    await this.task_run_publisher.publish_status(task_type, status, busy);
  }

  private emit_progress(task_type: TaskType): Promise<void> {
    return this.task_run_publisher.publish_progress_committed(task_type);
  }

  private async change_request_in_flight_count(task_type: TaskType, delta: number): Promise<void> {
    this.request_in_flight_count = Math.max(0, this.request_in_flight_count + delta);
    this.task_run_publisher.publish_request_pressure(task_type, this.request_in_flight_count);
  }

  private resolve_task_run_context(): TaskRunContext {
    const config_snapshot = this.app_setting_service.read_setting();
    const model = resolve_active_model(config_snapshot);
    if (model === null) {
      throw new AppErrors.ModelNotFoundError();
    }
    return { config_snapshot, model };
  }

  private async log_task_run_start(
    task_type: TaskType,
    run_context: TaskRunContext,
    quality_snapshot: ApiJsonValue,
    app_language: unknown,
  ): Promise<void> {
    const prompt_text = await this.build_task_start_prompt(
      task_type,
      run_context,
      quality_snapshot,
    );
    this.log_replay.task_run_start(run_context.model, app_language, prompt_text);
  }

  private async build_task_start_prompt(
    _task_type: TaskType,
    run_context: TaskRunContext,
    quality_snapshot: ApiJsonValue,
  ): Promise<string | null> {
    if (String(run_context.model["api_format"] ?? "") === "SakuraLLM") {
      return null;
    }
    const builder = new PromptBuilder(
      this.app_root,
      {
        app_language: this.read_optional_string(run_context.config_snapshot["app_language"]),
        source_language: this.read_optional_string(run_context.config_snapshot["source_language"]),
        target_language: this.read_optional_string(run_context.config_snapshot["target_language"]),
      },
      TextQualitySnapshotTool.from_api_value(quality_snapshot),
    );
    return await builder.build_glossary_analysis_main();
  }

  private build_analysis_checkpoints(
    context: AnalysisContext,
    status: "PROCESSED" | "ERROR",
  ): MutableJsonRecord[] {
    const updated_at = new Date().toISOString();
    return context.items.map((item) => ({
      item_id: item.item_id,
      status,
      updated_at,
      error_count: status === "ERROR" ? 1 : 0,
    }));
  }

  private build_checkpoint_status_map(checkpoints: MutableJsonRecord[]): Map<number, string> {
    const result = new Map<number, string>();
    for (const checkpoint of checkpoints) {
      const item_id = this.read_number(checkpoint["item_id"], 0);
      const status = String(checkpoint["status"] ?? "");
      if (item_id > 0 && (status === "NONE" || status === "PROCESSED" || status === "ERROR")) {
        result.set(item_id, status);
      }
    }
    return result;
  }

  private is_analyzable_item(item: TaskItemRecord): boolean {
    return (
      !is_task_skipped_item_status(this.read_status(item)) &&
      String(item["src"] ?? "").trim() !== ""
    );
  }

  private to_legacy_mode(mode: TaskStartMode | string): string {
    switch (mode) {
      case "continue":
        return "CONTINUE";
      case "reset":
        return "RESET";
      default:
        return "NEW";
    }
  }

  private resolve_worker_count(value: unknown): number {
    const worker_count = Number(value ?? DEFAULT_ANALYSIS_WORKER_COUNT);
    return Number.isFinite(worker_count)
      ? Math.max(1, Math.trunc(worker_count))
      : DEFAULT_ANALYSIS_WORKER_COUNT;
  }

  private read_item_id(item: TaskItemRecord): number {
    return this.read_number(item["id"] ?? item["item_id"], 0);
  }

  private read_status(item: TaskItemRecord): string {
    return String(item["status"] ?? "NONE");
  }

  private normalize_record_list(value: ApiJsonValue | undefined): MutableJsonRecord[] {
    if (!Array.isArray(value)) {
      return [];
    }
    return value
      .filter((item): item is JsonRecord => {
        return typeof item === "object" && item !== null && !Array.isArray(item);
      })
      .map((item) => ({ ...item }));
  }

  private normalize_record(value: ApiJsonValue | undefined): MutableJsonRecord {
    return typeof value === "object" && value !== null && !Array.isArray(value) ? { ...value } : {};
  }

  private read_number(value: ApiJsonValue | undefined, fallback: number): number {
    const number_value = Number(value ?? fallback);
    return Number.isFinite(number_value) ? Math.trunc(number_value) : fallback;
  }

  private read_optional_string(value: ApiJsonValue | undefined): string | undefined {
    return typeof value === "string" ? value : undefined;
  }
}
