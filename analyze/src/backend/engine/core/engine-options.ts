import type { MutableJsonRecord, TaskType } from "../run/task-run-types";
import { TASK_IDLE_STATUSES as BASE_TASK_IDLE_STATUSES } from "../../../domain/task";
import type { LogManager } from "../../log/log-manager";
import type { AppSettingService } from "../../app/app-setting-service";
import type { TaskRunPublisher } from "../run/task-run-publisher";
import type { ProjectTaskStore } from "../store/project-task-store";
import type { TaskPlanner } from "../planning/task-planner";
import type { WorkUnitExecutor } from "../work-unit/work-unit-executor";
import type { WorkUnitLogEntry } from "../protocol/work-unit";

export const TASK_IDLE_STATUSES = new Set<string>(BASE_TASK_IDLE_STATUSES);

export interface TaskEngineOptions {
  appRoot: string;
  taskStore: ProjectTaskStore;
  taskRunPublisher: TaskRunPublisher;
  executorClient: WorkUnitExecutor;
  taskPlanner: TaskPlanner;
  AppSettingService: AppSettingService;
  logManager: LogManager;
}

export interface TaskRunHandle {
  run_id: string;
  task_type: TaskType;
  signal: AbortSignal;
}

export interface TaskProgressSnapshot {
  start_time: number;
  time: number;
  total_line: number;
  line: number;
  processed_line: number;
  error_line: number;
  total_tokens: number;
  total_input_tokens: number;
  total_output_tokens: number;
}

export interface AnalysisWorkUnitResult {
  success: boolean;
  stopped: boolean;
  input_tokens: number;
  output_tokens: number;
  glossary_entries: MutableJsonRecord[];
  logs?: WorkUnitLogEntry[];
}

export interface TaskPipelineWorkerResult<TContext, TCommit> {
  commit_entries: TCommit[];
  retry_contexts: TContext[];
}
