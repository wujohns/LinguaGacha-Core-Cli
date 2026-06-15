/**
 * 集中维护当前模块的稳定常量。
 */
export const TASK_TYPES = ["analysis"] as const; // CLI 只暴露分析任务

/**
 * 集中维护当前模块的稳定常量。
 */
export const TASK_RUN_STATUSES = [
  "idle",
  "requested",
  "running",
  "stopping",
  "done",
  "error",
] as const;

/**
 * 集中维护当前模块的稳定常量。
 */
export const TASK_START_MODES = ["new", "continue", "reset"] as const;

/**
 * 集中维护当前模块的稳定常量。
 */
export const ANALYSIS_TASK_ACTIVE_STATUSES = ["requested", "running", "stopping"] as const;

/**
 * 集中维护当前模块的稳定常量。
 */
export const TASK_IDLE_STATUSES = ["done", "error", "idle"] as const;

/**
 * 集中维护当前模块的稳定常量。
 */
export const TASK_PROGRESS_STATUSES = ["NONE", "PROCESSED", "ERROR"] as const;

/**
 * 集中维护当前模块的稳定常量。
 */
export const TASK_SKIPPED_ITEM_STATUSES = [
  "EXCLUDED",
  "RULE_SKIPPED",
  "LANGUAGE_SKIPPED",
  "DUPLICATED",
] as const;

export type TaskType = (typeof TASK_TYPES)[number];
export type TaskRunStatus = (typeof TASK_RUN_STATUSES)[number];
export type TaskStartMode = (typeof TASK_START_MODES)[number];
export type TaskIdleStatus = (typeof TASK_IDLE_STATUSES)[number];
export type TaskProgressStatus = (typeof TASK_PROGRESS_STATUSES)[number];

const TASK_TYPE_SET = new Set<string>(TASK_TYPES);
const TASK_RUN_STATUS_SET = new Set<string>(TASK_RUN_STATUSES);
const TASK_START_MODE_SET = new Set<string>(TASK_START_MODES);
const TASK_IDLE_STATUS_SET = new Set<string>(TASK_IDLE_STATUSES);
const TASK_PROGRESS_STATUS_SET = new Set<string>(TASK_PROGRESS_STATUSES);
const TASK_SKIPPED_ITEM_STATUS_SET = new Set<string>(TASK_SKIPPED_ITEM_STATUSES);
const ANALYSIS_TASK_ACTIVE_STATUS_SET = new Set<string>(ANALYSIS_TASK_ACTIVE_STATUSES);

export function is_task_type(value: unknown): value is TaskType {
  return TASK_TYPE_SET.has(String(value));
}

export function is_task_run_status(value: unknown): value is TaskRunStatus {
  return TASK_RUN_STATUS_SET.has(String(value));
}

export function is_task_start_mode(value: unknown): value is TaskStartMode {
  return TASK_START_MODE_SET.has(String(value));
}

export function is_task_idle_status(value: unknown): value is TaskIdleStatus {
  return TASK_IDLE_STATUS_SET.has(String(value));
}

export function is_task_progress_status(value: unknown): value is TaskProgressStatus {
  return TASK_PROGRESS_STATUS_SET.has(value as TaskProgressStatus);
}

export function is_task_skipped_item_status(value: unknown): boolean {
  return TASK_SKIPPED_ITEM_STATUS_SET.has(String(value));
}

export function is_active_analysis_task_status(value: unknown): boolean {
  return ANALYSIS_TASK_ACTIVE_STATUS_SET.has(String(value));
}

export function normalize_task_type(value: unknown, fallback: TaskType = "analysis"): TaskType {
  return is_task_type(value) ? value : fallback;
}
