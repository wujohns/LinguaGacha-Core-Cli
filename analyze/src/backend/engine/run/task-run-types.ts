export {
  TASK_RUN_STATUSES,
  TASK_TYPES,
  is_task_run_status,
  is_task_type,
  normalize_task_type,
  type TaskRunStatus,
  type TaskType,
} from "../../../domain/task";
export type { JsonRecord, MutableJsonRecord } from "../protocol/json";

export interface TaskRunStateSnapshot {
  run_revision: number;
  status: import("../../../domain/task").TaskRunStatus;
  busy: boolean;
  request_in_flight_count: number;
  active_task_type: string;
}
