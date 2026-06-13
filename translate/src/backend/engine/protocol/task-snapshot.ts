import type { ApiJsonValue } from "../../api/api-types";
import type { TaskRunStatus, TranslationScope } from "../../../domain/task";

/** progress 只承载可累加的执行进度，任务差异字段必须放进 extras */
export type TaskProgress = {
  line: number;
  total_line: number;
  processed_line: number;
  error_line: number;
  total_tokens: number;
  total_output_tokens: number;
  total_input_tokens: number;
  time: number;
  start_time: number;
  [key: string]: ApiJsonValue;
};

/** translation extras 承载翻译专属语义。 */
export type TranslationExtras = {
  kind: "translation";
  scope: TranslationScope;
};

/** TaskSnapshot 是 CLI reporter 订阅的唯一公开形状 */
export type TaskSnapshot = {
  run_revision: number;
  task_type: "translation";
  status: TaskRunStatus;
  busy: boolean;
  request_in_flight_count: number;
  progress: TaskProgress;
  extras: TranslationExtras;
};
