import type { ApiJsonValue } from "../../api/api-types";
import type { TaskRunStatus, TaskType } from "../../../domain/task";

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

export type AnalysisExtras = {
  kind: "analysis";
  candidate_count: number;
};

export type TaskSnapshot = {
  run_revision: number;
  task_type: TaskType;
  status: TaskRunStatus;
  busy: boolean;
  request_in_flight_count: number;
  progress: TaskProgress;
  extras: AnalysisExtras;
};
