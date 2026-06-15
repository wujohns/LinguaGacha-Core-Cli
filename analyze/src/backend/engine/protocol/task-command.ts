import type { TaskStartMode } from "../../../domain/task";

export type StartTaskCommand = {
  task_type: "analysis";
  mode: TaskStartMode;
  expected_section_revisions: Record<string, number>;
  worker_count?: number;
};

export type StopTaskCommand = {
  task_type: "analysis";
};
