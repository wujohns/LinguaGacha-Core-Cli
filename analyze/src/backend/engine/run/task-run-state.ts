import type { ApiJsonValue } from "../../api/api-types";
import type { TaskRunStatus, TaskRunStateSnapshot, TaskType } from "./task-run-types";

const IDLE_TASK_TYPE = "idle";

export class TaskRunState {
  private status: TaskRunStatus = "idle";
  private busy = false;
  private active_task_type = IDLE_TASK_TYPE;
  private request_in_flight_count = 0;
  private run_revision = 0;

  public snapshot(): TaskRunStateSnapshot {
    return {
      run_revision: this.run_revision,
      status: this.status,
      busy: this.busy,
      request_in_flight_count: this.request_in_flight_count,
      active_task_type: this.active_task_type,
    };
  }

  public begin_task(task_type: TaskType): void {
    this.status = "requested";
    this.busy = true;
    this.active_task_type = task_type;
    this.bump_run_revision();
  }

  public restore(snapshot: TaskRunStateSnapshot): void {
    this.status = snapshot.status;
    this.busy = snapshot.busy;
    this.request_in_flight_count = snapshot.request_in_flight_count;
    this.active_task_type = snapshot.active_task_type;
    this.bump_run_revision();
  }

  public set_status(task_type: TaskType, status: TaskRunStatus, busy: boolean): void {
    this.status = status;
    this.busy = busy;
    this.active_task_type = this.busy ? task_type : IDLE_TASK_TYPE;
    if (!this.busy) {
      this.request_in_flight_count = 0;
    }
    this.bump_run_revision();
  }

  public mark_progress_committed(task_type: TaskType): void {
    if (this.busy) {
      this.active_task_type = task_type;
    }
    this.bump_run_revision();
  }

  public set_request_in_flight_count(task_type: TaskType, value: number): void {
    if (this.busy) {
      this.active_task_type = task_type;
    }
    this.request_in_flight_count = Math.max(0, this.read_number(value, 0));
    this.bump_run_revision();
  }

  private bump_run_revision(): void {
    this.run_revision += 1;
  }

  private read_number(value: ApiJsonValue | undefined, fallback: number): number {
    const number_value = Number(value ?? fallback);
    return Number.isFinite(number_value) ? Math.trunc(number_value) : fallback;
  }
}
