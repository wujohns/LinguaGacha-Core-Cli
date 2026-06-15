import type { ApiJsonValue } from "../api/api-types";
import type { ProjectOperationGate } from "../project/project-gate";
import { normalize_project_expected_section_revisions } from "../project/project-changes";
import { TaskEngine } from "../engine/core/engine";
import { TaskRunPublisher } from "../engine/run/task-run-publisher";
import { TaskSnapshotBuilder } from "../engine/run/task-snapshot-builder";
import { type JsonRecord, type MutableJsonRecord } from "../engine/run/task-run-types";
import type { StartTaskCommand, StopTaskCommand } from "../engine/protocol/task-command";
import * as AppErrors from "../../shared/error";
import { is_task_start_mode, is_task_type, type TaskStartMode } from "../../domain/task";

export class TaskService {
  private readonly task_engine: TaskEngine;
  private readonly snapshot_builder: TaskSnapshotBuilder;
  private readonly task_run_publisher: TaskRunPublisher;
  private readonly project_operation_gate: ProjectOperationGate;

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

  public async start_task(request: JsonRecord): Promise<MutableJsonRecord> {
    const command = this.normalize_start_command(request);
    const previous_state = this.task_run_publisher.snapshot_state();
    this.project_operation_gate.assert_task_start_allowed();
    await this.task_run_publisher.begin_task(command.task_type);
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

  public async get_task_snapshot(request: JsonRecord): Promise<MutableJsonRecord> {
    return {
      task: (await this.snapshot_builder.build_task_snapshot(request)) as unknown as ApiJsonValue,
    };
  }

  private normalize_start_command(request: JsonRecord): StartTaskCommand {
    const task_type = this.require_task_type(request["task_type"]);
    const mode = this.normalize_mode(request["mode"]);
    const worker_count = this.normalize_optional_positive_integer(request["worker_count"]);
    const expected_section_revisions = this.normalize_expected_section_revisions(
      request["expected_section_revisions"],
    );
    this.assert_expected_section_revisions(expected_section_revisions, ["prompts"]);
    return {
      task_type,
      mode,
      expected_section_revisions: expected_section_revisions ?? {},
      ...(worker_count === undefined ? {} : { worker_count }),
    };
  }

  private normalize_stop_command(request: JsonRecord): StopTaskCommand {
    return { task_type: this.require_task_type(request["task_type"]) };
  }

  private require_task_type(value: ApiJsonValue | undefined): "analysis" {
    if (is_task_type(value) && value === "analysis") {
      return "analysis";
    }
    throw new AppErrors.RequestValidationError();
  }

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

  private normalize_expected_section_revisions(
    value: ApiJsonValue | undefined,
  ): Record<string, number> | null {
    return normalize_project_expected_section_revisions(value);
  }

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
}
