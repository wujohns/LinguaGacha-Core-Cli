import type { ApiJsonValue } from "../backend/api/api-types";
import type { TaskRunStatus } from "../domain/task";
import { JsonTool } from "../shared/utils/json-tool";

type NowProvider = () => Date;
type JsonLineWriter = (line: string) => void;

export interface CLIProgressStats {
  total: number;
  skipped: number;
  failed: number;
  completed: number;
  pending: number;
  percent: number;
}

export interface CLIJsonStatusReporterOptions {
  now?: NowProvider;
  writeLine: JsonLineWriter;
}

interface CLIProgressInput {
  status: string;
  progress: Record<string, ApiJsonValue>;
}

export class CLIJsonStatusReporter {
  private readonly now: NowProvider;
  private readonly write_line: JsonLineWriter;
  private started = false;
  private finished = false;
  private last_progress_key: string | null = null;

  public constructor(options: CLIJsonStatusReporterOptions) {
    this.now = options.now ?? (() => new Date());
    this.write_line = options.writeLine;
  }

  public emit_started(): void {
    if (this.started) {
      return;
    }
    this.started = true;
    this.write_event({
      type: "started",
      command: "translate",
      timestamp: this.timestamp(),
    });
  }

  public emit_progress(snapshot: CLIProgressInput): void {
    const stats = build_cli_progress_stats(snapshot.progress);
    if (this.last_progress_key === null && is_empty_stats(stats)) {
      return;
    }
    const progress_key = JsonTool.stringifyStrict(stats);
    if (progress_key === this.last_progress_key) {
      return;
    }
    this.last_progress_key = progress_key;
    this.write_event({
      type: "progress",
      command: "translate",
      status: String(snapshot.status),
      timestamp: this.timestamp(),
      stats,
    });
  }

  public emit_finished(status: TaskRunStatus | "error", error?: unknown): void {
    if (this.finished) {
      return;
    }
    this.emit_started();
    this.finished = true;
    const event: Record<string, unknown> = {
      type: "finished",
      command: "translate",
      status,
      timestamp: this.timestamp(),
    };
    if (status === "error" && error !== undefined) {
      event["error"] = {
        message: error instanceof Error ? error.message : String(error),
      };
    }
    this.write_event(event);
  }

  private write_event(event: Record<string, unknown>): void {
    this.write_line(JsonTool.stringifyStrict(event));
  }

  private timestamp(): string {
    return this.now().toISOString();
  }
}

function is_empty_stats(stats: CLIProgressStats): boolean {
  return (
    stats.total === 0 &&
    stats.skipped === 0 &&
    stats.failed === 0 &&
    stats.completed === 0 &&
    stats.pending === 0
  );
}

export function build_cli_progress_stats(progress: Record<string, ApiJsonValue>): CLIProgressStats {
  const total = Math.max(0, read_progress_count(progress["total_line"]));
  const completed = clamp_count(
    read_progress_count(progress["processed_line"]) > 0
      ? read_progress_count(progress["processed_line"])
      : read_progress_count(progress["line"]),
    0,
    total,
  );
  const failed = clamp_count(read_progress_count(progress["error_line"]), 0, total - completed);
  const skipped = 0;
  const pending = Math.max(0, total - skipped - failed - completed);
  const percent = total > 0 ? ((completed + skipped) / total) * 100 : 0;
  return { total, skipped, failed, completed, pending, percent };
}

function read_progress_count(value: ApiJsonValue | undefined): number {
  const number_value = Number(value ?? 0);
  return Number.isFinite(number_value) ? Math.floor(number_value) : 0;
}

function clamp_count(value: number, min_value: number, max_value: number): number {
  return Math.min(max_value, Math.max(min_value, value));
}
