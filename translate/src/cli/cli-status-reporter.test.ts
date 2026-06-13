import { describe, expect, it } from "vitest";

import { CLIJsonStatusReporter, build_cli_progress_stats } from "./cli-status-reporter";

describe("CLIJsonStatusReporter", () => {
  it("emits machine-readable JSONL events and de-duplicates identical progress", () => {
    const lines: string[] = [];
    const reporter = new CLIJsonStatusReporter({
      now: () => new Date("2026-01-02T03:04:05.000Z"),
      writeLine: (line) => lines.push(line),
    });

    reporter.emit_started();
    reporter.emit_started();
    reporter.emit_progress({
      status: "running",
      progress: { total_line: 10, processed_line: 3, error_line: 1 },
    });
    reporter.emit_progress({
      status: "running",
      progress: { total_line: 10, processed_line: 3, error_line: 1 },
    });
    reporter.emit_finished("done");
    reporter.emit_finished("done");

    expect(lines.map((line) => JSON.parse(line))).toEqual([
      {
        type: "started",
        command: "translate",
        timestamp: "2026-01-02T03:04:05.000Z",
      },
      {
        type: "progress",
        command: "translate",
        status: "running",
        timestamp: "2026-01-02T03:04:05.000Z",
        stats: {
          total: 10,
          skipped: 0,
          failed: 1,
          completed: 3,
          pending: 6,
          percent: 30,
        },
      },
      {
        type: "finished",
        command: "translate",
        status: "done",
        timestamp: "2026-01-02T03:04:05.000Z",
      },
    ]);
  });

  it("emits started before a finished error event", () => {
    const lines: string[] = [];
    const reporter = new CLIJsonStatusReporter({
      now: () => new Date("2026-01-02T03:04:05.000Z"),
      writeLine: (line) => lines.push(line),
    });

    reporter.emit_finished("error", new Error("boom"));

    expect(lines.map((line) => JSON.parse(line))).toEqual([
      {
        type: "started",
        command: "translate",
        timestamp: "2026-01-02T03:04:05.000Z",
      },
      {
        type: "finished",
        command: "translate",
        status: "error",
        timestamp: "2026-01-02T03:04:05.000Z",
        error: { message: "boom" },
      },
    ]);
  });
});

describe("build_cli_progress_stats", () => {
  it("falls back to line when processed_line is absent", () => {
    expect(build_cli_progress_stats({ total_line: 4, line: 2 })).toEqual({
      total: 4,
      skipped: 0,
      failed: 0,
      completed: 2,
      pending: 2,
      percent: 50,
    });
  });
});
