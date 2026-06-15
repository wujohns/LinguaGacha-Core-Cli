import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { ApiStreamListener } from "../backend/api/api-stream-hub";
import type { TranslateCliOptions } from "./cli-parser";
import { CLIJsonStatusReporter } from "./cli-status-reporter";
import { run_translate_job } from "./translate-job-runner";
import type { TranslateRuntimeServices } from "./translate-runtime";

type TaskStartRecord = {
  task_type: string;
  mode: string;
  worker_count?: number;
};

class FakeStreamHub {
  private listeners = new Map<string, Set<ApiStreamListener>>();

  public subscribe(topic: string, listener: ApiStreamListener): () => void {
    let listeners = this.listeners.get(topic);
    if (listeners === undefined) {
      listeners = new Set<ApiStreamListener>();
      this.listeners.set(topic, listeners);
    }
    listeners.add(listener);
    return () => listeners?.delete(listener);
  }

  public publish_done(): void {
    const listeners = this.listeners.get("task.snapshot_changed") ?? new Set<ApiStreamListener>();
    for (const listener of Array.from(listeners)) {
      listener({
        topic: "task.snapshot_changed",
        payload: {
          task: {
            run_revision: 1,
            task_type: "translation",
            status: "done",
            busy: false,
            request_in_flight_count: 0,
            progress: {
              total_line: 1,
              processed_line: 1,
              error_line: 0,
            },
          },
        },
      });
    }
  }
}

class FakeServices {
  public readonly streams = new FakeStreamHub();
  public readonly start_records: TaskStartRecord[] = [];
  public readonly created_projects: unknown[] = [];
  public readonly loaded_projects: unknown[] = [];
  public readonly exported_dirs: string[] = [];
  public restore_failed_count = 0;
  public unloaded = false;
  public transient_overrides: unknown[] = [];

  public readonly appSettingService = {
    read_setting: () => ({}),
    set_transient_overrides: (value: unknown) => {
      this.transient_overrides.push(value);
    },
  };

  public readonly projectLifecycleService = {
    create_project_commit: async (payload: unknown) => {
      this.created_projects.push(payload);
    },
    load_project: async (payload: unknown) => {
      this.loaded_projects.push(payload);
    },
    unload_project: async () => {
      this.unloaded = true;
    },
  };

  public readonly taskService = {
    start_task: async (payload: TaskStartRecord) => {
      this.start_records.push(payload);
      queueMicrotask(() => this.streams.publish_done());
    },
  };

  public readonly translationFileExportService = {
    export_files_to_directory: async (output_dir: string) => {
      this.exported_dirs.push(output_dir);
    },
  };

  public build_expected_section_revisions(): Record<string, number> {
    return { quality: 1, prompts: 1 };
  }

  public async commit_cli_resource_operations(): Promise<void> {
    return undefined;
  }

  public async restore_failed_translation_items_for_continue(): Promise<number> {
    this.restore_failed_count += 1;
    return 0;
  }

  public as_runtime_services(): TranslateRuntimeServices {
    return this as unknown as TranslateRuntimeServices;
  }
}

describe("run_translate_job", () => {
  const temp_dirs: string[] = [];

  afterEach(() => {
    for (const dir of temp_dirs.splice(0)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("rejects mode new when the .lg project already exists", async () => {
    const { command } = create_fixture("new");
    fs.writeFileSync(command.projectPath, "existing");
    const services = new FakeServices();
    const lines: string[] = [];

    await expect(
      run_translate_job(services.as_runtime_services(), command, {
        statusReporter: new CLIJsonStatusReporter({ writeLine: (line) => lines.push(line) }),
      }),
    ).rejects.toThrow("Project already exists");

    expect(services.created_projects).toHaveLength(0);
    expect(services.start_records).toEqual([]);
    expect(services.unloaded).toBe(true);
    expect(JSON.parse(lines.at(-1) ?? "{}")).toMatchObject({
      type: "finished",
      status: "error",
    });
  });

  it("creates, runs, exports, and unloads a new project", async () => {
    const { command } = create_fixture("new");
    const services = new FakeServices();
    const lines: string[] = [];

    await run_translate_job(services.as_runtime_services(), command, {
      statusReporter: new CLIJsonStatusReporter({ writeLine: (line) => lines.push(line) }),
    });

    expect(services.created_projects).toHaveLength(1);
    expect(services.loaded_projects).toEqual([]);
    expect(services.start_records).toHaveLength(1);
	    expect(services.start_records[0]).toMatchObject({
	      task_type: "translation",
	      mode: "new",
	      scope: { kind: "all" },
	      worker_count: 3,
	    });
    expect(services.exported_dirs).toEqual([command.outputDir]);
    expect(services.unloaded).toBe(true);
    expect(services.transient_overrides.at(-1)).toBeNull();
    expect(lines.map((line) => JSON.parse(line)).at(-1)).toMatchObject({
      type: "finished",
      status: "done",
    });
  });

  it("loads existing project and forwards continue and reset modes", async () => {
    for (const mode of ["continue", "reset"] as const) {
      const { command } = create_fixture(mode);
      fs.writeFileSync(command.projectPath, "existing");
      const services = new FakeServices();

      await run_translate_job(services.as_runtime_services(), command, {
        statusReporter: new CLIJsonStatusReporter({ writeLine: () => undefined }),
      });

      expect(services.created_projects).toEqual([]);
      expect(services.loaded_projects).toEqual([{ path: path.resolve(command.projectPath) }]);
      expect(services.restore_failed_count).toBe(mode === "continue" ? 1 : 0);
      expect(services.start_records).toHaveLength(1);
      expect(services.start_records[0]).toMatchObject({
        task_type: "translation",
        mode,
        scope: { kind: "all" },
      });
      expect(services.exported_dirs).toEqual([command.outputDir]);
      expect(services.unloaded).toBe(true);
    }
  });

  function create_fixture(mode: TranslateCliOptions["mode"]): { command: TranslateCliOptions } {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "linguagacha-cli-job-"));
    temp_dirs.push(root);
    const input_path = path.join(root, "input.txt");
    const config_path = path.join(root, "config.json");
    fs.writeFileSync(input_path, "hello");
    fs.writeFileSync(config_path, "{}");
    return {
      command: {
        mode,
        projectPath: path.join(root, "project.lg"),
        configPath: config_path,
        inputPaths: mode === "new" ? [input_path] : [],
	        outputDir: path.join(root, "out"),
	        sourceLanguage: "JA",
	        targetLanguage: "ZH",
	        workerCount: mode === "new" ? 3 : null,
	        limiter: null,
	        resources: {
          promptPath: null,
          glossaryPath: null,
          preReplacementPath: null,
          postReplacementPath: null,
          textPreservePath: null,
        },
      },
    };
  }
});
