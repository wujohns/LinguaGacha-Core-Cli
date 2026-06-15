import fs from "node:fs";
import path from "node:path";

import type { ApiJsonValue } from "../backend/api/api-types";
import type { ApiStreamPayload } from "../backend/api/api-stream-hub";
import { normalize_project_settings_snapshot } from "../domain/setting";
import { is_task_run_status } from "../domain/task";
import type { TaskSnapshot } from "../backend/engine/protocol/task-snapshot";
import type { TranslateCliOptions } from "./cli-parser";
import type { CLIJsonStatusReporter } from "./cli-status-reporter";
import type { TranslateRuntimeServices } from "./translate-runtime";
import { apply_cli_resources } from "./cli-resource-applier";

export interface TranslateJobRunOptions {
  statusReporter: CLIJsonStatusReporter;
}

export async function run_translate_job(
  services: TranslateRuntimeServices,
  command: TranslateCliOptions,
  options: TranslateJobRunOptions,
): Promise<void> {
  options.statusReporter.emit_started();
  let transient_overrides_active = false;
  try {
    assert_existing_paths(command);
    fs.mkdirSync(command.outputDir, { recursive: true });
    await prepare_project(services, command);
    if (command.mode === "continue") {
      await services.restore_failed_translation_items_for_continue();
    }
    services.appSettingService.set_transient_overrides({
      ...build_cli_default_preset_overrides(),
      output_folder_open_on_finish: false,
      source_language: command.sourceLanguage,
      target_language: command.targetLanguage,
    });
    transient_overrides_active = true;
    await start_and_wait_for_translation(services, command, options);
    await services.translationFileExportService.export_files_to_directory(command.outputDir);
    options.statusReporter.emit_finished("done");
  } catch (error) {
    options.statusReporter.emit_finished("error", error);
    throw error;
  } finally {
    if (transient_overrides_active) {
      services.appSettingService.set_transient_overrides(null);
    }
    try {
      await services.projectLifecycleService.unload_project();
    } catch {
      // The database is closed by runtime shutdown; unload failure must not mask the job error.
    }
  }
}

async function prepare_project(
  services: TranslateRuntimeServices,
  command: TranslateCliOptions,
): Promise<void> {
  const project_path = path.resolve(command.projectPath);
  if (command.mode === "new") {
    if (fs.existsSync(project_path)) {
      throw new Error(`Project already exists: ${project_path}`);
    }
    await services.projectLifecycleService.create_project_commit({
      path: project_path,
      source_paths: command.inputPaths as unknown as ApiJsonValue,
      project_settings: build_project_settings(services, command) as unknown as ApiJsonValue,
    });
    await apply_cli_resources(services, command, project_path);
    return;
  }
  if (!fs.existsSync(project_path)) {
    throw new Error(`Project does not exist: ${project_path}`);
  }
  await services.projectLifecycleService.load_project({ path: project_path });
  if (has_any_resource(command)) {
    await apply_cli_resources(services, command, project_path);
  }
}

function build_cli_default_preset_overrides(): Record<string, ApiJsonValue> {
  return {
    glossary_default_preset: "",
    text_preserve_default_preset: "",
    pre_translation_replacement_default_preset: "",
    post_translation_replacement_default_preset: "",
    translation_custom_prompt_default_preset: "",
    analysis_custom_prompt_default_preset: "",
  };
}

function build_project_settings(
  services: TranslateRuntimeServices,
  command: TranslateCliOptions,
): Record<string, ApiJsonValue> {
  return normalize_project_settings_snapshot({
    ...services.appSettingService.read_setting(),
    source_language: command.sourceLanguage,
    target_language: command.targetLanguage,
  }) as unknown as Record<string, ApiJsonValue>;
}

function assert_existing_paths(command: TranslateCliOptions): void {
  if (!fs.existsSync(command.configPath)) {
    throw new Error(`Config file does not exist: ${command.configPath}`);
  }
  if (command.mode === "new") {
    for (const input_path of command.inputPaths) {
      if (!fs.existsSync(input_path)) {
        throw new Error(`Input path does not exist: ${input_path}`);
      }
    }
  }
  for (const resource_path of collect_resource_paths(command)) {
    if (!fs.existsSync(resource_path)) {
      throw new Error(`Resource file does not exist: ${resource_path}`);
    }
  }
}

function collect_resource_paths(command: TranslateCliOptions): string[] {
  return [
    command.resources.promptPath,
    command.resources.glossaryPath,
    command.resources.preReplacementPath,
    command.resources.postReplacementPath,
    command.resources.textPreservePath,
  ].filter((item): item is string => item !== null);
}

function has_any_resource(command: TranslateCliOptions): boolean {
  return collect_resource_paths(command).length > 0;
}

async function start_and_wait_for_translation(
  services: TranslateRuntimeServices,
  command: TranslateCliOptions,
  options: TranslateJobRunOptions,
): Promise<void> {
  const task_waiter = create_task_event_waiter(services, options);
  try {
    await services.taskService.start_task({
      task_type: "translation",
      mode: command.mode,
      scope: { kind: "all" } as unknown as ApiJsonValue,
      ...(command.workerCount === null ? {} : { worker_count: command.workerCount }),
      expected_section_revisions: services.build_expected_section_revisions([
        "quality",
        "prompts",
      ]) as unknown as ApiJsonValue,
    });
    await task_waiter.wait();
  } finally {
    task_waiter.dispose();
  }
}

function create_task_event_waiter(
  services: TranslateRuntimeServices,
  options: TranslateJobRunOptions,
): { wait: () => Promise<void>; dispose: () => void } {
  let resolve_wait: (() => void) | null = null;
  let reject_wait: ((error: Error) => void) | null = null;
  const wait_promise = new Promise<void>((resolve, reject) => {
    resolve_wait = resolve;
    reject_wait = reject;
  });
  const unsubscribe = services.streams.subscribe("task.snapshot_changed", (message) => {
    const snapshot = normalize_task_snapshot_payload(message.payload);
    if (snapshot === null || snapshot.task_type !== "translation") {
      return;
    }
    options.statusReporter.emit_progress(snapshot);
    if (snapshot.status === "done") {
      resolve_wait?.();
    } else if (snapshot.status === "error") {
      reject_wait?.(new Error("Translation task failed"));
    }
  });
  return {
    wait: () => wait_promise,
    dispose: unsubscribe,
  };
}

function normalize_task_snapshot_payload(payload: ApiStreamPayload): TaskSnapshot | null {
  const task = payload["task"];
  if (typeof task !== "object" || task === null || Array.isArray(task)) {
    return null;
  }
  const record = task as Record<string, ApiJsonValue>;
  const task_type = String(record["task_type"] ?? "");
  const status = String(record["status"] ?? "");
  const progress = record["progress"];
  if (
    task_type !== "translation" ||
    !is_task_run_status(status) ||
    typeof progress !== "object" ||
    progress === null ||
    Array.isArray(progress)
  ) {
    return null;
  }
  return {
    run_revision: Number(record["run_revision"] ?? 0),
    task_type,
    status,
    busy: Boolean(record["busy"]),
    request_in_flight_count: Number(record["request_in_flight_count"] ?? 0),
    progress: progress as TaskSnapshot["progress"],
    extras: { kind: "translation", scope: { kind: "all" } },
  };
}
