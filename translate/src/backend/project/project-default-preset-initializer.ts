import path from "node:path";

import type { AppPathService } from "../app/app-path-service";
import type { AppSettingService } from "../app/app-setting-service";
import type { ApiJsonValue } from "../api/api-types";
import type { DatabaseJsonValue, DatabaseOperation } from "../database/database-types";
import type { LogManager } from "../log/log-manager";
import { t_main_log } from "../log/log-text";
import { NativeFs } from "../../native/native-fs";
import { Prompt } from "../../domain/prompt";
import { QualityRule, type QualityRuleKind, type TextPreserveMode } from "../../domain/quality";
import * as AppErrors from "../../shared/error";
import { JsonTool } from "../../shared/utils/json-tool";

type MutableJsonRecord = Record<string, ApiJsonValue>;

const INITIAL_PRESET_REVISION = 1;
const DEFAULT_TEXT_PRESERVE_MODE = QualityRule.from_json("text_preserve").default_mode;
const LOADED_TEXT_PRESERVE_PRESET_MODE = "custom" satisfies TextPreserveMode;

const QUALITY_DEFAULT_PRESET_DISPLAY_NAMES: Record<QualityRuleKind, string> = {
  glossary: "术语表",
  text_preserve: "文本保护",
  pre_replacement: "译前替换",
  post_replacement: "译后替换",
};

export type ProjectDefaultPresetInitializationResult = {
  operations: DatabaseOperation[];
  loaded_names: string[];
};

export class ProjectDefaultPresetInitializer {
  private readonly app_setting_service: AppSettingService;
  private readonly paths: AppPathService;
  private readonly log_manager: LogManager;
  private readonly native_fs: NativeFs;

  public constructor(
    app_setting_service: AppSettingService,
    paths: AppPathService,
    log_manager: LogManager,
    native_fs: NativeFs,
  ) {
    this.app_setting_service = app_setting_service;
    this.paths = paths;
    this.log_manager = log_manager;
    this.native_fs = native_fs;
  }

  public build_operations(project_path: string): ProjectDefaultPresetInitializationResult {
    const config = this.app_setting_service.read_setting();
    const operations: DatabaseOperation[] = [
      this.op("setMeta", {
        projectPath: project_path,
        key: "text_preserve_mode",
        value: DEFAULT_TEXT_PRESERVE_MODE,
      }),
    ];
    const loaded_names: string[] = [];

    for (const rule of QualityRule.all()) {
      const virtual_id = this.string_value(config[rule.default_preset_setting_key]);
      if (virtual_id === "") {
        continue;
      }
      try {
        const entries = this.read_quality_rule_preset(rule, virtual_id);
        operations.push(...this.build_quality_rule_operations(project_path, rule, entries));
        loaded_names.push(QUALITY_DEFAULT_PRESET_DISPLAY_NAMES[rule.kind]);
      } catch (error) {
        this.log_non_blocking_warning(
          t_main_log("app.diagnostic.default_preset.quality_rule_load_failed"),
          error,
          { preset_directory: rule.preset_directory, virtual_id },
        );
      }
    }

    const translation_prompt = Prompt.translation();
    const prompt_virtual_id = this.string_value(config[translation_prompt.default_preset_setting_key]);
    if (prompt_virtual_id !== "") {
      try {
        const text = this.read_prompt_preset(translation_prompt, prompt_virtual_id);
        operations.push(...this.build_prompt_operations(project_path, translation_prompt, text));
        loaded_names.push("翻译提示词");
      } catch (error) {
        this.log_non_blocking_warning(
          t_main_log("app.diagnostic.default_preset.prompt_load_failed"),
          error,
          { task_type: translation_prompt.kind, virtual_id: prompt_virtual_id },
        );
      }
    }

    return { operations, loaded_names };
  }

  public log_loaded_names(loaded_names: string[]): void {
    if (loaded_names.length === 0) {
      return;
    }
    this.log_manager.info(
      t_main_log("app.log.default_preset_loaded", { NAMES: loaded_names.join(" | ") }),
      { source: "project-lifecycle" },
    );
  }

  private read_quality_rule_preset(rule: QualityRule, virtual_id: string): MutableJsonRecord[] {
    const preset_path = this.resolve_quality_rule_preset_path(rule, virtual_id);
    const data = JsonTool.parseStrict(this.native_fs.read_file(preset_path)) as unknown;
    if (!Array.isArray(data)) {
      throw new AppErrors.RequestValidationError({
        public_details: { filename: path.basename(preset_path) },
      });
    }
    return data.filter((entry): entry is MutableJsonRecord => this.is_record(entry));
  }

  private read_prompt_preset(prompt: ReturnType<typeof Prompt.translation>, virtual_id: string): string {
    const preset_path = this.resolve_prompt_preset_path(prompt, virtual_id);
    return this.native_fs
      .read_text_file(preset_path)
      .replace(/^\uFEFF/u, "")
      .trim();
  }

  private resolve_quality_rule_preset_path(rule: QualityRule, virtual_id: string): string {
    const { source, file_name } = this.split_virtual_id(virtual_id, rule.preset_extension);
    const directory =
      source === "builtin"
        ? this.paths.get_quality_rule_builtin_preset_dir(rule.preset_directory)
        : this.paths.get_quality_rule_user_preset_dir(rule.preset_directory);
    return path.join(directory, file_name);
  }

  private resolve_prompt_preset_path(
    prompt: ReturnType<typeof Prompt.translation>,
    virtual_id: string,
  ): string {
    const { source, file_name } = this.split_virtual_id(virtual_id, prompt.preset_extension);
    const directory =
      source === "builtin"
        ? this.paths.get_prompt_builtin_preset_dir(prompt.kind)
        : this.paths.get_prompt_user_preset_dir(prompt.kind);
    return path.join(directory, file_name);
  }

  private build_quality_rule_operations(
    project_path: string,
    rule: QualityRule,
    entries: MutableJsonRecord[],
  ): DatabaseOperation[] {
    const operations: DatabaseOperation[] = [
      this.op("setRules", {
        projectPath: project_path,
        ruleType: rule.database_type,
        rules: entries as unknown as DatabaseJsonValue,
      }),
    ];
    if (rule.enabled_meta_key !== null) {
      operations.push(
        this.op("setMeta", {
          projectPath: project_path,
          key: rule.enabled_meta_key,
          value: true,
        }),
      );
    }
    if (rule.mode_meta_key !== null) {
      operations.push(
        this.op("setMeta", {
          projectPath: project_path,
          key: rule.mode_meta_key,
          value: LOADED_TEXT_PRESERVE_PRESET_MODE,
        }),
      );
    }
    operations.push(
      this.op("setMeta", {
        projectPath: project_path,
        key: rule.revision_meta_key,
        value: INITIAL_PRESET_REVISION,
      }),
    );
    return operations;
  }

  private build_prompt_operations(
    project_path: string,
    prompt: ReturnType<typeof Prompt.translation>,
    text: string,
  ): DatabaseOperation[] {
    return [
      this.op("setRuleText", {
        projectPath: project_path,
        ruleType: prompt.database_type,
        text,
      }),
      this.op("setMeta", {
        projectPath: project_path,
        key: prompt.enabled_meta_key,
        value: true,
      }),
      this.op("setMeta", {
        projectPath: project_path,
        key: prompt.revision_meta_key,
        value: INITIAL_PRESET_REVISION,
      }),
    ];
  }

  private split_virtual_id(
    virtual_id: string,
    extension: ".json" | ".txt",
  ): { source: "builtin" | "user"; file_name: string } {
    const parts = virtual_id.split(":");
    if (parts.length !== 2 && !(extension === ".json" && parts.length === 3)) {
      throw new AppErrors.RequestValidationError();
    }
    const source = parts[0];
    const file_name = parts.at(-1) ?? "";
    if (source !== "builtin" && source !== "user") {
      throw new AppErrors.RequestValidationError();
    }
    this.ensure_preset_file_name(file_name, extension);
    return { source, file_name };
  }

  private ensure_preset_file_name(file_name: string, extension: ".json" | ".txt"): void {
    const has_path_boundary =
      path.basename(file_name) !== file_name ||
      path.win32.basename(file_name) !== file_name ||
      path.posix.basename(file_name) !== file_name ||
      path.isAbsolute(file_name) ||
      path.win32.isAbsolute(file_name) ||
      path.posix.isAbsolute(file_name);
    if (file_name === "" || has_path_boundary || !file_name.toLowerCase().endsWith(extension)) {
      throw new AppErrors.RequestValidationError();
    }
  }

  private log_non_blocking_warning(
    message: string,
    error: unknown,
    context: Record<string, unknown>,
  ): void {
    this.log_manager.warning(message, {
      source: "project-lifecycle",
      error,
      context,
    });
  }

  private is_record(value: unknown): value is MutableJsonRecord {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  private string_value(value: ApiJsonValue | DatabaseJsonValue | undefined): string {
    return typeof value === "string" ? value : String(value ?? "");
  }

  private op(name: string, args: Record<string, DatabaseJsonValue>): DatabaseOperation {
    return { name, args };
  }
}
