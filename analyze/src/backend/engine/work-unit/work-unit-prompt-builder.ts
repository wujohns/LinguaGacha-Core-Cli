import path from "node:path";

import type { TextQualitySnapshot } from "../../../shared/text/text-types";
import type { LLMMessage } from "../../llm/llm-types";
import { default_native_fs } from "../../../native/native-fs";
import { Prompt } from "../../../domain/prompt";
import { normalize_setting_snapshot } from "../../../domain/setting";
import { format_i18n_message, resolve_i18n_locale, type LocaleKey } from "../../../shared/i18n";
import {
  get_language_display_locale,
  get_prompt_source_language_name,
  get_prompt_target_language_name,
  normalize_language_code,
} from "../../../domain/language";

export interface PromptBuilderConfig {
  app_language?: string;
  source_language?: string;
  target_language?: string;
}

export interface PromptBuildResult {
  messages: LLMMessage[];
  console_log: string[];
}

export class PromptBuilder {
  private static readonly template_cache = new Map<string, string>();

  private readonly app_root: string;
  private readonly config: PromptBuilderConfig;
  private readonly quality_snapshot: TextQualitySnapshot;

  public constructor(
    app_root: string,
    config: PromptBuilderConfig,
    quality_snapshot: TextQualitySnapshot,
  ) {
    const setting_snapshot = normalize_setting_snapshot(config);
    this.app_root = app_root;
    this.config = {
      app_language: setting_snapshot.app_language,
      source_language: setting_snapshot.source_language,
      target_language: setting_snapshot.target_language,
    };
    this.quality_snapshot = quality_snapshot;
  }

  public static reset(): void {
    this.template_cache.clear();
  }

  public async generate_glossary_prompt(srcs: string[]): Promise<PromptBuildResult> {
    const instruction_text = await this.build_glossary_analysis_main();
    const inputs_text = this.build_analysis_inputs(srcs);
    return {
      messages: [
        { role: "system", content: instruction_text },
        { role: "user", content: inputs_text },
      ],
      console_log: [],
    };
  }

  public async build_glossary_analysis_main(): Promise<string> {
    const context = this.resolve_prompt_context();
    const prompt = Prompt.analysis();
    const prefix = await this.read_prompt_text(
      prompt.directory_name,
      context.prompt_language,
      "prefix.txt",
    );
    const base = this.quality_snapshot.analysis_prompt_enable
      ? this.quality_snapshot.analysis_prompt
      : await this.read_prompt_text(prompt.directory_name, context.prompt_language, "base.txt");
    const thinking = await this.read_prompt_text(
      prompt.directory_name,
      context.prompt_language,
      "thinking.txt",
    );
    const suffix = await this.read_prompt_text(
      prompt.directory_name,
      context.prompt_language,
      "suffix.txt",
    );
    return this.join_prompt_sections(prefix, base, thinking, suffix).replaceAll(
      "{target_language}",
      context.target_language,
    );
  }

  public build_analysis_inputs(srcs: string[]): string {
    if (srcs.length === 0) {
      return "";
    }
    return `${this.t("app.prompt.builder_input")}\n${srcs.join("\n")}`;
  }

  public join_prompt_sections(
    prefix: string,
    base: string,
    thinking: string,
    suffix: string,
  ): string {
    const parts = [`${prefix}\n${base}`];
    if (thinking !== "") {
      parts.push(thinking);
    }
    parts.push(suffix);
    return parts.join("\n\n");
  }

  private get_prompt_ui_language(): "zh" | "en" {
    return this.config.app_language === "EN" ? "en" : "zh";
  }

  private t(key: LocaleKey, params: Record<string, string> = {}): string {
    return format_i18n_message(resolve_i18n_locale(this.config.app_language), key, params);
  }

  private resolve_prompt_context(): {
    prompt_language: "zh" | "en";
    source_language: string;
    target_language: string;
  } {
    const prompt_language = this.get_prompt_ui_language();
    const display_locale = get_language_display_locale(this.config.app_language);
    const source_code = normalize_language_code(String(this.config.source_language));
    const target_code = normalize_language_code(String(this.config.target_language));
    return {
      prompt_language,
      source_language: get_prompt_source_language_name(source_code, display_locale),
      target_language: get_prompt_target_language_name(target_code, display_locale),
    };
  }

  private async read_prompt_text(
    task_dir_name: string,
    language: "zh" | "en",
    file_name: string,
  ): Promise<string> {
    const cache_key = `${this.app_root}\u0000${task_dir_name}\u0000${language}\u0000${file_name}`;
    const cached = PromptBuilder.template_cache.get(cache_key);
    if (cached !== undefined) {
      return cached;
    }
    const template_path = path.join(
      this.app_root,
      "resource",
      task_dir_name,
      "template",
      language,
      file_name,
    );
    const text = default_native_fs.read_text_file(template_path).trim();
    PromptBuilder.template_cache.set(cache_key, text);
    return text;
  }
}
