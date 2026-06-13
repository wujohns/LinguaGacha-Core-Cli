import path from "node:path";

import {
  ALL_LANGUAGE_CODE,
  SOURCE_LANGUAGE_CODES,
  TARGET_LANGUAGE_CODES,
  normalize_language_code,
  type SourceLanguageCode,
  type TargetLanguageCode,
} from "../domain/language";

export type TranslateCliMode = "new" | "continue" | "reset";

export type TranslateCliParseResult =
  | { kind: "help" }
  | { kind: "version" }
  | { kind: "command"; command: TranslateCliOptions };

export interface TranslateCliOptions {
  mode: TranslateCliMode;
  projectPath: string;
  configPath: string;
  inputPaths: string[];
  outputDir: string;
  sourceLanguage: SourceLanguageCode | typeof ALL_LANGUAGE_CODE;
  targetLanguage: TargetLanguageCode;
  resources: TranslateCliResources;
}

export interface TranslateCliResources {
  promptPath: string | null;
  glossaryPath: string | null;
  preReplacementPath: string | null;
  postReplacementPath: string | null;
  textPreservePath: string | null;
}

export class CLIUsageError extends Error {
  public readonly exitCode = 2;

  public constructor(message: string) {
    super(message);
    this.name = "CLIUsageError";
  }
}

const MODES = new Set<TranslateCliMode>(["new", "continue", "reset"]);
const SOURCE_LANGUAGE_SET = new Set<string>([ALL_LANGUAGE_CODE, ...SOURCE_LANGUAGE_CODES]);
const TARGET_LANGUAGE_SET = new Set<string>(TARGET_LANGUAGE_CODES);

export function parse_cli_args(argv: string[]): TranslateCliParseResult {
  if (argv.length === 0 || argv.includes("--help")) {
    return { kind: "help" };
  }
  if (argv.length === 1 && argv[0] === "--version") {
    return { kind: "version" };
  }
  if (argv[0]?.startsWith("--") !== true) {
    throw new CLIUsageError(`Unknown argument: ${argv[0] ?? ""}`);
  }
  return { kind: "command", command: parse_command_options(argv) };
}

function parse_command_options(tokens: string[]): TranslateCliOptions {
  let mode = "";
  let project_path = "";
  let config_path = "";
  let output_dir = "";
  let source_language = "";
  let target_language = "";
  const input_paths: string[] = [];
  const resources = create_empty_resources();

  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index] ?? "";
    const value = tokens[index + 1];
    if (token === "--mode") {
      mode = read_option_value(token, value);
      index += 1;
    } else if (token === "--project") {
      project_path = read_option_value(token, value);
      index += 1;
    } else if (token === "--config") {
      config_path = read_option_value(token, value);
      index += 1;
    } else if (token === "--input") {
      input_paths.push(read_option_value(token, value));
      index += 1;
    } else if (token === "--output-dir") {
      output_dir = read_option_value(token, value);
      index += 1;
    } else if (token === "--source-language") {
      source_language = read_option_value(token, value);
      index += 1;
    } else if (token === "--target-language") {
      target_language = read_option_value(token, value);
      index += 1;
    } else if (token === "--prompt") {
      resources.promptPath = normalize_resource_path(read_option_value(token, value), token, [
        ".txt",
      ]);
      index += 1;
    } else if (token === "--glossary") {
      resources.glossaryPath = normalize_resource_path(read_option_value(token, value), token, [
        ".json",
        ".xlsx",
      ]);
      index += 1;
    } else if (token === "--pre-replacement") {
      resources.preReplacementPath = normalize_resource_path(
        read_option_value(token, value),
        token,
        [".json", ".xlsx"],
      );
      index += 1;
    } else if (token === "--post-replacement") {
      resources.postReplacementPath = normalize_resource_path(
        read_option_value(token, value),
        token,
        [".json", ".xlsx"],
      );
      index += 1;
    } else if (token === "--text-preserve") {
      resources.textPreservePath = normalize_resource_path(read_option_value(token, value), token, [
        ".json",
        ".xlsx",
      ]);
      index += 1;
    } else {
      throw new CLIUsageError(`Unknown option: ${token}`);
    }
  }

  const parsed_mode = normalize_mode(mode);
  const normalized_input_paths = normalize_input_paths(input_paths);
  if (parsed_mode === "new" && normalized_input_paths.length === 0) {
    throw new CLIUsageError("Missing required option --input");
  }

  return {
    mode: parsed_mode,
    projectPath: require_lg_path(project_path),
    configPath: require_non_empty_text(config_path, "--config"),
    inputPaths: normalized_input_paths,
    outputDir: require_non_empty_text(output_dir, "--output-dir"),
    sourceLanguage: normalize_source_language(source_language),
    targetLanguage: normalize_target_language(target_language),
    resources,
  };
}

function create_empty_resources(): TranslateCliResources {
  return {
    promptPath: null,
    glossaryPath: null,
    preReplacementPath: null,
    postReplacementPath: null,
    textPreservePath: null,
  };
}

function normalize_mode(value: string): TranslateCliMode {
  const mode = value.trim().toLowerCase();
  if (MODES.has(mode as TranslateCliMode)) {
    return mode as TranslateCliMode;
  }
  throw new CLIUsageError(`Unsupported mode: ${value}`);
}

function normalize_resource_path(
  value: string,
  option_name: string,
  extensions: readonly string[],
): string {
  const normalized_value = require_non_empty_text(value, option_name);
  const lower_value = normalized_value.toLowerCase();
  if (!extensions.some((extension) => lower_value.endsWith(extension))) {
    throw new CLIUsageError(`${option_name} only supports ${extensions.join(" / ")} files`);
  }
  return normalized_value;
}

function read_option_value(option_name: string, value: string | undefined): string {
  if (value === undefined || value.startsWith("--")) {
    throw new CLIUsageError(`Missing value for ${option_name}`);
  }
  return value.trim();
}

function normalize_input_paths(values: string[]): string[] {
  return values.map((value) => value.trim()).filter((value) => value !== "");
}

function require_non_empty_text(value: string, option_name: string): string {
  const normalized_value = value.trim();
  if (normalized_value === "") {
    throw new CLIUsageError(`Missing required option ${option_name}`);
  }
  return normalized_value;
}

function require_lg_path(value: string): string {
  const project_path = require_non_empty_text(value, "--project");
  if (path.extname(project_path).toLowerCase() !== ".lg") {
    throw new CLIUsageError("--project must point to a .lg file");
  }
  return project_path;
}

function normalize_source_language(value: string): SourceLanguageCode | typeof ALL_LANGUAGE_CODE {
  const language = normalize_language_code(value);
  if (language !== null && SOURCE_LANGUAGE_SET.has(language)) {
    return language as SourceLanguageCode | typeof ALL_LANGUAGE_CODE;
  }
  throw new CLIUsageError(`Unsupported source language: ${value}`);
}

function normalize_target_language(value: string): TargetLanguageCode {
  const language = normalize_language_code(value);
  if (language !== null && TARGET_LANGUAGE_SET.has(language)) {
    return language as TargetLanguageCode;
  }
  throw new CLIUsageError(`Unsupported target language: ${value}`);
}
