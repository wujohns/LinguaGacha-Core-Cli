import type { JsonRecord } from "../shared/utils/json-tool";
import { UnknownPromptTypeError } from "../shared/error";

/**
 * 集中维护当前模块的稳定常量。
 */
export const PROMPT_KINDS = ["analysis"] as const;

export type PromptKind = (typeof PROMPT_KINDS)[number];
export type PromptDatabaseType = "analysis_prompt";

type PromptModel = {
  database_type: PromptDatabaseType;
  directory_name: "analysis_prompt";
  enabled_meta_key: string;
  revision_meta_key: string;
  default_preset_setting_key: "analysis_custom_prompt_default_preset";
  store_key: PromptKind;
  preset_extension: ".txt";
  template_files: readonly ["base.txt", "prefix.txt", "thinking.txt", "suffix.txt"];
};

const PROMPT_MODEL = {
  analysis: {
    database_type: "analysis_prompt",
    directory_name: "analysis_prompt",
    enabled_meta_key: "analysis_prompt_enable",
    revision_meta_key: "quality_prompt_revision.analysis",
    default_preset_setting_key: "analysis_custom_prompt_default_preset",
    store_key: "analysis",
    preset_extension: ".txt",
    template_files: ["base.txt", "prefix.txt", "thinking.txt", "suffix.txt"],
  },
} as const satisfies Record<PromptKind, PromptModel>;

const PROMPT_KIND_SET = new Set<PromptKind>(PROMPT_KINDS);

/**
 * Prompt 是提示词槽位实体，统一计算目录、rules 表类型、meta key 和项目 query key
 */
export class Prompt {
  public readonly kind: PromptKind;

  private constructor(kind: PromptKind) {
    this.kind = kind;
  }

  public static from_json(payload: unknown): Prompt {
    if (is_prompt_kind(payload)) {
      return new Prompt(payload);
    }
    const record = read_record(payload);
    const value = record["kind"] ?? record["task_type"] ?? record["type"];
    if (is_prompt_kind(value)) {
      return new Prompt(value);
    }
    throw new UnknownPromptTypeError(value);
  }

  public static analysis(): Prompt {
    return new Prompt("analysis");
  }

  public static all(): Prompt[] {
    return PROMPT_KINDS.map((kind) => new Prompt(kind));
  }

  public to_json(): JsonRecord {
    return { kind: this.kind };
  }

  public get database_type(): PromptDatabaseType {
    return PROMPT_MODEL[this.kind].database_type;
  }

  public get directory_name(): string {
    return PROMPT_MODEL[this.kind].directory_name;
  }

  public get enabled_meta_key(): string {
    return PROMPT_MODEL[this.kind].enabled_meta_key;
  }

  public get revision_meta_key(): string {
    return PROMPT_MODEL[this.kind].revision_meta_key;
  }

  public get default_preset_setting_key(): string {
    return PROMPT_MODEL[this.kind].default_preset_setting_key;
  }

  public get store_key(): PromptKind {
    return PROMPT_MODEL[this.kind].store_key;
  }

  public get preset_extension(): ".txt" {
    return PROMPT_MODEL[this.kind].preset_extension;
  }

  public get template_files(): readonly ["base.txt", "prefix.txt", "thinking.txt", "suffix.txt"] {
    return PROMPT_MODEL[this.kind].template_files;
  }

  public normalize_slice(value: unknown): { text: string; enabled: boolean; revision: number } {
    const record = read_record(value);
    return {
      text: String(record["text"] ?? ""),
      enabled: Boolean(record["enabled"]),
      revision: Number(record["revision"] ?? 0),
    };
  }
}

export function is_prompt_kind(value: unknown): value is PromptKind {
  return PROMPT_KIND_SET.has(value as PromptKind);
}

function read_record(value: unknown): JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as JsonRecord)
    : {};
}
