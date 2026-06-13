import type { TranslateRuntimeServices } from "./translate-runtime";
import type { DatabaseJsonValue, DatabaseOperation } from "../backend/database/database-types";
import { load_quality_rule_entries_from_file } from "../backend/quality/quality-rule-file-io";
import { default_native_fs } from "../native/native-fs";
import { Prompt } from "../domain/prompt";
import { QualityRule, type QualityRuleKind } from "../domain/quality";
import type { TranslateCliOptions } from "./cli-parser";

type CLIRuleResourceSpec = {
  resource_path: string | null;
  rule_kind: QualityRuleKind;
  enabled_meta_key: string | null;
  enabled_meta_value: DatabaseJsonValue;
};

export async function apply_cli_resources(
  services: TranslateRuntimeServices,
  command: TranslateCliOptions,
  project_path: string,
): Promise<void> {
  const operations = await build_cli_resource_operations(command, project_path);
  if (operations.length === 0) {
    return;
  }
  await services.commit_cli_resource_operations(project_path, operations);
}

async function build_cli_resource_operations(
  command: TranslateCliOptions,
  project_path: string,
): Promise<DatabaseOperation[]> {
  const operations: DatabaseOperation[] = [
    ...build_disabled_quality_operations(project_path),
    ...build_disabled_prompt_operations(project_path),
  ];
  operations.push(...(await build_rule_resource_operations(project_path, command)));
  operations.push(...read_prompt_resource_operations(project_path, command));
  operations.push(...build_revision_operations(project_path, command));
  return operations;
}

function build_disabled_quality_operations(project_path: string): DatabaseOperation[] {
  return [
    op("setMeta", { projectPath: project_path, key: "glossary_enable", value: false }),
    op("setMeta", {
      projectPath: project_path,
      key: "pre_translation_replacement_enable",
      value: false,
    }),
    op("setMeta", {
      projectPath: project_path,
      key: "post_translation_replacement_enable",
      value: false,
    }),
    op("setMeta", { projectPath: project_path, key: "text_preserve_mode", value: "off" }),
  ];
}

function build_disabled_prompt_operations(project_path: string): DatabaseOperation[] {
  return [
    op("setMeta", {
      projectPath: project_path,
      key: Prompt.translation().enabled_meta_key,
      value: false,
    }),
  ];
}

async function build_rule_resource_operations(
  project_path: string,
  command: TranslateCliOptions,
): Promise<DatabaseOperation[]> {
  const operations: DatabaseOperation[] = [];
  for (const spec of build_rule_resource_specs(command)) {
    if (spec.resource_path === null) {
      continue;
    }
    const rule = QualityRule.from_json(spec.rule_kind);
    const entries = await load_quality_rule_entries_from_file(spec.resource_path);
    operations.push(
      op("setRules", {
        projectPath: project_path,
        ruleType: rule.database_type,
        rules: entries as unknown as DatabaseJsonValue,
      }),
    );
    if (spec.enabled_meta_key !== null) {
      operations.push(
        op("setMeta", {
          projectPath: project_path,
          key: spec.enabled_meta_key,
          value: spec.enabled_meta_value,
        }),
      );
    }
  }
  return operations;
}

function build_rule_resource_specs(command: TranslateCliOptions): CLIRuleResourceSpec[] {
  return [
    {
      resource_path: command.resources.glossaryPath,
      rule_kind: "glossary",
      enabled_meta_key: "glossary_enable",
      enabled_meta_value: true,
    },
    {
      resource_path: command.resources.preReplacementPath,
      rule_kind: "pre_replacement",
      enabled_meta_key: "pre_translation_replacement_enable",
      enabled_meta_value: true,
    },
    {
      resource_path: command.resources.postReplacementPath,
      rule_kind: "post_replacement",
      enabled_meta_key: "post_translation_replacement_enable",
      enabled_meta_value: true,
    },
    {
      resource_path: command.resources.textPreservePath,
      rule_kind: "text_preserve",
      enabled_meta_key: "text_preserve_mode",
      enabled_meta_value: "custom",
    },
  ];
}

function read_prompt_resource_operations(
  project_path: string,
  command: TranslateCliOptions,
): DatabaseOperation[] {
  if (command.resources.promptPath === null) {
    return [];
  }
  const prompt = Prompt.translation();
  const text = default_native_fs
    .read_text_file(command.resources.promptPath)
    .replace(/^\uFEFF/u, "")
    .trim();
  return [
    op("setRuleText", {
      projectPath: project_path,
      ruleType: prompt.database_type,
      text,
    }),
    op("setMeta", {
      projectPath: project_path,
      key: prompt.enabled_meta_key,
      value: true,
    }),
  ];
}

function build_revision_operations(
  project_path: string,
  command: TranslateCliOptions,
): DatabaseOperation[] {
  const quality_revisions = QualityRule.all().map((rule) =>
    op("setMeta", {
      projectPath: project_path,
      key: rule.revision_meta_key,
      value: 1,
    }),
  );
  const prompt_revisions =
    command.resources.promptPath === null
      ? []
      : [
          op("setMeta", {
            projectPath: project_path,
            key: Prompt.translation().revision_meta_key,
            value: 1,
          }),
        ];
  return [...quality_revisions, ...prompt_revisions];
}

function op(name: string, args: Record<string, DatabaseJsonValue>): DatabaseOperation {
  return { name, args };
}
