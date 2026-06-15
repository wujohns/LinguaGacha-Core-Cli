import type { AnalyzeRuntimeServices } from "./analyze-runtime";
import type { DatabaseJsonValue, DatabaseOperation } from "../backend/database/database-types";
import { default_native_fs } from "../native/native-fs";
import { Prompt } from "../domain/prompt";
import { QualityRule } from "../domain/quality";
import type { AnalyzeCliOptions } from "./cli-parser";

export async function apply_cli_resources(
  services: AnalyzeRuntimeServices,
  command: AnalyzeCliOptions,
  project_path: string,
): Promise<void> {
  const operations = await build_cli_resource_operations(command, project_path);
  if (operations.length === 0) {
    return;
  }
  await services.commit_cli_resource_operations(project_path, operations);
}

async function build_cli_resource_operations(
  command: AnalyzeCliOptions,
  project_path: string,
): Promise<DatabaseOperation[]> {
  const operations: DatabaseOperation[] = [
    ...build_disabled_quality_operations(project_path),
    ...build_disabled_prompt_operations(project_path),
  ];
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
      key: Prompt.analysis().enabled_meta_key,
      value: false,
    }),
  ];
}

function read_prompt_resource_operations(
  project_path: string,
  command: AnalyzeCliOptions,
): DatabaseOperation[] {
  if (command.resources.promptPath === null) {
    return [];
  }
  const prompt = Prompt.analysis();
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
  command: AnalyzeCliOptions,
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
            key: Prompt.analysis().revision_meta_key,
            value: 1,
          }),
        ];
  return [...quality_revisions, ...prompt_revisions];
}

function op(name: string, args: Record<string, DatabaseJsonValue>): DatabaseOperation {
  return { name, args };
}
