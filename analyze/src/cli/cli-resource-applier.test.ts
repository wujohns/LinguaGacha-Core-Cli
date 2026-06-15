import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { DatabaseOperation } from "../backend/database/database-types";
import type { AnalyzeCliOptions } from "./cli-parser";
import { apply_cli_resources } from "./cli-resource-applier";
import type { AnalyzeRuntimeServices } from "./analyze-runtime";

class FakeServices {
  public operations: DatabaseOperation[] = [];

  public async commit_cli_resource_operations(
    _project_path: string,
    operations: DatabaseOperation[],
  ): Promise<void> {
    this.operations = operations;
  }

  public as_runtime_services(): AnalyzeRuntimeServices {
    return this as unknown as AnalyzeRuntimeServices;
  }
}

describe("apply_cli_resources", () => {
  const temp_dirs: string[] = [];

  afterEach(() => {
    for (const dir of temp_dirs.splice(0)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("disables implicit quality and prompt resources when no explicit resource is passed", async () => {
    const services = new FakeServices();
    await apply_cli_resources(services.as_runtime_services(), create_command(), "/work/project.lg");

    expect(read_meta(services.operations, "glossary_enable")).toBe(false);
    expect(read_meta(services.operations, "pre_translation_replacement_enable")).toBe(false);
    expect(read_meta(services.operations, "post_translation_replacement_enable")).toBe(false);
    expect(read_meta(services.operations, "text_preserve_mode")).toBe("off");
    expect(read_meta(services.operations, "analysis_prompt_enable")).toBe(false);
    expect(services.operations.some((operation) => operation.name === "setRules")).toBe(false);
  });

  it("imports explicit prompt text and enables only the analysis prompt", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "linguagacha-cli-resource-"));
    temp_dirs.push(root);
    const prompt_path = path.join(root, "prompt.txt");
    fs.writeFileSync(prompt_path, "\uFEFF custom prompt \n");
    const services = new FakeServices();

    await apply_cli_resources(
      services.as_runtime_services(),
      create_command({ promptPath: prompt_path }),
      "/work/project.lg",
    );

    expect(
      services.operations.find(
        (operation) =>
          operation.name === "setRuleText" &&
          operation.args?.["ruleType"] === "analysis_prompt",
      )?.args?.["text"],
    ).toBe("custom prompt");
    expect(read_meta(services.operations, "analysis_prompt_enable")).toBe(true);
    expect(read_meta(services.operations, "quality_prompt_revision.analysis")).toBe(1);
    expect(
      services.operations
        .filter((operation) => operation.name === "setRuleText")
        .map((operation) => operation.args?.["ruleType"]),
    ).toEqual(["analysis_prompt"]);
  });

  function create_command(
    resources: Partial<AnalyzeCliOptions["resources"]> = {},
  ): AnalyzeCliOptions {
    return {
      mode: "new",
      projectPath: "/work/project.lg",
      configPath: "/work/config.json",
      inputPaths: ["/work/input.txt"],
      outputDir: "/work/out",
      sourceLanguage: "JA",
      targetLanguage: "ZH",
      workerCount: null,
      limiter: null,
      resources: {
        promptPath: null,
        ...resources,
      },
    };
  }

  function read_meta(operations: DatabaseOperation[], key: string): unknown {
    return operations
      .filter((operation) => operation.name === "setMeta" && operation.args?.["key"] === key)
      .at(-1)?.args?.["value"];
  }
});
