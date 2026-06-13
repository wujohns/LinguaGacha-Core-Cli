import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { LLMClient } from "../backend/llm/llm-client";
import type { LLMRequestBody, LLMRequestResult } from "../backend/llm/llm-types";
import type { TranslateCliOptions } from "./cli-parser";
import { CLIJsonStatusReporter } from "./cli-status-reporter";
import { run_translate_job } from "./translate-job-runner";
import { TranslateRuntime } from "./translate-runtime";

describe("translate CLI integration", () => {
  const temp_dirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();
    for (const dir of temp_dirs.splice(0)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("creates a .lg project, translates with a mock LLM, exports txt, and can continue without reprocessing completed items", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "linguagacha-cli-integration-"));
    temp_dirs.push(root);
    const input_path = path.join(root, "input.txt");
    const config_path = path.join(root, "config.json");
    const project_path = path.join(root, "project.lg");
    const output_dir = path.join(root, "out");
    fs.writeFileSync(input_path, "hello\n", "utf-8");
    fs.writeFileSync(
      config_path,
      JSON.stringify({
        app_language: "EN",
        request_timeout: 10,
        activate_model_id: "mock",
        models: [
          {
            id: "mock",
            type: "CUSTOM",
            api_format: "OpenAI",
            api_key: "test-key",
            api_url: "https://example.invalid/v1",
            model_id: "mock-model",
          },
        ],
      }),
      "utf-8",
    );

    const request_bodies: LLMRequestBody[] = [];
    vi.spyOn(LLMClient.prototype, "request").mockImplementation(async (body) => {
      request_bodies.push(body);
      return {
        response_think: "",
        response_result: "{\"0\":\"你好\"}",
        input_tokens: 3,
        output_tokens: 2,
        cancelled: false,
        timeout: false,
        degraded: false,
      } satisfies LLMRequestResult;
    });

    const runtime = new TranslateRuntime({
      appRoot: process.cwd(),
      configPath: config_path,
      workerExecution: { kind: "in_process" },
    });
    try {
      const services = await runtime.start();
      const jsonl: string[] = [];
      await run_translate_job(services, create_command("new"), {
        statusReporter: new CLIJsonStatusReporter({ writeLine: (line) => jsonl.push(line) }),
      });

      expect(fs.existsSync(project_path)).toBe(true);
      expect(fs.readFileSync(path.join(output_dir, "input.txt"), "utf-8")).toBe("你好");
      expect(fs.readFileSync(path.join(output_dir, "bilingual", "input.txt"), "utf-8")).toBe(
        "hello\n你好",
      );
      expect(request_bodies).toHaveLength(1);
      expect(jsonl.map((line) => JSON.parse(line)).at(-1)).toMatchObject({
        type: "finished",
        status: "done",
      });
    } finally {
      await runtime.stop();
    }

    const continue_runtime = new TranslateRuntime({
      appRoot: process.cwd(),
      configPath: config_path,
      workerExecution: { kind: "in_process" },
    });
    try {
      const services = await continue_runtime.start();
      await run_translate_job(services, create_command("continue"), {
        statusReporter: new CLIJsonStatusReporter({ writeLine: () => undefined }),
      });
      expect(request_bodies).toHaveLength(1);
      expect(fs.existsSync(`${project_path}-wal`)).toBe(false);
      expect(fs.existsSync(`${project_path}-shm`)).toBe(false);
    } finally {
      await continue_runtime.stop();
    }

    function create_command(mode: TranslateCliOptions["mode"]): TranslateCliOptions {
      return {
        mode,
        projectPath: project_path,
        configPath: config_path,
        inputPaths: mode === "new" ? [input_path] : [],
        outputDir: output_dir,
        sourceLanguage: "EN",
        targetLanguage: "ZH",
        resources: {
          promptPath: null,
          glossaryPath: null,
          preReplacementPath: null,
          postReplacementPath: null,
          textPreservePath: null,
        },
      };
    }
  });

  it("continue restores ERROR items and retries them without reprocessing completed items", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "linguagacha-cli-retry-error-"));
    temp_dirs.push(root);
    const input_path = path.join(root, "input.txt");
    const config_path = path.join(root, "config.json");
    const project_path = path.join(root, "project.lg");
    const output_dir = path.join(root, "out");
    fs.writeFileSync(input_path, "ok\nfail\n", "utf-8");
    write_mock_config(config_path);

    const request_bodies: LLMRequestBody[] = [];
    let request_count = 0;
    vi.spyOn(LLMClient.prototype, "request").mockImplementation(async (body) => {
      request_bodies.push(body);
      request_count += 1;
      return {
        response_think: "",
        response_result: request_count === 1 ? "{\"0\":\"好\"}\n{\"1\":\"\"}" : "",
        input_tokens: 3,
        output_tokens: 2,
        cancelled: false,
        timeout: false,
        degraded: false,
      } satisfies LLMRequestResult;
    });

    const runtime = new TranslateRuntime({
      appRoot: process.cwd(),
      configPath: config_path,
      workerExecution: { kind: "in_process" },
    });
    try {
      const services = await runtime.start();
      await run_translate_job(services, create_command("new"), {
        statusReporter: new CLIJsonStatusReporter({ writeLine: () => undefined }),
      });
      expect(request_bodies.length).toBeGreaterThan(1);
      expect(fs.readFileSync(path.join(output_dir, "input.txt"), "utf-8")).toBe("好\nfail");
    } finally {
      await runtime.stop();
    }

    vi.spyOn(LLMClient.prototype, "request").mockImplementation(async (body) => {
      request_bodies.push(body);
      return {
        response_think: "",
        response_result: "{\"0\":\"失败已恢复\"}",
        input_tokens: 3,
        output_tokens: 2,
        cancelled: false,
        timeout: false,
        degraded: false,
      } satisfies LLMRequestResult;
    });

    const continue_runtime = new TranslateRuntime({
      appRoot: process.cwd(),
      configPath: config_path,
      workerExecution: { kind: "in_process" },
    });
    try {
      const services = await continue_runtime.start();
      const before_continue_request_count = request_bodies.length;
      await run_translate_job(services, create_command("continue"), {
        statusReporter: new CLIJsonStatusReporter({ writeLine: () => undefined }),
      });
      expect(request_bodies).toHaveLength(before_continue_request_count + 1);
      expect(fs.readFileSync(path.join(output_dir, "input.txt"), "utf-8")).toBe("好\n失败已恢复");
      expect(fs.existsSync(`${project_path}-wal`)).toBe(false);
      expect(fs.existsSync(`${project_path}-shm`)).toBe(false);
    } finally {
      await continue_runtime.stop();
    }

    function create_command(mode: TranslateCliOptions["mode"]): TranslateCliOptions {
      return {
        mode,
        projectPath: project_path,
        configPath: config_path,
        inputPaths: mode === "new" ? [input_path] : [],
        outputDir: output_dir,
        sourceLanguage: "EN",
        targetLanguage: "ZH",
        resources: {
          promptPath: null,
          glossaryPath: null,
          preReplacementPath: null,
          postReplacementPath: null,
          textPreservePath: null,
        },
      };
    }
  });
});

function write_mock_config(config_path: string): void {
  fs.writeFileSync(
    config_path,
    JSON.stringify({
      app_language: "EN",
      request_timeout: 10,
      activate_model_id: "mock",
      models: [
        {
          id: "mock",
          type: "CUSTOM",
          api_format: "OpenAI",
          api_key: "test-key",
          api_url: "https://example.invalid/v1",
          model_id: "mock-model",
        },
      ],
    }),
    "utf-8",
  );
}
