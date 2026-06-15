import { describe, expect, it } from "vitest";

import { CLIUsageError, parse_cli_args } from "./cli-parser";

describe("parse_cli_args", () => {
  it("accepts a full new analyze command", () => {
    const result = parse_cli_args([
      "--mode",
      "new",
      "--project",
      "./work/game.lg",
      "--config",
      "./config.json",
      "--input",
      "./src-a",
      "--input",
      "./src-b",
      "--output-dir",
      "./out",
      "--source-language",
      "JA",
      "--target-language",
      "ZH",
      "--worker-count",
      "6",
      "--limiter-url",
      "http://127.0.0.1:8787/",
      "--limiter-resource",
      "deepseek",
      "--prompt",
      "./analysis-prompt.txt",
    ]);

    expect(result).toEqual({
      kind: "command",
      command: {
        mode: "new",
        projectPath: "./work/game.lg",
        configPath: "./config.json",
        inputPaths: ["./src-a", "./src-b"],
        outputDir: "./out",
        sourceLanguage: "JA",
        targetLanguage: "ZH",
        workerCount: 6,
        limiter: {
          url: "http://127.0.0.1:8787",
          resource: "deepseek",
        },
        resources: {
          promptPath: "./analysis-prompt.txt",
        },
      },
    });
  });

  it("accepts continue without input paths", () => {
    const result = parse_cli_args([
      "--mode",
      "continue",
      "--project",
      "./work/game.lg",
      "--config",
      "./config.json",
      "--output-dir",
      "./out",
      "--source-language",
      "ALL",
      "--target-language",
      "ZH-HANT",
    ]);

    expect(result.kind).toBe("command");
    if (result.kind === "command") {
      expect(result.command.inputPaths).toEqual([]);
      expect(result.command.sourceLanguage).toBe("ALL");
      expect(result.command.targetLanguage).toBe("ZH-HANT");
      expect(result.command.workerCount).toBeNull();
      expect(result.command.limiter).toBeNull();
    }
  });

  it("accepts reset without input paths", () => {
    const result = parse_cli_args([
      "--mode",
      "reset",
      "--project",
      "./work/game.lg",
      "--config",
      "./config.json",
      "--output-dir",
      "./out",
      "--source-language",
      "EN",
      "--target-language",
      "ZH",
    ]);

    expect(result.kind).toBe("command");
    if (result.kind === "command") {
      expect(result.command.mode).toBe("reset");
      expect(result.command.inputPaths).toEqual([]);
    }
  });

  it("returns help and version sentinel results", () => {
    expect(parse_cli_args([])).toEqual({ kind: "help" });
    expect(parse_cli_args(["--help"])).toEqual({ kind: "help" });
    expect(parse_cli_args(["--version"])).toEqual({ kind: "version" });
  });

  it("rejects invalid modes and missing new inputs", () => {
    expect(() =>
      parse_cli_args([
        "--mode",
        "analyze",
        "--project",
        "./work/game.lg",
        "--config",
        "./config.json",
        "--output-dir",
        "./out",
        "--source-language",
        "JA",
        "--target-language",
        "ZH",
      ]),
    ).toThrow(CLIUsageError);

    expect(() =>
      parse_cli_args([
        "--mode",
        "new",
        "--project",
        "./work/game.lg",
        "--config",
        "./config.json",
        "--output-dir",
        "./out",
        "--source-language",
        "JA",
        "--target-language",
        "ZH",
      ]),
    ).toThrow("Missing required option --input");
  });

  it("rejects invalid project, resources, target language, and empty values", () => {
    const base = [
      "--mode",
      "continue",
      "--project",
      "./work/game.lg",
      "--config",
      "./config.json",
      "--output-dir",
      "./out",
      "--source-language",
      "JA",
      "--target-language",
      "ZH",
    ];

    expect(() => parse_cli_args([...base.slice(0, 3), "./work/game.sqlite", ...base.slice(4)])).toThrow(
      "--project must point to a .lg file",
    );
    expect(() => parse_cli_args([...base, "--prompt", "./bad.json"])).toThrow(
      "--prompt only supports .txt files",
    );
    expect(() => parse_cli_args([...base, "--glossary", "./glossary.json"])).toThrow(
      "Unknown option: --glossary",
    );
    expect(() =>
      parse_cli_args([
        "--mode",
        "continue",
        "--project",
        "./work/game.lg",
        "--config",
        "./config.json",
        "--output-dir",
        "./out",
        "--source-language",
        "JA",
        "--target-language",
        "ALL",
      ]),
    ).toThrow("Unsupported target language: ALL");
    expect(() => parse_cli_args([...base, "--prompt"])).toThrow("Missing value for --prompt");
    expect(() => parse_cli_args([...base, "--input", "   "])).not.toThrow();
    expect(() => parse_cli_args([...base, "--worker-count", "0"])).toThrow(
      "--worker-count must be a positive integer",
    );
    expect(() => parse_cli_args([...base, "--limiter-url", "ftp://127.0.0.1"])).toThrow(
      "--limiter-url must be an http(s) URL",
    );
    expect(() => parse_cli_args([...base, "--limiter-resource", "deepseek"])).toThrow(
      "--limiter-url is required when limiter options are provided",
    );
    expect(() => parse_cli_args([...base, "--limiter-lease-ttl-ms", "300000"])).toThrow(
      "Unknown option: --limiter-lease-ttl-ms",
    );
    expect(() => parse_cli_args([...base, "--limiter-acquire-wait-ms", "15000"])).toThrow(
      "Unknown option: --limiter-acquire-wait-ms",
    );
  });
});
