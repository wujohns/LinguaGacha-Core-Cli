import { describe, expect, it } from "vitest";

import { CLIUsageError, parse_cli_args } from "./cli-parser";

describe("parse_cli_args", () => {
  it("accepts a full new translate command", () => {
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
      "--prompt",
      "./prompt.txt",
      "--glossary",
      "./glossary.json",
      "--pre-replacement",
      "./pre.xlsx",
      "--post-replacement",
      "./post.json",
      "--text-preserve",
      "./preserve.xlsx",
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
        resources: {
          promptPath: "./prompt.txt",
          glossaryPath: "./glossary.json",
          preReplacementPath: "./pre.xlsx",
          postReplacementPath: "./post.json",
          textPreservePath: "./preserve.xlsx",
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

  it("rejects invalid project, resource, target language, and empty values", () => {
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
    expect(() => parse_cli_args([...base, "--glossary", "./bad.txt"])).toThrow(
      "--glossary only supports .json / .xlsx files",
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
  });
});
