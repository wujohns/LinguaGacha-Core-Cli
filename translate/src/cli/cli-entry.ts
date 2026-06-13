import fs from "node:fs";
import path from "node:path";

import { build_cli_help, write_stderr, write_stdout } from "./cli-output";
import { CLIUsageError, parse_cli_args } from "./cli-parser";

export async function run_cli_entry(argv: string[], app_root: string): Promise<number> {
  try {
    const parse_result = parse_cli_args(argv);
    if (parse_result.kind === "help") {
      write_stdout(build_cli_help());
      return 0;
    }
    if (parse_result.kind === "version") {
      write_stdout(read_cli_version(app_root));
      return 0;
    }
    const { run_translate_cli_command } = await import(
      new URL("./translate-runner.js", import.meta.url).href
    );
    await run_translate_cli_command(app_root, parse_result.command);
    return 0;
  } catch (error) {
    if (error instanceof CLIUsageError) {
      write_stderr(error.message);
      write_stderr(build_cli_help());
      return error.exitCode;
    }
    write_stderr(error instanceof Error ? error.message : String(error));
    return 1;
  }
}

function read_cli_version(app_root: string): string {
  const version_path = path.join(app_root, "version.txt");
  if (fs.existsSync(version_path)) {
    return fs.readFileSync(version_path, "utf-8").trim();
  }
  return "0.0.0";
}
