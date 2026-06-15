import { CLIJsonStatusReporter } from "./cli-status-reporter";
import type { AnalyzeCliOptions } from "./cli-parser";
import { write_stdout } from "./cli-output";
import { run_analyze_job } from "./analyze-job-runner";
import { AnalyzeRuntime } from "./analyze-runtime";

export async function run_analyze_cli_command(
  app_root: string,
  command: AnalyzeCliOptions,
): Promise<void> {
  const runtime = new AnalyzeRuntime({
    appRoot: app_root,
    configPath: command.configPath,
    workerCount: command.workerCount ?? undefined,
    limiter: command.limiter,
  });
  try {
    const services = await runtime.start();
    await run_analyze_job(services, command, {
      statusReporter: new CLIJsonStatusReporter({ writeLine: write_stdout }),
    });
  } finally {
    await runtime.stop();
  }
}
