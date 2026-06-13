import { CLIJsonStatusReporter } from "./cli-status-reporter";
import type { TranslateCliOptions } from "./cli-parser";
import { write_stdout } from "./cli-output";
import { run_translate_job } from "./translate-job-runner";
import { TranslateRuntime } from "./translate-runtime";

export async function run_translate_cli_command(
  app_root: string,
  command: TranslateCliOptions,
): Promise<void> {
  const runtime = new TranslateRuntime({
    appRoot: app_root,
    configPath: command.configPath,
  });
  try {
    const services = await runtime.start();
    await run_translate_job(services, command, {
      statusReporter: new CLIJsonStatusReporter({ writeLine: write_stdout }),
    });
  } finally {
    await runtime.stop();
  }
}
