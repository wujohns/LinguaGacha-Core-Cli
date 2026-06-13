import path from "node:path";
import { fileURLToPath } from "node:url";

import { run_cli_entry } from "./cli/cli-entry";

const module_dir = path.dirname(fileURLToPath(import.meta.url));
const app_root = path.resolve(module_dir, "..");
const exit_code = await run_cli_entry(process.argv.slice(2), app_root);
process.exitCode = exit_code;
