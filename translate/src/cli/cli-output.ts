export function build_cli_help(executable_name = "linguagacha-translate"): string {
  return `Usage:
	  ${executable_name} --mode <new|continue|reset> --project <file.lg> --config <config.json> --output-dir <dir> --source-language <code|ALL> --target-language <code> [--input <file-or-dir>...]

Options:
  --mode                 Required; new, continue, or reset
  --project              Required; persistent .lg state file
  --config               Required; LinguaGacha config.json with models and activate_model_id
  --input                Required for new; repeatable source file or directory
  --output-dir           Required; translated output directory
	  --source-language      Required; allows ALL
	  --target-language      Required; does not allow ALL
	  --worker-count         Optional positive integer; local work-unit worker limit
	  --limiter-url          Optional http(s) limiter endpoint; disabled when omitted
	  --limiter-resource     Optional limiter resource name; defaults to default
	  --prompt               Optional .txt translation prompt
  --glossary             Optional .json / .xlsx glossary
  --pre-replacement      Optional .json / .xlsx pre-translation replacements
  --post-replacement     Optional .json / .xlsx post-translation replacements
  --text-preserve        Optional .json / .xlsx text preserve rules

Samples:
  ${executable_name} --mode new --project ./work/game.lg --config ./config.json --input ./game --output-dir ./out --source-language JA --target-language ZH
  ${executable_name} --mode continue --project ./work/game.lg --config ./config.json --output-dir ./out --source-language JA --target-language ZH
`;
}

export function write_stdout(message: string): void {
  process.stdout.write(message.endsWith("\n") ? message : `${message}\n`);
}

export function write_stderr(message: string): void {
  process.stderr.write(message.endsWith("\n") ? message : `${message}\n`);
}
