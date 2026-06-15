#!/usr/bin/env node

import { loadConfig, resolveConfigPath } from "./config.js";
import { startLimiterServer } from "./server.js";

async function main() {
  if (process.argv.includes("--help")) {
    process.stdout.write(
      [
        "Usage:",
        "  linguagacha-limiter --config <config.yaml>",
        "",
        "Endpoints:",
        "  GET  /healthz",
        "  GET  /v1/status",
        "  POST /v1/leases/acquire",
        "  POST /v1/leases/release",
        "",
      ].join("\n"),
    );
    return;
  }
  const configPath = resolveConfigPath(process.argv.slice(2));
  const config = loadConfig(configPath);
  const { server } = startLimiterServer(config);
  await new Promise((resolve) => server.once("listening", resolve));
  process.stderr.write(`limiter listening on http://${config.host}:${config.port}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
