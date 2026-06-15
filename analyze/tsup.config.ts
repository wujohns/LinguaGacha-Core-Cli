import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    "analyze-runner": "src/cli/analyze-runner.ts",
    "work-unit-worker-entry": "src/backend/engine/work-unit/work-unit-worker-entry.ts",
    "planning-worker-entry": "src/backend/engine/planning/planning-worker-entry.ts",
  },
  banner: {
    js: "#!/usr/bin/env node",
  },
  clean: true,
  dts: false,
  external: ["node:sqlite"],
  format: ["esm"],
  minify: false,
  outDir: "dist",
  platform: "node",
  shims: false,
  sourcemap: false,
  splitting: false,
  target: "node24",
});
