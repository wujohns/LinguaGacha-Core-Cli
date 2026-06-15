import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: [
      "src/cli/cli-parser.test.ts",
      "src/cli/cli-resource-applier.test.ts",
      "src/cli/cli-status-reporter.test.ts",
    ],
  },
});
