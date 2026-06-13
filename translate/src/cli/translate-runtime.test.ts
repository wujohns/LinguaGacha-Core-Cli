import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { resolve_active_model } from "../backend/model/model-config-resolver";
import { TranslateRuntime } from "./translate-runtime";

describe("TranslateRuntime config validation", () => {
  const temp_dirs: string[] = [];

  afterEach(() => {
    for (const dir of temp_dirs.splice(0)) {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("fails before startup when --config has no usable models", async () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), "linguagacha-runtime-config-"));
    temp_dirs.push(root);
    const config_path = path.join(root, "config.json");
    fs.writeFileSync(config_path, JSON.stringify({ models: [], activate_model_id: "" }));

    const runtime = new TranslateRuntime({
      appRoot: path.resolve("."),
      configPath: config_path,
      workerExecution: { kind: "in_process" },
    });

    await expect(runtime.start()).rejects.toThrow("at least one usable model");
    await runtime.stop();
  });

  it("selects activate_model_id when present and falls back to the first model", () => {
    const first = { id: "first", model_id: "first-model" };
    const active = { id: "active", model_id: "active-model" };

    expect(
      resolve_active_model({
        activate_model_id: "active",
        models: [first, active],
      }),
    ).toEqual(active);
    expect(
      resolve_active_model({
        activate_model_id: "missing",
        models: [first, active],
      }),
    ).toEqual(first);
    expect(resolve_active_model({ models: [] })).toBeNull();
  });
});
