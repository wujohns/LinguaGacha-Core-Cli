import { describe, expect, it } from "vitest";

import type { ApiJsonValue } from "../../api/api-types";
import type { CacheReadPort } from "../../cache/cache-types";
import type { ProjectWriteStore } from "../../project/project-write-store";
import { ProjectSessionState } from "../../project/project-session";
import { TaskPlanner } from "../planning/task-planner";
import { ProjectTaskStore } from "./project-task-store";

type MutableJsonRecord = Record<string, ApiJsonValue>;

class FakeDatabase {
  public checkpoints: MutableJsonRecord[] = [
    {
      item_id: 1,
      status: "PROCESSED",
      updated_at: "2026-01-01T00:00:00.000Z",
      error_count: 0,
    },
    {
      item_id: 2,
      status: "ERROR",
      updated_at: "2026-01-01T00:00:00.000Z",
      error_count: 1,
    },
    {
      item_id: 3,
      status: "NONE",
      updated_at: "2026-01-01T00:00:00.000Z",
      error_count: 0,
    },
  ];

  public execute(operation: { name: string }): ApiJsonValue {
    if (operation.name === "getAnalysisItemCheckpoints") {
      return this.checkpoints.map((checkpoint) => ({ ...checkpoint }));
    }
    if (operation.name === "getAllMeta") {
      return {};
    }
    return null;
  }
}

class FakeWriteStore {
  public restored_checkpoints: MutableJsonRecord[] = [];

  public async restore_failed_analysis_checkpoints_for_continue(request: {
    checkpoints: ApiJsonValue | undefined;
  }): Promise<MutableJsonRecord> {
    this.restored_checkpoints = Array.isArray(request.checkpoints)
      ? request.checkpoints
          .filter((checkpoint): checkpoint is MutableJsonRecord => is_record(checkpoint))
          .map((checkpoint) => ({ ...checkpoint }))
      : [];
    return { restored_count: this.restored_checkpoints.length };
  }
}

class FakePlanningWorkerPool {
  public async count_items(
    items: Array<{ cache_key: string; text: string }>,
  ): Promise<Array<{ cache_key: string; token_count: number }>> {
    return items.map((item) => ({
      cache_key: item.cache_key,
      token_count: Math.max(1, item.text.length),
    }));
  }
}

describe("ProjectTaskStore analysis continue restore", () => {
  it("restores only ERROR checkpoints to NONE", async () => {
    const session_state = new ProjectSessionState();
    session_state.mark_loaded("/work/project.lg");
    const database = new FakeDatabase();
    const write_store = new FakeWriteStore();
    const task_store = new ProjectTaskStore(
      database as never,
      session_state,
      {},
      create_cache(),
      write_store as unknown as ProjectWriteStore,
    );

    const result = await task_store.restore_failed_analysis_items_for_continue();

    expect(result["restored_count"]).toBe(1);
    expect(write_store.restored_checkpoints).toEqual([
      expect.objectContaining({
        item_id: 2,
        status: "NONE",
        error_count: 0,
      }),
    ]);
  });

  it("plans restored ERROR items and skips PROCESSED items", async () => {
    const planner = new TaskPlanner({
      planningWorkerPool: new FakePlanningWorkerPool() as never,
    });
    const contexts = await planner.build_analysis_contexts(
      [
        build_item(1, "done"),
        build_item(2, "retry"),
        build_item(3, "pending"),
      ],
      [
        { item_id: 1, status: "PROCESSED" },
        { item_id: 2, status: "NONE" },
        { item_id: 3, status: "NONE" },
      ],
      {},
      new AbortController().signal,
    );

    expect(contexts.flatMap((context) => context.items.map((item) => item.item_id))).toEqual([
      2,
      3,
    ]);
  });

  function create_cache(): CacheReadPort {
    return {
      items: {
        readItems: () => [],
        readItem: () => null,
      },
      files: {
        readFileEntries: () => [],
      },
      quality: {
        readBlock: () => ({}),
      },
      prompts: {
        readBlock: () => ({}),
      },
      readSectionRevisions: () => ({}),
      snapshot: () => ({
        projectPath: "/work/project.lg",
        epoch: 1,
        freshness: "fresh",
        sectionRevisions: {},
        itemCount: 0,
      }),
    };
  }

  function build_item(id: number, src: string): MutableJsonRecord {
    return {
      id,
      item_id: id,
      file_path: "input.txt",
      src,
      status: "NONE",
    };
  }
});

function is_record(value: unknown): value is MutableJsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
