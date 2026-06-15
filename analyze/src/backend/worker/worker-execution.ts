import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const BUNDLED_CHUNK_DIRECTORY_NAME = "chunks";
const WORK_UNIT_WORKER_ENTRY_FILE_NAME = "work-unit-worker-entry.js";
const PLANNING_WORKER_ENTRY_FILE_NAME = "planning-worker-entry.js";

export type BackendWorkerExecution =
  | {
      kind: "worker_threads";
      workUnitWorkerEntryUrl: URL;
      planningWorkerEntryUrl: URL;
    }
  | {
      kind: "in_process";
    };

export function resolve_node_bundle_dir_from_module_url(module_url: string): string {
  const module_dir = path.dirname(fileURLToPath(module_url));
  if (path.basename(module_dir) === BUNDLED_CHUNK_DIRECTORY_NAME) {
    return path.dirname(module_dir);
  }
  return module_dir;
}

export function build_worker_threads_backend_worker_execution_from_bundle_dir(
  bundle_dir: string,
): BackendWorkerExecution {
  return {
    kind: "worker_threads",
    workUnitWorkerEntryUrl: pathToFileURL(path.join(bundle_dir, WORK_UNIT_WORKER_ENTRY_FILE_NAME)),
    planningWorkerEntryUrl: pathToFileURL(path.join(bundle_dir, PLANNING_WORKER_ENTRY_FILE_NAME)),
  };
}
