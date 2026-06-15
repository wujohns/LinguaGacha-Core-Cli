import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

import { AppMetadataService } from "../backend/app/app-metadata-service";
import { AppPathService } from "../backend/app/app-path-service";
import { AppSettingService } from "../backend/app/app-setting-service";
import type { CacheFileEntry, CacheFreshness, CacheReadPort, CacheSnapshot } from "../backend/cache/cache-types";
import { ProjectDatabase } from "../backend/database/database-operations";
import { TaskEngine } from "../backend/engine/core/engine";
import { PlanningWorkerPool } from "../backend/engine/planning/planning-worker-pool";
import { TaskRunPublisher } from "../backend/engine/run/task-run-publisher";
import { TaskRunState } from "../backend/engine/run/task-run-state";
import { TaskSnapshotBuilder } from "../backend/engine/run/task-snapshot-builder";
import { ProjectTaskStore } from "../backend/engine/store/project-task-store";
import { TaskPlanner } from "../backend/engine/planning/task-planner";
import { WorkUnitWorkerPool } from "../backend/engine/work-unit/work-unit-worker-pool";
import { TaskService } from "../backend/engine/task-service";
import { LogManager } from "../backend/log/log-manager";
import { set_main_log_language_reader } from "../backend/log/log-text";
import { migration_orchestrator } from "../backend/migration/migration-orchestrator";
import { ProjectChangeEventAdapter, ProjectChangePublisher } from "../backend/project/project-changes";
import { ProjectDataReader, build_section_revisions_from_meta } from "../backend/project/project-data";
import { ProjectEventBus } from "../backend/project/project-events";
import { ProjectOperationGate } from "../backend/project/project-gate";
import { ProjectLifecycleService, ProjectSessionState } from "../backend/project/project-session";
import { ProjectWriteStore } from "../backend/project/project-write-store";
import { TranslationFileExportService } from "../backend/translation/translation-file-export-service";
import type { BackendWorkerExecution } from "../backend/worker/worker-execution";
import { resolve_active_model } from "../backend/model/model-config-resolver";
import { ApiStreamHub } from "../backend/api/api-stream-hub";
import type { ApiJsonValue } from "../backend/api/api-types";
import type { DatabaseOperation } from "../backend/database/database-types";
import type { ProjectDataRecord } from "../backend/project/project-data";
import type { ProjectEvent, ProjectEventType } from "../backend/project/project-events";
import type { ProjectDataSectionRevisions } from "../shared/project-event";
import { JsonTool } from "../shared/utils/json-tool";
import type { TranslateCliLimiterOptions } from "./cli-parser";

export interface TranslateRuntimeStartOptions {
  appRoot: string;
  configPath: string;
  workerExecution?: BackendWorkerExecution;
  workerCount?: number;
  limiter?: TranslateCliLimiterOptions | null;
}

export class TranslateRuntime {
  private readonly app_root: string;
  private readonly config_path: string;
  private readonly worker_execution: BackendWorkerExecution;
  private readonly worker_count: number | undefined;
  private readonly limiter: TranslateCliLimiterOptions | null;
  private readonly data_root: string;
  private database: ProjectDatabase | null = null;
  private readonly stream_hub = new ApiStreamHub();
  private log_manager: LogManager | null = null;
  private services: TranslateRuntimeServices | null = null;

  public constructor(options: TranslateRuntimeStartOptions) {
    this.app_root = path.resolve(options.appRoot);
    this.config_path = path.resolve(options.configPath);
    this.worker_execution = options.workerExecution ?? build_default_worker_execution(this.app_root);
    this.worker_count =
      options.workerCount === undefined ? undefined : Math.max(1, Math.trunc(options.workerCount));
    this.limiter = options.limiter ?? null;
    this.data_root = path.join(path.dirname(this.config_path), ".linguagacha-translate-runtime");
  }

  public async start(): Promise<TranslateRuntimeServices> {
    assert_existing_file(this.config_path, "Config file does not exist");
    const config = read_config_file(this.config_path);
    assert_config_has_model(config, this.config_path);
    const database = create_project_database();
    this.database = database;
    const paths = new AppPathService({
      appRoot: this.app_root,
      env: { ...process.env, LINGUAGACHA_DATA_ROOT: this.data_root },
    });
    const metadata = new AppMetadataService(paths);
    const log_manager = new LogManager({
      logDir: path.join(this.data_root, "log"),
      targets: { console: false, window: false },
    });
    this.log_manager = log_manager;
    migration_orchestrator.run_startup_migrations({ paths, log_manager });
    const app_setting_service = new AppSettingService(paths);
    app_setting_service.save_setting(config);
    set_main_log_language_reader(() => app_setting_service.read_app_language());

    const project_session_state = new ProjectSessionState();
    const task_run_state = new TaskRunState();
    const project_data_reader = new ProjectDataReader(database);
    const project_event_bus = new ProjectEventBus();
    const cache_manager = new TranslateCache(database);
    cache_manager.subscribe(project_event_bus);
    const project_change_adapter = new ProjectChangeEventAdapter(
      database,
      project_session_state,
      project_data_reader,
    );
    const project_change_publisher = new ProjectChangePublisher(
      project_change_adapter,
      this.stream_hub,
    );
    const project_write_store = new ProjectWriteStore(
      database,
      project_event_bus,
      project_change_publisher,
    );
    const project_lifecycle_service = new ProjectLifecycleService(
      database,
      project_session_state,
      app_setting_service,
      paths,
      log_manager,
      project_event_bus,
    );
    const task_snapshot_builder = new TaskSnapshotBuilder(
      database,
      task_run_state,
      project_session_state,
      project_data_reader,
    );
    const task_run_publisher = new TaskRunPublisher(
      this.stream_hub,
      task_run_state,
      task_snapshot_builder,
    );
    const project_task_store = new ProjectTaskStore(
      database,
      project_session_state,
      task_run_state,
      cache_manager,
      project_write_store,
    );
    const work_unit_worker_pool = new WorkUnitWorkerPool({
      appRoot: this.app_root,
      execution: this.worker_execution,
      systemProxySnapshot: null,
      workerCount: this.worker_count,
      maxInFlight: this.worker_count,
      limiter: this.limiter,
    });
    const planning_worker_pool = new PlanningWorkerPool({ execution: this.worker_execution });
    const task_engine = new TaskEngine({
      appRoot: this.app_root,
      taskStore: project_task_store,
      taskRunPublisher: task_run_publisher,
      executorClient: work_unit_worker_pool,
      taskPlanner: new TaskPlanner({ planningWorkerPool: planning_worker_pool }),
      AppSettingService: app_setting_service,
      logManager: log_manager,
    });
    const task_service = new TaskService(
      task_engine,
      task_snapshot_builder,
      task_run_publisher,
      new ProjectOperationGate(task_run_state),
      project_session_state,
    );
    const translation_file_export_service = new TranslationFileExportService(
      database,
      app_setting_service,
      project_session_state,
      async () => undefined,
      log_manager,
    );

    this.stream_hub.start();
    app_setting_service.set_stream_publisher(this.stream_hub);
    this.services = new TranslateRuntimeServices({
      appSettingService: app_setting_service,
      database,
      projectEventBus: project_event_bus,
      projectTaskStore: project_task_store,
      projectLifecycleService: project_lifecycle_service,
      taskService: task_service,
      taskSnapshotBuilder: task_snapshot_builder,
      translationFileExportService: translation_file_export_service,
      streams: this.stream_hub,
      workUnitWorkerPool: work_unit_worker_pool,
      planningWorkerPool: planning_worker_pool,
    });
    metadata.read_version();
    return this.services;
  }

  public async stop(): Promise<void> {
    await this.services?.dispose();
    this.services = null;
    this.database?.close();
    this.database = null;
    await this.log_manager?.shutdown();
    this.log_manager = null;
    set_main_log_language_reader(null);
  }
}

export class TranslateRuntimeServices {
  public readonly appSettingService: AppSettingService;
  public readonly projectLifecycleService: ProjectLifecycleService;
  public readonly taskService: TaskService;
  public readonly translationFileExportService: TranslationFileExportService;
  public readonly streams: ApiStreamHub;
  private readonly database: ProjectDatabase;
  private readonly project_event_bus: ProjectEventBus;
  private readonly project_task_store: ProjectTaskStore;
  private readonly task_snapshot_builder: TaskSnapshotBuilder;
  private readonly work_unit_worker_pool: WorkUnitWorkerPool;
  private readonly planning_worker_pool: PlanningWorkerPool;

  public constructor(options: {
    appSettingService: AppSettingService;
    database: ProjectDatabase;
    projectEventBus: ProjectEventBus;
    projectTaskStore: ProjectTaskStore;
    projectLifecycleService: ProjectLifecycleService;
    taskService: TaskService;
    taskSnapshotBuilder: TaskSnapshotBuilder;
    translationFileExportService: TranslationFileExportService;
    streams: ApiStreamHub;
    workUnitWorkerPool: WorkUnitWorkerPool;
    planningWorkerPool: PlanningWorkerPool;
  }) {
    this.appSettingService = options.appSettingService;
    this.database = options.database;
    this.project_event_bus = options.projectEventBus;
    this.project_task_store = options.projectTaskStore;
    this.projectLifecycleService = options.projectLifecycleService;
    this.taskService = options.taskService;
    this.task_snapshot_builder = options.taskSnapshotBuilder;
    this.translationFileExportService = options.translationFileExportService;
    this.streams = options.streams;
    this.work_unit_worker_pool = options.workUnitWorkerPool;
    this.planning_worker_pool = options.planningWorkerPool;
  }

  public build_expected_section_revisions(sections: string[]): Record<string, number> {
    const revisions: Record<string, number> = {};
    for (const section of sections) {
      revisions[section] = this.task_snapshot_builder.get_section_revision(section);
    }
    return revisions;
  }

  public async commit_cli_resource_operations(
    project_path: string,
    operations: DatabaseOperation[],
  ): Promise<void> {
    if (operations.length === 0) {
      return;
    }
    this.database.execute_transaction(operations);
    const meta = this.database.execute({
      name: "getAllMeta",
      args: { projectPath: project_path },
    });
    const section_revisions = build_section_revisions_from_meta(
      typeof meta === "object" && meta !== null && !Array.isArray(meta) ? meta : {},
    );
    await this.project_event_bus.publish({
      type: "project.quality.changed",
      projectPath: project_path,
      source: "cli",
      affectedSections: ["quality", "prompts"],
      sectionRevisions: section_revisions,
      scope: "quality-full",
    });
    await this.project_event_bus.publish({
      type: "project.prompts.changed",
      projectPath: project_path,
      source: "cli",
      affectedSections: ["quality", "prompts"],
      sectionRevisions: section_revisions,
      scope: "prompts-full",
    });
  }

  public async restore_failed_translation_items_for_continue(): Promise<number> {
    const result = await this.project_task_store.restore_failed_translation_items_for_continue();
    return Number(result["restored_count"] ?? 0);
  }

  public async dispose(): Promise<void> {
    this.appSettingService.set_stream_publisher(null);
    this.streams.stop();
    await Promise.all([
      this.work_unit_worker_pool.dispose(),
      this.planning_worker_pool.dispose(),
    ]);
  }
}

function build_default_worker_execution(app_root: string): BackendWorkerExecution {
  return {
    kind: "worker_threads",
    workUnitWorkerEntryUrl: pathToFileURL(path.join(app_root, "dist", "work-unit-worker-entry.js")),
    planningWorkerEntryUrl: pathToFileURL(path.join(app_root, "dist", "planning-worker-entry.js")),
  };
}

function assert_existing_file(file_path: string, message: string): void {
  if (!fs.existsSync(file_path) || !fs.statSync(file_path).isFile()) {
    throw new Error(`${message}: ${file_path}`);
  }
}

function assert_config_has_model(config: Record<string, ApiJsonValue>, config_path: string): void {
  if (resolve_active_model(config) === null) {
    throw new Error(
      `Config file must contain at least one usable model in models/activate_model_id: ${config_path}`,
    );
  }
}

function create_project_database(): ProjectDatabase {
  try {
    return new ProjectDatabase();
  } catch (error) {
    throw new Error(
      "Node runtime lacks required node:sqlite DatabaseSync or node:zlib Zstd support",
      { cause: error },
    );
  }
}

const CACHE_EVENT_TYPES: ProjectEventType[] = [
  "project.opened_for_cache",
  "project.unloaded",
  "project.items.changed",
  "project.quality.changed",
  "project.prompts.changed",
  "project.settings.changed",
];

class TranslateCache implements CacheReadPort {
  private readonly data_reader: ProjectDataReader;
  private project_path = "";
  private epoch = 0;
  private freshness: CacheFreshness = "empty";
  private item_records: ProjectDataRecord[] = [];
  private item_by_id = new Map<number, ProjectDataRecord>();
  private file_entries: CacheFileEntry[] = [];
  private quality_block: ProjectDataRecord = {};
  private prompt_block: ProjectDataRecord = {};
  private section_revisions: ProjectDataSectionRevisions = {};

  public readonly items = {
    readItems: (): ProjectDataRecord[] => this.read_items(),
    readItem: (itemId: number): ProjectDataRecord | null => this.read_item(itemId),
  };
  public readonly files = {
    readFileEntries: (): CacheFileEntry[] => this.file_entries.map((entry) => ({ ...entry })),
  };
  public readonly quality = {
    readBlock: (): ProjectDataRecord => ({ ...this.quality_block }),
  };
  public readonly prompts = {
    readBlock: (): ProjectDataRecord => ({ ...this.prompt_block }),
  };
  public readonly analysis = {
    readBlock: (): ProjectDataRecord => ({}),
  };

  public constructor(database: ProjectDatabase) {
    this.data_reader = new ProjectDataReader(database);
  }

  public subscribe(project_event_bus: ProjectEventBus): void {
    for (const event_type of CACHE_EVENT_TYPES) {
      project_event_bus.subscribe(event_type, async (event) => {
        this.handle_event(event);
      });
    }
  }

  public readSectionRevisions(): ProjectDataSectionRevisions {
    return { ...this.section_revisions };
  }

  public snapshot(): CacheSnapshot {
    return {
      projectPath: this.project_path,
      epoch: this.epoch,
      freshness: this.freshness,
      sectionRevisions: { ...this.section_revisions },
      itemCount: this.item_records.length,
    };
  }

  private handle_event(event: ProjectEvent): void {
    if (event.type === "project.unloaded") {
      this.clear(event.projectPath);
      return;
    }
    if (event.projectPath === "") {
      return;
    }
    this.rebuild(event.projectPath);
  }

  private rebuild(project_path: string): void {
    const meta = this.data_reader.get_all_meta(project_path);
    const items_snapshot = this.data_reader.build_runtime_items_snapshot(project_path);
    const files_block = this.data_reader.build_files_record_block(project_path, items_snapshot);
    const quality_block = this.data_reader.build_quality_block(project_path, meta);
    const prompt_block = this.data_reader.build_prompts_block(project_path, meta);
    const item_records = items_snapshot.item_records.map((item) => ({ ...item }));
    this.project_path = project_path;
    this.epoch += 1;
    this.freshness = "fresh";
    this.item_records = item_records;
    this.item_by_id = new Map(
      item_records.map((item) => [this.read_number(item["item_id"], 0), item]),
    );
    this.file_entries = Object.values(files_block)
      .filter((value): value is ProjectDataRecord => {
        return typeof value === "object" && value !== null && !Array.isArray(value);
      })
      .map((file) => ({
        rel_path: String(file["rel_path"] ?? ""),
        file_type: String(file["file_type"] ?? ""),
        sort_index: this.read_number(file["sort_index"], 0),
      }));
    this.quality_block = quality_block;
    this.prompt_block = prompt_block;
    this.section_revisions = this.data_reader.build_section_revisions(meta);
  }

  private clear(project_path?: string): void {
    if (project_path !== undefined && this.project_path !== "" && this.project_path !== project_path) {
      return;
    }
    this.project_path = "";
    this.epoch += 1;
    this.freshness = "empty";
    this.item_records = [];
    this.item_by_id = new Map();
    this.file_entries = [];
    this.quality_block = {};
    this.prompt_block = {};
    this.section_revisions = {};
  }

  private read_items(): ProjectDataRecord[] {
    return this.item_records.map((item) => ({ ...item }));
  }

  private read_item(item_id: number): ProjectDataRecord | null {
    const item = this.item_by_id.get(item_id);
    return item === undefined ? null : { ...item };
  }

  private read_number(value: unknown, fallback: number): number {
    const number_value = Number(value ?? fallback);
    return Number.isFinite(number_value) ? Math.trunc(number_value) : fallback;
  }
}

function read_config_file(config_path: string): Record<string, ApiJsonValue> {
  const payload = JsonTool.parseStrict<ApiJsonValue>(fs.readFileSync(config_path, "utf-8"));
  if (typeof payload !== "object" || payload === null || Array.isArray(payload)) {
    throw new Error(`Config file must contain a JSON object: ${config_path}`);
  }
  return payload as Record<string, ApiJsonValue>;
}
