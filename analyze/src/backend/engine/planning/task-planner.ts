import crypto from "node:crypto";

import type { ApiJsonValue } from "../../api/api-types";
import type { MutableJsonRecord } from "../run/task-run-types";
import { is_task_skipped_item_status } from "../../../domain/task";
import { read_item_name_text } from "../../../shared/item-name";
import type { PlanningWorkerPool } from "./planning-worker-pool";
import {
  build_task_token_metric_cache_key,
  count_non_empty_source_lines,
  TaskTokenMetricCache,
  type TaskTokenCountInput,
  type TaskTokenMetric,
} from "./token-metric-cache";
import type {
  AnalysisContext,
  AnalysisItemContext,
  TaskItemRecord,
} from "./task-plan-types";

const DEFAULT_ANALYSIS_INPUT_TOKEN_LIMIT = 512; // 分析 prompt 额外包含术语抽取说明，默认 token 门槛独立保留。
const HASH_YIELD_EVERY_ITEMS = 1024; // 主线程计算 hash 时分批让出事件循环，避免大项目启动阶段长时间无响应。

interface MetricSeed {
  item_id: number;
  cache_key: string;
  src: string;
  line_count: number;
}

/**
 * TaskPlanner 是后台任务唯一规划器：它复用进程内 token 缓存，并把精确计数交给 planning worker。
 */
export class TaskPlanner {
  private readonly planning_worker_pool: PlanningWorkerPool; // 只做纯计算，不接触数据库和事件。
  private readonly metric_cache: TaskTokenMetricCache; // 进程内计算指标缓存，随 BackendServices 生命周期释放。

  /**
   * 注入 planning worker 和可选 cache，保证规划结果仍由 Backend 主线程解释。
   */
  public constructor(options: {
    planningWorkerPool: PlanningWorkerPool;
    metricCache?: TaskTokenMetricCache;
  }) {
    this.planning_worker_pool = options.planningWorkerPool;
    this.metric_cache = options.metricCache ?? new TaskTokenMetricCache();
  }

  /**
   * 构建分析上下文，checkpoint 已完成或错误的条目不会重复调度。
   */
  public async build_analysis_contexts(
    items: TaskItemRecord[],
    checkpoints: MutableJsonRecord[],
    model: MutableJsonRecord,
    signal: AbortSignal,
  ): Promise<AnalysisContext[]> {
    const checkpoint_status_by_id = this.build_checkpoint_status_map(checkpoints);
    const pending_items = items
      .map((item) => this.build_analysis_item_context(item, checkpoint_status_by_id))
      .filter((item): item is AnalysisItemContext => item !== null)
      .filter((item) => item.previous_status !== "PROCESSED" && item.previous_status !== "ERROR");
    const seed_items = pending_items.map((item) => ({
      item_id: item.item_id,
      id: item.item_id,
      src: item.src_text,
      file_path: item.file_path,
      status: "NONE",
    }));
    const context_by_id = new Map(pending_items.map((item) => [item.item_id, item]));
    const chunks = await this.generate_item_chunks(
      seed_items,
      this.get_input_token_limit(model, DEFAULT_ANALYSIS_INPUT_TOKEN_LIMIT),
      signal,
    );
    return chunks
      .map(({ chunk_items }) => {
        const chunk_contexts = chunk_items
          .map((item) => context_by_id.get(this.read_item_id(item)))
          .filter((item): item is AnalysisItemContext => item !== undefined);
        return {
          work_unit_id: crypto.randomUUID(),
          file_path: chunk_contexts[0]?.file_path ?? "",
          items: chunk_contexts,
          retry_count: 0,
        };
      })
      .filter((context) => context.items.length > 0);
  }

  /**
   * 共享切块实现，只依赖 item 快照和已解析 token 指标，不在主线程执行 tokenizer。
   */
  private async generate_item_chunks(
    items: TaskItemRecord[],
    input_token_threshold: number,
    signal: AbortSignal,
  ): Promise<Array<{ chunk_items: TaskItemRecord[] }>> {
    const metric_by_id = await this.resolve_item_metrics(items, signal);
    const line_limit = Math.max(8, Math.trunc(input_token_threshold / 16));
    const chunks: Array<{ chunk_items: TaskItemRecord[] }> = [];
    let skipped_count = 0;
    let line_length = 0;
    let token_length = 0;
    let chunk: TaskItemRecord[] = [];
    for (const [index, item] of items.entries()) {
      this.throw_if_aborted(signal);
      if (this.read_status(item) !== "NONE") {
        skipped_count += 1;
        continue;
      }
      const metric = metric_by_id.get(this.read_item_id(item));
      if (metric === undefined) {
        skipped_count += 1;
        continue;
      }
      if (
        chunk.length > 0 &&
        (line_length + metric.line_count > line_limit ||
          token_length + metric.token_count > input_token_threshold ||
          String(item["file_path"] ?? "") !== String(chunk[chunk.length - 1]?.["file_path"] ?? ""))
      ) {
        chunks.push({ chunk_items: chunk });
        skipped_count = 0;
        line_length = 0;
        token_length = 0;
        chunk = [];
      }
      chunk.push(item);
      line_length += metric.line_count;
      token_length += metric.token_count;
    }
    if (chunk.length > 0) {
      chunks.push({ chunk_items: chunk });
    }
    return chunks;
  }

  /**
   * 为当前 item 快照解析 token/行数指标，cache 命中直接复用，缺失才调用 worker。
   */
  private async resolve_item_metrics(
    items: TaskItemRecord[],
    signal: AbortSignal,
  ): Promise<Map<number, TaskTokenMetric>> {
    const seeds = await this.build_metric_seeds(items, signal);
    const metric_by_id = new Map<number, TaskTokenMetric>();
    const missing_by_key = new Map<string, TaskTokenCountInput>();
    for (const seed of seeds) {
      const cached_metric = this.metric_cache.get(seed.cache_key);
      if (cached_metric !== null) {
        metric_by_id.set(seed.item_id, cached_metric);
        continue;
      }
      missing_by_key.set(seed.cache_key, { cache_key: seed.cache_key, text: seed.src });
    }
    if (missing_by_key.size > 0) {
      await this.count_missing_metrics([...missing_by_key.values()], seeds, metric_by_id, signal);
    }
    return metric_by_id;
  }

  /**
   * 批量调用 planning worker，并把结果写入进程内 cache 后回填本次规划所需指标。
   */
  private async count_missing_metrics(
    missing_inputs: TaskTokenCountInput[],
    seeds: MetricSeed[],
    metric_by_id: Map<number, TaskTokenMetric>,
    signal: AbortSignal,
  ): Promise<void> {
    const results = await this.planning_worker_pool.count_items(missing_inputs, signal);
    const token_count_by_key = new Map(
      results.map((result) => [result.cache_key, result.token_count]),
    );
    for (const seed of seeds) {
      if (metric_by_id.has(seed.item_id)) {
        continue;
      }
      const token_count = token_count_by_key.get(seed.cache_key);
      if (token_count === undefined) {
        continue;
      }
      const metric = { token_count, line_count: seed.line_count };
      this.metric_cache.set(seed.cache_key, metric);
      metric_by_id.set(seed.item_id, metric);
    }
  }

  /**
   * 构建 cache 校验种子；hash 计算留在主线程但分批让出，避免大项目启动阶段长卡顿。
   */
  private async build_metric_seeds(
    items: TaskItemRecord[],
    signal: AbortSignal,
  ): Promise<MetricSeed[]> {
    const seeds: MetricSeed[] = [];
    const seen_item_ids = new Set<number>();
    for (const [index, item] of items.entries()) {
      this.throw_if_aborted(signal);
      if (this.read_status(item) !== "NONE") {
        continue;
      }
      const item_id = this.read_item_id(item);
      if (item_id <= 0 || seen_item_ids.has(item_id)) {
        continue;
      }
      seen_item_ids.add(item_id);
      const src = String(item["src"] ?? "");
      seeds.push({
        item_id,
        cache_key: build_task_token_metric_cache_key(src),
        src,
        line_count: count_non_empty_source_lines(src),
      });
      if (index > 0 && index % HASH_YIELD_EVERY_ITEMS === 0) {
        await new Promise<void>((resolve) => setImmediate(resolve));
      }
    }
    return seeds;
  }

  /**
   * 输入 token 阈值读取集中处理，保护模型配置缺字段场景。
   */
  private get_input_token_limit(model: MutableJsonRecord, fallback: number): number {
    const threshold = this.normalize_record(model["threshold"]);
    return Math.max(16, this.read_number(threshold["input_token_limit"], fallback));
  }

  /**
   * 从 item 和 checkpoint map 构建不可变分析输入快照。
   */
  private build_analysis_item_context(
    item: TaskItemRecord,
    checkpoint_status_by_id: Map<number, string>,
  ): AnalysisItemContext | null {
    if (!this.is_analyzable_item(item)) {
      return null;
    }
    const item_id = this.read_item_id(item);
    if (item_id <= 0) {
      return null;
    }
    return {
      item_id,
      file_path: String(item["file_path"] ?? ""),
      src_text: this.build_analysis_source_text(item),
      previous_status: checkpoint_status_by_id.get(item_id) ?? null,
    };
  }

  /**
   * 分析输入在规划期渲染姓名前缀，后续 worker 继续只消费纯文本快照。
   */
  private build_analysis_source_text(item: TaskItemRecord): string {
    const src = String(item["src"] ?? "").trim();
    if (src === "") {
      return "";
    }
    const name = read_item_name_text(item["name_src"]).trim();
    return name === "" ? src : `【${name}】${src}`;
  }

  /**
   * checkpoint 只接受三态状态，坏数据不会影响调度。
   */
  private build_checkpoint_status_map(checkpoints: MutableJsonRecord[]): Map<number, string> {
    const result = new Map<number, string>();
    for (const checkpoint of checkpoints) {
      const item_id = this.read_number(checkpoint["item_id"], 0);
      const status = String(checkpoint["status"] ?? "");
      if (item_id > 0 && (status === "NONE" || status === "PROCESSED" || status === "ERROR")) {
        result.set(item_id, status);
      }
    }
    return result;
  }

  /**
   * 分析跳过规则保持稳定语义。
   */
  private is_analyzable_item(item: TaskItemRecord): boolean {
    return (
      !is_task_skipped_item_status(this.read_status(item)) &&
      String(item["src"] ?? "").trim() !== ""
    );
  }

  /**
   * item id 同时兼容数据库内部 id 和公开 item_id。
   */
  private read_item_id(item: TaskItemRecord): number {
    return this.read_number(item["id"] ?? item["item_id"], 0);
  }

  /**
   * 读取 item 当前状态事实。
   */
  private read_status(item: TaskItemRecord): string {
    return String(item["status"] ?? "NONE");
  }

  /**
   * JSON 普通对象归一，避免数组和 null 进入业务分支。
   */
  private normalize_record(value: ApiJsonValue | undefined): MutableJsonRecord {
    return typeof value === "object" && value !== null && !Array.isArray(value) ? { ...value } : {};
  }

  /**
   * 数字字段统一截断，坏值回退到调用方默认值。
   */
  private read_number(value: ApiJsonValue | undefined, fallback: number): number {
    const number_value = Number(value ?? fallback);
    return Number.isFinite(number_value) ? Math.trunc(number_value) : fallback;
  }

  /**
   * 规划阶段主动响应停止信号，避免已取消任务继续规划。
   */
  private throw_if_aborted(signal: AbortSignal): void {
    if (signal.aborted) {
      throw signal.reason instanceof Error ? signal.reason : new Error("任务规划已取消。");
    }
  }
}
