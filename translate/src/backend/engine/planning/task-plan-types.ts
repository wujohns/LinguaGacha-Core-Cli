import type { MutableJsonRecord } from "../run/task-run-types";

/**
 * 任务 item 快照仍是数据库 JSON 行的可变副本，规划器只读取字段，提交阶段才允许改状态。
 */
export type TaskItemRecord = MutableJsonRecord;

/**
 * 翻译 context 是 pipeline 的最小工作单元，包含 chunk、preceding 与重试元信息。
 */
export interface TranslationContext {
  work_unit_id: string;
  items: TaskItemRecord[];
  precedings: TaskItemRecord[];
  token_threshold: number;
  split_count: number;
  retry_count: number;
  is_initial: boolean;
}

/**
 * 翻译提交项只携带可批量写库的数据和 token 累计值。
 */
export interface TranslationCommitEntry {
  items: TaskItemRecord[];
  input_tokens: number;
  output_tokens: number;
}

/**
 * 翻译拆分重试会同时产生新 context 和强制失败条目，两者必须分开提交。
 */
export interface TranslationRetryPlan {
  retry_contexts: TranslationContext[];
  forced_error_items: TaskItemRecord[];
}
