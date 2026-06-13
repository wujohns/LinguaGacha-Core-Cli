import type { TaskStartMode, TranslationScope } from "../../../domain/task";

/** StartTaskCommand 是 API 命令层交给 Engine 的唯一启动命令形状 */
export type StartTaskCommand = {
  task_type: "translation";
  mode: TaskStartMode; // 只描述本轮启动语义，不参与状态机
  scope: TranslationScope; // CLI 当前固定为全量翻译
  expected_section_revisions: Record<string, number>; // revision 锁保护后台任务输入不基于旧快照运行
};

/** StopTaskCommand 只按 translation 停止。 */
export type StopTaskCommand = {
  task_type: "translation";
};
