import type { ApiJsonValue } from "../../api/api-types";

/** TaskArtifact 是 Engine 到 ProjectTaskStore 的唯一提交载荷，隔离数据库 operation 细节 */
export type TaskArtifact = {
  kind: "item_updates";
  source: "translation";
  items: ApiJsonValue;
};
