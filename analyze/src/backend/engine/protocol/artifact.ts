import type { ApiJsonValue } from "../../api/api-types";

export type TaskArtifact =
  | {
      kind: "analysis_checkpoints";
      checkpoints: ApiJsonValue;
    }
  | {
      kind: "analysis_candidates";
      entries: ApiJsonValue;
    };
