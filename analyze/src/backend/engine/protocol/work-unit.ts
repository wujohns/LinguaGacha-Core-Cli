import type { ApiJsonValue } from "../../api/api-types";
import type { LogError } from "../../../shared/error";

export type WorkUnitLogEntry = {
  level: "info" | "warning" | "error";
  message: string;
  error?: LogError;
  context?: Record<string, ApiJsonValue>;
};

export type AnalysisWorkUnit = {
  unit_id: string;
  run_id: string;
  kind: "analysis";
  model: ApiJsonValue;
  config_snapshot: ApiJsonValue;
  quality_snapshot: ApiJsonValue;
  payload: {
    file_path: string;
    items: ApiJsonValue;
  };
  diagnostics: {
    retry_count: number;
  };
};

export type WorkUnit = AnalysisWorkUnit;
