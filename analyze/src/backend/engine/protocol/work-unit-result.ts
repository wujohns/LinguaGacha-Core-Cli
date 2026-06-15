import type { ApiJsonValue } from "../../api/api-types";
import type { WorkUnitLogEntry } from "./work-unit";

export type AnalysisWorkUnitOutput = {
  kind: "analysis";
  glossary_entries: ApiJsonValue;
  valid_empty_result: boolean;
};

export type WorkUnitExecutionResult = {
  unit_id: string;
  kind: "analysis";
  outcome: "success" | "failed" | "stopped";
  metrics: {
    input_tokens: number;
    output_tokens: number;
  };
  output: AnalysisWorkUnitOutput;
  logs: WorkUnitLogEntry[];
};
