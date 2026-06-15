import type { WorkUnit } from "../protocol/work-unit";
import type { WorkUnitExecutionResult } from "../protocol/work-unit-result";
import { LLMClient } from "../../llm/llm-client";
import {
  LLMRequestLimiterClient,
  type LLMRequestLimiterOptions,
} from "../../llm/llm-request-limiter-client";
import { AppMetadataService } from "../../app/app-metadata-service";
import { AppPathService } from "../../app/app-path-service";
import { AnalysisWorkUnitRunner } from "./runners/analysis-runner";
import * as AppErrors from "../../../shared/error";

export interface WorkUnitRunnerOptions {
  appRoot: string;
  limiter?: LLMRequestLimiterOptions | null;
}

export class WorkUnitRunner {
  private readonly analysis_runner: AnalysisWorkUnitRunner;

  public constructor(options: WorkUnitRunnerOptions) {
    const paths = new AppPathService({ appRoot: options.appRoot });
    const metadata = new AppMetadataService(paths);
    const llm_client = new LLMClient({ userAgent: metadata.build_linguagacha_user_agent() });
    const request_client =
      options.limiter === null || options.limiter === undefined
        ? llm_client
        : new LLMRequestLimiterClient(llm_client, options.limiter);
    this.analysis_runner = new AnalysisWorkUnitRunner(options.appRoot, request_client);
  }

  public async run(unit: WorkUnit, signal: AbortSignal): Promise<WorkUnitExecutionResult> {
    if (unit.kind === "analysis") {
      return this.analysis_runner.execute_unit(unit, signal);
    }
    throw new AppErrors.WorkerExecutionFailedError({
      diagnostic_context: { expected_kind: "analysis", result_kind: unit.kind },
    });
  }
}
