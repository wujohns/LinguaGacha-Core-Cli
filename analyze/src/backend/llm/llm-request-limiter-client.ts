import { setTimeout as sleep } from "node:timers/promises";

import { JsonTool } from "../../shared/utils/json-tool";
import type { LLMClientPort, LLMRequestBody, LLMRequestResult } from "./llm-types";

export interface LLMRequestLimiterOptions {
  url: string;
  resource: string;
}

interface NormalizedLLMRequestLimiterOptions {
  url: string;
  resource: string;
}

type LimiterAcquireResponse = {
  ok: boolean;
  lease_id?: string;
  retry_after_ms?: number;
};

const DEFAULT_RETRY_AFTER_MS = 1_000;

/**
 * Optional LLM request limiter decorator. It only protects the real request boundary;
 * translation retry/error semantics stay in the caller.
 */
export class LLMRequestLimiterClient implements LLMClientPort {
  private readonly inner: LLMClientPort;
  private readonly options: NormalizedLLMRequestLimiterOptions;

  public constructor(inner: LLMClientPort, options: LLMRequestLimiterOptions) {
    this.inner = inner;
    this.options = {
      url: options.url.replace(/\/+$/u, ""),
      resource: options.resource,
    };
  }

  public async request(body: LLMRequestBody, signal: AbortSignal): Promise<LLMRequestResult> {
    const lease_id = await this.acquire(body, signal);
    try {
      return await this.inner.request(body, signal);
    } finally {
      await this.release(lease_id, signal);
    }
  }

  private async acquire(body: LLMRequestBody, signal: AbortSignal): Promise<string> {
    for (;;) {
      if (signal.aborted) {
        throw new Error("LLM limiter acquire cancelled");
      }
      const response = await this.try_acquire_once(body, signal);
      if (response.ok && typeof response.lease_id === "string" && response.lease_id !== "") {
        return response.lease_id;
      }
      const retry_after_ms = this.normalize_retry_after_ms(response.retry_after_ms);
      await sleep(retry_after_ms, undefined, { signal });
    }
  }

  private async release(lease_id: string, signal: AbortSignal): Promise<void> {
    try {
      await this.post_json(
        "/v1/leases/release",
        { lease_id },
        signal.aborted ? undefined : signal,
      );
    } catch {
      // Release is best-effort; limiter TTL is the crash/transport fallback.
    }
  }

  private async try_acquire_once(
    body: LLMRequestBody,
    signal: AbortSignal,
  ): Promise<LimiterAcquireResponse> {
    try {
      return await this.post_json<LimiterAcquireResponse>(
        "/v1/leases/acquire",
        {
          resource: this.options.resource,
          request_id: `${body.run_id}:${body.work_unit_id}`,
        },
        signal,
      );
    } catch (error) {
      if (signal.aborted) {
        throw error;
      }
      return { ok: false, retry_after_ms: DEFAULT_RETRY_AFTER_MS };
    }
  }

  private async post_json<T = unknown>(
    pathname: string,
    body: Record<string, unknown>,
    signal: AbortSignal | undefined,
  ): Promise<T> {
    const response = await fetch(`${this.options.url}${pathname}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JsonTool.stringifyStrict(body),
      signal,
    });
    const text = await response.text();
    const data = text === "" ? {} : JsonTool.parseStrict<unknown>(text);
    if (!response.ok) {
      return this.normalize_error_response(data, response.status) as T;
    }
    return data as T;
  }

  private normalize_error_response(data: unknown, status: number): LimiterAcquireResponse {
    if (typeof data === "object" && data !== null && !Array.isArray(data)) {
      const record = data as Record<string, unknown>;
      return {
        ok: false,
        retry_after_ms: this.normalize_retry_after_ms(record["retry_after_ms"]),
      };
    }
    return {
      ok: false,
      retry_after_ms: DEFAULT_RETRY_AFTER_MS,
    };
  }

  private normalize_retry_after_ms(value: unknown): number {
    const number_value = Number(value ?? DEFAULT_RETRY_AFTER_MS);
    return Number.isFinite(number_value)
      ? Math.max(1, Math.trunc(number_value))
      : DEFAULT_RETRY_AFTER_MS;
  }
}
