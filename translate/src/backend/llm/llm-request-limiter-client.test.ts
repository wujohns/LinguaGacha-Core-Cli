import http from "node:http";
import { once } from "node:events";

import { afterEach, describe, expect, it } from "vitest";

import { LLMRequestLimiterClient } from "./llm-request-limiter-client";
import type { LLMClientPort, LLMRequestBody, LLMRequestResult } from "./llm-types";

interface LimiterRequest {
  pathname: string;
  body: Record<string, unknown>;
}

const request_body: LLMRequestBody = {
  run_id: "run-1",
  work_unit_id: "unit-1",
  model: {},
  config_snapshot: {},
  messages: [{ role: "user", content: "hello" }],
};

const success_result: LLMRequestResult = {
  response_think: "",
  response_result: "ok",
  input_tokens: 1,
  output_tokens: 1,
  cancelled: false,
  timeout: false,
  degraded: false,
};

describe("LLMRequestLimiterClient", () => {
  const servers: http.Server[] = [];

  afterEach(async () => {
    await Promise.all(
      servers.splice(0).map(
        (server) =>
          new Promise<void>((resolve, reject) => {
            server.close((error) => (error === undefined ? resolve() : reject(error)));
          }),
      ),
    );
  });

  it("acquires a lease before the LLM request and releases it afterwards", async () => {
    const requests: LimiterRequest[] = [];
    const url = await start_limiter_server(servers, requests, (pathname) => {
      if (pathname === "/v1/leases/acquire") {
        return { ok: true, lease_id: "lease-1" };
      }
      return { ok: true };
    });
    const inner = create_inner_client(async () => success_result);
    const client = new LLMRequestLimiterClient(inner, { url, resource: "deepseek" });

    await expect(client.request(request_body, new AbortController().signal)).resolves.toEqual(
      success_result,
    );

    expect(requests).toEqual([
      {
        pathname: "/v1/leases/acquire",
        body: {
          resource: "deepseek",
          request_id: "run-1:unit-1",
        },
      },
      {
        pathname: "/v1/leases/release",
        body: { lease_id: "lease-1" },
      },
    ]);
  });

  it("waits and retries when the limiter is busy", async () => {
    const requests: LimiterRequest[] = [];
    let acquire_count = 0;
    const url = await start_limiter_server(servers, requests, (pathname) => {
      if (pathname === "/v1/leases/acquire") {
        acquire_count += 1;
        return acquire_count === 1
          ? { ok: false, retry_after_ms: 1 }
          : { ok: true, lease_id: "lease-2" };
      }
      return { ok: true };
    });
    const inner = create_inner_client(async () => success_result);
    const client = new LLMRequestLimiterClient(inner, {
      url,
      resource: "deepseek",
    });

    await client.request(request_body, new AbortController().signal);

    expect(requests.map((request) => request.pathname)).toEqual([
      "/v1/leases/acquire",
      "/v1/leases/acquire",
      "/v1/leases/release",
    ]);
    expect(requests[0]?.body).toEqual({
      resource: "deepseek",
      request_id: "run-1:unit-1",
    });
  });

  it("releases the lease when the inner LLM request throws", async () => {
    const requests: LimiterRequest[] = [];
    const url = await start_limiter_server(servers, requests, (pathname) => {
      if (pathname === "/v1/leases/acquire") {
        return { ok: true, lease_id: "lease-3" };
      }
      return { ok: true };
    });
    const client = new LLMRequestLimiterClient(
      create_inner_client(async () => {
        throw new Error("llm failed");
      }),
      { url, resource: "deepseek" },
    );

    await expect(client.request(request_body, new AbortController().signal)).rejects.toThrow(
      "llm failed",
    );

    expect(requests.map((request) => request.pathname)).toEqual([
      "/v1/leases/acquire",
      "/v1/leases/release",
    ]);
  });
});

function create_inner_client(request: LLMClientPort["request"]): LLMClientPort {
  return { request };
}

async function start_limiter_server(
  servers: http.Server[],
  requests: LimiterRequest[],
  respond: (pathname: string, body: Record<string, unknown>) => Record<string, unknown>,
): Promise<string> {
  const server = http.createServer((request, response) => {
    const chunks: Buffer[] = [];
    request.on("data", (chunk: Buffer) => chunks.push(chunk));
    request.on("end", () => {
      const body_text = Buffer.concat(chunks).toString("utf8");
      const body = body_text === "" ? {} : (JSON.parse(body_text) as Record<string, unknown>);
      const pathname = new URL(request.url ?? "/", "http://127.0.0.1").pathname;
      requests.push({ pathname, body });
      response.setHeader("content-type", "application/json");
      response.end(JSON.stringify(respond(pathname, body)));
    });
  });
  server.listen(0, "127.0.0.1");
  await once(server, "listening");
  servers.push(server);
  const address = server.address();
  if (typeof address !== "object" || address === null) {
    throw new Error("failed to start limiter test server");
  }
  return `http://127.0.0.1:${address.port.toString()}`;
}
