import express from "express";

import { LeaseStore } from "./lease-store.js";

export function createLimiterApp(config, options = {}) {
  const store = options.store ?? new LeaseStore(config);
  const app = express();
  app.disable("x-powered-by");
  app.use(express.json({ limit: "1mb" }));

  app.get("/healthz", (_request, response) => {
    response.json({ ok: true });
  });

  app.get("/v1/status", (_request, response) => {
    response.json(store.snapshot());
  });

  app.post("/v1/leases/acquire", (request, response) => {
    const body = normalizeRecord(request.body);
    const resource = readText(body.resource, "default");
    const requestId = readText(body.request_id, "");
    if (requestId === "") {
      response.status(400).json({ ok: false, error: "request_id is required" });
      return;
    }
    const result = store.acquire({ resource, requestId });
    if (result.ok) {
      response.json({
        ok: true,
        lease_id: result.leaseId,
        resource: result.resource,
        expires_in_ms: result.expiresInMs,
      });
      return;
    }
    response.status(429).json({
      ok: false,
      resource: result.resource,
      retry_after_ms: result.retryAfterMs,
    });
  });

  app.post("/v1/leases/release", (request, response) => {
    const body = normalizeRecord(request.body);
    const leaseId = readText(body.lease_id, "");
    if (leaseId === "") {
      response.status(400).json({ ok: false, error: "lease_id is required" });
      return;
    }
    response.json(store.release(leaseId));
  });

  app.use((_request, response) => {
    response.status(404).json({ ok: false, error: "not found" });
  });

  app.use((error, _request, response, _next) => {
    response.status(400).json({
      ok: false,
      error: error instanceof Error ? error.message : "invalid request",
    });
  });

  return { app, store };
}

export function startLimiterServer(config, options = {}) {
  const { app, store } = createLimiterApp(config, options);
  const server = app.listen(config.port, config.host);
  return { app, server, store };
}

function normalizeRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value : {};
}

function readText(value, fallback) {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : fallback;
}
