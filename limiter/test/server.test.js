import assert from "node:assert/strict";
import { once } from "node:events";
import test from "node:test";

import { normalizeConfig } from "../src/config.js";
import { createLimiterApp } from "../src/server.js";

test("limiter server serves health, acquire, busy, release, and reacquire", async (t) => {
  const config = normalizeConfig({
    host: "127.0.0.1",
    port: 0,
    default_concurrency: 1,
    default_lease_ttl_ms: 300000,
    default_retry_after_ms: 7,
  });
  const { app } = createLimiterApp(config);
  const server = app.listen(0, "127.0.0.1");
  t.after(() => {
    server.close();
  });
  await once(server, "listening");
  const baseUrl = getServerUrl(server);

  assert.deepEqual(await getJson(`${baseUrl}/healthz`), { ok: true });

  const first = await postJson(`${baseUrl}/v1/leases/acquire`, {
    resource: "default",
    request_id: "request-a",
  });
  assert.equal(first.status, 200);
  assert.equal(first.body.ok, true);
  assert.equal(first.body.resource, "default");

  const busy = await postJson(`${baseUrl}/v1/leases/acquire`, {
    resource: "default",
    request_id: "request-b",
  });
  assert.deepEqual(busy, {
    status: 429,
    body: {
      ok: false,
      resource: "default",
      retry_after_ms: 7,
    },
  });

  const released = await postJson(`${baseUrl}/v1/leases/release`, {
    lease_id: first.body.lease_id,
  });
  assert.deepEqual(released, {
    status: 200,
    body: { ok: true, released: true },
  });

  const reacquired = await postJson(`${baseUrl}/v1/leases/acquire`, {
    resource: "default",
    request_id: "request-b",
  });
  assert.equal(reacquired.status, 200);
  assert.equal(reacquired.body.ok, true);
});

function getServerUrl(server) {
  const address = server.address();
  if (typeof address !== "object" || address === null) {
    throw new Error("server is not listening");
  }
  return `http://127.0.0.1:${address.port}`;
}

async function getJson(url) {
  const response = await fetch(url);
  return await response.json();
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return {
    status: response.status,
    body: await response.json(),
  };
}
