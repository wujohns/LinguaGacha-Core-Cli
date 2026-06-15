import assert from "node:assert/strict";
import test from "node:test";

import { normalizeConfig } from "../src/config.js";
import { LeaseStore } from "../src/lease-store.js";

test("LeaseStore grants up to resource concurrency and releases leases", () => {
  const config = normalizeConfig({
    default_concurrency: 1,
    default_lease_ttl_ms: 1000,
    default_retry_after_ms: 25,
    resources: {
      deepseek: {
        concurrency: 1,
        lease_ttl_ms: 1000,
        retry_after_ms: 25,
      },
    },
  });
  const store = new LeaseStore(config, { now: () => 100 });

  const first = store.acquire({ resource: "deepseek", requestId: "request-a" });
  assert.equal(first.ok, true);

  const duplicate = store.acquire({ resource: "deepseek", requestId: "request-a" });
  assert.deepEqual(
    { ok: duplicate.ok, leaseId: duplicate.leaseId },
    { ok: true, leaseId: first.leaseId },
  );

  assert.deepEqual(store.acquire({ resource: "deepseek", requestId: "request-b" }), {
    ok: false,
    resource: "deepseek",
    retryAfterMs: 25,
  });

  assert.deepEqual(store.release(first.leaseId), { ok: true, released: true });
  assert.equal(store.acquire({ resource: "deepseek", requestId: "request-b" }).ok, true);
});

test("LeaseStore expires leases by resource TTL", () => {
  let now = 100;
  const config = normalizeConfig({
    default_concurrency: 1,
    default_lease_ttl_ms: 10,
    default_retry_after_ms: 5,
  });
  const store = new LeaseStore(config, { now: () => now });

  assert.equal(store.acquire({ resource: "default", requestId: "request-a" }).ok, true);
  assert.deepEqual(store.acquire({ resource: "default", requestId: "request-b" }), {
    ok: false,
    resource: "default",
    retryAfterMs: 5,
  });

  now = 111;
  assert.deepEqual(
    pick(store.acquire({ resource: "default", requestId: "request-b" }), ["ok", "resource"]),
    { ok: true, resource: "default" },
  );
});

function pick(record, keys) {
  return Object.fromEntries(keys.map((key) => [key, record[key]]));
}
