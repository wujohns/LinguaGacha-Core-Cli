import fs from "node:fs";
import path from "node:path";

import yaml from "js-yaml";

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8787;
const DEFAULT_RESOURCE_NAME = "default";
const DEFAULT_CONCURRENCY = 1;
const DEFAULT_LEASE_TTL_MS = 300_000;
const DEFAULT_RETRY_AFTER_MS = 1_000;

export function loadConfig(configPath) {
  const rawText = fs.readFileSync(configPath, "utf8");
  return normalizeConfig(yaml.load(rawText));
}

export function resolveConfigPath(argv) {
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] !== "--config") {
      continue;
    }
    const value = argv[index + 1];
    if (value === undefined || value.startsWith("--") || value.trim() === "") {
      throw new Error("Missing value for --config");
    }
    return path.resolve(value);
  }
  return path.resolve("config.yaml");
}

export function normalizeConfig(value) {
  const record = normalizeRecord(value);
  const defaultConcurrency = readPositiveInteger(
    record.default_concurrency ?? record.concurrency,
    DEFAULT_CONCURRENCY,
  );
  const defaultLeaseTtlMs = readPositiveInteger(
    record.default_lease_ttl_ms ?? record.lease_ttl_ms,
    DEFAULT_LEASE_TTL_MS,
  );
  const defaultRetryAfterMs = readPositiveInteger(
    record.default_retry_after_ms ?? record.retry_after_ms,
    DEFAULT_RETRY_AFTER_MS,
  );
  const resources = normalizeResources(record.resources, {
    concurrency: defaultConcurrency,
    leaseTtlMs: defaultLeaseTtlMs,
    retryAfterMs: defaultRetryAfterMs,
  });
  if (!resources.has(DEFAULT_RESOURCE_NAME)) {
    resources.set(DEFAULT_RESOURCE_NAME, {
      concurrency: defaultConcurrency,
      leaseTtlMs: defaultLeaseTtlMs,
      retryAfterMs: defaultRetryAfterMs,
    });
  }
  return {
    host: readText(record.host, DEFAULT_HOST),
    port: readPort(record.port, DEFAULT_PORT),
    defaultConcurrency,
    defaultLeaseTtlMs,
    defaultRetryAfterMs,
    resources,
  };
}

function normalizeResources(value, defaults) {
  const resources = new Map();
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return resources;
  }
  for (const [name, rawResource] of Object.entries(value)) {
    const normalizedName = name.trim();
    if (normalizedName === "") {
      continue;
    }
    const resource = normalizeRecord(rawResource);
    resources.set(normalizedName, {
      concurrency: readPositiveInteger(resource.concurrency, defaults.concurrency),
      leaseTtlMs: readPositiveInteger(
        resource.lease_ttl_ms ?? resource.leaseTtlMs,
        defaults.leaseTtlMs,
      ),
      retryAfterMs: readPositiveInteger(
        resource.retry_after_ms ?? resource.retryAfterMs,
        defaults.retryAfterMs,
      ),
    });
  }
  return resources;
}

function normalizeRecord(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? value : {};
}

function readText(value, fallback) {
  return typeof value === "string" && value.trim() !== "" ? value.trim() : fallback;
}

function readPort(value, fallback) {
  const numberValue = readPositiveInteger(value, fallback);
  return numberValue <= 65_535 ? numberValue : fallback;
}

function readPositiveInteger(value, fallback) {
  const numberValue = Number(value ?? fallback);
  return Number.isInteger(numberValue) && numberValue > 0 ? numberValue : fallback;
}
