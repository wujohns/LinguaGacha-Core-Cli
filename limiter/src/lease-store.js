import crypto from "node:crypto";

export class LeaseStore {
  constructor(config, options = {}) {
    this.config = config;
    this.now = options.now ?? (() => Date.now());
    this.leases = new Map();
    this.requestIndex = new Map();
  }

  acquire(input) {
    const resource = input.resource.trim() === "" ? "default" : input.resource.trim();
    const requestId = input.requestId.trim();
    this.pruneExpired();
    const existing = this.findExistingRequestLease(resource, requestId);
    if (existing !== null) {
      return this.toSuccess(existing);
    }
    const resourceConfig = this.resolveResource(resource);
    const inUse = this.countResourceLeases(resource);
    if (inUse >= resourceConfig.concurrency) {
      return {
        ok: false,
        resource,
        retryAfterMs: resourceConfig.retryAfterMs,
      };
    }
    const lease = {
      leaseId: crypto.randomUUID(),
      resource,
      requestId,
      expiresAt: this.now() + resourceConfig.leaseTtlMs,
    };
    this.leases.set(lease.leaseId, lease);
    this.requestIndex.set(this.buildRequestKey(resource, requestId), lease.leaseId);
    return this.toSuccess(lease);
  }

  release(leaseId) {
    this.pruneExpired();
    const lease = this.leases.get(leaseId);
    if (lease === undefined) {
      return { ok: true, released: false };
    }
    this.deleteLease(lease);
    return { ok: true, released: true };
  }

  snapshot() {
    this.pruneExpired();
    const resources = {};
    for (const [name, resourceConfig] of this.config.resources.entries()) {
      resources[name] = {
        concurrency: resourceConfig.concurrency,
        in_use: this.countResourceLeases(name),
        lease_ttl_ms: resourceConfig.leaseTtlMs,
        retry_after_ms: resourceConfig.retryAfterMs,
      };
    }
    return { ok: true, resources };
  }

  findExistingRequestLease(resource, requestId) {
    const leaseId = this.requestIndex.get(this.buildRequestKey(resource, requestId));
    if (leaseId === undefined) {
      return null;
    }
    return this.leases.get(leaseId) ?? null;
  }

  toSuccess(lease) {
    return {
      ok: true,
      leaseId: lease.leaseId,
      resource: lease.resource,
      expiresInMs: Math.max(1, lease.expiresAt - this.now()),
    };
  }

  resolveResource(resource) {
    return (
      this.config.resources.get(resource) ?? {
        concurrency: this.config.defaultConcurrency,
        leaseTtlMs: this.config.defaultLeaseTtlMs,
        retryAfterMs: this.config.defaultRetryAfterMs,
      }
    );
  }

  countResourceLeases(resource) {
    let count = 0;
    for (const lease of this.leases.values()) {
      if (lease.resource === resource) {
        count += 1;
      }
    }
    return count;
  }

  pruneExpired() {
    const now = this.now();
    for (const lease of [...this.leases.values()]) {
      if (lease.expiresAt <= now) {
        this.deleteLease(lease);
      }
    }
  }

  deleteLease(lease) {
    this.leases.delete(lease.leaseId);
    this.requestIndex.delete(this.buildRequestKey(lease.resource, lease.requestId));
  }

  buildRequestKey(resource, requestId) {
    return `${resource}\0${requestId}`;
  }
}
