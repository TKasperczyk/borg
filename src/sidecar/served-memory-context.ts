import { randomUUID } from "node:crypto";

import type { CommitmentRecord } from "../memory/commitments/index.js";
import type { RetrievedEpisode } from "../retrieval/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import type { EntityId, SessionId } from "../util/ids.js";

type ServedMemoryContext = {
  sessionId: SessionId;
  audienceEntityId: EntityId;
  senderEntityId: EntityId;
  episodes: readonly RetrievedEpisode[];
  commitments: readonly CommitmentRecord[];
};

const DEFAULT_TTL_MS = 30 * 60_000;
const DEFAULT_MAX_PER_TENANT = 16;
const DEFAULT_MAX_TENANTS = 64;

// Existing caches own embeddings, waiters, or semantic verdicts. This registry owns only
// served records; it never keeps a pooled Borg alive. IDs are independent of completion order.
export class ServedMemoryContextRegistry {
  private readonly tenants = new Map<
    string,
    Map<string, ServedMemoryContext & { expiresAt: number }>
  >();
  private readonly clock: Clock;
  private readonly ttlMs: number;
  private readonly maxPerTenant: number;
  private readonly maxTenants: number;

  constructor(
    options: {
      clock?: Clock;
      ttlMs?: number;
      maxPerTenant?: number;
      maxTenants?: number;
    } = {},
  ) {
    this.clock = options.clock ?? new SystemClock();
    const ttlMs = options.ttlMs ?? DEFAULT_TTL_MS;
    this.ttlMs = Number.isFinite(ttlMs) && ttlMs > 0 ? ttlMs : DEFAULT_TTL_MS;
    const maxPerTenant = options.maxPerTenant ?? DEFAULT_MAX_PER_TENANT;
    this.maxPerTenant =
      Number.isSafeInteger(maxPerTenant) && maxPerTenant > 0
        ? maxPerTenant
        : DEFAULT_MAX_PER_TENANT;
    const maxTenants = options.maxTenants ?? DEFAULT_MAX_TENANTS;
    this.maxTenants =
      Number.isSafeInteger(maxTenants) && maxTenants > 0 ? maxTenants : DEFAULT_MAX_TENANTS;
  }

  private prune(): number {
    const now = this.clock.now();
    for (const [tenant, snapshots] of this.tenants) {
      for (const [id, snapshot] of snapshots) {
        if (snapshot.expiresAt <= now) snapshots.delete(id);
      }
      if (snapshots.size === 0) this.tenants.delete(tenant);
    }
    return now;
  }

  put(tenant: string, snapshot: ServedMemoryContext): string {
    const now = this.prune();
    const snapshots = this.tenants.get(tenant) ?? new Map();
    const id = randomUUID();
    snapshots.set(id, {
      ...structuredClone(snapshot),
      expiresAt: now + this.ttlMs,
    });
    while (snapshots.size > this.maxPerTenant) {
      snapshots.delete(snapshots.keys().next().value!);
    }
    this.tenants.delete(tenant);
    this.tenants.set(tenant, snapshots);
    while (this.tenants.size > this.maxTenants) {
      this.tenants.delete(this.tenants.keys().next().value!);
    }
    return id;
  }

  get(tenant: string, sessionId: SessionId, contextId: string): ServedMemoryContext | undefined {
    this.prune();
    const snapshots = this.tenants.get(tenant);
    const snapshot = snapshots?.get(contextId);
    if (snapshot?.sessionId !== sessionId) return undefined;
    this.tenants.delete(tenant);
    this.tenants.set(tenant, snapshots!);
    snapshots!.delete(contextId);
    snapshots!.set(contextId, snapshot);
    return snapshot;
  }
}
