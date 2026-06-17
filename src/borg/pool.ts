// A pool of per-tenant Borg "beings", each rooted at its own dataDir under one
// shared root (<root>/<tenantId>). This is how multi-tenancy is achieved WITHOUT
// any change to borg's storage/identity model: one being == one tenant == one
// dataDir, all colocated on one volume. Cross-tenant isolation is the filesystem
// boundary (separate sqlite file + lancedb dir per tenant), not a per-query
// filter, so there is no path along which one tenant's memory can leak into
// another's recall.
//
// The pool lazily opens beings on first use, dedupes concurrent opens of the
// same tenant, and evicts least-recently-used beings (closing them via the
// existing Borg.close()) once a max-open bound is exceeded. It NEVER closes a
// being that has an in-flight operation: LRU eviction skips in-use beings, and
// the deliberate close paths (evict / closeAll) DRAIN in-flight operations
// before tearing storage down. Schedulers are never started (Borg.open builds
// but does not start them), so idle beings cost no CPU.
//
// Access is exclusively through withTenant(): the being is reserved for the
// duration of the callback, so eviction can never close it mid-operation. There
// is intentionally no get() that hands back a bare being -- that would let a
// caller hold a reference the pool later closes (use-after-close).

import { join, resolve, sep } from "node:path";

import { Borg } from "../borg.js";
import { ConfigError } from "../util/errors.js";
import type { BorgOpenOptions } from "./types.js";

// Conservative tenant-id slug: lowercase alnum start, then alnum / _ / - up to
// 64 chars. Blocks "/", ".", ".." and anything else that could escape the root.
const DEFAULT_TENANT_ID_PATTERN = /^[a-z0-9][a-z0-9_-]{0,63}$/;

export type BorgPoolOptions = {
  // Root directory under which every tenant's dataDir lives: <root>/<tenantId>.
  root: string;
  // Per-tenant Borg.open options MINUS dataDir (the pool derives that per
  // tenant). Put the shared, stateless clients here (production AnthropicLLMClient
  // / CachingEmbeddingClient -- both tenant-independent). NOTE: a STATEFUL fake
  // such as FakeLLMClient holds one mutable response queue and must NOT be shared
  // across tenants in tests. Schedulers are never started regardless.
  openOptions?: Omit<BorgOpenOptions, "dataDir" | "tracerPath">;
  // Soft cap on simultaneously-open beings. Beyond this, least-recently-used
  // beings with no in-flight operation are closed. The bound can be temporarily
  // exceeded when every other being is in use (we never force-close in-flight
  // work); it reconciles as soon as a being is released. undefined / <= 0 =
  // unbounded.
  maxOpen?: number;
  // Validate tenant ids before path-joining (traversal guard). Defaults to a
  // conservative slug pattern.
  tenantIdPattern?: RegExp;
  // Per-tenant tracer file path. Returning a per-tenant path enables tracing
  // without the shared-BORG_TRACE interleave (N beings appending one file).
  tracerPathFor?: (tenantId: string) => string | undefined;
};

type Deferred = { readonly promise: Promise<void>; readonly resolve: () => void };

function deferred(): Deferred {
  let resolve!: () => void;
  const promise = new Promise<void>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

type PoolEntry = {
  readonly tenantId: string;
  readonly promise: Promise<Borg>;
  inUse: number;
  // Monotonic LRU stamp. Recency means LAST-COMPLETION by design (stamped on both
  // acquire and release), so a being that just finished a long op ranks as hot.
  lastUsed: number;
  // A deliberate close (evict/closeAll) requested while the being was in use.
  closeRequested: boolean;
  // Set once the close is actually running; idempotent dedupe for concurrent closers.
  closing?: Promise<void>;
  // Resolves when a deferred close finally completes (drain barrier for evict()).
  drained?: Deferred;
};

export class BorgPool {
  private readonly root: string;
  private readonly openOptions: Omit<BorgOpenOptions, "dataDir" | "tracerPath">;
  private readonly maxOpen: number | undefined;
  private readonly tenantIdPattern: RegExp;
  private readonly tracerPathFor: ((tenantId: string) => string | undefined) | undefined;

  private readonly entries = new Map<string, PoolEntry>();
  // Monotonic counter for LRU ordering -- deterministic, no clock dependency.
  private seq = 0;

  constructor(options: BorgPoolOptions) {
    if (!options.root.trim()) {
      throw new ConfigError("BorgPool requires a non-empty root directory");
    }
    this.root = options.root;
    this.openOptions = options.openOptions ?? {};
    this.maxOpen = options.maxOpen;
    this.tenantIdPattern = options.tenantIdPattern ?? DEFAULT_TENANT_ID_PATTERN;
    this.tracerPathFor = options.tracerPathFor;
  }

  /**
   * Run `fn` against the tenant's being, holding it open for the duration. The
   * being is reserved (inUse) across the call, so eviction can never close it
   * mid-operation. This is the only access path -- do not retain the `borg`
   * reference beyond the callback (the pool may close it afterwards).
   */
  async withTenant<T>(tenantId: string, fn: (borg: Borg) => T | Promise<T>): Promise<T> {
    const { entry, created } = this.acquire(tenantId);
    entry.inUse += 1;
    try {
      if (created) {
        await this.enforceMaxOpen(entry.tenantId);
      }
      const borg = await entry.promise;
      return await fn(borg);
    } finally {
      entry.inUse -= 1;
      entry.lastUsed = (this.seq += 1);
      if (entry.inUse === 0) {
        if (entry.closeRequested) {
          // Trigger the deferred close. The deliberate caller (evict/closeAll)
          // awaits the result via requestClose -> entry.closing, so we must not
          // await/throw here: a finally that rejects would mask fn's own result.
          // entry.closing is handled (its drain .then + the requestClose await),
          // so this is not an unhandled rejection.
          void this.finishClose(entry);
        } else {
          // Reconcile the soft bound on the release boundary (a concurrent burst
          // may have temporarily exceeded it while every being was in use).
          await this.enforceMaxOpen(entry.tenantId);
        }
      }
    }
  }

  /** Close a tenant's being and drop it from the pool, draining any in-flight op first. */
  async evict(tenantId: string): Promise<void> {
    if (!this.isValidTenantId(tenantId)) {
      return; // idempotent no-op for malformed/legacy ids during offboarding
    }
    const entry = this.entries.get(tenantId);
    if (entry === undefined) {
      return;
    }
    await this.requestClose(entry);
  }

  /** Close every open being, draining in-flight operations. */
  async closeAll(): Promise<void> {
    const results = await Promise.allSettled(
      [...this.entries.values()].map((entry) => this.requestClose(entry)),
    );
    const errors = results
      .filter((r): r is PromiseRejectedResult => r.status === "rejected")
      .map((r) => r.reason);
    if (errors.length > 0) {
      throw new AggregateError(errors, "BorgPool.closeAll: one or more beings failed to close");
    }
  }

  /** Whether a being is currently open (or opening) for this tenant. */
  has(tenantId: string): boolean {
    return this.isValidTenantId(tenantId) && this.entries.has(tenantId);
  }

  /** Number of currently-open (or opening/closing) beings. */
  size(): number {
    return this.entries.size;
  }

  /** Tenant ids with a currently-open (or opening/closing) being. */
  openTenantIds(): string[] {
    return [...this.entries.keys()];
  }

  private acquire(tenantId: string): { entry: PoolEntry; created: boolean } {
    const validated = this.validateTenantId(tenantId);
    const existing = this.entries.get(validated);
    if (existing !== undefined && !existing.closeRequested && existing.closing === undefined) {
      existing.lastUsed = (this.seq += 1);
      return { entry: existing, created: false };
    }

    // No reusable entry. Open a fresh being; if a prior one is closing or
    // pending close, serialize the open AFTER it fully closes so two writers
    // never share one dataDir.
    const priorClose =
      existing === undefined ? undefined : (existing.closing ?? existing.drained?.promise);
    const open =
      priorClose === undefined
        ? this.openBeing(validated)
        : priorClose.then(
            () => this.openBeing(validated),
            () => this.openBeing(validated),
          );
    const entry: PoolEntry = {
      tenantId: validated,
      inUse: 0,
      lastUsed: (this.seq += 1),
      closeRequested: false,
      // Cache the in-flight promise so concurrent acquires share one open. On
      // failure, drop the entry so a later call retries.
      promise: open.catch((error: unknown) => {
        if (this.entries.get(validated) === entry) {
          this.entries.delete(validated);
        }
        throw error;
      }),
    };
    this.entries.set(validated, entry);
    return { entry, created: true };
  }

  private openBeing(tenantId: string): Promise<Borg> {
    const tracerPath = this.tracerPathFor?.(tenantId);
    // With no explicit per-tenant tracer path, strip ambient BORG_TRACE so an
    // operator's debug env var doesn't commingle every tenant's trace content
    // into one shared file.
    const env = tracerPath === undefined ? this.tenantSafeEnv() : this.openOptions.env;
    return Borg.open({
      ...this.openOptions,
      ...(env === undefined ? {} : { env }),
      dataDir: join(this.root, tenantId),
      ...(tracerPath === undefined ? {} : { tracerPath }),
    });
  }

  private tenantSafeEnv(): NodeJS.ProcessEnv | undefined {
    const base = this.openOptions.env ?? process.env;
    if (base.BORG_TRACE === undefined && base.BORG_TRACE_PROMPTS === undefined) {
      return this.openOptions.env; // nothing to strip; preserve caller's choice
    }
    const clone: NodeJS.ProcessEnv = { ...base };
    delete clone.BORG_TRACE;
    delete clone.BORG_TRACE_PROMPTS;
    return clone;
  }

  // Deliberate close (evict/closeAll): drain in-flight ops, then close.
  private async requestClose(entry: PoolEntry): Promise<void> {
    if (entry.inUse === 0) {
      await this.finishClose(entry);
      return;
    }
    entry.closeRequested = true;
    entry.drained ??= deferred();
    await entry.drained.promise;
    // The drain barrier resolves on success or failure; await the actual close so
    // a deliberate evict()/closeAll() surfaces a close error rather than swallowing it.
    if (entry.closing !== undefined) {
      await entry.closing;
    }
  }

  private async enforceMaxOpen(exclude: string): Promise<void> {
    if (this.maxOpen === undefined || this.maxOpen <= 0) {
      return;
    }
    while (this.entries.size > this.maxOpen) {
      const victim = this.leastRecentlyUsedEvictable(exclude);
      if (victim === undefined) {
        return; // everything else is in use; soft bound temporarily exceeded
      }
      try {
        await this.finishClose(victim);
      } catch (error) {
        // Best-effort: a victim's close failure must not fail the unrelated
        // caller that triggered eviction. The entry is removed regardless.
        console.error(`BorgPool: failed to close evicted being "${victim.tenantId}"`, error);
      }
    }
  }

  private leastRecentlyUsedEvictable(exclude: string): PoolEntry | undefined {
    let best: PoolEntry | undefined;
    for (const entry of this.entries.values()) {
      if (
        entry.tenantId === exclude ||
        entry.inUse > 0 ||
        entry.closeRequested ||
        entry.closing !== undefined
      ) {
        continue;
      }
      if (best === undefined || entry.lastUsed < best.lastUsed) {
        best = entry;
      }
    }
    return best;
  }

  // Actually close the being. Idempotent. Keeps the entry in the map (marked
  // `closing`) until close completes so a concurrent same-tenant acquire chains
  // its open after this close rather than opening a second handle on the dataDir.
  private finishClose(entry: PoolEntry): Promise<void> {
    if (entry.closing !== undefined) {
      return entry.closing;
    }
    entry.closing = (async () => {
      let borg: Borg | undefined;
      try {
        borg = await entry.promise;
      } catch {
        return; // open failed; the entry was already de-registered, nothing to close
      }
      try {
        await borg.close();
      } finally {
        if (this.entries.get(entry.tenantId) === entry) {
          this.entries.delete(entry.tenantId);
        }
      }
    })();
    if (entry.drained !== undefined) {
      const drain = entry.drained;
      void entry.closing.then(
        () => drain.resolve(),
        () => drain.resolve(),
      );
    }
    return entry.closing;
  }

  private isValidTenantId(tenantId: string): boolean {
    if (!this.tenantIdPattern.test(tenantId)) {
      return false;
    }
    // Defense in depth: the pattern already blocks separators and "..", but
    // confirm the resolved dataDir is a single component strictly under the root.
    const rootResolved = resolve(this.root);
    const dir = resolve(rootResolved, tenantId);
    return dir === join(rootResolved, tenantId) && dir.startsWith(rootResolved + sep);
  }

  private validateTenantId(tenantId: string): string {
    if (!this.isValidTenantId(tenantId)) {
      throw new ConfigError(`Invalid tenantId for BorgPool: ${JSON.stringify(tenantId)}`);
    }
    return tenantId;
  }
}
