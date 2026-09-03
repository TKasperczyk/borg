import type { StreamEntry } from "../stream/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import type { SessionId, StreamEntryId } from "../util/ids.js";

export type InboxAwaitResponse =
  | { status: "pending" }
  | { status: "generating" }
  | {
      status: "answered";
      terminal_id: StreamEntryId;
      entry_ids: StreamEntryId[];
      reply: string;
    }
  | {
      status: "observed";
      terminal_id: StreamEntryId;
      entry_ids: StreamEntryId[];
    };

type Waiter = {
  acceptGenerating: boolean;
  resolve: (response: InboxAwaitResponse) => void;
  timer: ReturnType<typeof setTimeout>;
};

export type ResponseWaiterHandle = {
  promise: Promise<InboxAwaitResponse>;
  cancel(): void;
};

export type ResponseWaiterLease = {
  release(): void;
};

export type ResponseWaiterRegistryOptions = {
  acquireTenantLease?: (tenant: string) => ResponseWaiterLease;
  clock?: Clock;
  generatingTtlMs?: number;
  maxGeneratingEntries?: number;
  maxTerminalTombstones?: number;
};

const DEFAULT_GENERATING_TTL_MS = 130_000;
const TERMINAL_TOMBSTONE_TTL_MS = 10 * 60_000;
const DEFAULT_MAX_LIFECYCLE_ENTRIES = 10_000;

function waiterKey(tenant: string, sessionId: SessionId, entryId: StreamEntryId): string {
  return JSON.stringify([tenant, sessionId, entryId]);
}

export function awaitResponseForTerminal(input: {
  terminalEntry: StreamEntry;
}): Exclude<InboxAwaitResponse, { status: "pending" } | { status: "generating" }> {
  const responseTo = input.terminalEntry.response_to;
  if (responseTo === undefined) {
    throw new Error("terminal entry is missing response_to");
  }
  const common = {
    terminal_id: input.terminalEntry.id,
    entry_ids: [...responseTo.source_entry_ids],
  };
  if (input.terminalEntry.kind === "agent_msg") {
    if (typeof input.terminalEntry.content !== "string") {
      throw new Error("agent_msg terminal content must be text");
    }
    return { status: "answered", ...common, reply: input.terminalEntry.content };
  }
  return { status: "observed", ...common };
}

export class ResponseWaiterRegistry {
  private readonly waiters = new Map<string, Set<Waiter>>();
  private readonly generating = new Map<string, number>();
  private readonly terminalTombstones = new Map<string, number>();
  private readonly clock: Clock;
  private readonly generatingTtlMs: number;
  private readonly maxGeneratingEntries: number;
  private readonly maxTerminalTombstones: number;
  private shuttingDown = false;

  constructor(private readonly options: ResponseWaiterRegistryOptions = {}) {
    this.clock = options.clock ?? new SystemClock();
    const generatingTtlMs = options.generatingTtlMs ?? DEFAULT_GENERATING_TTL_MS;
    this.generatingTtlMs =
      Number.isFinite(generatingTtlMs) && generatingTtlMs > 0
        ? generatingTtlMs
        : DEFAULT_GENERATING_TTL_MS;
    const maxGeneratingEntries = options.maxGeneratingEntries ?? DEFAULT_MAX_LIFECYCLE_ENTRIES;
    this.maxGeneratingEntries =
      Number.isSafeInteger(maxGeneratingEntries) && maxGeneratingEntries > 0
        ? maxGeneratingEntries
        : DEFAULT_MAX_LIFECYCLE_ENTRIES;
    const maxTerminalTombstones = options.maxTerminalTombstones ?? DEFAULT_MAX_LIFECYCLE_ENTRIES;
    this.maxTerminalTombstones =
      Number.isSafeInteger(maxTerminalTombstones) && maxTerminalTombstones > 0
        ? maxTerminalTombstones
        : DEFAULT_MAX_LIFECYCLE_ENTRIES;
  }

  register(input: {
    tenant: string;
    sessionId: SessionId;
    entryId: StreamEntryId;
    timeoutMs: number;
    seenGenerating?: boolean;
  }): ResponseWaiterHandle {
    if (this.shuttingDown) {
      return { promise: Promise.resolve({ status: "pending" }), cancel() {} };
    }

    this.pruneLifecycleState();
    const key = waiterKey(input.tenant, input.sessionId, input.entryId);
    if (input.seenGenerating !== true && this.generating.has(key)) {
      return { promise: Promise.resolve({ status: "generating" }), cancel() {} };
    }
    const lease = this.options.acquireTenantLease?.(input.tenant);
    let settle!: (response: InboxAwaitResponse) => void;
    const promise = new Promise<InboxAwaitResponse>((resolve) => {
      settle = resolve;
    });
    const timer = setTimeout(() => removeAndResolve({ status: "pending" }), input.timeoutMs);
    timer.unref?.();
    const waiter: Waiter = {
      acceptGenerating: input.seenGenerating !== true,
      resolve: settle,
      timer,
    };
    const bucket = this.waiters.get(key) ?? new Set<Waiter>();
    bucket.add(waiter);
    this.waiters.set(key, bucket);
    let settled = false;

    const removeAndResolve = (response: InboxAwaitResponse) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timer);
      bucket.delete(waiter);
      if (bucket.size === 0) {
        this.waiters.delete(key);
      }
      lease?.release();
      settle(response);
    };
    waiter.resolve = removeAndResolve;

    return {
      promise,
      cancel: () => removeAndResolve({ status: "pending" }),
    };
  }

  markGenerating(input: {
    tenant: string;
    sessionId: SessionId;
    entryIds: readonly StreamEntryId[];
  }): void {
    if (this.shuttingDown) {
      return;
    }

    const now = this.clock.now();
    this.pruneLifecycleState(now);
    for (const entryId of new Set(input.entryIds)) {
      const key = waiterKey(input.tenant, input.sessionId, entryId);
      if (this.terminalTombstones.has(key)) {
        continue;
      }
      if (!this.generating.has(key)) {
        this.generating.set(key, now + this.generatingTtlMs);
        this.trimLifecycleMap(this.generating, this.maxGeneratingEntries);
      }
      const bucket = this.waiters.get(key);
      if (bucket === undefined) {
        continue;
      }
      for (const waiter of [...bucket]) {
        if (waiter.acceptGenerating) {
          waiter.resolve({ status: "generating" });
        }
      }
    }
  }

  resolveTerminal(tenant: string, terminalEntry: StreamEntry): void {
    if (this.shuttingDown) {
      return;
    }

    const response = awaitResponseForTerminal({ terminalEntry });
    const now = this.clock.now();
    this.pruneLifecycleState(now);
    for (const entryId of response.entry_ids) {
      const key = waiterKey(tenant, terminalEntry.session_id, entryId);
      this.generating.delete(key);
      this.terminalTombstones.delete(key);
      this.terminalTombstones.set(key, now + TERMINAL_TOMBSTONE_TTL_MS);
      this.trimLifecycleMap(this.terminalTombstones, this.maxTerminalTombstones);
      const bucket = this.waiters.get(key);
      if (bucket === undefined) {
        continue;
      }
      for (const waiter of [...bucket]) {
        waiter.resolve(response);
      }
    }
  }

  shutdown(): void {
    this.shuttingDown = true;
    this.generating.clear();
    this.terminalTombstones.clear();
    for (const bucket of [...this.waiters.values()]) {
      for (const waiter of [...bucket]) {
        waiter.resolve({ status: "pending" });
      }
    }
  }

  size(): number {
    let count = 0;
    for (const bucket of this.waiters.values()) {
      count += bucket.size;
    }
    return count;
  }

  private pruneLifecycleState(now = this.clock.now()): void {
    for (const [key, expiresAt] of this.generating) {
      if (expiresAt <= now) {
        this.generating.delete(key);
      }
    }
    for (const [key, expiresAt] of this.terminalTombstones) {
      if (expiresAt <= now) {
        this.terminalTombstones.delete(key);
      }
    }
  }

  private trimLifecycleMap(entries: Map<string, number>, maxEntries: number): void {
    while (entries.size > maxEntries) {
      const oldest = entries.keys().next().value;
      if (oldest === undefined) {
        return;
      }
      entries.delete(oldest);
    }
  }
}
