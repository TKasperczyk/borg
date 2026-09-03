import type { StreamEntry } from "../stream/index.js";
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
};

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
  private readonly generating = new Set<string>();
  private shuttingDown = false;

  constructor(private readonly options: ResponseWaiterRegistryOptions = {}) {}

  register(input: {
    tenant: string;
    sessionId: SessionId;
    entryId: StreamEntryId;
    timeoutMs: number;
  }): ResponseWaiterHandle {
    if (this.shuttingDown) {
      return { promise: Promise.resolve({ status: "pending" }), cancel() {} };
    }

    const key = waiterKey(input.tenant, input.sessionId, input.entryId);
    if (this.generating.has(key)) {
      return { promise: Promise.resolve({ status: "generating" }), cancel() {} };
    }
    const lease = this.options.acquireTenantLease?.(input.tenant);
    let settle!: (response: InboxAwaitResponse) => void;
    const promise = new Promise<InboxAwaitResponse>((resolve) => {
      settle = resolve;
    });
    const timer = setTimeout(() => removeAndResolve({ status: "pending" }), input.timeoutMs);
    timer.unref?.();
    const waiter: Waiter = { resolve: settle, timer };
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

    for (const entryId of new Set(input.entryIds)) {
      const key = waiterKey(input.tenant, input.sessionId, entryId);
      this.generating.add(key);
      const bucket = this.waiters.get(key);
      if (bucket === undefined) {
        continue;
      }
      for (const waiter of [...bucket]) {
        waiter.resolve({ status: "generating" });
      }
    }
  }

  resolveTerminal(tenant: string, terminalEntry: StreamEntry): void {
    const response = awaitResponseForTerminal({ terminalEntry });
    for (const entryId of response.entry_ids) {
      const key = waiterKey(tenant, terminalEntry.session_id, entryId);
      this.generating.delete(key);
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
}
