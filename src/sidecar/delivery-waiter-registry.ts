import type { SessionId } from "../util/ids.js";
import type { ResponseWaiterLease } from "./response-waiter-registry.js";

type Wake = "available" | "timeout" | "closed";
type Waiter = { resolve(wake: Wake): void };

/** Availability is only a signal: every awakened claimer must lease through SQLite. */
export class DeliveryWaiterRegistry {
  private readonly waiters = new Map<string, Set<Waiter>>();
  private shuttingDown = false;

  constructor(
    private readonly options: {
      acquireTenantLease?: (tenant: string) => ResponseWaiterLease;
    } = {},
  ) {}

  register(input: { tenant: string; sessionIds: readonly SessionId[]; timeoutMs: number }): {
    promise: Promise<Wake>;
    cancel(): void;
  } {
    if (this.shuttingDown) return { promise: Promise.resolve("closed"), cancel() {} };
    const lease = this.options.acquireTenantLease?.(input.tenant);
    const keys = [...new Set(input.sessionIds)].map((id) => JSON.stringify([input.tenant, id]));
    let settle!: (wake: Wake) => void;
    const promise = new Promise<Wake>((resolve) => {
      settle = resolve;
    });
    let settled = false;
    const waiter: Waiter = {
      resolve: (wake) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        for (const key of keys) {
          const bucket = this.waiters.get(key);
          bucket?.delete(waiter);
          if (bucket?.size === 0) this.waiters.delete(key);
        }
        lease?.release();
        settle(wake);
      },
    };
    const timer = setTimeout(() => waiter.resolve("timeout"), input.timeoutMs);
    timer.unref?.();
    for (const key of keys) {
      const bucket = this.waiters.get(key) ?? new Set<Waiter>();
      bucket.add(waiter);
      this.waiters.set(key, bucket);
    }
    return { promise, cancel: () => waiter.resolve("closed") };
  }

  notify(tenant: string, sessionId: SessionId): void {
    const bucket = this.waiters.get(JSON.stringify([tenant, sessionId]));
    for (const waiter of [...(bucket ?? [])]) waiter.resolve("available");
  }

  shutdown(): void {
    this.shuttingDown = true;
    for (const bucket of [...this.waiters.values()]) {
      for (const waiter of [...bucket]) waiter.resolve("closed");
    }
  }

  size(): number {
    return new Set([...this.waiters.values()].flatMap((bucket) => [...bucket])).size;
  }
}
