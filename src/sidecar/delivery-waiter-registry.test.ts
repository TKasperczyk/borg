import { afterEach, describe, expect, it, vi } from "vitest";
import { createSessionId } from "../util/ids.js";
import { DeliveryWaiterRegistry } from "./delivery-waiter-registry.js";

afterEach(() => vi.useRealTimers());

describe("DeliveryWaiterRegistry", () => {
  it("wakes only matching tenants/sessions and releases each lease once", async () => {
    const release = vi.fn();
    const acquireTenantLease = vi.fn(() => ({ release }));
    const registry = new DeliveryWaiterRegistry({ acquireTenantLease });
    const sessionId = createSessionId();
    const other = createSessionId();
    const first = registry.register({
      tenant: "a",
      sessionIds: [sessionId, other],
      timeoutMs: 1000,
    });
    const second = registry.register({ tenant: "b", sessionIds: [sessionId], timeoutMs: 1000 });
    registry.notify("a", other);
    expect(await first.promise).toBe("available");
    first.cancel();
    expect(release).toHaveBeenCalledTimes(1);
    expect(registry.size()).toBe(1);
    registry.shutdown();
    expect(await second.promise).toBe("closed");
    expect(release).toHaveBeenCalledTimes(2);
    expect(registry.size()).toBe(0);
    expect(
      await registry.register({ tenant: "b", sessionIds: [sessionId], timeoutMs: 1000 }).promise,
    ).toBe("closed");
    expect(acquireTenantLease).toHaveBeenCalledTimes(2);
  });

  it("cleans every session bucket on timeout and cancellation", async () => {
    vi.useFakeTimers();
    const registry = new DeliveryWaiterRegistry();
    const input = {
      tenant: "a",
      sessionIds: [createSessionId(), createSessionId()],
      timeoutMs: 10,
    };
    const waiter = registry.register(input);
    await vi.advanceTimersByTimeAsync(10);
    expect(await waiter.promise).toBe("timeout");
    expect(registry.size()).toBe(0);
    const cancelled = registry.register(input);
    cancelled.cancel();
    expect(await cancelled.promise).toBe("closed");
    expect(registry.size()).toBe(0);
  });
});
