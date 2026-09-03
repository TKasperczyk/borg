import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { StreamEntry } from "../stream/index.js";
import { createSessionId, createStreamEntryId } from "../util/ids.js";
import { ResponseWaiterRegistry } from "./response-waiter-registry.js";

describe("ResponseWaiterRegistry", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("resolves every waiter for a covered entry after the terminal commit", async () => {
    const registry = new ResponseWaiterRegistry();
    const sessionId = createSessionId();
    const sourceId = createStreamEntryId();
    const terminalId = createStreamEntryId();
    const first = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: sourceId,
      timeoutMs: 10,
    });
    const second = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: sourceId,
      timeoutMs: 10,
    });
    registry.resolveTerminal("tenant", {
      id: terminalId,
      session_id: sessionId,
      timestamp: 2,
      kind: "agent_msg",
      content: "reply",
      sender_entity_id: null,
      reply_target_entity_id: null,
      compressed: false,
      response_to: {
        kind: "stream_backlog",
        from_cursor_exclusive: null,
        through_cursor_inclusive: { ts: 1, entryId: sourceId },
        source_entry_ids: [sourceId],
        count: 1,
      },
    } satisfies StreamEntry);

    await expect(first.promise).resolves.toEqual({
      status: "answered",
      terminal_id: terminalId,
      entry_ids: [sourceId],
      reply: "reply",
    });
    await expect(second.promise).resolves.toMatchObject({ status: "answered" });
    expect(registry.size()).toBe(0);
  });

  it("cleans up on timeout, cancellation, and shutdown", async () => {
    const registry = new ResponseWaiterRegistry();
    const sessionId = createSessionId();
    const timeout = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: createStreamEntryId(),
      timeoutMs: 5,
    });
    await vi.advanceTimersByTimeAsync(5);
    await expect(timeout.promise).resolves.toEqual({ status: "pending" });

    const cancelled = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: createStreamEntryId(),
      timeoutMs: 50,
    });
    cancelled.cancel();
    await expect(cancelled.promise).resolves.toEqual({ status: "pending" });

    const shutdown = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: createStreamEntryId(),
      timeoutMs: 50,
    });
    registry.shutdown();
    await expect(shutdown.promise).resolves.toEqual({ status: "pending" });
    expect(registry.size()).toBe(0);
  });

  it("holds one tenant lease per registered waiter until that waiter settles", async () => {
    const releases: Array<ReturnType<typeof vi.fn>> = [];
    const acquireTenantLease = vi.fn(() => {
      const release = vi.fn();
      releases.push(release);
      return { release };
    });
    const registry = new ResponseWaiterRegistry({ acquireTenantLease });
    const sessionId = createSessionId();
    const first = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: createStreamEntryId(),
      timeoutMs: 5,
    });
    const second = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: createStreamEntryId(),
      timeoutMs: 50,
    });

    expect(acquireTenantLease).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(5);
    await first.promise;
    expect(releases[0]).toHaveBeenCalledTimes(1);
    expect(releases[1]).not.toHaveBeenCalled();

    second.cancel();
    await second.promise;
    expect(releases[1]).toHaveBeenCalledTimes(1);
  });
});
