import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { StreamEntry } from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { createSessionId, createStreamEntryId } from "../util/ids.js";
import { ResponseWaiterRegistry } from "./response-waiter-registry.js";

describe("ResponseWaiterRegistry", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("observes registration, generation, timeout and cancellation independently per tenant", async () => {
    const observer = vi.fn();
    const registry = new ResponseWaiterRegistry({ observer });
    const sessionId = createSessionId();
    const entryId = createStreamEntryId();
    const input = { tenant: "a", sessionId, entryId, timeoutMs: 10 };
    const generating = registry.register(input);
    const timeout = registry.register({ ...input, tenant: "b" });
    registry.markGenerating({ tenant: "a", sessionId, entryIds: [entryId] });
    await expect(generating.promise).resolves.toEqual({ status: "generating" });
    await vi.advanceTimersByTimeAsync(10);
    await expect(timeout.promise).resolves.toEqual({ status: "pending" });
    const cancelled = registry.register({ ...input, seenGenerating: true });
    cancelled.cancel();
    cancelled.cancel();
    const shutdown = registry.register({ ...input, seenGenerating: true });
    registry.shutdown();
    await expect(shutdown.promise).resolves.toEqual({ status: "pending" });
    expect(observer.mock.calls.map(([event]) => [event.tenant, event.action])).toEqual([
      ["a", "registered"], ["b", "registered"], ["a", "generating"], ["b", "timeout"],
      ["a", "registered"], ["a", "cancelled"], ["a", "registered"], ["a", "cancelled"],
    ]);
  });

  it("does not retain a waiter or lease when its observer fails", async () => {
    const log = vi.spyOn(console, "error").mockImplementation(() => {});
    const release = vi.fn();
    try {
      const registry = new ResponseWaiterRegistry({
        observer: () => { throw new Error("observer"); }, acquireTenantLease: () => ({ release }),
      });
      const waiter = registry.register({ tenant: "a", sessionId: createSessionId(), entryId: createStreamEntryId(), timeoutMs: 10 });
      waiter.cancel();
      await expect(waiter.promise).resolves.toEqual({ status: "pending" });
      expect(registry.size()).toBe(0);
      expect(release).toHaveBeenCalledOnce();
    } finally { log.mockRestore(); }
  });

  it("resolves every waiter for a covered entry after the terminal commit", async () => {
    const observer = vi.fn();
    const registry = new ResponseWaiterRegistry({ observer });
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
    expect(observer.mock.calls.map(([event]) => event.action)).toEqual([
      "registered", "registered", "resolved", "resolved",
    ]);
  });

  it("serves remembered generating state until a terminal clears it and rejects late progress", async () => {
    const clock = new ManualClock(100);
    const registry = new ResponseWaiterRegistry({ clock });
    const sessionId = createSessionId();
    const generatingEntryId = createStreamEntryId();
    const otherEntryId = createStreamEntryId();
    registry.markGenerating({
      tenant: "tenant",
      sessionId,
      entryIds: [generatingEntryId, otherEntryId],
    });

    const late = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: generatingEntryId,
      timeoutMs: 50,
    });
    await expect(late.promise).resolves.toEqual({ status: "generating" });
    expect(registry.size()).toBe(0);

    registry.resolveTerminal("tenant", {
      id: createStreamEntryId(),
      session_id: sessionId,
      timestamp: 2,
      kind: "agent_observed",
      content: { reason: "done" },
      sender_entity_id: null,
      reply_target_entity_id: null,
      compressed: false,
      response_to: {
        kind: "stream_backlog",
        from_cursor_exclusive: null,
        through_cursor_inclusive: { ts: 1, entryId: generatingEntryId },
        source_entry_ids: [generatingEntryId],
        count: 1,
      },
    } satisfies StreamEntry);
    registry.markGenerating({
      tenant: "tenant",
      sessionId,
      entryIds: [generatingEntryId],
    });

    const cleared = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: generatingEntryId,
      timeoutMs: 5,
    });
    const stillGenerating = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: otherEntryId,
      timeoutMs: 5,
    });
    await expect(stillGenerating.promise).resolves.toEqual({ status: "generating" });
    await vi.advanceTimersByTimeAsync(5);
    await expect(cleared.promise).resolves.toEqual({ status: "pending" });

    clock.advance(10 * 60_000);
    registry.markGenerating({
      tenant: "tenant",
      sessionId,
      entryIds: [generatingEntryId],
    });
    const afterTombstoneExpiry = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: generatingEntryId,
      timeoutMs: 5,
    });
    await expect(afterTombstoneExpiry.promise).resolves.toEqual({ status: "generating" });
  });

  it("does not return or re-signal generating to a waiter that has already seen it", async () => {
    const registry = new ResponseWaiterRegistry();
    const sessionId = createSessionId();
    const entryId = createStreamEntryId();
    registry.markGenerating({ tenant: "tenant", sessionId, entryIds: [entryId] });

    const waiter = registry.register({
      tenant: "tenant",
      sessionId,
      entryId,
      timeoutMs: 5,
      seenGenerating: true,
    });
    registry.markGenerating({ tenant: "tenant", sessionId, entryIds: [entryId] });

    expect(registry.size()).toBe(1);
    await vi.advanceTimersByTimeAsync(5);
    await expect(waiter.promise).resolves.toEqual({ status: "pending" });
  });

  it("expires remembered generating state against the injected clock", async () => {
    const clock = new ManualClock(100);
    const registry = new ResponseWaiterRegistry({ clock, generatingTtlMs: 10 });
    const sessionId = createSessionId();
    const entryId = createStreamEntryId();
    registry.markGenerating({ tenant: "tenant", sessionId, entryIds: [entryId] });
    clock.advance(5);
    registry.markGenerating({ tenant: "tenant", sessionId, entryIds: [entryId] });
    clock.advance(5);

    const waiter = registry.register({
      tenant: "tenant",
      sessionId,
      entryId,
      timeoutMs: 5,
    });

    expect(registry.size()).toBe(1);
    await vi.advanceTimersByTimeAsync(5);
    await expect(waiter.promise).resolves.toEqual({ status: "pending" });
  });

  it("caps generating markers and terminal tombstones", async () => {
    const registry = new ResponseWaiterRegistry({
      maxGeneratingEntries: 2,
      maxTerminalTombstones: 2,
    });
    const sessionId = createSessionId();
    const entryIds = [createStreamEntryId(), createStreamEntryId(), createStreamEntryId()];
    registry.markGenerating({ tenant: "tenant", sessionId, entryIds });
    const evictedGenerating = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: entryIds[0]!,
      timeoutMs: 5,
    });
    const retainedGenerating = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: entryIds[2]!,
      timeoutMs: 5,
    });
    await expect(retainedGenerating.promise).resolves.toEqual({ status: "generating" });
    await vi.advanceTimersByTimeAsync(5);
    await expect(evictedGenerating.promise).resolves.toEqual({ status: "pending" });

    registry.resolveTerminal("tenant", {
      id: createStreamEntryId(),
      session_id: sessionId,
      timestamp: 2,
      kind: "agent_observed",
      content: { reason: "done" },
      sender_entity_id: null,
      reply_target_entity_id: null,
      compressed: false,
      response_to: {
        kind: "stream_backlog",
        from_cursor_exclusive: null,
        through_cursor_inclusive: { ts: 1, entryId: entryIds[2]! },
        source_entry_ids: entryIds,
        count: entryIds.length,
      },
    } satisfies StreamEntry);

    registry.markGenerating({ tenant: "tenant", sessionId, entryIds: [entryIds[0]!] });
    registry.markGenerating({ tenant: "tenant", sessionId, entryIds: [entryIds[2]!] });
    const evictedTombstone = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: entryIds[0]!,
      timeoutMs: 5,
    });
    const retainedTombstone = registry.register({
      tenant: "tenant",
      sessionId,
      entryId: entryIds[2]!,
      timeoutMs: 5,
    });

    await expect(evictedTombstone.promise).resolves.toEqual({ status: "generating" });
    await vi.advanceTimersByTimeAsync(5);
    await expect(retainedTombstone.promise).resolves.toEqual({ status: "pending" });
  });

  it("idempotently wakes current waiters when entries start generating", async () => {
    const registry = new ResponseWaiterRegistry();
    const sessionId = createSessionId();
    const entryId = createStreamEntryId();
    const first = registry.register({
      tenant: "tenant",
      sessionId,
      entryId,
      timeoutMs: 50,
    });
    const second = registry.register({
      tenant: "tenant",
      sessionId,
      entryId,
      timeoutMs: 50,
    });

    registry.markGenerating({ tenant: "tenant", sessionId, entryIds: [entryId, entryId] });
    registry.markGenerating({ tenant: "tenant", sessionId, entryIds: [entryId] });

    await expect(first.promise).resolves.toEqual({ status: "generating" });
    await expect(second.promise).resolves.toEqual({ status: "generating" });
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
