import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_SESSION_ID, createSessionId } from "borg";

import { createLiveBridge, LEDGER_CACHE_MAX } from "./live.js";

afterEach(() => vi.useRealTimers());

describe("live diagnostic retention", () => {
  it("evicts old ledgers and retains the most recently produced ledgers", () => {
    const live = createLiveBridge();
    for (let i = 0; i < LEDGER_CACHE_MAX + 10; i++) {
      live.tracer.emit("evidence_ledger.built", {
        turnId: `turn_${i}`,
        ledger: { sections: [], marker: i },
      });
    }
    expect(live.ledgerCache.size).toBe(LEDGER_CACHE_MAX);
    expect(live.ledgerCache.has("turn_0")).toBe(false);
    expect(live.ledgerCache.get(`turn_${LEDGER_CACHE_MAX + 9}`)).toMatchObject({
      marker: LEDGER_CACHE_MAX + 9,
    });
  });

  it("bounds session buffers and expires inactive sessions without clearing active turn state", () => {
    vi.useFakeTimers();
    const live = createLiveBridge();
    for (let i = 0; i < 200; i++) {
      live.broadcaster.broadcast({ type: "test", ts: Date.now(), session_id: createSessionId() });
    }
    expect(live.broadcaster.memoryStats().bufferedSessions).toBe(128);
    live.tracer.emit("turn_phase.started", {
      turnId: "turn_a",
      turn_id: "turn_a",
      session_id: DEFAULT_SESSION_ID,
      phase: "delib",
    });
    vi.advanceTimersByTime(60001);
    expect(live.broadcaster.memoryStats()).toMatchObject({ activeTurns: 1, bufferedSessions: 0 });
    live.tracer.emit("turn.terminal", {
      turnId: "turn_a",
      turn_id: "turn_a",
      session_id: DEFAULT_SESSION_ID,
    });
    expect(live.broadcaster.memoryStats().activeTurns).toBe(0);
  });

  it("does not create empty retained buffers when clients subscribe to unseen sessions", () => {
    const live = createLiveBridge();
    const client = { send: vi.fn() };
    live.broadcaster.add(client);
    for (let i = 0; i < 200; i++) {
      live.broadcaster.handleSubscriptionMessage(client, {
        type: "subscribe",
        session_id: createSessionId(),
      });
    }
    expect(live.broadcaster.memoryStats().bufferedSessions).toBe(0);
    expect(client.send).not.toHaveBeenCalled();
  });
});
