import { afterEach, describe, expect, it, vi } from "vitest";
import {
  runWithTimeoutPolicy,
  type TimeoutLoadSignals,
  type TimeoutObservation,
} from "./timeout-policy.js";

describe("bounded timeout policy", () => {
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("counts event-loop holds against the original budget even when the result beats the timer", async () => {
    let now = 0;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    const result = runWithTimeoutPolicy({
      timeoutMs: 100,
      maxTimeoutMs: 200,
      run: async () => {
        now = 150;
        return "late result";
      },
    });
    await expect(result).rejects.toMatchObject({ timeoutMs: 100 });
  });

  it.each([
    [{ maintenanceActive: true }, "maintenance_active"],
    [{ sessionBusy: true }, "session_busy"],
    [{ embeddingRetryAt: 10_000 }, "embedding_backoff"],
    [{ embeddingRequestStartedAt: 0 }, "embedding_slow"],
  ] as const)(
    "extends for structural state %j, once from the original start",
    async (state, reason) => {
      vi.useFakeTimers();
      vi.setSystemTime(1000);
      let signals: TimeoutLoadSignals = {};
      let signal!: AbortSignal;
      const observations: TimeoutObservation[] = [];
      const run = vi.fn(async (value: AbortSignal) => {
        signal = value;
        return await new Promise<never>(() => {});
      });
      const result = runWithTimeoutPolicy({
        timeoutMs: 100,
        maxTimeoutMs: 250,
        loadSignals: () => signals,
        observe: (event) => observations.push(event),
        run,
      });
      const rejected = expect(result).rejects.toMatchObject({ timeoutMs: 250 });
      await vi.advanceTimersByTimeAsync(80);
      signals = state;
      await vi.advanceTimersByTimeAsync(20);
      expect(signal.aborted).toBe(false);
      expect(observations.at(-1)).toMatchObject({
        outcome: "extended",
        budgetMs: 250,
        extensionReasons: [reason],
      });
      await vi.advanceTimersByTimeAsync(150);
      await rejected;
      expect(signal.aborted).toBe(true);
      expect(run).toHaveBeenCalledTimes(1);
      expect(observations.at(-1)).toMatchObject({ outcome: "timed_out", elapsedMs: 250 });
      expect(vi.getTimerCount()).toBe(0);
    },
  );

  it("keeps an explicit absolute deadline authoritative and cancels timers after success", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(1000);
    const result = runWithTimeoutPolicy({
      timeoutMs: 100,
      maxTimeoutMs: 500,
      deadlineAt: 1040,
      loadSignals: () => ({ maintenanceActive: true }),
      run: async () => await new Promise<never>(() => {}),
    });
    const rejected = expect(result).rejects.toMatchObject({ timeoutMs: 40 });
    await vi.advanceTimersByTimeAsync(40);
    await rejected;
    await expect(
      runWithTimeoutPolicy({ timeoutMs: 100, maxTimeoutMs: 200, run: async () => 42 }),
    ).resolves.toBe(42);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("does not start expired or cancelled work and keeps healthy reads at their base ceiling", async () => {
    vi.useFakeTimers();
    const run = vi.fn(async () => 1);
    const controller = new AbortController();
    controller.abort(new Error("cancelled"));
    await expect(
      runWithTimeoutPolicy({ timeoutMs: 100, maxTimeoutMs: 200, run, signal: controller.signal }),
    ).rejects.toThrow("cancelled");
    await expect(
      runWithTimeoutPolicy({ timeoutMs: 100, maxTimeoutMs: 200, run, deadlineAt: Date.now() }),
    ).rejects.toMatchObject({ timeoutMs: 0 });
    expect(run).not.toHaveBeenCalled();
    const result = runWithTimeoutPolicy({
      timeoutMs: 100,
      maxTimeoutMs: 200,
      run: async () => await new Promise<never>(() => {}),
    });
    const rejected = expect(result).rejects.toMatchObject({ timeoutMs: 100 });
    await vi.advanceTimersByTimeAsync(100);
    await rejected;
  });
});
