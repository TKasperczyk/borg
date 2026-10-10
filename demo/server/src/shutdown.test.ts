import { afterEach, describe, expect, it, vi } from "vitest";
import { createDemoShutdownHandler } from "./shutdown.js";

describe("demo shutdown", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("waits for the complete drain before closing stores", async () => {
    let release!: () => void;
    const drain = new Promise<void>((resolve) => {
      release = resolve;
    });
    const beginDrain = vi.fn(() => drain);
    const closeStores = vi.fn(async () => {});
    const exit = vi.fn();
    const shutdown = createDemoShutdownHandler({
      beginDrain,
      closeStores,
      exit,
      forceCloseConnections: vi.fn(),
      logger: { log: vi.fn(), error: vi.fn() },
    });
    const pending = shutdown("SIGTERM");
    expect(beginDrain).toHaveBeenCalledOnce();
    expect(closeStores).not.toHaveBeenCalled();
    expect(exit).not.toHaveBeenCalled();
    release();
    await pending;
    expect(closeStores).toHaveBeenCalledOnce();
    expect(closeStores).toHaveBeenCalledWith({ drained: true });
    expect(exit).toHaveBeenCalledWith(0);
  });

  it("bounds the whole drain to 75 seconds and store closure to ten more", async () => {
    vi.useFakeTimers();
    let rejectDrain!: (error: Error) => void;
    const drain = new Promise<void>((_resolve, reject) => {
      rejectDrain = reject;
    });
    const closeStores = vi.fn(async () => await new Promise<void>(() => {}));
    const exit = vi.fn();
    const shutdown = createDemoShutdownHandler({
      beginDrain: () => drain,
      closeStores,
      exit,
      forceCloseConnections: vi.fn(),
      logger: { log: vi.fn(), error: vi.fn() },
    });
    const pending = shutdown("SIGTERM");
    await vi.advanceTimersByTimeAsync(74_999);
    expect(closeStores).not.toHaveBeenCalled();
    await vi.advanceTimersByTimeAsync(1);
    expect(closeStores).toHaveBeenCalledOnce();
    expect(closeStores).toHaveBeenCalledWith({ drained: false });
    expect(exit).not.toHaveBeenCalled();
    rejectDrain(new Error("late tail failure"));
    await vi.advanceTimersByTimeAsync(10_000);
    await pending;
    expect(exit).toHaveBeenCalledWith(1);
  });

  it.each([
    ["SIGTERM", 1_000],
    ["SIGINT", 0],
  ] as const)(
    "exits immediately on a repeated %s %i ms after the first",
    async (signal, afterMs) => {
      vi.useFakeTimers();
      const exit = vi.fn();
      const shutdown = createDemoShutdownHandler({
        beginDrain: async () => await new Promise<void>(() => {}),
        closeStores: async () => {},
        exit,
        forceCloseConnections: vi.fn(),
        logger: { log: vi.fn(), error: vi.fn() },
      });
      const pending = shutdown("SIGTERM");
      await vi.advanceTimersByTimeAsync(afterMs);
      await shutdown(signal);
      expect(exit).toHaveBeenCalledWith(1);
      await vi.advanceTimersByTimeAsync(75_000);
      await pending;
    },
  );

  it("keeps draining through a copy of the first signal relayed within a second", async () => {
    vi.useFakeTimers();
    let release!: () => void;
    const drain = new Promise<void>((resolve) => {
      release = resolve;
    });
    const closeStores = vi.fn(async () => {});
    const exit = vi.fn();
    const shutdown = createDemoShutdownHandler({
      beginDrain: () => drain,
      closeStores,
      exit,
      forceCloseConnections: vi.fn(),
      logger: { log: vi.fn(), error: vi.fn() },
    });
    const pending = shutdown("SIGTERM");
    await vi.advanceTimersByTimeAsync(30);
    await shutdown("SIGTERM");
    await vi.advanceTimersByTimeAsync(969);
    await shutdown("SIGTERM");
    expect(exit).not.toHaveBeenCalled();
    release();
    await pending;
    expect(closeStores).toHaveBeenCalledOnce();
    expect(exit).toHaveBeenCalledOnce();
    expect(exit).toHaveBeenCalledWith(0);
  });

  it("closes stores and exits even if draining and connection close fail", async () => {
    const closeStores = vi.fn(async () => {});
    const exit = vi.fn();
    const shutdown = createDemoShutdownHandler({
      beginDrain: async () => {
        throw new Error("connector stop failed");
      },
      closeStores,
      exit,
      forceCloseConnections: () => {
        throw new Error("connection close failed");
      },
      logger: { log: vi.fn(), error: vi.fn() },
    });
    await shutdown("SIGTERM");
    expect(closeStores).toHaveBeenCalledOnce();
    expect(closeStores).toHaveBeenCalledWith({ drained: false });
    expect(exit).toHaveBeenCalledWith(1);
  });
});
