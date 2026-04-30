import { afterEach, describe, expect, it, vi } from "vitest";

import type { BorgDependencies } from "./types.js";
import { closeBorgDependencies } from "./lifecycle.js";

describe("borg lifecycle", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("attempts LanceDB close when SQLite close throws", async () => {
    const sqliteError = new Error("sqlite close failed");
    const flushEnqueueHooks = vi.fn().mockResolvedValue(undefined);
    const sqliteClose = vi.fn(() => {
      throw sqliteError;
    });
    const lanceClose = vi.fn().mockResolvedValue(undefined);
    vi.spyOn(console, "error").mockImplementation(() => {});

    const deps = {
      autonomyScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      maintenanceScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      reviewQueueRepository: {
        flushEnqueueHooks,
      },
      sqlite: {
        close: sqliteClose,
      },
      lance: {
        close: lanceClose,
      },
    } as unknown as BorgDependencies;

    await expect(closeBorgDependencies(deps)).rejects.toThrow(AggregateError);
    expect(flushEnqueueHooks.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER).toBeLessThan(
      sqliteClose.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER,
    );
    expect(lanceClose).toHaveBeenCalledTimes(1);
  });

  it("attempts both scheduler stops when one stop throws", async () => {
    const autonomyError = new Error("autonomy stop failed");
    const maintenanceStop = vi.fn().mockResolvedValue(undefined);
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});

    const deps = {
      autonomyScheduler: {
        stop: vi.fn(() => {
          throw autonomyError;
        }),
      },
      maintenanceScheduler: {
        stop: maintenanceStop,
      },
      reviewQueueRepository: {
        flushEnqueueHooks: vi.fn().mockResolvedValue(undefined),
      },
      sqlite: {
        close: vi.fn(),
      },
      lance: {
        close: vi.fn().mockResolvedValue(undefined),
      },
    } as unknown as BorgDependencies;

    await expect(closeBorgDependencies(deps)).rejects.toThrow(AggregateError);
    expect(maintenanceStop).toHaveBeenCalledWith({ graceful: true });
    expect(consoleError).toHaveBeenCalledWith("Failed to close autonomy scheduler", autonomyError);
  });

  it("drains review queue enqueue hooks before closing SQLite", async () => {
    const flushEnqueueHooks = vi.fn().mockResolvedValue(undefined);
    const sqliteClose = vi.fn();

    const deps = {
      autonomyScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      maintenanceScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      reviewQueueRepository: {
        flushEnqueueHooks,
      },
      sqlite: {
        close: sqliteClose,
      },
      lance: {
        close: vi.fn().mockResolvedValue(undefined),
      },
    } as unknown as BorgDependencies;

    await closeBorgDependencies(deps);

    expect(flushEnqueueHooks).toHaveBeenCalledTimes(1);
    expect(flushEnqueueHooks.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER).toBeLessThan(
      sqliteClose.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER,
    );
  });
});
