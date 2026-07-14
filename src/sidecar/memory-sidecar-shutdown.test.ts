import { afterEach, describe, expect, it, vi } from "vitest";

import { createMaintenanceRunId } from "../util/ids.js";
import { drainMemorySidecar } from "./memory-sidecar-shutdown.js";

afterEach(() => {
  vi.useRealTimers();
});

describe("memory sidecar shutdown", () => {
  it("shares one absolute deadline across HTTP drain and pool shutdown", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(0);
    const runId = createMaintenanceRunId();
    const events: string[] = [];
    const draining = drainMemorySidecar({
      timeoutMs: 100,
      beginShutdown: () => {
        events.push("begin");
        return [runId];
      },
      forceFinalizeMaintenance: () => {
        events.push("force");
        return [runId];
      },
      onAbandoned: (runIds) => events.push(`abandoned:${runIds.join(",")}`),
      closeIdleConnections: () => events.push("idle"),
      closeHttp: () =>
        new Promise<void>((resolve) => {
          events.push("http");
          setTimeout(resolve, 60);
        }),
      shutdownPool: () =>
        new Promise<void>(() => {
          events.push("pool");
        }),
    });

    expect(events).toEqual(["begin", "idle", "http"]);
    await vi.advanceTimersByTimeAsync(60);
    expect(events).toEqual(["begin", "idle", "http", "pool"]);
    await vi.advanceTimersByTimeAsync(40);
    const result = await draining;

    expect(events).toEqual(["begin", "idle", "http", "pool", "force", `abandoned:${runId}`]);
    expect(result).toEqual({
      abandonedRunIds: [runId],
      http: { status: "completed" },
      pool: { status: "timed_out" },
    });
    expect(Date.now()).toBe(100);
  });

  it("does not force-finalize maintenance when HTTP and pool drain cleanly", async () => {
    const runId = createMaintenanceRunId();
    const events: string[] = [];

    const result = await drainMemorySidecar({
      timeoutMs: 100,
      beginShutdown: () => [runId],
      forceFinalizeMaintenance: () => {
        events.push("force");
        return [runId];
      },
      onAbandoned: () => events.push("abandoned"),
      closeIdleConnections: () => {},
      closeHttp: async () => {},
      shutdownPool: async () => {},
    });

    expect(result).toEqual({
      abandonedRunIds: [],
      http: { status: "completed" },
      pool: { status: "completed" },
    });
    expect(events).toEqual([]);
  });
});
