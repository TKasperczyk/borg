import { afterEach, describe, expect, it, vi } from "vitest";

import { createMaintenanceRunId } from "../util/ids.js";

import { AuditLog, ReverserRegistry } from "./audit-log.js";
import { createOfflineTestHarness } from "./test-support.js";

describe("offline audit log", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("records, lists, and reverts audit rows idempotently", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const registry = new ReverserRegistry();
    const auditLog = new AuditLog({
      db: harness.db,
      clock: harness.clock,
      registry,
    });
    const reverser = vi.fn();

    registry.register("curator", "archive", reverser);

    const recorded = auditLog.record({
      run_id: createMaintenanceRunId(),
      process: "curator",
      action: "archive",
      targets: {
        episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      },
      reversal: {
        previous: [],
      },
    });

    expect(auditLog.list({ process: "curator", reverted: false })).toEqual([
      expect.objectContaining({
        id: recorded.id,
        process: "curator",
      }),
    ]);

    const reverted = await auditLog.revert(recorded.id, "test");
    expect(reverted).toMatchObject({
      id: recorded.id,
      reverted_by: "test",
    });
    expect(reverser).toHaveBeenCalledTimes(1);

    const second = await auditLog.revert(recorded.id, "again");
    expect(second?.reverted_at).toBe(reverted?.reverted_at ?? null);
    expect(reverser).toHaveBeenCalledTimes(1);
    expect(auditLog.list({ reverted: true })).toHaveLength(1);
  });
});
