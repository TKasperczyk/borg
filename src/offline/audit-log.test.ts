import { afterEach, describe, expect, it, vi } from "vitest";

import { createMaintenanceRunId } from "../util/ids.js";

import { AuditLog, ReverserRegistry } from "./audit-log.js";
import { createOfflineTestHarness } from "./test-support.js";
import { refreshSerializedEmbeddings } from "../embeddings/serialized.js";

describe("offline audit log", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("projects completion targets without reading or parsing reversal blobs", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const runId = createMaintenanceRunId();
    const audit = harness.auditLog.record({
      run_id: runId,
      process: "semantic-extractor",
      action: "extract",
      targets: { episode_ids: ["ep_aaaaaaaaaaaaaaaa"] },
      reversal: { created_node_ids: [], updated_nodes: [], created_edge_ids: [] },
    });
    // A malformed reversal proves this projection never hydrates it.
    harness.db
      .prepare("UPDATE maintenance_audit SET reversal = ? WHERE id = ?")
      .run("not JSON", audit.id);
    const prepare = vi.spyOn(harness.db, "prepare");
    expect(
      harness.auditLog.listTargets({
        process: "semantic-extractor",
        run_id: runId,
        reverted: false,
      }),
    ).toEqual([{ episode_ids: ["ep_aaaaaaaaaaaaaaaa"] }]);
    expect(
      prepare.mock.calls.some(([sql]) => /SELECT targets\s+FROM maintenance_audit/.test(sql)),
    ).toBe(true);
    expect(harness.auditLog.listTargets({ reverted: true })).toEqual([]);
    expect(harness.auditLog.listTargets({ process: "curator" })).toEqual([]);
  });

  it("rejects an unrecoverable serialized vector before starting a reversal transaction", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);
    const registry = new ReverserRegistry();
    const reverser = vi.fn();
    registry.register("curator", "archive", reverser);
    const auditLog = new AuditLog({
      db: harness.db,
      registry,
      prepareSerializedEmbeddings: (payload) =>
        refreshSerializedEmbeddings(payload, harness.createContext().embeddingClient),
    });
    const audit = auditLog.record({
      run_id: createMaintenanceRunId(),
      process: "curator",
      action: "archive",
      targets: {},
      reversal: { previous: [{ embedding: [1, 2, 3, 4] }] },
    });
    await expect(auditLog.revert(audit.id)).rejects.toMatchObject({
      code: "SERIALIZED_EMBEDDING_INCOMPATIBLE",
    });
    expect(reverser).not.toHaveBeenCalled();
    expect(harness.db.raw.inTransaction).toBe(false);
    expect(auditLog.get(audit.id)?.reverted_at).toBeNull();
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

  it("rolls back reverser database changes when marking reverted fails", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const registry = new ReverserRegistry();
    const auditLog = new AuditLog({
      db: harness.db,
      clock: harness.clock,
      registry,
    });
    harness.db.exec("CREATE TABLE audit_revert_probe (id INTEGER PRIMARY KEY)");
    registry.register("curator", "archive", () => {
      harness.db.prepare("INSERT INTO audit_revert_probe DEFAULT VALUES").run();
    });

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
    const originalPrepare = harness.db.prepare.bind(harness.db);
    vi.spyOn(harness.db, "prepare").mockImplementation((sql) => {
      if (sql.includes("UPDATE maintenance_audit SET reverted_at")) {
        throw new Error("mark reverted failed");
      }

      return originalPrepare(sql);
    });

    await expect(auditLog.revert(recorded.id, "test")).rejects.toThrow("mark reverted failed");
    expect(
      (
        harness.db.prepare("SELECT COUNT(*) AS count FROM audit_revert_probe").get() as {
          count: number;
        }
      ).count,
    ).toBe(0);
    expect(auditLog.get(recorded.id)?.reverted_at).toBeNull();
  });
});
