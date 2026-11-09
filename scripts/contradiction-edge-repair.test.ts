import { afterEach, describe, expect, it } from "vitest";
import { SemanticEdgeRepository } from "../src/memory/semantic/repository.js";
import {
  registerContradictionRetirementReverser,
  CONTRADICTION_RETIREMENT_ACTION,
} from "../src/memory/semantic/contradiction-retirement.js";
import { AuditLog, ReverserRegistry } from "../src/offline/audit-log.js";
import { createEpisodeId, createSemanticNodeId } from "../src/util/ids.js";
import { repairFixture } from "./audited-data-repair-test-support.js";
import { formatDataRepairPlan, loadRepairPlan } from "./audited-data-repair.js";
import {
  applyContradictionRepair,
  planContradictionRepair,
  type ContradictionRepairPlan,
} from "./contradiction-edge-repair.js";

const cleanups: Array<() => void> = [];
afterEach(() => {
  while (cleanups.length) cleanups.pop()!();
});
function fixture() {
  const h = repairFixture();
  cleanups.push(h.cleanup);
  const nodes = [createSemanticNodeId(), createSemanticNodeId()];
  for (const id of nodes)
    h.db
      .prepare(
        "INSERT INTO semantic_nodes(id, kind, label, description, aliases, confidence, source_episode_ids, created_at, updated_at, last_verified_at) VALUES (?, 'proposition', 'opaque', 'opaque', '[]', 0.8, '[]', 100, 100, 100)",
      )
      .run(id);
  const edges = new SemanticEdgeRepository(h);
  const edge = edges.addEdge({
    from_node_id: nodes[0]!,
    to_node_id: nodes[1]!,
    relation: "contradicts",
    confidence: 0.8,
    evidence_episode_ids: [createEpisodeId()],
    created_at: 100,
    last_verified_at: 100,
  });
  const review = (resolution: string, extra = {}) =>
    h.db
      .prepare(
        "INSERT INTO review_queue(kind, refs, reason, created_at, resolved_at, resolution) VALUES ('contradiction', ?, 'opaque', 100, 200, ?)",
      )
      .run(JSON.stringify({ node_ids: nodes, edge_id: edge.id, ...extra }), resolution);
  const registry = new ReverserRegistry();
  registerContradictionRetirementReverser(registry, edges);
  return {
    ...h,
    edges,
    edge,
    nodes,
    review,
    audit: new AuditLog({ db: h.db, clock: h.clock, registry }),
  };
}

describe("contradiction backfills", () => {
  it.each(["keep_both", "dismiss"])(
    "dry-runs judged %s, applies once, and reverses via maintenance audit",
    async (resolution) => {
      const h = fixture();
      h.review(resolution);
      const input = { dataDir: h.dataDir, kind: "backfill-judged-contradiction-edges" as const };
      const plan = planContradictionRepair(input, 1000);
      expect(plan.summary.changed).toBe(1);
      expect(JSON.parse(formatDataRepairPlan(plan))).toMatchObject({
        kind: input.kind,
        rows: [{ before: h.edge }],
        journal: { pending: expect.any(String) },
        completion_audit: { process: "correction" },
      });
      expect(h.edges.getEdge(h.edge.id)).toEqual(h.edge);
      expect(h.audit.list({})).toEqual([]);
      await applyContradictionRepair(plan);
      expect(h.edges.getEdge(h.edge.id)).toEqual(plan.rows[0]!.after);
      expect(planContradictionRepair(input).summary.changed).toBe(0);
      await applyContradictionRepair(planContradictionRepair(input));
      const audit = h.audit.list({}).find((a) => a.action === CONTRADICTION_RETIREMENT_ACTION)!;
      expect(audit.targets).toMatchObject({ resolution, source_process: input.kind });
      await h.audit.revert(audit.id);
      expect(h.edges.getEdge(h.edge.id)).toEqual(h.edge);
    },
  );
  it.each(["needs_manual", "apply_failed"])(
    "includes judged dismiss after %s, excluding only handler bypasses and mismatched refs",
    async (verdict) => {
      const h = fixture();
      h.review("dismiss", {
        __borg_review_resolver_repair: { mode: "resolver_finalized_without_handler" },
      });
      h.review("keep_both", { node_ids: [h.nodes[0], createSemanticNodeId()] });
      const attempted = h.review("needs_manual", {
        __borg_review_resolver_diagnostic: { verdict, attempts: 1 },
      });
      h.db
        .prepare("UPDATE review_queue SET resolution = NULL, resolved_at = NULL WHERE id = ?")
        .run(attempted.lastInsertRowid);
      // Historical successful judgment finalizes the row but retains the earlier
      // diagnostic refs. It did not stamp the handler-bypass repair marker.
      h.db
        .prepare("UPDATE review_queue SET resolution = 'dismiss', resolved_at = 300 WHERE id = ?")
        .run(attempted.lastInsertRowid);
      const plan = planContradictionRepair({
        dataDir: h.dataDir,
        kind: "backfill-judged-contradiction-edges",
      });
      const counts = { changed: 1, skipped_diagnostic: 1, skipped_mismatched_refs: 1 };
      expect(plan.summary).toEqual(counts);
      expect(JSON.parse(formatDataRepairPlan(plan))).toMatchObject({ summary: counts });
      expect(h.edges.getEdge(h.edge.id)).toEqual(h.edge);
      await applyContradictionRepair(plan);
      expect(h.edges.getEdge(h.edge.id)?.invalidated_by_review_id).toBe(
        Number(attempted.lastInsertRowid),
      );
      expect(
        h.audit.list({}).find((a) => a.action === CONTRADICTION_RETIREMENT_ACTION)?.targets,
      ).toMatchObject({ resolution: "dismiss", review_item_id: Number(attempted.lastInsertRowid) });
    },
  );
  it("deduplicates two superseded endpoints, excludes contradicted nodes, resumes a committed repair and reverses", async () => {
    const h = fixture();
    // Seed a legacy shape without going through the new retirement writer.
    h.db
      .prepare(
        "UPDATE semantic_nodes SET status = 'superseded', corrected_by = ?, superseded_at = 300",
      )
      .run(h.nodes[1]!);
    const input = { dataDir: h.dataDir, kind: "backfill-superseded-contradiction-edges" as const };
    const plan = planContradictionRepair(input, 1000);
    expect(plan.rows).toHaveLength(1);
    await expect(
      applyContradictionRepair(plan, {
        checkpoint: (p) => {
          if (p === "after_sqlite") throw new Error("interrupted");
        },
      }),
    ).rejects.toThrow("interrupted");
    const resumed = loadRepairPlan<ContradictionRepairPlan>(h.dataDir, input.kind);
    await applyContradictionRepair(resumed, { resume: true });
    expect(
      h.audit.list({}).filter((a) => a.action === CONTRADICTION_RETIREMENT_ACTION),
    ).toHaveLength(1);
    const audit = h.audit.list({}).find((a) => a.action === CONTRADICTION_RETIREMENT_ACTION)!;
    await h.audit.revert(audit.id);
    expect(h.edges.getEdge(h.edge.id)).toEqual(h.edge);
    h.db.exec("UPDATE semantic_nodes SET status = 'contradicted'");
    expect(planContradictionRepair(input).rows).toEqual([]);
  });
  it("refuses changes after planning before writing an edge or audit", async () => {
    const h = fixture();
    h.review("keep_both");
    const plan = planContradictionRepair({
      dataDir: h.dataDir,
      kind: "backfill-judged-contradiction-edges",
    });
    h.db.prepare("UPDATE semantic_edges SET confidence = 0.2 WHERE id = ?").run(h.edge.id);
    await expect(applyContradictionRepair(plan)).rejects.toThrow("Changed since planning");
    expect(h.edges.getEdge(h.edge.id)?.valid_to).toBeNull();
    expect(h.audit.list({})).toEqual([]);
  });
});
