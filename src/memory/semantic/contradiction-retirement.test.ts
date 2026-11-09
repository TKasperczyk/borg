import { afterEach, describe, expect, it } from "vitest";
import {
  createOfflineTestHarness,
  createSemanticNodeFixture,
  createSemanticEdgeFixture,
} from "../../offline/test-support.js";
import { CONTRADICTION_RETIREMENT_ACTION } from "./contradiction-retirement.js";

const cleanups: Array<() => Promise<void>> = [];
afterEach(async () => {
  while (cleanups.length) await cleanups.pop()!();
});
async function fixture() {
  const h = await createOfflineTestHarness({ reviewOpenQuestionExtractor: null });
  cleanups.push(h.cleanup);
  const a = await h.semanticNodeRepository.insert(createSemanticNodeFixture());
  const b = await h.semanticNodeRepository.insert(createSemanticNodeFixture());
  const edge = h.semanticEdgeRepository.addEdge(
    createSemanticEdgeFixture({
      from_node_id: a.id,
      to_node_id: b.id,
      relation: "contradicts",
    }),
  );
  const item = h.reviewQueueRepository.list({ kind: "contradiction", openOnly: true })[0]!;
  return { ...h, a, b, edge, item };
}
function audits(h: Awaited<ReturnType<typeof fixture>>) {
  return h.auditLog.list({}).filter((a) => a.action === CONTRADICTION_RETIREMENT_ACTION);
}

describe("audited contradiction retirement", () => {
  it("reverses only its own event when two closures occur in the same clock tick", async () => {
    const h = await fixture();
    h.semanticEdgeRepository.invalidateEdge(h.edge.id, {
      at: h.clock.now(),
      by_process: "manual",
    });
    h.semanticEdgeRepository.restoreEdge(h.edge);
    await h.reviewQueueRepository.resolve(h.item.id, "dismiss");
    const events = h.db.prepare("SELECT * FROM semantic_edge_invalidation_events ORDER BY id");
    const before = events.all();
    expect(before).toHaveLength(2);
    expect(before[0]).toEqual({ ...before[1], id: before[0]!.id });
    expect(before[0]!.id).not.toBe(before[1]!.id);
    const audit = audits(h)[0]!;
    expect(audit.reversal).toMatchObject({
      invalidation_events: [{ edge_id: h.edge.id, event_id: before[1]!.id }],
    });

    const reversed = await h.auditLog.revert(audit.id);
    expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toEqual(h.edge);
    expect(events.all()).toEqual([before[0]]);
    expect(await h.auditLog.revert(audit.id)).toEqual(reversed);
    expect(events.all()).toEqual([before[0]]);
  });

  it("refuses legacy reversal metadata without guessing which event it owns", async () => {
    const h = await fixture();
    await h.reviewQueueRepository.resolve(h.item.id, "dismiss");
    const audit = audits(h)[0]!;
    const reversal = { ...audit.reversal };
    delete reversal.invalidation_events;
    h.db
      .prepare("UPDATE maintenance_audit SET reversal = ? WHERE id = ?")
      .run(JSON.stringify(reversal), audit.id);
    const retired = h.semanticEdgeRepository.getEdge(h.edge.id);
    const events = h.db.prepare("SELECT * FROM semantic_edge_invalidation_events").all();
    await expect(h.auditLog.revert(audit.id)).rejects.toThrow("recorded invalidation event IDs");
    expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toEqual(retired);
    expect(h.db.prepare("SELECT * FROM semantic_edge_invalidation_events").all()).toEqual(events);
    expect(h.auditLog.get(audit.id)?.reverted_at).toBeNull();
  });

  it("cancels only the matching pending event and keeps reversal idempotent", async () => {
    const h = await fixture();
    await h.reviewQueueRepository.resolve(h.item.id, "keep_both");
    const retired = h.semanticEdgeRepository.getEdge(h.edge.id)!;
    const events = h.db.prepare("SELECT * FROM semantic_edge_invalidation_events ORDER BY id");
    const pending = events.all()[0]!;
    const insertEvent = h.db.prepare(
      "INSERT INTO semantic_edge_invalidation_events(edge_id, valid_to, invalidated_at, processed_at) VALUES (?, ?, ?, ?)",
    );
    // Earlier/different closures and already processed history must survive.
    insertEvent.run(h.edge.id, retired.valid_to! - 1, retired.invalidated_at, null);
    insertEvent.run(h.edge.id, retired.valid_to, retired.invalidated_at! - 1, null);
    insertEvent.run(h.edge.id, retired.valid_to, retired.invalidated_at, h.clock.now());
    const before = events.all();
    const audit = audits(h)[0]!;
    const reversed = await h.auditLog.revert(audit.id);
    expect(events.all()).toEqual(before.filter((event) => event.id !== pending.id));
    expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toEqual(h.edge);
    const changes = h.db.prepare("SELECT total_changes() AS count").get();
    expect(await h.auditLog.revert(audit.id)).toEqual(reversed);
    expect(h.db.prepare("SELECT total_changes() AS count").get()).toEqual(changes);
    h.semanticEdgeRepository.invalidateEdge(h.edge.id, {
      at: h.clock.now() + 1,
      by_process: "manual",
    });
    const afterNewClosure = events.all();
    await h.auditLog.revert(audit.id);
    expect(events.all()).toEqual(afterNewClosure);
    expect(h.semanticEdgeRepository.getEdge(h.edge.id)?.valid_to).not.toBeNull();
  });

  it("rolls back event cancellation and restoration if audit finalization fails", async () => {
    const h = await fixture();
    await h.reviewQueueRepository.resolve(h.item.id, "dismiss");
    const retired = h.semanticEdgeRepository.getEdge(h.edge.id);
    const pending = h.db.prepare("SELECT * FROM semantic_edge_invalidation_events").all();
    const audit = audits(h)[0]!;
    h.db.exec(
      "CREATE TRIGGER reject_audit_reversal BEFORE UPDATE OF reverted_at ON maintenance_audit WHEN NEW.reverted_at IS NOT NULL BEGIN SELECT RAISE(ABORT, 'audit finalization failed'); END",
    );
    await expect(h.auditLog.revert(audit.id)).rejects.toThrow("audit finalization failed");
    expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toEqual(retired);
    expect(h.db.prepare("SELECT * FROM semantic_edge_invalidation_events").all()).toEqual(pending);
    expect(h.auditLog.get(audit.id)?.reverted_at).toBeNull();
  });

  it.each(["review", "supersession"])(
    "preserves derived dependencies when reversing %s retirement",
    async (source) => {
      const h = await fixture();
      const dependencies = h.semanticBeliefDependencyRepository;
      dependencies.addDependency({
        target_type: "semantic_node",
        target_id: h.b.id,
        source_edge_id: h.edge.id,
        dependency_kind: "derived_from",
      });
      const before = dependencies.listBySourceEdge(h.edge.id);
      if (source === "review") await h.reviewQueueRepository.resolve(h.item.id, "keep_both");
      else await h.semanticNodeRepository.markSuperseded(h.a.id, h.b.id, h.clock.now());
      expect(dependencies.listBySourceEdge(h.edge.id)).toEqual(before);
      const derivedEdge = h.semanticEdgeRepository.addEdge(
        createSemanticEdgeFixture({
          from_node_id: h.a.id,
          to_node_id: h.b.id,
          relation: "related_to",
        }),
      );
      dependencies.addDependency({
        target_type: "semantic_edge",
        target_id: derivedEdge.id,
        source_edge_id: h.edge.id,
        dependency_kind: "derived_from",
      });
      const beforeReversal = dependencies.listBySourceEdge(h.edge.id);
      expect(beforeReversal).toHaveLength(2);
      await h.auditLog.revert(audits(h)[0]!.id);
      expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toEqual(h.edge);
      expect(dependencies.listBySourceEdge(h.edge.id)).toEqual(beforeReversal);
    },
  );

  it.each(["keep_both", "dismiss"] as const)(
    "retires only the referenced edge for judged %s, once, and reverses",
    async (decision) => {
      const h = await fixture();
      const other = h.semanticEdgeRepository.addEdge(
        createSemanticEdgeFixture({
          from_node_id: h.b.id,
          to_node_id: h.a.id,
          relation: "contradicts",
        }),
      );
      await h.reviewQueueRepository.resolve(h.item.id, decision);
      expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toMatchObject({
        valid_to: expect.any(Number),
        invalidated_by_process: "review",
        invalidated_by_review_id: h.item.id,
      });
      expect(h.semanticEdgeRepository.getEdge(other.id)).toEqual(other);
      expect((await h.semanticNodeRepository.get(h.a.id))?.status).toBe("active");
      const audit = audits(h)[0]!;
      expect(audit.targets).toMatchObject({ review_item_id: h.item.id, resolution: decision });
      await h.reviewQueueRepository.resolve(h.item.id, decision);
      expect(audits(h)).toHaveLength(1);
      await h.auditLog.revert(audit.id);
      expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toEqual(h.edge);
    },
  );

  it("rolls back closure, invalidation outbox, identity audit and maintenance audit if finalization fails", async () => {
    const h = await fixture();
    h.db.exec(
      "CREATE TRIGGER reject_review_finalization BEFORE UPDATE OF resolved_at ON review_queue WHEN NEW.resolved_at IS NOT NULL BEGIN SELECT RAISE(ABORT, 'finalization failed'); END",
    );
    await expect(h.reviewQueueRepository.resolve(h.item.id, "dismiss")).rejects.toThrow(
      "finalization failed",
    );
    expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toEqual(h.edge);
    expect(h.reviewQueueRepository.get(h.item.id)?.resolved_at).toBeNull();
    expect(audits(h)).toEqual([]);
    expect(h.db.prepare("SELECT * FROM semantic_edge_invalidation_events").all()).toEqual([]);
    expect(
      h.db.prepare("SELECT * FROM identity_events WHERE record_type = 'semantic_edge'").all(),
    ).toEqual([]);
  });

  it("rejects mismatched relation or endpoints without closing an unrelated edge", async () => {
    const h = await fixture();
    h.db.prepare("UPDATE semantic_edges SET relation = 'supports' WHERE id = ?").run(h.edge.id);
    await expect(h.reviewQueueRepository.resolve(h.item.id, "keep_both")).rejects.toThrow(
      "does not match",
    );
    expect(h.semanticEdgeRepository.getEdge(h.edge.id)?.valid_to).toBeNull();
    expect(audits(h)).toEqual([]);
  });

  it.each(["markSuperseded", "update", "restore", "insert"] as const)(
    "retires both directions via %s and preserves other relations",
    async (writer) => {
      const h = await fixture();
      const reverse = h.semanticEdgeRepository.addEdge(
        createSemanticEdgeFixture({
          from_node_id: h.b.id,
          to_node_id: h.a.id,
          relation: "contradicts",
        }),
      );
      const supports = h.semanticEdgeRepository.addEdge(
        createSemanticEdgeFixture({
          from_node_id: h.a.id,
          to_node_id: h.b.id,
          relation: "supports",
        }),
      );
      const lifecycle = {
        status: "superseded" as const,
        corrected_by: h.b.id,
        superseded_at: h.clock.now(),
      };
      if (writer === "markSuperseded")
        await h.semanticNodeRepository.markSuperseded(h.a.id, h.b.id, h.clock.now());
      else if (writer === "update") await h.semanticNodeRepository.update(h.a.id, lifecycle);
      else if (writer === "restore")
        await h.semanticNodeRepository.restore({ ...h.a, ...lifecycle });
      else await h.semanticNodeRepository.insert({ ...h.a, ...lifecycle });
      for (const edge of [h.edge, reverse])
        expect(h.semanticEdgeRepository.getEdge(edge.id)?.valid_to).not.toBeNull();
      expect(h.semanticEdgeRepository.getEdge(supports.id)).toEqual(supports);
      expect(audits(h)).toHaveLength(1);
      expect(audits(h)[0]!.targets).toMatchObject({ node_id: h.a.id, corrected_by: h.b.id });
      expect(() =>
        h.semanticEdgeRepository.addEdge(
          createSemanticEdgeFixture({
            from_node_id: h.a.id,
            to_node_id: h.b.id,
            relation: "contradicts",
          }),
        ),
      ).toThrow("superseded");
      expect(() => h.semanticEdgeRepository.restoreEdge(h.edge)).toThrow("superseded");
      expect(() => h.semanticEdgeRepository.mergeOpenEdgeEvidence(h.edge)).toThrow("superseded");
      // Explicit audit undo restores the recorded pre-repair graph, even its stale edge.
      await h.auditLog.revert(audits(h)[0]!.id);
      expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toEqual(h.edge);
      expect(h.semanticEdgeRepository.getEdge(reverse.id)).toEqual(reverse);
    },
  );

  it("rolls back the supersession when its maintenance audit cannot be written", async () => {
    const h = await fixture();
    h.db.exec(
      "CREATE TRIGGER reject_retirement_audit BEFORE INSERT ON maintenance_audit BEGIN SELECT RAISE(ABORT, 'audit failed'); END",
    );
    await expect(
      h.semanticNodeRepository.markSuperseded(h.a.id, h.b.id, h.clock.now()),
    ).rejects.toThrow("audit failed");
    expect((await h.semanticNodeRepository.get(h.a.id))?.status).toBe("active");
    expect(h.semanticEdgeRepository.getEdge(h.edge.id)).toEqual(h.edge);
  });

  it("refuses audit reversal over a later edge write", async () => {
    const h = await fixture();
    await h.reviewQueueRepository.resolve(h.item.id, "keep_both");
    const audit = audits(h)[0]!;
    h.db.prepare("UPDATE semantic_edges SET confidence = 0.9 WHERE id = ?").run(h.edge.id);
    await expect(h.auditLog.revert(audit.id)).rejects.toThrow("changed since retirement");
    expect(h.semanticEdgeRepository.getEdge(h.edge.id)?.valid_to).not.toBeNull();
    expect(h.auditLog.get(audit.id)?.reverted_at).toBeNull();
  });
});
