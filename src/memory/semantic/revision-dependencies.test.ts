import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { createSemanticEdgeId, createSemanticNodeId } from "../../util/ids.js";
import { semanticMigrations } from "./migrations.js";
import { SemanticBeliefDependencyRepository } from "./revision-dependencies.js";

describe("semantic belief dependency repository", () => {
  const cleanup: Array<() => void> = [];

  afterEach(() => {
    while (cleanup.length > 0) {
      cleanup.pop()?.();
    }
  });

  function createFixture() {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: semanticMigrations,
    });
    const clock = new ManualClock(1_000);
    const repository = new SemanticBeliefDependencyRepository({
      db,
      clock,
    });

    cleanup.push(() => {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    return { db, clock, repository };
  }

  it("adds dependencies idempotently", () => {
    const { db, clock, repository } = createFixture();
    const input = {
      target_type: "semantic_node" as const,
      target_id: createSemanticNodeId(),
      source_edge_id: createSemanticEdgeId(),
      dependency_kind: "supports" as const,
    };
    const first = repository.addDependency(input);

    clock.set(2_000);
    const second = repository.addDependency(input);
    const row = db.prepare("SELECT COUNT(*) AS count FROM semantic_belief_dependencies").get() as {
      count: number;
    };

    expect(second).toEqual(first);
    expect(first.created_at).toBe(1_000);
    expect(row.count).toBe(1);
  });

  it("lists dependencies by source edge and target", () => {
    const { repository } = createFixture();
    const sourceEdgeId = createSemanticEdgeId();
    const nodeTargetId = createSemanticNodeId();
    const edgeTargetId = createSemanticEdgeId();
    const otherSourceEdgeId = createSemanticEdgeId();

    const nodeDependency = repository.addDependency({
      target_type: "semantic_node",
      target_id: nodeTargetId,
      source_edge_id: sourceEdgeId,
      dependency_kind: "supports",
    });
    const edgeDependency = repository.addDependency({
      target_type: "semantic_edge",
      target_id: edgeTargetId,
      source_edge_id: sourceEdgeId,
      dependency_kind: "derived_from",
    });
    repository.addDependency({
      target_type: "semantic_node",
      target_id: createSemanticNodeId(),
      source_edge_id: otherSourceEdgeId,
      dependency_kind: "supports",
    });

    expect(repository.listBySourceEdge(sourceEdgeId)).toEqual([edgeDependency, nodeDependency]);
    expect(repository.listByTarget("semantic_node", nodeTargetId)).toEqual([nodeDependency]);
    expect(repository.listByTarget("semantic_edge", edgeTargetId)).toEqual([edgeDependency]);
  });

  it("removes dependencies", () => {
    const { repository } = createFixture();
    const input = {
      target_type: "semantic_node" as const,
      target_id: createSemanticNodeId(),
      source_edge_id: createSemanticEdgeId(),
      dependency_kind: "supports" as const,
    };

    repository.addDependency(input);

    expect(repository.removeDependency(input)).toBe(true);
    expect(repository.removeDependency(input)).toBe(false);
    expect(repository.listBySourceEdge(input.source_edge_id)).toEqual([]);
  });
});
