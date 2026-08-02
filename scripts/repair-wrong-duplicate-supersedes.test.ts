import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { ReviewQueueItem } from "../src/memory/review-queue/index.js";
import {
  semanticMigrations,
  type SemanticNode,
  type SemanticNodeRepository,
} from "../src/memory/semantic/index.js";
import { offlineMigrations } from "../src/offline/index.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
  type OfflineTestHarness,
} from "../src/offline/test-support.js";
import { composeMigrations, openDatabase } from "../src/storage/sqlite/index.js";
import {
  createMaintenanceRunId,
  createSemanticEdgeId,
  createSemanticNodeId,
} from "../src/util/ids.js";
import {
  main,
  parseRepairCliArgs,
  repairReportExitCode,
  repairWrongDuplicateSupersedes,
  WRONG_DUPLICATE_RESTORE_AUDIT_ACTION,
} from "./repair-wrong-duplicate-supersedes.js";

describe("repair wrong duplicate supersedes", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  async function insertNode(
    harness: OfflineTestHarness,
    sourceEpisodeId: SemanticNode["source_episode_ids"][number],
    ticket: number,
  ): Promise<SemanticNode> {
    return harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: `Ticket AININJAS-${ticket}`,
        description: `AININJAS-${ticket} is a distinct ticket record.`,
        source_episode_ids: [sourceEpisodeId],
      }),
    );
  }

  async function resolveSupersede(
    harness: OfflineTestHarness,
    loser: SemanticNode,
    winner: SemanticNode,
  ): Promise<number> {
    const review = harness.reviewQueueRepository.enqueue({
      kind: "duplicate",
      reason: "Historical duplicate resolution",
      refs: {
        node_ids: [loser.id, winner.id],
        node_labels: [loser.label, winner.label],
      },
    });
    await harness.reviewQueueRepository.resolve(review.id, {
      decision: "supersede",
      winner_node_id: winner.id,
      reason: "Historical resolver chose a winner.",
    });
    return review.id;
  }

  it("defaults to dry-run and accepts either flag or positional data directories", () => {
    expect(parseRepairCliArgs(["--data-dir", "/tmp/example-bank"])).toMatchObject({
      help: false,
      apply: false,
      dataDir: "/tmp/example-bank",
    });
    expect(parseRepairCliArgs(["/tmp/example-bank", "--apply"])).toMatchObject({
      help: false,
      apply: true,
      dataDir: "/tmp/example-bank",
    });
  });

  it("uses a read-only minimal SQLite open for dry-run and no LanceDB for apply", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-wrong-duplicate-repair-"));
    cleanup.push(async () => {
      rmSync(tempDir, { recursive: true, force: true });
    });
    const databasePath = join(tempDir, "borg.db");
    const db = openDatabase(databasePath, {
      migrations: composeMigrations(semanticMigrations, offlineMigrations),
    });
    const loserId = createSemanticNodeId();
    const winnerId = createSemanticNodeId();
    const insertNode = db.prepare(
      `
        INSERT INTO semantic_nodes (
          id, kind, label, description, domain, aliases, observation_metadata, confidence,
          source_episode_ids, created_at, updated_at, last_verified_at, archived, superseded_by,
          status, corrected_by, superseded_at
        ) VALUES (?, 'proposition', ?, ?, NULL, '[]', NULL, 0.8, '["ep_placeholder"]',
                  1, 1, 1, 0, NULL, ?, ?, ?)
      `,
    );
    insertNode.run(
      loserId,
      "Ticket AININJAS-8000",
      "The first sanitized maintenance target.",
      "superseded",
      winnerId,
      2,
    );
    insertNode.run(
      winnerId,
      "Ticket AININJAS-8001",
      "The second sanitized maintenance target.",
      "active",
      null,
      null,
    );
    db.prepare(
      `
        INSERT INTO review_queue (kind, refs, reason, created_at, resolved_at, resolution)
        VALUES ('duplicate', ?, 'Sanitized historical duplicate decision', 1, 2, 'supersede')
      `,
    ).run(
      JSON.stringify({
        node_ids: [loserId, winnerId],
        node_labels: ["Ticket AININJAS-8000", "Ticket AININJAS-8001"],
      }),
    );
    db.close();
    const beforeDryRun = readFileSync(databasePath);
    const stdout = vi
      .spyOn(process.stdout, "write")
      .mockImplementation((() => true) as typeof process.stdout.write);
    const stderr = vi
      .spyOn(process.stderr, "write")
      .mockImplementation((() => true) as typeof process.stderr.write);

    await expect(main(["--data-dir", tempDir])).resolves.toBe(0);
    expect(readFileSync(databasePath)).toEqual(beforeDryRun);
    await expect(main(["--data-dir", tempDir, "--apply"])).resolves.toBe(0);

    const inspectionDb = openDatabase(databasePath);
    const restored = inspectionDb
      .prepare("SELECT status, corrected_by, superseded_at FROM semantic_nodes WHERE id = ?")
      .get(loserId);
    const auditCount = inspectionDb
      .prepare("SELECT COUNT(*) AS count FROM maintenance_audit")
      .get() as { count: number };
    inspectionDb.close();

    expect(restored).toEqual({
      status: "active",
      corrected_by: null,
      superseded_at: null,
    });
    expect(Number(auditCount.count)).toBe(1);
    expect(stdout).toHaveBeenCalled();
    expect(stderr).toHaveBeenCalled();
    expect(readFileSync(databasePath)).not.toEqual(beforeDryRun);
    expect(existsSync(join(tempDir, "lancedb"))).toBe(false);
  });

  it("restores 13 unique current losers from 14 matching rows, audits them, and is idempotent", async () => {
    const harness = await createOfflineTestHarness({
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const episode = await harness.episodicRepository.createEpisode(createEpisodeFixture());
    let ticket = 2_000;
    const nextNode = () => insertNode(harness, episode.id, ticket++);

    const chainFirst = await nextNode();
    const chainMiddle = await nextNode();
    const chainLast = await nextNode();
    const reviewIds = [
      await resolveSupersede(harness, chainFirst, chainMiddle),
      await resolveSupersede(harness, chainMiddle, chainLast),
    ];

    const sharedLoser = await nextNode();
    const sharedFirstWinner = await nextNode();
    const sharedSecondWinner = await nextNode();
    reviewIds.push(
      await resolveSupersede(harness, sharedLoser, sharedFirstWinner),
      await resolveSupersede(harness, sharedLoser, sharedSecondWinner),
    );

    for (let index = 0; index < 10; index += 1) {
      reviewIds.push(await resolveSupersede(harness, await nextNode(), await nextNode()));
    }

    const runId = createMaintenanceRunId();
    const dependencies = {
      db: harness.db,
      reviewQueueRepository: harness.reviewQueueRepository,
      semanticNodeRepository: harness.semanticNodeRepository,
      auditLog: harness.auditLog,
      runId,
    };
    const dryRun = await repairWrongDuplicateSupersedes(dependencies);

    expect(reviewIds).toHaveLength(14);
    expect(dryRun).toMatchObject({
      dryRun: true,
      restored: [],
      currentOnlyReviewIds: [],
      malformedReviews: [],
      outOfScopeReviews: [],
      missingTargets: [],
    });
    expect(dryRun.matchingReviews).toHaveLength(14);
    expect(dryRun.candidates).toHaveLength(13);
    expect(harness.auditLog.list({ process: "review-resolver" })).toEqual([]);
    expect(
      (
        await harness.semanticNodeRepository.getMany(
          dryRun.candidates.map((candidate) => candidate.nodeId),
          { includeArchived: true },
        )
      ).every((node) => node?.status === "superseded"),
    ).toBe(true);

    const applied = await repairWrongDuplicateSupersedes(dependencies, { apply: true });
    const restoredNodes = await harness.semanticNodeRepository.getMany(
      applied.restored.map((candidate) => candidate.nodeId),
      { includeArchived: true },
    );
    const audit = harness.auditLog.list({ run_id: runId, process: "review-resolver" });

    expect(applied.matchingReviews).toHaveLength(14);
    expect(applied.candidates).toHaveLength(13);
    expect(applied.restored).toHaveLength(13);
    expect(
      restoredNodes.every(
        (node) =>
          node?.status === "active" && node.corrected_by === null && node.superseded_at === null,
      ),
    ).toBe(true);
    expect(audit).toHaveLength(13);
    expect(
      audit.every(
        (row) =>
          row.action === WRONG_DUPLICATE_RESTORE_AUDIT_ACTION && row.reversal.no_reverser === true,
      ),
    ).toBe(true);
    expect(
      reviewIds.map((reviewId) => harness.reviewQueueRepository.get(reviewId)?.resolution),
    ).toEqual(Array.from({ length: 14 }, () => "supersede"));

    const secondApply = await repairWrongDuplicateSupersedes(
      { ...dependencies, runId: createMaintenanceRunId() },
      { apply: true },
    );

    expect(secondApply.matchingReviews).toHaveLength(14);
    expect(secondApply.candidates).toEqual([]);
    expect(secondApply.restored).toEqual([]);
    expect(harness.auditLog.list({ process: "review-resolver" })).toHaveLength(13);
  });

  it("reports a current-label-only conflict without changing it", async () => {
    const harness = await createOfflineTestHarness({
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const episode = await harness.episodicRepository.createEpisode(createEpisodeFixture());
    const first = await insertNode(harness, episode.id, 3_000);
    const second = await insertNode(harness, episode.id, 3_001);
    const review = harness.reviewQueueRepository.enqueue({
      kind: "duplicate",
      reason: "Historical duplicate resolution",
      refs: {
        node_ids: [first.id, second.id],
        node_labels: ["Ticket AININJAS-3000", "AININJAS-3000"],
      },
    });
    await harness.reviewQueueRepository.resolve(review.id, {
      decision: "supersede",
      winner_node_id: second.id,
      reason: "Historical resolver chose a winner.",
    });

    const report = await repairWrongDuplicateSupersedes(
      {
        db: harness.db,
        reviewQueueRepository: harness.reviewQueueRepository,
        semanticNodeRepository: harness.semanticNodeRepository,
        auditLog: harness.auditLog,
        runId: createMaintenanceRunId(),
      },
      { apply: true },
    );

    expect(report.matchingReviews).toEqual([]);
    expect(report.currentOnlyReviewIds).toEqual([review.id]);
    expect(report.candidates).toEqual([]);
    expect((await harness.semanticNodeRepository.get(first.id))?.status).toBe("superseded");
    expect(harness.auditLog.list({ process: "review-resolver" })).toEqual([]);
  });

  it("revalidates lifecycle state immediately before applying a restore", async () => {
    const harness = await createOfflineTestHarness({
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const episode = await harness.episodicRepository.createEpisode(createEpisodeFixture());
    const loser = await insertNode(harness, episode.id, 4_000);
    const winner = await insertNode(harness, episode.id, 4_001);
    await resolveSupersede(harness, loser, winner);
    let getManyCalls = 0;
    const unrelatedCorrectionId = createSemanticNodeId();
    const restoreActive = vi.fn(
      harness.semanticNodeRepository.restoreActive.bind(harness.semanticNodeRepository),
    );
    const getMany = vi.fn(async (...args: Parameters<SemanticNodeRepository["getMany"]>) => {
      const nodes = await harness.semanticNodeRepository.getMany(...args);
      getManyCalls += 1;

      if (getManyCalls === 2 && nodes[0] !== null && nodes[0] !== undefined) {
        return nodes.map((node, index) =>
          index === 0 && node !== null ? { ...node, corrected_by: unrelatedCorrectionId } : node,
        );
      }

      return nodes;
    });

    await expect(
      repairWrongDuplicateSupersedes(
        {
          db: harness.db,
          reviewQueueRepository: harness.reviewQueueRepository,
          semanticNodeRepository: { getMany, restoreActive },
          auditLog: harness.auditLog,
          runId: createMaintenanceRunId(),
        },
        { apply: true },
      ),
    ).rejects.toThrow(
      `Semantic node ${loser.id} is no longer superseded by its reviewed counterpart`,
    );
    expect(restoreActive).not.toHaveBeenCalled();
    expect(harness.auditLog.list({ process: "review-resolver" })).toEqual([]);
  });

  it("does not restore a reviewed winner later superseded by an unrelated valid review", async () => {
    const harness = await createOfflineTestHarness({
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const episode = await harness.episodicRepository.createEpisode(createEpisodeFixture());
    const loser = await insertNode(harness, episode.id, 5_000);
    const oldWinner = await insertNode(harness, episode.id, 5_001);
    const unrelatedWinner = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Canonical replacement record",
        description: "This unrelated later review is treated as valid.",
        source_episode_ids: [episode.id],
      }),
    );
    const wrongReviewId = await resolveSupersede(harness, loser, oldWinner);
    await resolveSupersede(harness, oldWinner, unrelatedWinner);

    const report = await repairWrongDuplicateSupersedes(
      {
        db: harness.db,
        reviewQueueRepository: harness.reviewQueueRepository,
        semanticNodeRepository: harness.semanticNodeRepository,
        auditLog: harness.auditLog,
        runId: createMaintenanceRunId(),
      },
      { apply: true },
    );

    expect(report.matchingReviews.map((review) => review.reviewId)).toEqual([wrongReviewId]);
    expect(report.candidates.map((candidate) => candidate.nodeId)).toEqual([loser.id]);
    expect(await harness.semanticNodeRepository.get(loser.id)).toMatchObject({
      status: "active",
      corrected_by: null,
    });
    expect(await harness.semanticNodeRepository.get(oldWinner.id)).toMatchObject({
      status: "superseded",
      corrected_by: unrelatedWinner.id,
    });
    expect(harness.auditLog.list({ process: "review-resolver" })).toHaveLength(1);
  });

  it("rolls back the lifecycle restore when audit insertion fails", async () => {
    const harness = await createOfflineTestHarness({
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const episode = await harness.episodicRepository.createEpisode(createEpisodeFixture());
    const loser = await insertNode(harness, episode.id, 6_000);
    const winner = await insertNode(harness, episode.id, 6_001);
    await resolveSupersede(harness, loser, winner);

    await expect(
      repairWrongDuplicateSupersedes(
        {
          db: harness.db,
          reviewQueueRepository: harness.reviewQueueRepository,
          semanticNodeRepository: harness.semanticNodeRepository,
          auditLog: {
            record: vi.fn(() => {
              throw new Error("simulated audit failure");
            }),
          },
          runId: createMaintenanceRunId(),
        },
        { apply: true },
      ),
    ).rejects.toThrow("simulated audit failure");

    expect(await harness.semanticNodeRepository.get(loser.id)).toMatchObject({
      status: "superseded",
      corrected_by: winner.id,
    });
    expect(harness.auditLog.list({ process: "review-resolver" })).toEqual([]);
  });

  it("separates malformed, out-of-scope, and missing-target rows and fails unsafe apply reports", async () => {
    const harness = await createOfflineTestHarness({
      reviewOpenQuestionExtractor: null,
    });
    cleanup.push(harness.cleanup);
    const missingFirst = createSemanticNodeId();
    const missingSecond = createSemanticNodeId();
    const rows = [
      {
        id: 101,
        kind: "duplicate",
        refs: { node_ids: ["not-a-node-id"] },
        reason: "Malformed historical row",
        created_at: 1,
        resolved_at: 2,
        resolution: "supersede",
      },
      {
        id: 102,
        kind: "duplicate",
        refs: {
          loser_edge_id: createSemanticEdgeId(),
          reason: "Valid edge closure",
        },
        reason: "Out-of-scope historical row",
        created_at: 3,
        resolved_at: 4,
        resolution: "supersede",
      },
      {
        id: 103,
        kind: "duplicate",
        refs: {
          node_ids: [missingFirst, missingSecond],
          node_labels: ["Ticket AININJAS-7000", "Ticket AININJAS-7001"],
        },
        reason: "Missing historical targets",
        created_at: 5,
        resolved_at: 6,
        resolution: "supersede",
      },
    ] satisfies ReviewQueueItem[];
    const reviewQueueRepository = {
      list: ({ kind }: { kind?: ReviewQueueItem["kind"] }) => (kind === "duplicate" ? rows : []),
    };
    const semanticNodeRepository = {
      getMany: vi.fn(async () => [null, null]),
      restoreActive: vi.fn(async () => null),
    };
    const dependencies = {
      db: harness.db,
      reviewQueueRepository,
      semanticNodeRepository,
      auditLog: harness.auditLog,
      runId: createMaintenanceRunId(),
    };

    const dryRun = await repairWrongDuplicateSupersedes(dependencies);
    const applied = await repairWrongDuplicateSupersedes(dependencies, { apply: true });

    expect(dryRun.malformedReviews).toEqual([
      { reviewId: 101, reason: "refs failed semantic-pair validation" },
    ]);
    expect(dryRun.outOfScopeReviews).toEqual([
      { reviewId: 102, reason: "semantic edge-closure refs" },
    ]);
    expect(dryRun.missingTargets).toEqual([
      {
        reviewId: 103,
        nodeIds: [missingFirst, missingSecond],
        missingNodeIds: [missingFirst, missingSecond],
      },
    ]);
    expect(repairReportExitCode(dryRun)).toBe(0);
    expect(repairReportExitCode(applied)).toBe(1);
    expect(semanticNodeRepository.restoreActive).not.toHaveBeenCalled();
  });
});
