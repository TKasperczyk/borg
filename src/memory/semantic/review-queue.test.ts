import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { LanceDbStore } from "../../storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createAutobiographicalPeriodId,
  createSemanticNodeId,
  type EpisodeId,
  type SkillId,
} from "../../util/ids.js";
import { OpenQuestionsRepository, selfMigrations } from "../self/index.js";
import {
  enqueueOpenQuestionForReview,
  type ReviewOpenQuestionExtractorLike,
} from "../self/review-open-question-hook.js";
import { deriveProceduralContextKey } from "../procedural/index.js";
import { semanticMigrations } from "./migrations.js";
import {
  ReviewQueueRepository,
  type ReviewQueueHandler,
  type ReviewResolution,
} from "./review-queue.js";
import { createCorrectionReviewHandler } from "./review-handlers/correction.js";
import { createSkillSplitReviewQueueHandler } from "./review-handlers/skill-split.js";
import { registerBuiltinReviewQueueHandlers } from "./review-handlers/defaults.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "./repository.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../../offline/test-support.js";

const TYPESCRIPT_DEBUG_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "code_debugging",
  domain_tags: ["typescript"],
  audience_scope: "self",
});
const ROADMAP_PLANNING_CONTEXT_KEY = deriveProceduralContextKey({
  problem_kind: "planning",
  domain_tags: ["roadmap"],
  audience_scope: "self",
});

function createPendingInsightRefs(input: {
  nodeId: string;
  episodeId: EpisodeId;
  description: string;
  confidence: number;
  lastVerifiedAt: number;
  embedding: readonly number[];
  clusterKey?: string;
}) {
  const clusterKey = input.clusterKey ?? "cluster:insight";

  return {
    node_ids: [input.nodeId],
    episode_ids: [input.episodeId],
    evidence_cluster_key: clusterKey,
    evidence_cluster_size: 1,
    reflector_pending_insight: {
      target: {
        mode: "update" as const,
        node_id: input.nodeId,
        patch: {
          description: input.description,
          confidence: input.confidence,
          source_episode_ids: [input.episodeId],
          last_verified_at: input.lastVerifiedAt,
          embedding: [...input.embedding],
          archived: false,
        },
      },
      candidate_support_edges: [],
      evidence_cluster: {
        key: clusterKey,
        episode_ids: [input.episodeId],
        size: 1,
      },
    },
  };
}

function createDeferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((next) => {
    resolve = next;
  });

  return {
    promise,
    resolve,
  };
}

describe("review queue", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("enqueues contradiction reviews on conflicting support paths and resolves them", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(semanticMigrations, selfMigrations),
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      clock,
    });
    const pendingHooks: Promise<void>[] = [];
    const openQuestionExtractor = {
      extract: vi.fn(async (_item, context) => ({
        question: "¿Qué afirmación debería conservarse?",
        urgency: 0.82,
        related_episode_ids: [],
        related_semantic_node_ids: [...context.allowed_semantic_node_ids],
      })),
    } satisfies ReviewOpenQuestionExtractorLike;
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock,
      semanticNodeRepository: nodeRepository,
      onEnqueue: (item) => {
        const pending = enqueueOpenQuestionForReview(openQuestionsRepository, item, {
          extractor: openQuestionExtractor,
        });
        pendingHooks.push(pending);
        return pending;
      },
    });
    registerBuiltinReviewQueueHandlers(reviewQueue);
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
      enqueueReview: (input) => reviewQueue.enqueue(input),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episodeIds = ["ep_aaaaaaaaaaaaaaaa" as EpisodeId];
    const first = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas succeeds",
      description: "Atlas succeeds",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const second = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas fails",
      description: "Atlas does not succeed",
      aliases: [],
      confidence: 0.6,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    edgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    edgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });

    await Promise.all(pendingHooks);

    const openItems = reviewQueue.getOpen();
    const openQuestions = openQuestionsRepository.list({ status: "open" });

    expect(openItems).toHaveLength(1);
    expect(openItems[0]?.kind).toBe("contradiction");
    expect(openItems[0]?.reason).toContain("conflicts_with_support_chain");
    expect(openQuestionExtractor.extract).toHaveBeenCalledOnce();
    expect(openQuestions[0]?.source).toBe("contradiction");
    expect(openQuestions[0]?.question).toBe("¿Qué afirmación debería conservarse?");
    expect(openQuestions[0]?.urgency).toBe(0.82);
    expect(openQuestions[0]?.related_semantic_node_ids).toEqual([first.id, second.id]);

    const resolved = await reviewQueue.resolve(openItems[0]!.id, {
      decision: "invalidate",
      winner_node_id: first.id,
    });
    const updatedSecond = await nodeRepository.get(second.id);

    expect(resolved?.resolution).toBe("invalidate");
    expect(updatedSecond?.archived).toBe(true);
    expect(updatedSecond?.confidence).toBe(0);
  });

  it("closes loser semantic edges on contradiction review invalidation", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(1_000),
    });
    cleanup.push(harness.cleanup);

    const episodeId = createEpisodeFixture().id;
    const first = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Atlas is stable",
          description: "Atlas is stable after the rollback.",
          source_episode_ids: [episodeId],
          confidence: 0.8,
        },
        [1, 0, 0, 0],
      ),
    );
    const second = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Rollback completed",
          description: "Rollback completed before the review.",
          source_episode_ids: [episodeId],
          confidence: 0.7,
        },
        [0, 1, 0, 0],
      ),
    );
    const loserEdge = harness.semanticEdgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "supports",
      confidence: 0.73,
      evidence_episode_ids: [episodeId],
      created_at: 1_000,
      last_verified_at: 1_000,
    });
    const item = harness.reviewQueueRepository.enqueue({
      kind: "contradiction",
      refs: {
        loser_edge_id: loserEdge.id,
        suggested_valid_to: 1_200,
        reason: "newer review evidence superseded the support edge",
      },
      reason: "support edge lost the contradiction review",
    });

    const resolved = await harness.reviewQueueRepository.resolve(item.id, "invalidate");
    const closedEdge = harness.semanticEdgeRepository.getEdge(loserEdge.id);
    const firstAfter = await harness.semanticNodeRepository.get(first.id);
    const secondAfter = await harness.semanticNodeRepository.get(second.id);
    const auditEvents = harness.identityEventRepository.list({
      recordType: "semantic_edge",
      recordId: loserEdge.id,
    });

    expect(resolved?.resolution).toBe("invalidate");
    expect(closedEdge).toEqual(
      expect.objectContaining({
        id: loserEdge.id,
        confidence: 0.73,
        evidence_episode_ids: [episodeId],
        valid_to: 1_200,
        invalidated_by_process: "review",
        invalidated_by_review_id: item.id,
        invalidated_reason: "newer review evidence superseded the support edge",
      }),
    );
    expect(firstAfter).toEqual(expect.objectContaining({ archived: false, confidence: 0.8 }));
    expect(secondAfter).toEqual(expect.objectContaining({ archived: false, confidence: 0.7 }));
    expect(auditEvents[0]).toEqual(
      expect.objectContaining({
        action: "edge_invalidate",
        review_item_id: item.id,
        new_value: expect.objectContaining({
          edge_id: loserEdge.id,
          prior_valid_to: null,
          new_valid_to: 1_200,
          by_process: "review",
          by_review_id: item.id,
          reason: "newer review evidence superseded the support edge",
        }),
      }),
    );
  });

  it("logs hook failures without aborting edge insertion", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(semanticMigrations, selfMigrations),
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock,
      semanticNodeRepository: nodeRepository,
      onEnqueue() {
        throw new Error("hook exploded");
      },
      onEnqueueError: (error) => {
        void writer.append({
          kind: "internal_event",
          content: {
            hook: "review_queue_open_question",
            error: error instanceof Error ? error.message : String(error),
          },
        });
      },
    });
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
      enqueueReview: (input) => reviewQueue.enqueue(input),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episodeIds = ["ep_cccccccccccccccc" as EpisodeId];
    const first = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas works",
      description: "Atlas works",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const second = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas breaks",
      description: "Atlas breaks",
      aliases: [],
      confidence: 0.6,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    const edge = edgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });

    await new Promise((resolve) => {
      setTimeout(resolve, 25);
    });

    const entries = new StreamReader({
      dataDir: tempDir,
    }).tail(1);

    expect(edge.relation).toBe("contradicts");
    expect(reviewQueue.getOpen()).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      kind: "internal_event",
      content: {
        hook: "review_queue_open_question",
      },
    });
  });

  it("requires an explicit winner for contradiction invalidation", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(semanticMigrations, selfMigrations),
    });
    const table = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(1_000);
    const nodeRepository = new SemanticNodeRepository({
      table,
      db,
      clock,
    });
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock,
      semanticNodeRepository: nodeRepository,
    });
    registerBuiltinReviewQueueHandlers(reviewQueue);
    const edgeRepository = new SemanticEdgeRepository({
      db,
      clock,
      enqueueReview: (input) => reviewQueue.enqueue(input),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episodeIds = ["ep_aaaaaaaaaaaaaaaa" as EpisodeId];
    const first = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas succeeds",
      description: "Atlas succeeds",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const second = await nodeRepository.insert({
      id: createSemanticNodeId(),
      kind: "proposition",
      label: "Atlas fails",
      description: "Atlas does not succeed",
      aliases: [],
      confidence: 0.6,
      source_episode_ids: episodeIds,
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });

    edgeRepository.addEdge({
      from_node_id: first.id,
      to_node_id: second.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: episodeIds,
      created_at: 1_000,
      last_verified_at: 1_000,
    });

    const openItem = reviewQueue.getOpen()[0];

    await expect(reviewQueue.resolve(openItem!.id, "invalidate")).rejects.toMatchObject({
      code: "REVIEW_QUEUE_WINNER_REQUIRED",
    });
  });

  it("applies misattribution repairs to episodes on accept", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(5_000),
    });
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture(
        {
          title: "Misattributed review",
          narrative: "Alex led the review, but only the team is listed.",
          participants: ["team"],
          tags: ["review"],
          created_at: 100,
          updated_at: 100,
        },
        [1, 0, 0, 0],
      ),
    );

    const item = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs: {
        target_type: "episode",
        target_id: episode.id,
        patch: {
          participants: ["team", "Alex"],
          narrative: "Alex led the review.",
          tags: ["review", "alex"],
        },
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
      reason: "episode attribution is wrong",
    });

    await harness.reviewQueueRepository.resolve(item.id, "accept");
    const updated = await harness.episodicRepository.get(episode.id);

    expect(updated?.participants).toEqual(["team", "Alex"]);
    expect(updated?.narrative).toBe("Alex led the review.");
    expect(updated?.tags).toEqual(["review", "alex"]);
  });

  it("replaces semantic node source episodes when accepting misattribution repairs", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(5_500),
    });
    cleanup.push(harness.cleanup);

    const firstEpisodeId = "ep_aaaaaaaaaaaaaaaa" as EpisodeId;
    const secondEpisodeId = "ep_bbbbbbbbbbbbbbbb" as EpisodeId;
    const node = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture({
        label: "Misattributed semantic node",
        source_episode_ids: [firstEpisodeId, secondEpisodeId],
      }),
    );
    const item = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs: {
        target_type: "semantic_node",
        target_id: node.id,
        patch: {
          source_episode_ids: [firstEpisodeId],
        },
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
      reason: "semantic node attribution includes the wrong episode",
    });

    await harness.reviewQueueRepository.resolve(item.id, "accept");

    expect((await harness.semanticNodeRepository.get(node.id))?.source_episode_ids).toEqual([
      firstEpisodeId,
    ]);
  });

  it("applies temporal drift repairs to semantic nodes on accept", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(6_000),
    });
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture().id;
    const node = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Atlas postmortem",
          description: "This happened before the rollback.",
          source_episode_ids: [episode],
          last_verified_at: 100,
        },
        [0, 0, 1, 0],
      ),
    );

    const item = harness.reviewQueueRepository.enqueue({
      kind: "temporal_drift",
      refs: {
        target_type: "semantic_node",
        target_id: node.id,
        patch_description: "This happened after the rollback.",
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
      reason: "temporal claim drifted",
    });

    await harness.reviewQueueRepository.resolve(item.id, "accept");
    const updated = await harness.semanticNodeRepository.get(node.id);

    expect(updated?.description).toBe("This happened after the rollback.");
    expect(updated?.last_verified_at).toBe(6_000);
  });

  it("applies identity inconsistency reinforce, contradict, and patch repairs on accept", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(7_000),
    });
    cleanup.push(harness.cleanup);

    const evidenceEpisode = createEpisodeFixture().id;
    const value = harness.valuesRepository.add({
      label: "groundedness",
      description: "Stay grounded.",
      priority: 5,
      provenance: {
        kind: "manual",
      },
    });
    const trait = harness.traitsRepository.reinforce({
      label: "patient",
      delta: 0.1,
      provenance: {
        kind: "manual",
      },
      timestamp: 6_900,
    });
    const goal = harness.goalsRepository.add({
      description: "Stabilize Atlas",
      priority: 8,
      provenance: {
        kind: "episodes",
        episode_ids: [evidenceEpisode],
      },
    });
    const period = harness.autobiographicalRepository.upsertPeriod({
      label: "2026-Q2",
      start_ts: 1_000,
      narrative: "Initial narrative.",
      key_episode_ids: [evidenceEpisode],
      themes: ["stability"],
      provenance: {
        kind: "episodes",
        episode_ids: [evidenceEpisode],
      },
    });

    const reinforce = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      refs: {
        target_type: "value",
        target_id: value.id,
        repair_op: "reinforce",
        evidence_episode_ids: [evidenceEpisode],
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
      reason: "episode reinforces the value",
    });
    const contradict = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      refs: {
        target_type: "trait",
        target_id: trait.id,
        repair_op: "contradict",
        evidence_episode_ids: [evidenceEpisode],
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
      reason: "episode contradicts the trait",
    });
    const patchGoal = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      refs: {
        target_type: "goal",
        target_id: goal.id,
        repair_op: "patch",
        patch: {
          progress_notes: "Reviewed progress.",
          last_progress_ts: 7_000,
        },
        proposed_provenance: {
          kind: "online",
          process: "reflector",
        },
      },
      reason: "goal progress should be revised",
    });
    const patchPeriod = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      refs: {
        target_type: "autobiographical_period",
        target_id: period.id,
        repair_op: "patch",
        patch: {
          narrative: "Reconciled narrative.",
        },
        proposed_provenance: {
          kind: "offline",
          process: "self-narrator",
        },
        evidence_episode_ids: [evidenceEpisode],
      },
      reason: "period narrative should be revised",
    });

    await harness.reviewQueueRepository.resolve(reinforce.id, "accept");
    await harness.reviewQueueRepository.resolve(contradict.id, "accept");
    await harness.reviewQueueRepository.resolve(patchGoal.id, "accept");
    await harness.reviewQueueRepository.resolve(patchPeriod.id, "accept");

    expect(harness.valuesRepository.get(value.id)?.support_count).toBe(1);
    expect(harness.valuesRepository.get(value.id)?.evidence_episode_ids).toEqual([evidenceEpisode]);
    expect(harness.traitsRepository.get(trait.id)?.contradiction_count).toBe(1);
    expect(harness.goalsRepository.get(goal.id)).toEqual(
      expect.objectContaining({
        progress_notes: "Reviewed progress.",
        provenance: {
          kind: "online",
          process: "reflector",
        },
      }),
    );
    expect(
      harness.identityEventRepository
        .list({
          recordType: "goal",
          recordId: goal.id,
        })
        .find((event) => event.review_item_id === patchGoal.id),
    ).toEqual(
      expect.objectContaining({
        provenance: {
          kind: "online",
          process: "reflector",
        },
      }),
    );
    expect(harness.autobiographicalRepository.getPeriod(period.id)).toEqual(
      expect.objectContaining({
        narrative: "Reconciled narrative.",
        provenance: {
          kind: "offline",
          process: "self-narrator",
        },
      }),
    );
  });

  it("records identity events for accepted autobiographical period rollovers", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(7_250),
    });
    cleanup.push(harness.cleanup);

    const evidenceEpisode = createEpisodeFixture().id;
    const currentPeriod = harness.autobiographicalRepository.upsertPeriod({
      label: "2026-Q2",
      start_ts: 1_000,
      narrative: "Current period.",
      key_episode_ids: [evidenceEpisode],
      themes: ["stability"],
      provenance: {
        kind: "episodes",
        episode_ids: [evidenceEpisode],
      },
    });
    const nextPeriodId = createAutobiographicalPeriodId();
    const rollover = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      refs: {
        target_type: "autobiographical_period",
        target_id: currentPeriod.id,
        repair_op: "patch",
        patch: {
          end_ts: 7_000,
        },
        proposed_provenance: {
          kind: "offline",
          process: "self-narrator",
        },
        next_period_open_payload: {
          id: nextPeriodId,
          label: "2026-Q3",
          start_ts: 7_000,
          end_ts: null,
          narrative: "Next period.",
          key_episode_ids: [evidenceEpisode],
          themes: ["rollover"],
          provenance: {
            kind: "offline",
            process: "self-narrator",
          },
          created_at: 7_000,
          last_updated: 7_000,
        },
      },
      reason: "period should roll over",
    });

    await harness.reviewQueueRepository.resolve(rollover.id, "accept");

    expect(harness.autobiographicalRepository.getPeriod(currentPeriod.id)).toEqual(
      expect.objectContaining({
        end_ts: 7_000,
      }),
    );
    expect(harness.autobiographicalRepository.getPeriod(nextPeriodId)).toEqual(
      expect.objectContaining({
        label: "2026-Q3",
      }),
    );
    expect(
      harness.identityEventRepository.list({
        recordType: "autobiographical_period",
        recordId: nextPeriodId,
      }),
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          action: "create",
        }),
      ]),
    );
  });

  it("rolls back SQLite-only repair side effects when resolution application fails", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(7_500),
    });
    cleanup.push(harness.cleanup);

    const evidenceEpisode = createEpisodeFixture().id;
    const value = harness.valuesRepository.add({
      label: "carefulness",
      description: "Prefer careful changes.",
      priority: 5,
      provenance: {
        kind: "manual",
      },
    });
    const item = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      refs: {
        target_type: "value",
        target_id: value.id,
        repair_op: "reinforce",
        evidence_episode_ids: [evidenceEpisode],
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
      reason: "episode reinforces carefulness",
    });
    const originalReinforce = harness.valuesRepository.reinforce.bind(harness.valuesRepository);
    const reinforceSpy = vi
      .spyOn(harness.valuesRepository, "reinforce")
      .mockImplementation((...args: Parameters<typeof harness.valuesRepository.reinforce>) => {
        originalReinforce(...args);
        throw new Error("reinforce failed after write");
      });

    await expect(harness.reviewQueueRepository.resolve(item.id, "accept")).rejects.toThrow(
      "reinforce failed after write",
    );
    expect(harness.valuesRepository.listReinforcementEvents(value.id)).toHaveLength(0);
    expect(harness.valuesRepository.get(value.id)?.support_count).toBe(0);
    expect(harness.reviewQueueRepository.getOpen().map((openItem) => openItem.id)).toContain(
      item.id,
    );

    reinforceSpy.mockRestore();
    await harness.reviewQueueRepository.resolve(item.id, "accept");

    expect(harness.valuesRepository.listReinforcementEvents(value.id)).toHaveLength(1);
    expect(harness.valuesRepository.get(value.id)?.support_count).toBe(1);
  });

  it("applies accepted pending reflector insights and archives invalidated ones", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(8_000),
    });
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture().id;
    const freshNode = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Fresh insight",
          description: "A newly reflected proposition.",
          source_episode_ids: [episode],
          confidence: 0.6,
          last_verified_at: 50,
        },
        [0, 0, 1, 0],
      ),
    );

    const newInsight = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      refs: createPendingInsightRefs({
        nodeId: freshNode.id,
        episodeId: episode,
        description: "An updated reflected proposition.",
        confidence: 0.7,
        lastVerifiedAt: 8_000,
        embedding: [0, 0, 1, 0],
      }),
      reason: "fresh insight should be reconsidered",
    });

    await harness.reviewQueueRepository.resolve(newInsight.id, "accept");

    expect(await harness.semanticNodeRepository.get(freshNode.id)).toEqual(
      expect.objectContaining({
        archived: false,
        description: "An updated reflected proposition.",
        confidence: 0.7,
        last_verified_at: 8_000,
      }),
    );
    const invalidatedInsight = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      refs: createPendingInsightRefs({
        nodeId: freshNode.id,
        episodeId: episode,
        description: "This patch should not be applied.",
        confidence: 0.95,
        lastVerifiedAt: 9_000,
        embedding: [1, 0, 0, 0],
        clusterKey: "cluster:invalidated",
      }),
      reason: "fresh insight should be reconsidered again",
    });

    await harness.reviewQueueRepository.resolve(invalidatedInsight.id, "invalidate");

    expect(await harness.semanticNodeRepository.get(freshNode.id)).toEqual(
      expect.objectContaining({
        archived: true,
        description: "An updated reflected proposition.",
        last_verified_at: 8_000,
        confidence: 0.7,
      }),
    );
  });

  it("rejects legacy new insight refs even for dismiss", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(8_500),
    });
    cleanup.push(harness.cleanup);

    const legacy = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      refs: {
        node_ids: ["semn_cccccccccccccccc"],
      },
      reason: "legacy reflector insight",
    });

    await expect(harness.reviewQueueRepository.resolve(legacy.id, "dismiss")).rejects.toThrow();
    expect(harness.reviewQueueRepository.getOpen().map((item) => item.id)).toContain(legacy.id);
  });

  it("rejects accept on under-specified repair rows and leaves them open", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(9_000),
    });
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(createEpisodeFixture());
    const goal = harness.goalsRepository.add({
      description: "Stabilize Atlas",
      priority: 8,
      provenance: {
        kind: "manual",
      },
    });
    const underSpecifiedMisattribution = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs: {
        target_type: "episode",
        target_id: episode.id,
      },
      reason: "under-specified row missing patch",
    });
    const underSpecifiedTemporalDrift = harness.reviewQueueRepository.enqueue({
      kind: "temporal_drift",
      refs: {
        target_type: "episode",
        target_id: episode.id,
      },
      reason: "under-specified row missing corrected timestamps",
    });
    const underSpecifiedIdentityRepair = harness.reviewQueueRepository.enqueue({
      kind: "identity_inconsistency",
      refs: {
        target_type: "goal",
        target_id: goal.id,
        repair_op: "patch",
        proposed_provenance: {
          kind: "offline",
          process: "overseer",
        },
      },
      reason: "under-specified row missing identity patch",
    });

    await expect(
      harness.reviewQueueRepository.resolve(underSpecifiedMisattribution.id, "accept"),
    ).rejects.toThrow();
    await expect(
      harness.reviewQueueRepository.resolve(underSpecifiedTemporalDrift.id, "accept"),
    ).rejects.toThrow();
    await expect(
      harness.reviewQueueRepository.resolve(underSpecifiedIdentityRepair.id, "accept"),
    ).rejects.toThrow();

    expect(harness.reviewQueueRepository.getOpen().map((item) => item.id)).toEqual(
      expect.arrayContaining([
        underSpecifiedMisattribution.id,
        underSpecifiedTemporalDrift.id,
        underSpecifiedIdentityRepair.id,
      ]),
    );
  });

  it("rejects malformed semantic pair refs before resolving", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(9_000),
    });
    cleanup.push(harness.cleanup);

    const malformedDuplicate = harness.reviewQueueRepository.enqueue({
      kind: "duplicate",
      refs: {
        node_ids: ["semn_aaaaaaaaaaaaaaaa"],
      },
      reason: "under-specified row lost one side of the pair",
    });
    const malformedContradiction = harness.reviewQueueRepository.enqueue({
      kind: "contradiction",
      refs: {},
      reason: "under-specified row lost the pair refs",
    });

    await expect(
      harness.reviewQueueRepository.resolve(malformedDuplicate.id, {
        decision: "supersede",
        winner_node_id: "semn_aaaaaaaaaaaaaaaa" as never,
      }),
    ).rejects.toThrow();
    await expect(
      harness.reviewQueueRepository.resolve(malformedContradiction.id, {
        decision: "invalidate",
        winner_node_id: "semn_bbbbbbbbbbbbbbbb" as never,
      }),
    ).rejects.toThrow();
    await expect(
      harness.reviewQueueRepository.resolve(malformedDuplicate.id, "dismiss"),
    ).rejects.toThrow();

    expect(harness.reviewQueueRepository.getOpen().map((item) => item.id)).toEqual(
      expect.arrayContaining([malformedDuplicate.id, malformedContradiction.id]),
    );
  });

  it("keeps semantic pair reviews open when required nodes are missing", async () => {
    const db = openDatabase(":memory:", {
      migrations: [...semanticMigrations],
    });
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock: new FixedClock(9_250),
      semanticNodeRepository: {
        getMany: vi.fn(async () => [null, null]),
      } as never,
    });
    registerBuiltinReviewQueueHandlers(reviewQueue);

    try {
      const item = reviewQueue.enqueue({
        kind: "duplicate",
        refs: {
          node_ids: ["semn_aaaaaaaaaaaaaaaa", "semn_bbbbbbbbbbbbbbbb"],
        },
        reason: "duplicate pair points at missing nodes",
      });

      await expect(
        reviewQueue.resolve(item.id, {
          decision: "supersede",
          winner_node_id: "semn_aaaaaaaaaaaaaaaa" as never,
        }),
      ).rejects.toMatchObject({
        name: "SemanticError",
        code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
      });

      expect(reviewQueue.get(item.id)).toMatchObject({
        resolved_at: null,
        resolution: null,
      });
    } finally {
      db.close();
    }
  });

  it("guards cross-store applying-state acquisition with compare-and-set", async () => {
    const db = openDatabase(":memory:", {
      migrations: [...semanticMigrations],
    });
    const prepareGate = createDeferred();
    const applyGate = createDeferred();
    const apply = vi.fn(async () => {
      await applyGate.promise;
    });
    const handler: ReviewQueueHandler<
      "correction",
      unknown,
      { decision: "accept"; started_at: number }
    > = {
      kind: "correction",
      refsSchema: z.object({}).strict(),
      allowedResolutions: new Set<ReviewResolution>(["accept"]),
      transactionScope: () => "cross_store_applying_state",
      applyingState: {
        key: "__test_applying",
        schema: z
          .object({
            decision: z.literal("accept"),
            started_at: z.number().finite(),
          })
          .strict(),
        async prepare({ ctx }) {
          await prepareGate.promise;
          return {
            decision: "accept",
            started_at: ctx.clock.now(),
          };
        },
        matches: (state, resolution) => state.decision === resolution.decision,
      },
      apply,
    };
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock: new FixedClock(9_500),
    });
    reviewQueue.registerHandler(handler);

    try {
      const item = reviewQueue.enqueue({
        kind: "correction",
        refs: {},
        reason: "custom cross-store correction",
      });
      const first = reviewQueue.resolve(item.id, "accept");
      const second = reviewQueue.resolve(item.id, "accept");
      const settled = Promise.allSettled([first, second]);

      prepareGate.resolve();
      await new Promise((resolve) => {
        setTimeout(resolve, 0);
      });

      applyGate.resolve();
      const results = await settled;
      const fulfilled = results.filter((result) => result.status === "fulfilled");
      const rejected = results.filter((result) => result.status === "rejected");

      expect(apply).toHaveBeenCalledTimes(1);
      expect(fulfilled).toHaveLength(1);
      expect(rejected).toHaveLength(1);
      expect(rejected[0]).toMatchObject({
        reason: expect.objectContaining({
          code: "REVIEW_QUEUE_RESOLUTION_RACE",
        }),
      });
      expect(reviewQueue.get(item.id)).toMatchObject({
        resolved_at: 9_500,
        resolution: "accept",
      });
    } finally {
      db.close();
    }
  });

  it("guards external review finalization with compare-and-set", async () => {
    const db = openDatabase(":memory:", {
      migrations: [...semanticMigrations],
    });
    const applyGate = createDeferred();
    const apply = vi.fn(async () => {
      await applyGate.promise;
    });
    const handler: ReviewQueueHandler<"correction", unknown> = {
      kind: "correction",
      refsSchema: z.object({}).strict(),
      allowedResolutions: new Set<ReviewResolution>(["accept"]),
      transactionScope: () => "external",
      apply,
    };
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock: new FixedClock(9_750),
    });
    reviewQueue.registerHandler(handler);

    try {
      const item = reviewQueue.enqueue({
        kind: "correction",
        refs: {},
        reason: "custom external correction",
      });
      const first = reviewQueue.resolve(item.id, "accept");
      const second = reviewQueue.resolve(item.id, "accept");
      const settled = Promise.allSettled([first, second]);

      // External handlers run before the review row finalization CAS. The queue
      // guarantees that only one resolver wins the resolved row; external
      // side-effect safety belongs to the handler itself, such as skill_split's
      // claimedAt validation inside supersedeWithSplits().
      applyGate.resolve();
      const results = await settled;
      const fulfilled = results.filter((result) => result.status === "fulfilled");
      const rejected = results.filter((result) => result.status === "rejected");

      expect(fulfilled).toHaveLength(1);
      expect(rejected).toHaveLength(1);
      expect(rejected[0]).toMatchObject({
        reason: expect.objectContaining({
          code: "REVIEW_QUEUE_ALREADY_RESOLVED",
        }),
      });
      expect(reviewQueue.get(item.id)).toMatchObject({
        resolved_at: 9_750,
        resolution: "accept",
      });
    } finally {
      db.close();
    }
  });

  it("resolves harness correction rows through the registered correction handler", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(11_500),
    });
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Correction target episode",
        narrative: "Original narrative.",
        created_at: 11_000,
        updated_at: 11_000,
      }),
    );
    const correction = harness.reviewQueueRepository.enqueue({
      kind: "correction",
      refs: {
        target_type: "episode",
        target_id: episode.id,
        patch: {
          narrative: "Corrected narrative.",
        },
        proposed_provenance: {
          kind: "manual",
        },
      },
      reason: "user corrected the episode narrative",
    });

    const resolved = await harness.reviewQueueRepository.resolve(correction.id, "accept");

    expect(resolved?.resolution).toBe("accept");
    expect(await harness.episodicRepository.get(episode.id)).toEqual(
      expect.objectContaining({
        narrative: "Corrected narrative.",
      }),
    );
    expect(
      harness.identityEventRepository.list({
        recordType: "episode",
        recordId: episode.id,
        limit: 10,
      }),
    ).toEqual([
      expect.objectContaining({
        action: "correction_apply",
        review_item_id: correction.id,
      }),
    ]);
  });

  it("only allows manual dismiss for belief revision reviews", async () => {
    const db = openDatabase(":memory:", {
      migrations: [...semanticMigrations],
    });
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock: new FixedClock(11_750),
    });
    registerBuiltinReviewQueueHandlers(reviewQueue);
    const refs = {
      target_type: "semantic_node",
      target_id: "semn_aaaaaaaaaaaaaaaa",
      invalidated_edge_id: "seme_aaaaaaaaaaaaaaaa",
      dependency_path_edge_ids: ["seme_aaaaaaaaaaaaaaaa"],
      surviving_support_edge_ids: [],
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      audience_entity_id: null,
    };

    try {
      const dismissedItem = reviewQueue.enqueue({
        kind: "belief_revision",
        refs,
        reason: "support chain collapsed",
      });
      const dismissed = await reviewQueue.resolve(dismissedItem.id, "dismiss");

      expect(dismissed?.resolution).toBe("dismiss");

      for (const decision of ["keep", "weaken", "archive_node", "invalidate_edge"] as const) {
        const item = reviewQueue.enqueue({
          kind: "belief_revision",
          refs,
          reason: "support chain collapsed",
        });

        await expect(reviewQueue.resolve(item.id, decision)).rejects.toMatchObject({
          name: "SemanticError",
          code: "REVIEW_QUEUE_RESOLUTION_INVALID",
        });
      }
    } finally {
      db.close();
    }
  });

  it("rejects incompatible review decisions and allows valid pairings", async () => {
    const db = openDatabase(":memory:", {
      migrations: [...semanticMigrations],
    });
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock: new FixedClock(1_000),
    });
    registerBuiltinReviewQueueHandlers(reviewQueue);
    reviewQueue.registerHandler(
      createCorrectionReviewHandler({
        applyCorrection: vi.fn(),
      }),
    );
    reviewQueue.registerHandler(
      createSkillSplitReviewQueueHandler({
        accept: () => ({
          status: "applied",
          newSkillIds: ["skl_bbbbbbbbbbbbbbbb" as SkillId],
        }),
        reject: () => undefined,
      }),
    );

    try {
      const correction = reviewQueue.enqueue({
        kind: "correction",
        refs: {
          target_type: "value",
          target_id: "val_aaaaaaaaaaaaaaaa",
          patch: {
            description: "Prefer grounded claims.",
          },
        },
        reason: "user corrected the record",
      });
      const contradiction = reviewQueue.enqueue({
        kind: "contradiction",
        refs: {
          node_ids: ["semn_aaaaaaaaaaaaaaaa", "semn_bbbbbbbbbbbbbbbb"],
        },
        reason: "conflicting support paths",
      });
      const newInsight = reviewQueue.enqueue({
        kind: "new_insight",
        refs: createPendingInsightRefs({
          nodeId: "semn_dddddddddddddddd",
          episodeId: "ep_aaaaaaaaaaaaaaaa" as EpisodeId,
          description: "Fresh reflector insight.",
          confidence: 0.7,
          lastVerifiedAt: 1_000,
          embedding: [0, 0, 0, 1],
        }),
        reason: "fresh reflector insight",
      });
      const skillSplit = reviewQueue.enqueue({
        kind: "skill_split",
        refs: {
          target_type: "skill",
          target_id: "skl_aaaaaaaaaaaaaaaa",
          original_skill_id: "skl_aaaaaaaaaaaaaaaa",
          proposed_children: [
            {
              label: "Debug comparison",
              problem: "Debug comparison",
              approach: "Compare the failing state.",
              context_stats: [
                {
                  skill_id: "skl_aaaaaaaaaaaaaaaa",
                  context_key: TYPESCRIPT_DEBUG_CONTEXT_KEY,
                  alpha: 2,
                  beta: 1,
                  attempts: 1,
                  successes: 1,
                  failures: 0,
                  last_used: 1_000,
                  last_successful: 1_000,
                  updated_at: 1_000,
                },
              ],
            },
            {
              label: "Planning comparison",
              problem: "Planning comparison",
              approach: "Compare the roadmap state.",
              context_stats: [
                {
                  skill_id: "skl_aaaaaaaaaaaaaaaa",
                  context_key: ROADMAP_PLANNING_CONTEXT_KEY,
                  alpha: 1,
                  beta: 2,
                  attempts: 1,
                  successes: 0,
                  failures: 1,
                  last_used: 1_000,
                  last_successful: null,
                  updated_at: 1_000,
                },
              ],
            },
          ],
          rationale: "The contexts diverge.",
          evidence_summary: {
            source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            divergence: 0.5,
            min_posterior_mean: 0.25,
            max_posterior_mean: 0.75,
            buckets: [
              {
                context_key: TYPESCRIPT_DEBUG_CONTEXT_KEY,
                posterior_mean: 0.75,
                alpha: 2,
                beta: 1,
                attempts: 1,
                successes: 1,
                failures: 0,
                last_used: 1_000,
                last_successful: 1_000,
              },
              {
                context_key: ROADMAP_PLANNING_CONTEXT_KEY,
                posterior_mean: 0.25,
                alpha: 1,
                beta: 2,
                attempts: 1,
                successes: 0,
                failures: 1,
                last_used: 1_000,
                last_successful: null,
              },
            ],
          },
          cooldown: {
            proposed_at: 1_000,
            claimed_at: 900,
            claim_expires_at: 1_800_900,
            split_cooldown_days: 7,
            split_claim_stale_sec: 1_800,
            last_split_attempt_at: null,
            split_failure_count: 0,
            last_split_error: null,
          },
        },
        reason: "split proposed",
      });

      await expect(reviewQueue.resolve(correction.id, "keep_both")).rejects.toMatchObject({
        name: "SemanticError",
        code: "REVIEW_QUEUE_RESOLUTION_INVALID",
      });
      await expect(reviewQueue.resolve(contradiction.id, "accept")).rejects.toMatchObject({
        name: "SemanticError",
        code: "REVIEW_QUEUE_RESOLUTION_INVALID",
      });
      await expect(reviewQueue.resolve(newInsight.id, "keep_both")).rejects.toMatchObject({
        name: "SemanticError",
        code: "REVIEW_QUEUE_RESOLUTION_INVALID",
      });
      await expect(reviewQueue.resolve(skillSplit.id, "dismiss")).rejects.toMatchObject({
        name: "SemanticError",
        code: "REVIEW_QUEUE_RESOLUTION_INVALID",
      });

      const rejectedCorrection = await reviewQueue.resolve(correction.id, "reject");
      const dismissedContradiction = await reviewQueue.resolve(contradiction.id, "dismiss");
      const dismissedInsight = await reviewQueue.resolve(newInsight.id, "dismiss");
      const rejectedSkillSplit = await reviewQueue.resolve(skillSplit.id, {
        decision: "reject",
        reason: "operator rejected",
      });

      expect(rejectedCorrection?.resolution).toBe("reject");
      expect(dismissedContradiction?.resolution).toBe("dismiss");
      expect(dismissedInsight?.resolution).toBe("dismiss");
      expect(rejectedSkillSplit).toMatchObject({
        resolution: "reject",
        refs: expect.objectContaining({
          review_resolution: expect.objectContaining({
            reason: "operator rejected",
          }),
        }),
      });
    } finally {
      db.close();
    }
  });
});
