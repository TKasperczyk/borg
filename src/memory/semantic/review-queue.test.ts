import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { createSemanticNodeId, type EpisodeId } from "../../util/ids.js";
import { OpenQuestionsRepository, selfMigrations } from "../self/index.js";
import { enqueueOpenQuestionForReview } from "../self/review-open-question-hook.js";
import { semanticMigrations } from "./migrations.js";
import { ReviewQueueRepository } from "./review-queue.js";
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
      migrations: [...semanticMigrations, ...selfMigrations],
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
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock,
      semanticNodeRepository: nodeRepository,
      onEnqueue: (item) => enqueueOpenQuestionForReview(openQuestionsRepository, item),
    });
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

    const openItems = reviewQueue.getOpen();
    const openQuestions = openQuestionsRepository.list({ status: "open" });

    expect(openItems).toHaveLength(1);
    expect(openItems[0]?.kind).toBe("contradiction");
    expect(openItems[0]?.reason).toContain("conflicts_with_support_chain");
    expect(openQuestions[0]?.source).toBe("contradiction");
    expect(openQuestions[0]?.related_semantic_node_ids).toEqual([first.id, second.id]);

    const resolved = await reviewQueue.resolve(openItems[0]!.id, "invalidate");
    const updatedSecond = await nodeRepository.get(second.id);

    expect(resolved?.resolution).toBe("invalidate");
    expect(updatedSecond?.archived).toBe(true);
    expect(updatedSecond?.confidence).toBe(0);
  });

  it("logs hook failures without aborting edge insertion", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...semanticMigrations, ...selfMigrations],
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
          kind: "offline",
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
          kind: "offline",
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

  it("verifies accepted new insights, archives invalidated ones, and refreshes stale semantic nodes", async () => {
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
    const staleNode = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Stale claim",
          description: "Needs a verification pass.",
          source_episode_ids: [episode],
          confidence: 0.8,
          last_verified_at: 10,
        },
        [0, 1, 0, 0],
      ),
    );

    const newInsight = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      refs: {
        node_ids: [freshNode.id],
      },
      reason: "fresh insight should be reconsidered",
    });
    const stale = harness.reviewQueueRepository.enqueue({
      kind: "stale",
      refs: {
        target_type: "semantic_node",
        target_id: staleNode.id,
      },
      reason: "stale node needs refresh",
    });

    await harness.reviewQueueRepository.resolve(newInsight.id, "accept");
    await harness.reviewQueueRepository.resolve(stale.id, "accept");

    expect(await harness.semanticNodeRepository.get(freshNode.id)).toEqual(
      expect.objectContaining({
        archived: false,
        confidence: 0.7,
        last_verified_at: 8_000,
      }),
    );
    const invalidatedInsight = harness.reviewQueueRepository.enqueue({
      kind: "new_insight",
      refs: {
        node_ids: [freshNode.id],
      },
      reason: "fresh insight should be reconsidered again",
    });

    await harness.reviewQueueRepository.resolve(invalidatedInsight.id, "invalidate");

    expect((await harness.semanticNodeRepository.get(freshNode.id))?.archived).toBe(true);
    expect(await harness.semanticNodeRepository.get(staleNode.id)).toEqual(
      expect.objectContaining({
        last_verified_at: 8_000,
        confidence: 0.75,
      }),
    );
  });

  it("rejects accept on legacy under-specified repair rows and leaves them open", async () => {
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
    const legacyMisattribution = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs: {
        target_type: "episode",
        target_id: episode.id,
      },
      reason: "legacy row missing patch",
    });
    const legacyTemporalDrift = harness.reviewQueueRepository.enqueue({
      kind: "temporal_drift",
      refs: {
        target_type: "episode",
        target_id: episode.id,
      },
      reason: "legacy row missing corrected timestamps",
    });
    const legacyIdentityRepair = harness.reviewQueueRepository.enqueue({
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
      reason: "legacy row missing identity patch",
    });

    await expect(harness.reviewQueueRepository.resolve(legacyMisattribution.id, "accept")).rejects.toMatchObject(
      {
        name: "SemanticError",
        code: "REVIEW_QUEUE_REPAIR_REQUIRES_STRUCTURED_REFS",
      },
    );
    await expect(
      harness.reviewQueueRepository.resolve(legacyTemporalDrift.id, "accept"),
    ).rejects.toMatchObject({
      name: "SemanticError",
      code: "REVIEW_QUEUE_REPAIR_REQUIRES_STRUCTURED_REFS",
    });
    await expect(
      harness.reviewQueueRepository.resolve(legacyIdentityRepair.id, "accept"),
    ).rejects.toMatchObject({
      name: "SemanticError",
      code: "REVIEW_QUEUE_REPAIR_REQUIRES_STRUCTURED_REFS",
    });

    expect(harness.reviewQueueRepository.getOpen().map((item) => item.id)).toEqual(
      expect.arrayContaining([
        legacyMisattribution.id,
        legacyTemporalDrift.id,
        legacyIdentityRepair.id,
      ]),
    );
  });

  it("rejects incompatible review decisions and allows valid pairings", async () => {
    const db = openDatabase(":memory:", {
      migrations: [...semanticMigrations],
    });
    const reviewQueue = new ReviewQueueRepository({
      db,
      clock: new FixedClock(1_000),
    });

    try {
      const correction = reviewQueue.enqueue({
        kind: "correction",
        refs: {
          record_id: "val_aaaaaaaaaaaaaaaa",
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
        refs: {
          node_ids: ["semn_dddddddddddddddd"],
        },
        reason: "fresh reflector insight",
      });
      const stale = reviewQueue.enqueue({
        kind: "stale",
        refs: {
          node_id: "semn_cccccccccccccccc",
        },
        reason: "needs refresh",
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

      const rejectedCorrection = await reviewQueue.resolve(correction.id, "reject");
      const dismissedContradiction = await reviewQueue.resolve(contradiction.id, "dismiss");
      const acceptedInsight = await reviewQueue.resolve(newInsight.id, "accept");
      const acceptedStale = await reviewQueue.resolve(stale.id, "accept");

      expect(rejectedCorrection?.resolution).toBe("reject");
      expect(dismissedContradiction?.resolution).toBe("dismiss");
      expect(acceptedInsight?.resolution).toBe("accept");
      expect(acceptedStale?.resolution).toBe("accept");
    } finally {
      db.close();
    }
  });
});
