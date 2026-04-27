import { describe, expect, it } from "vitest";

import { summarizeSemanticContext } from "../../cognition/deliberation/prompt/retrieval.js";
import { SemanticGraph } from "../../memory/semantic/index.js";
import { resolveSemanticContext, toRetrievedSemantic } from "../../retrieval/semantic-retrieval.js";
import { StreamReader } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import type { EpisodeId } from "../../util/ids.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
  type OfflineTestHarness,
} from "../test-support.js";
import { BeliefReviserProcess } from "./index.js";

type TestNode = Awaited<ReturnType<OfflineTestHarness["semanticNodeRepository"]["insert"]>>;

const EPISODE_ID = "ep_aaaaaaaaaaaaaaaa" as EpisodeId;
const STALE_EPISODE_ID = "ep_bbbbbbbbbbbbbbbb" as EpisodeId;
const FRESH_EPISODE_ID = "ep_cccccccccccccccc" as EpisodeId;

async function insertNode(
  harness: OfflineTestHarness,
  label: string,
  vector: number[],
  sourceEpisodeIds: EpisodeId[] = [EPISODE_ID],
) {
  return harness.semanticNodeRepository.insert(
    createSemanticNodeFixture(
      {
        label,
        description: `${label} description`,
        source_episode_ids: sourceEpisodeIds,
      },
      vector,
    ),
  );
}

function addEdge(
  harness: OfflineTestHarness,
  from: TestNode,
  to: TestNode,
  relation: "supports" | "causes" = "supports",
  evidenceEpisodeIds: EpisodeId[] = [EPISODE_ID],
) {
  return harness.semanticEdgeRepository.addEdge({
    from_node_id: from.id,
    to_node_id: to.id,
    relation,
    confidence: 0.7,
    evidence_episode_ids: evidenceEpisodeIds,
    created_at: 1_000,
    last_verified_at: 1_000,
  });
}

async function runBeliefReviser(
  harness: OfflineTestHarness,
  options: {
    maxReviewsPerEvent?: number;
    confidenceDropMultiplier?: number;
    confidenceFloor?: number;
  } = {},
) {
  const process = new BeliefReviserProcess({
    db: harness.db,
    ...options,
  });

  return process.run(harness.createContext(), {});
}

function openBeliefRevisionItems(harness: OfflineTestHarness) {
  return harness.reviewQueueRepository.list({
    kind: "belief_revision",
    openOnly: true,
  });
}

function invalidationEvents(harness: OfflineTestHarness) {
  return harness.db
    .prepare(
      `
        SELECT edge_id, processed_at
        FROM semantic_edge_invalidation_events
        ORDER BY id ASC
      `,
    )
    .all() as Array<{ edge_id: string; processed_at: number | null }>;
}

describe("belief reviser process", () => {
  it("enqueues a belief_revision review for an invalidated support target", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0]);
      const target = await insertNode(harness, "Target", [0, 1, 0, 0], [
        FRESH_EPISODE_ID,
      ]);
      const support = addEdge(harness, anchor, target);

      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);

      const [review] = openBeliefRevisionItems(harness);
      const updatedTarget = await harness.semanticNodeRepository.get(target.id);
      const confidenceDropEvent = new StreamReader({
        dataDir: harness.tempDir,
      })
        .tail(10)
        .find(
          (entry) =>
            entry.kind === "internal_event" &&
            typeof entry.content === "object" &&
            entry.content !== null &&
            "hook" in entry.content &&
            entry.content.hook === "belief_reviser_confidence_dropped",
        );

      expect(review).toEqual(
        expect.objectContaining({
          kind: "belief_revision",
          refs: expect.objectContaining({
            target_type: "semantic_node",
            target_id: target.id,
            invalidated_edge_id: support.id,
            dependency_path_edge_ids: [support.id],
            surviving_support_edge_ids: [],
            evidence_episode_ids: [FRESH_EPISODE_ID],
          }),
        }),
      );
      expect(updatedTarget?.confidence).toBeCloseTo(0.25);
      expect(confidenceDropEvent?.content).toMatchObject({
        hook: "belief_reviser_confidence_dropped",
        target_id: target.id,
        previous_confidence: 0.5,
        next_confidence: 0.25,
      });
      expect(invalidationEvents(harness)).toEqual([
        {
          edge_id: support.id,
          processed_at: 2_000,
        },
      ]);
    } finally {
      await harness.cleanup();
    }
  });

  it("does not drop confidence when a target still has surviving support", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0]);
      const survivingAnchor = await insertNode(harness, "Surviving anchor", [0, 0, 1, 0]);
      const target = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Supported target",
            description: "Supported target description",
            confidence: 0.8,
            source_episode_ids: [FRESH_EPISODE_ID],
          },
          [0, 1, 0, 0],
        ),
      );
      const invalidatedSupport = addEdge(harness, anchor, target);
      const survivingSupport = addEdge(harness, survivingAnchor, target);

      harness.semanticEdgeRepository.invalidateEdge(invalidatedSupport.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);

      const [review] = openBeliefRevisionItems(harness);
      const updatedTarget = await harness.semanticNodeRepository.get(target.id);

      expect(review?.refs).toEqual(
        expect.objectContaining({
          target_id: target.id,
          invalidated_edge_id: invalidatedSupport.id,
          surviving_support_edge_ids: [survivingSupport.id],
        }),
      );
      expect(updatedTarget?.confidence).toBeCloseTo(0.8);
    } finally {
      await harness.cleanup();
    }
  });

  it("does not drop confidence when a target has other valid incoming evidence edges", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const supportAnchor = await insertNode(harness, "Support anchor", [1, 0, 0, 0]);
      const causeAnchor = await insertNode(harness, "Cause anchor", [0, 0, 1, 0]);
      const target = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Causally grounded target",
            description: "Causally grounded target description",
            confidence: 0.8,
            source_episode_ids: [FRESH_EPISODE_ID],
          },
          [0, 1, 0, 0],
        ),
      );
      const support = addEdge(harness, supportAnchor, target);

      addEdge(harness, causeAnchor, target, "causes");
      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);

      expect(openBeliefRevisionItems(harness)).toHaveLength(1);
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.8);
    } finally {
      await harness.cleanup();
    }
  });

  it("enforces the configured confidence floor when dropping unsupported nodes", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0]);
      const target = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Near-floor target",
            description: "Near-floor target description",
            confidence: 0.08,
            source_episode_ids: [FRESH_EPISODE_ID],
          },
          [0, 1, 0, 0],
        ),
      );
      const support = addEdge(harness, anchor, target);

      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);

      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.05);
    } finally {
      await harness.cleanup();
    }
  });

  it("walks support descendants to depth 2 and not beyond", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const a = await insertNode(harness, "A", [1, 0, 0, 0]);
      const b = await insertNode(harness, "B", [0, 1, 0, 0]);
      const c = await insertNode(harness, "C", [0, 0, 1, 0]);
      const d = await insertNode(harness, "D", [0, 0, 0, 1]);
      const e = await insertNode(harness, "E", [1, 1, 0, 0]);
      const ab = addEdge(harness, a, b);
      const bc = addEdge(harness, b, c);
      const cd = addEdge(harness, c, d);

      addEdge(harness, d, e);
      harness.semanticEdgeRepository.invalidateEdge(ab.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);

      const reviews = openBeliefRevisionItems(harness);
      const targetIds = new Set(reviews.map((item) => item.refs.target_id));
      const cReview = reviews.find((item) => item.refs.target_id === c.id);
      const dReview = reviews.find((item) => item.refs.target_id === d.id);

      expect(targetIds).toEqual(new Set([b.id, c.id, d.id]));
      expect(cReview?.refs).toEqual(
        expect.objectContaining({
          dependency_path_edge_ids: [ab.id, bc.id],
        }),
      );
      expect(dReview?.refs).toEqual(
        expect.objectContaining({
          dependency_path_edge_ids: [ab.id, bc.id, cd.id],
        }),
      );
      expect(targetIds.has(e.id)).toBe(false);
    } finally {
      await harness.cleanup();
    }
  });

  it("filters invalidated edge evidence from belief revision review evidence", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0], [STALE_EPISODE_ID]);
      const target = await insertNode(harness, "Target", [0, 1, 0, 0], [
        STALE_EPISODE_ID,
        FRESH_EPISODE_ID,
      ]);
      const support = addEdge(harness, anchor, target, "supports", [STALE_EPISODE_ID]);

      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);

      const [review] = openBeliefRevisionItems(harness);

      expect(review?.refs).toEqual(
        expect.objectContaining({
          evidence_episode_ids: [FRESH_EPISODE_ID],
        }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("does not walk transitive supports for non-support dependencies", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const a = await insertNode(harness, "Cause source", [1, 0, 0, 0]);
      const b = await insertNode(harness, "Explicit dependent", [0, 1, 0, 0]);
      const c = await insertNode(harness, "Supported descendant", [0, 0, 1, 0]);
      const cause = addEdge(harness, a, b, "causes");

      addEdge(harness, b, c);
      harness.semanticBeliefDependencyRepository.addDependency({
        target_type: "semantic_node",
        target_id: b.id,
        source_edge_id: cause.id,
        dependency_kind: "derived_from",
      });
      harness.semanticEdgeRepository.invalidateEdge(cause.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);

      const targetIds = new Set(
        openBeliefRevisionItems(harness).map((item) => item.refs.target_id),
      );

      expect(targetIds).toEqual(new Set([b.id]));
      expect(targetIds.has(c.id)).toBe(false);
    } finally {
      await harness.cleanup();
    }
  });

  it("does not re-enqueue reviews for already processed events", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0]);
      const target = await insertNode(harness, "Target", [0, 1, 0, 0]);
      const support = addEdge(harness, anchor, target);

      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);
      await runBeliefReviser(harness);

      expect(openBeliefRevisionItems(harness)).toHaveLength(1);
      expect(invalidationEvents(harness)[0]?.processed_at).toBe(2_000);
    } finally {
      await harness.cleanup();
    }
  });

  it("does not double-drop confidence when a processed event is retried after review deletion", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0]);
      const target = await insertNode(harness, "Target", [0, 1, 0, 0]);
      const support = addEdge(harness, anchor, target);

      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);

      const [review] = openBeliefRevisionItems(harness);
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.25);

      if (review !== undefined) {
        harness.reviewQueueRepository.delete(review.id);
      }

      await runBeliefReviser(harness);

      expect(openBeliefRevisionItems(harness)).toHaveLength(0);
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.25);
    } finally {
      await harness.cleanup();
    }
  });

  it("dedupes open reviews by target and invalidated edge", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0]);
      const target = await insertNode(harness, "Target", [0, 1, 0, 0]);
      const support = addEdge(harness, anchor, target);

      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });
      harness.db
        .prepare(
          `
            INSERT INTO semantic_edge_invalidation_events (
              edge_id,
              valid_to,
              invalidated_at,
              processed_at
            ) VALUES (?, ?, ?, NULL)
          `,
        )
        .run(support.id, 2_000, 2_000);

      await runBeliefReviser(harness);

      expect(openBeliefRevisionItems(harness)).toHaveLength(1);
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.25);
      expect(invalidationEvents(harness)).toEqual([
        {
          edge_id: support.id,
          processed_at: 2_000,
        },
        {
          edge_id: support.id,
          processed_at: 2_000,
        },
      ]);
    } finally {
      await harness.cleanup();
    }
  });

  it("caps event fanout, logs it, and still marks the event processed", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0]);
      const target = await insertNode(harness, "Target", [0, 1, 0, 0]);
      const childOne = await insertNode(harness, "Child one", [0, 0, 1, 0]);
      const childTwo = await insertNode(harness, "Child two", [0, 0, 0, 1]);
      const childThree = await insertNode(harness, "Child three", [1, 1, 0, 0]);
      const support = addEdge(harness, anchor, target);

      addEdge(harness, target, childOne);
      addEdge(harness, target, childTwo);
      addEdge(harness, target, childThree);
      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness, {
        maxReviewsPerEvent: 2,
      });

      const entries = new StreamReader({
        dataDir: harness.tempDir,
      }).tail(10);
      const fanoutEvent = entries.find(
        (entry) =>
          entry.kind === "internal_event" &&
          typeof entry.content === "object" &&
          entry.content !== null &&
          "hook" in entry.content &&
          entry.content.hook === "belief_reviser_fanout_capped",
      );

      expect(openBeliefRevisionItems(harness)).toHaveLength(2);
      expect(invalidationEvents(harness)).toEqual([
        {
          edge_id: support.id,
          processed_at: 2_000,
        },
      ]);
      expect(fanoutEvent?.content).toMatchObject({
        hook: "belief_reviser_fanout_capped",
        invalidated_edge_id: support.id,
        review_cap: 2,
        planned_reviews: 2,
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("marks propagated belief revisions in semantic retrieval and prompt rendering", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      const episode = createEpisodeFixture({
        id: FRESH_EPISODE_ID,
        title: "Target support evidence",
        tags: ["target"],
      });
      await harness.episodicRepository.insert(episode);
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0], [FRESH_EPISODE_ID]);
      const target = await insertNode(harness, "Target belief", [0, 1, 0, 0], [
        FRESH_EPISODE_ID,
      ]);
      const support = addEdge(harness, anchor, target, "supports", [FRESH_EPISODE_ID]);

      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);

      const semanticGraph = new SemanticGraph({
        nodeRepository: harness.semanticNodeRepository,
        edgeRepository: harness.semanticEdgeRepository,
      });
      const retrieved = toRetrievedSemantic(
        await resolveSemanticContext(
          "Target belief",
          {
            graphWalkDepth: 1,
            maxGraphNodes: 4,
            underReviewMultiplier: 0.5,
          },
          {
            embeddingClient: harness.embeddingClient,
            episodicRepository: harness.episodicRepository,
            semanticNodeRepository: harness.semanticNodeRepository,
            semanticGraph,
            reviewQueueRepository: harness.reviewQueueRepository,
          },
        ),
      );
      const retrievedTarget = retrieved.matched_nodes.find((node) => node.id === target.id);
      const promptBlock = summarizeSemanticContext(retrieved, 1_000, 2_000);

      expect(retrievedTarget?.under_review).toMatchObject({
        reason: "Supporting semantic edge was invalidated; target needs re-evaluation",
      });
      expect(retrievedTarget?.retrieval_score).toBeCloseTo(
        (retrievedTarget?.base_retrieval_score ?? 0) * 0.5,
      );
      expect(promptBlock).toContain("[under re-evaluation:");
      expect(promptBlock).toContain("Target belief");
    } finally {
      await harness.cleanup();
    }
  });
});
