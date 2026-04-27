import { describe, expect, it } from "vitest";

import { summarizeSemanticContext } from "../../cognition/deliberation/prompt/retrieval.js";
import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import { SemanticGraph, type ReviewQueueItem, type SemanticEdge } from "../../memory/semantic/index.js";
import { resolveSemanticContext, toRetrievedSemantic } from "../../retrieval/semantic-retrieval.js";
import { StreamReader } from "../../stream/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import type { EntityId, EpisodeId } from "../../util/ids.js";
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
    regradeBatchSize?: number;
    maxEventsPerRun?: number;
    maxReviewsPerRun?: number;
    claimStaleSec?: number;
    maxParseFailures?: number;
    budget?: number;
    consecutiveParseFailureLimit?: number;
  } = {},
) {
  const process = new BeliefReviserProcess({
    db: harness.db,
    ...options,
  });

  return process.run(harness.createContext(), {});
}

function beliefRevisionResponse(input: Record<string, unknown>): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 5,
    output_tokens: 3,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_belief_revision",
        name: "EmitBeliefRevision",
        input,
      },
    ],
  };
}

function enqueueNodeBeliefRevision(
  harness: OfflineTestHarness,
  node: TestNode,
  overrides: Partial<Record<string, unknown>> = {},
): ReviewQueueItem {
  return harness.reviewQueueRepository.enqueue({
    kind: "belief_revision",
    refs: {
      target_type: "semantic_node",
      target_id: node.id,
      invalidated_edge_id: "seme_aaaaaaaaaaaaaaaa" as SemanticEdge["id"],
      dependency_path_edge_ids: ["seme_aaaaaaaaaaaaaaaa" as SemanticEdge["id"]],
      surviving_support_edge_ids: [],
      evidence_episode_ids: [],
      ...overrides,
    },
    reason: "Supporting semantic edge was invalidated; target needs re-evaluation",
  });
}

function openBeliefRevisionItems(harness: OfflineTestHarness) {
  return harness.reviewQueueRepository.list({
    kind: "belief_revision",
    openOnly: true,
  });
}

function beliefRevisionItems(harness: OfflineTestHarness) {
  return harness.reviewQueueRepository.list({
    kind: "belief_revision",
  });
}

type BeliefReviserInternals = {
  claimReview(
    ctx: ReturnType<OfflineTestHarness["createContext"]>,
    reviewId: ReviewQueueItem["id"],
  ): ReviewQueueItem | null;
  prepareNodeVectorSync(
    ctx: ReturnType<OfflineTestHarness["createContext"]>,
    item: ReviewQueueItem,
    verdict: Record<string, unknown>,
    expectedClaim: { run_id: string; claimed_at: number },
  ): { verdict: Record<string, unknown>; nodeSyncs: unknown[] };
  syncNodeToVectorStore(
    ctx: ReturnType<OfflineTestHarness["createContext"]>,
    sync: unknown,
  ): Promise<boolean>;
  applyVerdict(
    ctx: ReturnType<OfflineTestHarness["createContext"]>,
    item: ReviewQueueItem,
    verdict: Record<string, unknown>,
    expectedClaim: { run_id: string; claimed_at: number },
  ): { applied: boolean };
};

function claimRef(item: ReviewQueueItem): { run_id: string; claimed_at: number } {
  return item.refs.__borg_belief_revision_claim as { run_id: string; claimed_at: number };
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

  it("applies a keep verdict and restores an auto-dropped node confidence", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "keep",
          rationale: "The surviving target-local evidence still supports the claim.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0]);
      const target = await insertNode(harness, "Target", [0, 1, 0, 0], [FRESH_EPISODE_ID]);
      const support = addEdge(harness, anchor, target);

      harness.semanticEdgeRepository.invalidateEdge(support.id, {
        at: 2_000,
        by_process: "manual",
      });

      await runBeliefReviser(harness);
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.25);

      await runBeliefReviser(harness, { regradeBatchSize: 1 });

      const [review] = beliefRevisionItems(harness);
      expect(review).toMatchObject({
        resolved_at: 2_000,
        resolution: "keep",
      });
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.5);
      expect(llmClient.requests).toHaveLength(1);
    } finally {
      await harness.cleanup();
    }
  });

  it("applies a weaken verdict by reducing node confidence", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "weaken",
          rationale: "The claim is still plausible but less supported.",
          confidence_delta: -0.2,
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const target = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Weaken target",
            confidence: 0.8,
          },
          [0, 1, 0, 0],
        ),
      );
      enqueueNodeBeliefRevision(harness, target);

      await runBeliefReviser(harness, { regradeBatchSize: 1 });

      const [review] = beliefRevisionItems(harness);
      expect(review).toMatchObject({
        resolved_at: 2_000,
        resolution: "weaken",
      });
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.6);
    } finally {
      await harness.cleanup();
    }
  });

  it("clamps weaken verdict confidence at the configured floor", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "weaken",
          rationale: "The claim is almost unsupported.",
          confidence_delta: -0.2,
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const target = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Floor target",
            confidence: 0.1,
          },
          [0, 1, 0, 0],
        ),
      );
      enqueueNodeBeliefRevision(harness, target);

      await runBeliefReviser(harness, {
        regradeBatchSize: 1,
        confidenceFloor: 0.05,
      });

      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.05);
      expect(beliefRevisionItems(harness)[0]).toMatchObject({
        resolution: "weaken",
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("applies an archive_node verdict and removes the node from normal retrieval", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "archive_node",
          rationale: "The target claim has no remaining local support.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const target = await insertNode(harness, "Archived target", [0, 1, 0, 0]);
      enqueueNodeBeliefRevision(harness, target);

      await runBeliefReviser(harness, { regradeBatchSize: 1 });

      const updated = await harness.semanticNodeRepository.get(target.id);
      const matches = await harness.semanticNodeRepository.findByLabelOrAlias("Archived target");

      expect(beliefRevisionItems(harness)[0]).toMatchObject({
        resolution: "archive_node",
      });
      expect(updated?.archived).toBe(true);
      expect(matches).toHaveLength(0);
    } finally {
      await harness.cleanup();
    }
  });

  it("applies an invalidate_edge verdict to edge targets and writes a new invalidation event", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "invalidate_edge",
          rationale: "The dependent edge no longer follows from valid evidence.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const source = await insertNode(harness, "Source", [1, 0, 0, 0]);
      const target = await insertNode(harness, "Target", [0, 1, 0, 0]);
      const edge = addEdge(harness, source, target, "causes");

      harness.reviewQueueRepository.enqueue({
        kind: "belief_revision",
        refs: {
          target_type: "semantic_edge",
          target_id: edge.id,
          invalidated_edge_id: "seme_aaaaaaaaaaaaaaaa" as SemanticEdge["id"],
          dependency_path_edge_ids: ["seme_aaaaaaaaaaaaaaaa" as SemanticEdge["id"]],
          surviving_support_edge_ids: [],
          evidence_episode_ids: [],
        },
        reason: "Supporting semantic edge was invalidated; target needs re-evaluation",
      });

      await runBeliefReviser(harness, { regradeBatchSize: 1 });

      const updated = harness.semanticEdgeRepository.getEdge(edge.id);

      expect(updated?.valid_to).toBe(2_000);
      expect(beliefRevisionItems(harness)[0]).toMatchObject({
        resolution: "invalidate_edge",
      });
      expect(invalidationEvents(harness)).toEqual([
        {
          edge_id: edge.id,
          processed_at: null,
        },
      ]);
    } finally {
      await harness.cleanup();
    }
  });

  it("treats invalidate_edge for a node target as manual review", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "invalidate_edge",
          rationale: "This verdict is invalid for a node target.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const target = await insertNode(harness, "Node target", [0, 1, 0, 0]);
      enqueueNodeBeliefRevision(harness, target);

      await runBeliefReviser(harness, { regradeBatchSize: 1 });

      const [review] = openBeliefRevisionItems(harness);

      expect(review?.resolved_at).toBeNull();
      expect(review?.refs).toEqual(
        expect.objectContaining({
          belief_revision_escalated_at: 2_000,
          belief_revision_llm: expect.objectContaining({
            verdict: "manual_review",
            original_verdict: "invalidate_edge",
          }),
        }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("keeps manual_review verdicts open and escalated", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "manual_review",
          rationale: "The local evidence is ambiguous.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const target = await insertNode(harness, "Manual target", [0, 1, 0, 0]);
      enqueueNodeBeliefRevision(harness, target);

      await runBeliefReviser(harness, { regradeBatchSize: 1 });

      const [review] = openBeliefRevisionItems(harness);

      expect(review?.resolution).toBeNull();
      expect(review?.refs).toEqual(
        expect.objectContaining({
          belief_revision_escalated_at: 2_000,
          belief_revision_llm: expect.objectContaining({
            verdict: "manual_review",
          }),
        }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("filters LLM evidence episodes to the review's audience", async () => {
    const audienceA = "ent_aaaaaaaaaaaaaaaa" as EntityId;
    const audienceB = "ent_bbbbbbbbbbbbbbbb" as EntityId;
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "keep",
          rationale: "Audience-local evidence is sufficient.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const privateA = createEpisodeFixture({
        title: "Private A episode",
        narrative: "Only audience A should see this narrative.",
        audience_entity_id: audienceA,
        shared: false,
      });
      const privateB = createEpisodeFixture({
        title: "Private B episode",
        narrative: "Only audience B should see this narrative.",
        audience_entity_id: audienceB,
        shared: false,
      });
      await harness.episodicRepository.insert(privateA);
      await harness.episodicRepository.insert(privateB);
      const target = await insertNode(harness, "Audience target", [0, 1, 0, 0], [privateA.id]);

      enqueueNodeBeliefRevision(harness, target, {
        evidence_episode_ids: [privateB.id],
      });

      await runBeliefReviser(harness, { regradeBatchSize: 1 });

      const prompt = llmClient.requests[0]?.messages[0]?.content ?? "";

      expect(prompt).toContain("Private B episode");
      expect(prompt).toContain("Only audience B should see this narrative.");
      expect(prompt).not.toContain("Private A episode");
      expect(prompt).not.toContain("Only audience A should see this narrative.");
    } finally {
      await harness.cleanup();
    }
  });

  it("sanitizes target and edge episode ids in LLM input by review audience", async () => {
    const audienceA = "ent_aaaaaaaaaaaaaaaa" as EntityId;
    const audienceB = "ent_bbbbbbbbbbbbbbbb" as EntityId;
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "keep",
          rationale: "Audience-local evidence is sufficient.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const visible = createEpisodeFixture({
        title: "Audience A support",
        audience_entity_id: audienceA,
        shared: false,
      });
      const hidden = createEpisodeFixture({
        title: "Audience B private support",
        audience_entity_id: audienceB,
        shared: false,
      });
      await harness.episodicRepository.insert(visible);
      await harness.episodicRepository.insert(hidden);
      const anchor = await insertNode(harness, "Anchor", [1, 0, 0, 0], [visible.id, hidden.id]);
      const survivor = await insertNode(harness, "Survivor", [0, 0, 1, 0], [visible.id]);
      const target = await insertNode(harness, "Sanitized target", [0, 1, 0, 0], [
        visible.id,
        hidden.id,
      ]);
      const invalidated = addEdge(harness, anchor, target, "supports", [visible.id, hidden.id]);
      const surviving = addEdge(harness, survivor, target, "supports", [visible.id, hidden.id]);

      enqueueNodeBeliefRevision(harness, target, {
        invalidated_edge_id: invalidated.id,
        dependency_path_edge_ids: [invalidated.id],
        surviving_support_edge_ids: [surviving.id],
        evidence_episode_ids: [visible.id, hidden.id],
        audience_entity_id: audienceA,
      });

      await runBeliefReviser(harness, { regradeBatchSize: 1 });

      const prompt = llmClient.requests[0]?.messages[0]?.content ?? "";

      expect(prompt).toContain(visible.id);
      expect(prompt).not.toContain(hidden.id);
      expect(prompt).toContain("Audience A support");
      expect(prompt).not.toContain("Audience B private support");
    } finally {
      await harness.cleanup();
    }
  });

  it("caps propagated invalidation events per run and leaves excess events pending", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
    });

    try {
      for (let index = 0; index < 100; index += 1) {
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
          .run("seme_aaaaaaaaaaaaaaaa", 2_000, 2_000);
      }

      await runBeliefReviser(harness, {
        maxEventsPerRun: 32,
        maxReviewsPerRun: 128,
      });

      const events = invalidationEvents(harness);
      const entries = new StreamReader({
        dataDir: harness.tempDir,
      }).tail(10);
      const capEvent = entries.find(
        (entry) =>
          entry.kind === "internal_event" &&
          typeof entry.content === "object" &&
          entry.content !== null &&
          "hook" in entry.content &&
          entry.content.hook === "belief_reviser_run_capped",
      );

      expect(events.filter((event) => event.processed_at !== null)).toHaveLength(32);
      expect(events.filter((event) => event.processed_at === null)).toHaveLength(68);
      expect(capEvent?.content).toMatchObject({
        hook: "belief_reviser_run_capped",
        planned_events: 32,
        pending_event_count: 68,
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("reclaims stale belief-revision claims", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "keep",
          rationale: "The stale claim can be safely reclaimed.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(1_000_000),
      llmClient,
    });

    try {
      const target = await insertNode(harness, "Stale claim target", [0, 1, 0, 0]);

      enqueueNodeBeliefRevision(harness, target, {
        __borg_belief_revision_claim: {
          run_id: "old-run",
          claimed_at: 1,
        },
      });

      await runBeliefReviser(harness, {
        regradeBatchSize: 1,
        claimStaleSec: 600,
      });

      expect(llmClient.requests).toHaveLength(1);
      expect(beliefRevisionItems(harness)[0]).toMatchObject({
        resolution: "keep",
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("drops stale-owner cleanup after another run reclaims and resolves the review", async () => {
    let resolveFirstStarted!: () => void;
    let releaseFirst!: () => void;
    const firstResponseStarted = new Promise<void>((resolve) => {
      resolveFirstStarted = resolve;
    });
    const releaseFirstResponse = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    const llmClient = new FakeLLMClient({
      responses: [
        async () => {
          resolveFirstStarted();
          await releaseFirstResponse;
          return beliefRevisionResponse({
            verdict: "weaken",
            rationale: "Malformed stale owner response.",
          });
        },
        beliefRevisionResponse({
          verdict: "keep",
          rationale: "The reclaimed run keeps the belief.",
        }),
      ],
    });
    const clock = new ManualClock(1_000);
    const harness = await createOfflineTestHarness({
      clock,
      llmClient,
    });

    try {
      const target = await insertNode(harness, "Claim owner target", [0, 1, 0, 0]);
      enqueueNodeBeliefRevision(harness, target);

      const process = new BeliefReviserProcess({
        db: harness.db,
        claimStaleSec: 1,
      });
      const firstRun = process.run(harness.createContext(), {});
      await firstResponseStarted;

      clock.advance(2_000);
      await process.run(harness.createContext(), {});
      releaseFirst();
      await firstRun;

      const [review] = beliefRevisionItems(harness);
      const mismatchEvent = new StreamReader({
        dataDir: harness.tempDir,
      })
        .tail(20)
        .find(
          (entry) =>
            entry.kind === "internal_event" &&
            typeof entry.content === "object" &&
            entry.content !== null &&
            "hook" in entry.content &&
            entry.content.hook === "claim_ownership_mismatch_dropped",
        );

      expect(review).toMatchObject({
        resolution: "keep",
      });
      expect(review?.refs).not.toHaveProperty("belief_revision_failure_count");
      expect(mismatchEvent?.content).toMatchObject({
        hook: "claim_ownership_mismatch_dropped",
        action: "record_parse_failure",
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("drops stale-owner node sync after another run reclaims and resolves the review", async () => {
    const clock = new ManualClock(1_000);
    const harness = await createOfflineTestHarness({
      clock,
    });

    try {
      const target = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Deferred sync target",
            confidence: 0.8,
          },
          [0, 1, 0, 0],
        ),
      );
      const review = enqueueNodeBeliefRevision(harness, target);
      const process = new BeliefReviserProcess({
        db: harness.db,
        claimStaleSec: 1,
      });
      const internals = process as unknown as BeliefReviserInternals;

      const ctxA = harness.createContext();
      const claimedA = internals.claimReview(ctxA, review.id);
      expect(claimedA).not.toBeNull();
      const expectedA = claimRef(claimedA!);
      const preparedA = internals.prepareNodeVectorSync(
        ctxA,
        claimedA!,
        {
          verdict: "weaken",
          rationale: "The stale worker weakens the claim.",
          confidence_delta: -0.2,
        },
        expectedA,
      );
      expect(preparedA.nodeSyncs).toHaveLength(1);

      clock.advance(2_000);
      const ctxB = harness.createContext();
      const claimedB = internals.claimReview(ctxB, review.id);
      expect(claimedB).not.toBeNull();
      const expectedB = claimRef(claimedB!);
      const preparedB = internals.prepareNodeVectorSync(
        ctxB,
        claimedB!,
        {
          verdict: "archive_node",
          rationale: "The reclaimed worker archives the claim.",
        },
        expectedB,
      );
      expect(preparedB.nodeSyncs).toHaveLength(1);
      await internals.syncNodeToVectorStore(ctxB, preparedB.nodeSyncs[0]);
      expect(internals.applyVerdict(ctxB, claimedB!, preparedB.verdict, expectedB).applied).toBe(
        true,
      );

      await expect(internals.syncNodeToVectorStore(ctxA, preparedA.nodeSyncs[0])).resolves.toBe(
        false,
      );

      const updated = await harness.semanticNodeRepository.get(target.id);
      const mismatchEvent = new StreamReader({
        dataDir: harness.tempDir,
      })
        .tail(20)
        .find(
          (entry) =>
            entry.kind === "internal_event" &&
            typeof entry.content === "object" &&
            entry.content !== null &&
            "hook" in entry.content &&
            entry.content.hook === "claim_ownership_mismatch_dropped" &&
            "action" in entry.content &&
            entry.content.action === "sync_node",
        );

      expect(updated).toMatchObject({
        archived: true,
        confidence: 0.8,
      });
      expect(beliefRevisionItems(harness)[0]).toMatchObject({
        resolution: "archive_node",
      });
      expect(mismatchEvent).toBeDefined();
    } finally {
      await harness.cleanup();
    }
  });

  it("rejects positive confidence_delta values from the LLM", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "weaken",
          rationale: "Positive deltas are invalid for weakening.",
          confidence_delta: 0.2,
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const target = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Positive delta target",
            confidence: 0.8,
          },
          [0, 1, 0, 0],
        ),
      );
      enqueueNodeBeliefRevision(harness, target);

      const result = await runBeliefReviser(harness, { regradeBatchSize: 1 });
      const [review] = openBeliefRevisionItems(harness);

      expect(result.errors).toEqual([
        expect.objectContaining({
          code: "belief_reviser_regrade_failed",
        }),
      ]);
      expect(review?.refs).toEqual(
        expect.objectContaining({
          belief_revision_failure_count: 1,
        }),
      );
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.8);
    } finally {
      await harness.cleanup();
    }
  });

  it("applies weaken verdicts to semantic edge targets", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "weaken",
          rationale: "The edge is less reliable but still useful.",
          confidence_delta: -0.2,
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const source = await insertNode(harness, "Edge source", [1, 0, 0, 0]);
      const target = await insertNode(harness, "Edge target", [0, 1, 0, 0]);
      const edge = addEdge(harness, source, target, "causes");

      harness.reviewQueueRepository.enqueue({
        kind: "belief_revision",
        refs: {
          target_type: "semantic_edge",
          target_id: edge.id,
          invalidated_edge_id: "seme_aaaaaaaaaaaaaaaa" as SemanticEdge["id"],
          dependency_path_edge_ids: ["seme_aaaaaaaaaaaaaaaa" as SemanticEdge["id"]],
          surviving_support_edge_ids: [],
          evidence_episode_ids: [],
          audience_entity_id: null,
        },
        reason: "Supporting semantic edge was invalidated; target needs re-evaluation",
      });

      await runBeliefReviser(harness, { regradeBatchSize: 1 });

      expect(harness.semanticEdgeRepository.getEdge(edge.id)?.confidence).toBeCloseTo(0.5);
      expect(beliefRevisionItems(harness)[0]).toMatchObject({
        resolution: "weaken",
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("escalates malformed LLM output after the configured parse-failure limit", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "weaken",
          rationale: "Missing delta.",
        }),
        beliefRevisionResponse({
          verdict: "weaken",
          rationale: "Still missing delta.",
        }),
        beliefRevisionResponse({
          verdict: "weaken",
          rationale: "Still malformed.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const target = await insertNode(harness, "Escalated malformed target", [0, 1, 0, 0]);
      enqueueNodeBeliefRevision(harness, target);

      await runBeliefReviser(harness, {
        regradeBatchSize: 1,
        maxParseFailures: 3,
      });
      await runBeliefReviser(harness, {
        regradeBatchSize: 1,
        maxParseFailures: 3,
      });
      await runBeliefReviser(harness, {
        regradeBatchSize: 1,
        maxParseFailures: 3,
      });

      const [review] = openBeliefRevisionItems(harness);

      expect(llmClient.requests).toHaveLength(3);
      expect(review?.resolution).toBeNull();
      expect(review?.refs).toEqual(
        expect.objectContaining({
          belief_revision_failure_count: 3,
          belief_revision_escalated_at: 2_000,
          belief_revision_llm: expect.objectContaining({
            verdict: "manual_review",
            original_verdict: "parse_failure",
          }),
        }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("only applies one verdict when concurrent re-graders race for the same review", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        async () => {
          await new Promise((resolve) => setTimeout(resolve, 20));
          return beliefRevisionResponse({
            verdict: "weaken",
            rationale: "One re-grader should apply this once.",
            confidence_delta: -0.2,
          });
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const target = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Concurrent target",
            confidence: 0.8,
          },
          [0, 1, 0, 0],
        ),
      );
      enqueueNodeBeliefRevision(harness, target);

      await Promise.all([
        runBeliefReviser(harness, { regradeBatchSize: 1 }),
        runBeliefReviser(harness, { regradeBatchSize: 1 }),
      ]);

      expect(llmClient.requests).toHaveLength(1);
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.6);
      expect(beliefRevisionItems(harness)[0]).toMatchObject({
        resolution: "weaken",
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("leaves reviews open and unapplied when the LLM emits malformed schema", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        beliefRevisionResponse({
          verdict: "weaken",
          rationale: "Missing delta should fail schema validation.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(2_000),
      llmClient,
    });

    try {
      const target = await harness.semanticNodeRepository.insert(
        createSemanticNodeFixture(
          {
            label: "Malformed target",
            confidence: 0.8,
          },
          [0, 1, 0, 0],
        ),
      );
      enqueueNodeBeliefRevision(harness, target);

      const result = await runBeliefReviser(harness, { regradeBatchSize: 1 });
      const [review] = openBeliefRevisionItems(harness);
      const failureEvent = new StreamReader({
        dataDir: harness.tempDir,
      })
        .tail(10)
        .find(
          (entry) =>
            entry.kind === "internal_event" &&
            typeof entry.content === "object" &&
            entry.content !== null &&
            "hook" in entry.content &&
            entry.content.hook === "belief_reviser_regrade_failed",
        );

      expect(result.errors).toEqual([
        expect.objectContaining({
          code: "belief_reviser_regrade_failed",
        }),
      ]);
      expect(review?.resolution).toBeNull();
      expect(review?.refs).not.toHaveProperty("__borg_belief_revision_claim");
      expect((await harness.semanticNodeRepository.get(target.id))?.confidence).toBeCloseTo(0.8);
      expect(failureEvent?.content).toMatchObject({
        hook: "belief_reviser_regrade_failed",
        review_id: review?.id,
      });
    } finally {
      await harness.cleanup();
    }
  });
});
