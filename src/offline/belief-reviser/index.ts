import { z } from "zod";

import { entityIdSchema } from "../../memory/commitments/index.js";
import { episodeIdSchema, type Episode } from "../../memory/episodic/index.js";
import {
  semanticEdgeIdSchema,
  semanticNodeIdSchema,
  type ReviewQueueItem,
  type ReviewResolution,
  type ReviewQueueInsertInput,
  type SemanticEdge,
  type SemanticNode,
} from "../../memory/semantic/index.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import type { EntityId } from "../../util/ids.js";
import { serializeJsonValue } from "../../util/json-value.js";

import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessRunOptions,
  OfflineResult,
} from "../types.js";
import {
  BeliefRevisionParseError,
  evaluateBeliefRevision,
  inferBeliefRevisionAudience,
  type BeliefRevisionLlmInput,
  type BeliefRevisionVerdict,
  visibleBeliefRevisionEpisodes,
} from "./llm-reviewer.js";

const DEFAULT_MAX_REVIEWS_PER_EVENT = 64;
const DEFAULT_CONFIDENCE_DROP_MULTIPLIER = 0.5;
const DEFAULT_CONFIDENCE_FLOOR = 0.05;
const DEFAULT_REGRADE_BATCH_SIZE = 10;
const DEFAULT_MAX_EVENTS_PER_RUN = 32;
const DEFAULT_MAX_REVIEWS_PER_RUN = 128;
const DEFAULT_CLAIM_STALE_SEC = 600;
const DEFAULT_MAX_PARSE_FAILURES = 3;
const DEFAULT_BUDGET = 20;
const DEFAULT_CONSECUTIVE_PARSE_FAILURE_LIMIT = 5;
const MAX_SUPPORT_DESCENDANT_HOPS = 2;
const BELIEF_REVISION_REASON =
  "Supporting semantic edge was invalidated; target needs re-evaluation";
const CLAIM_REF_KEY = "__borg_belief_revision_claim";
const APPLYING_REF_KEY = "__borg_belief_revision_applying";
const AUTO_DROP_REF_KEY = "auto_confidence_drop";
const LLM_REVIEW_REF_KEY = "belief_revision_llm";
const ESCALATED_AT_REF_KEY = "belief_revision_escalated_at";
const FAILURE_COUNT_REF_KEY = "belief_revision_failure_count";
const LAST_ERROR_REF_KEY = "belief_revision_last_error";

const invalidationEventSchema = z.object({
  id: z.number().int().positive(),
  edge_id: semanticEdgeIdSchema,
  valid_to: z.number().finite(),
  invalidated_at: z.number().finite(),
});

const beliefRevisionTargetSchema = z.discriminatedUnion("target_type", [
  z.object({
    target_type: z.literal("semantic_node"),
    target_id: semanticNodeIdSchema,
  }),
  z.object({
    target_type: z.literal("semantic_edge"),
    target_id: semanticEdgeIdSchema,
  }),
]);

const beliefRevisionReviewSchema = beliefRevisionTargetSchema.and(
  z.object({
    invalidated_edge_id: semanticEdgeIdSchema,
    dependency_path_edge_ids: z.array(semanticEdgeIdSchema).min(1),
    surviving_support_edge_ids: z.array(semanticEdgeIdSchema),
    evidence_episode_ids: z.array(episodeIdSchema),
    audience_entity_id: entityIdSchema.nullable().default(null),
  }),
);

const beliefReviserPlanItemSchema = z.object({
  event_id: z.number().int().positive(),
  invalidated_edge_id: semanticEdgeIdSchema,
  reviews: z.array(beliefRevisionReviewSchema),
  fanout_capped: z.boolean().default(false),
  review_cap: z.number().int().positive().default(DEFAULT_MAX_REVIEWS_PER_EVENT),
});

const beliefRevisionRegradeItemSchema = z.object({
  review_id: z.number().int().positive(),
});

const autoConfidenceDropSchema = z
  .object({
    previous_confidence: z.number().min(0).max(1),
    next_confidence: z.number().min(0).max(1),
    applied_at: z.number().finite(),
  })
  .strict();

const beliefRevisionClaimSchema = z
  .object({
    run_id: z.string().min(1),
    claimed_at: z.number().finite(),
  })
  .strict();

const beliefRevisionApplyingSchema = z
  .object({
    verdict: z.enum(["keep", "weaken", "archive_node"]),
    target_id: semanticNodeIdSchema,
    confidence: z.number().min(0).max(1).optional(),
    archived: z.boolean().optional(),
  })
  .strict();

export const beliefReviserPlanSchema = z.object({
  process: z.literal("belief-reviser"),
  items: z.array(beliefReviserPlanItemSchema),
  regrade_items: z.array(beliefRevisionRegradeItemSchema).default([]),
  run_capped: z.boolean().default(false),
  event_cap: z.number().int().positive().default(DEFAULT_MAX_EVENTS_PER_RUN),
  review_run_cap: z.number().int().positive().default(DEFAULT_MAX_REVIEWS_PER_RUN),
  pending_event_count: z.number().int().nonnegative().default(0),
  errors: z
    .array(
      z.object({
        process: z.literal("belief-reviser"),
        message: z.string(),
        code: z.string().optional(),
      }),
    )
    .default([]),
  tokens_used: z.number().int().nonnegative(),
  budget_exhausted: z.boolean().default(false),
});

export type BeliefReviserPlan = z.infer<typeof beliefReviserPlanSchema>;
export type BeliefReviserProcessOptions = {
  db: SqliteDatabase;
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
  llmTimeoutMs?: number;
};

type InvalidationEvent = z.infer<typeof invalidationEventSchema>;
type BeliefRevisionReview = z.infer<typeof beliefRevisionReviewSchema>;
type ReviewCollector = {
  reviewsByKey: Map<string, BeliefRevisionReview>;
  maxReviews: number;
  fanoutCapped: boolean;
};
type AddReviewResult = "added" | "duplicate" | "missing" | "capped";
type BuildReviewsResult = {
  reviews: BeliefRevisionReview[];
  fanoutCapped: boolean;
};
type AutoConfidenceDrop = {
  previous_confidence: number;
  next_confidence: number;
  applied_at: number;
};
type ConfidenceDrop = {
  targetId: SemanticNode["id"];
  previousConfidence: number;
  nextConfidence: number;
};
type ApplyEventResult = {
  enqueued: BeliefRevisionReview[];
  confidenceDrops: ConfidenceDrop[];
  processed: boolean;
};
type NodeVectorSync =
  | {
      kind: "confidence";
      targetId: SemanticNode["id"];
      confidence: number;
    }
  | {
      kind: "archive";
      targetId: SemanticNode["id"];
    };
type PreparedVerdict = {
  verdict: BeliefRevisionVerdict;
  nodeSyncs: NodeVectorSync[];
};
type ApplyRegradeResult = {
  applied: boolean;
  resolution: ReviewResolution | "manual_review";
  nodeSyncs: NodeVectorSync[];
  change: OfflineChange | null;
};

function uniqueEpisodeIds(ids: readonly z.infer<typeof episodeIdSchema>[]) {
  return [...new Set(ids)];
}

function uniqueEdgeIds(ids: readonly z.infer<typeof semanticEdgeIdSchema>[]) {
  return [...new Set(ids)];
}

function collectEpisodeIds(value: unknown, collector = new Set<Episode["id"]>()): Set<Episode["id"]> {
  if (typeof value === "string") {
    const parsed = episodeIdSchema.safeParse(value);

    if (parsed.success) {
      collector.add(parsed.data);
    }

    return collector;
  }

  if (Array.isArray(value)) {
    for (const entry of value) {
      collectEpisodeIds(entry, collector);
    }

    return collector;
  }

  if (value !== null && typeof value === "object") {
    for (const entry of Object.values(value)) {
      collectEpisodeIds(entry, collector);
    }
  }

  return collector;
}

async function inferAudienceForEpisodeIds(
  ctx: OfflineContext,
  episodeIds: readonly Episode["id"][],
): Promise<EntityId | null> {
  const episodes = (await ctx.episodicRepository.getMany(episodeIds)).filter(
    (episode): episode is Episode => episode !== null,
  );

  return inferBeliefRevisionAudience(episodes);
}

async function visibleEpisodeIdsForAudience(
  ctx: OfflineContext,
  episodeIds: readonly Episode["id"][],
  audienceEntityId: EntityId | null,
): Promise<Episode["id"][]> {
  const episodes = (await ctx.episodicRepository.getMany([...new Set(episodeIds)])).filter(
    (episode): episode is Episode => episode !== null,
  );

  return visibleBeliefRevisionEpisodes(episodes, audienceEntityId).map((episode) => episode.id);
}

function normalizeMaxReviewsPerEvent(value: number | undefined): number {
  if (value === undefined) {
    return DEFAULT_MAX_REVIEWS_PER_EVENT;
  }

  if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
    throw new TypeError("maxReviewsPerEvent must be a positive integer");
  }

  return value;
}

function normalizeRegradeBatchSize(value: number | undefined): number {
  if (value === undefined) {
    return DEFAULT_REGRADE_BATCH_SIZE;
  }

  if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
    throw new TypeError("regradeBatchSize must be a positive integer");
  }

  return value;
}

function normalizePositiveIntegerOption(
  value: number | undefined,
  fallback: number,
  label: string,
): number {
  if (value === undefined) {
    return fallback;
  }

  if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
    throw new TypeError(`${label} must be a positive integer`);
  }

  return value;
}

function normalizePositiveNumberOption(
  value: number | undefined,
  fallback: number,
  label: string,
): number {
  if (value === undefined) {
    return fallback;
  }

  if (!Number.isFinite(value) || value <= 0) {
    throw new TypeError(`${label} must be a positive number`);
  }

  return value;
}

function normalizeUnitIntervalOption(
  value: number | undefined,
  fallback: number,
  label: string,
): number {
  if (value === undefined) {
    return fallback;
  }

  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new TypeError(`${label} must be between 0 and 1`);
  }

  return value;
}

function evidenceEpisodeIdsWithoutStaleEvidence(
  ids: readonly z.infer<typeof episodeIdSchema>[],
  staleEvidenceEpisodeIds: readonly z.infer<typeof episodeIdSchema>[],
) {
  const staleIds = new Set(staleEvidenceEpisodeIds);

  return uniqueEpisodeIds(ids).filter((id) => !staleIds.has(id));
}

function reviewKey(
  review: Pick<BeliefRevisionReview, "target_type" | "target_id" | "invalidated_edge_id">,
) {
  return JSON.stringify([review.target_type, review.target_id, review.invalidated_edge_id]);
}

function buildChange(review: BeliefRevisionReview, eventId: number): OfflineChange {
  return {
    process: "belief-reviser",
    action: "enqueue_belief_revision",
    targets: {
      event_id: eventId,
      target_type: review.target_type,
      target_id: review.target_id,
      invalidated_edge_id: review.invalidated_edge_id,
    },
    preview: {
      dependency_path_edge_ids: review.dependency_path_edge_ids,
      surviving_support_edge_ids: review.surviving_support_edge_ids,
      evidence_episode_ids: review.evidence_episode_ids,
    },
  };
}

function buildConfidenceDropChange(drop: ConfidenceDrop, eventId: number): OfflineChange {
  return {
    process: "belief-reviser",
    action: "drop_semantic_node_confidence",
    targets: {
      event_id: eventId,
      target_type: "semantic_node",
      target_id: drop.targetId,
    },
    preview: {
      previous_confidence: drop.previousConfidence,
      next_confidence: drop.nextConfidence,
    },
  };
}

function buildRegradeChange(input: {
  reviewId: number;
  verdict: BeliefRevisionVerdict["verdict"];
  targetType: BeliefRevisionReview["target_type"];
  targetId: BeliefRevisionReview["target_id"];
}): OfflineChange {
  return {
    process: "belief-reviser",
    action: "regrade_belief_revision",
    targets: {
      review_id: input.reviewId,
      target_type: input.targetType,
      target_id: input.targetId,
    },
    preview: {
      verdict: input.verdict,
    },
  };
}

function listUnprocessedEvents(db: SqliteDatabase): InvalidationEvent[] {
  const rows = db
    .prepare(
      `
        SELECT id, edge_id, valid_to, invalidated_at
        FROM semantic_edge_invalidation_events
        WHERE processed_at IS NULL
        ORDER BY id ASC
      `,
    )
    .all() as Record<string, unknown>[];

  return rows.map((row) =>
    invalidationEventSchema.parse({
      id: Number(row.id),
      edge_id: row.edge_id,
      valid_to: Number(row.valid_to),
      invalidated_at: Number(row.invalidated_at),
    }),
  );
}

function collectOpenBeliefRevisionKeys(ctx: OfflineContext): Set<string> {
  const keys = new Set<string>();

  for (const item of ctx.reviewQueueRepository.list({
    kind: "belief_revision",
    openOnly: true,
  })) {
    const parsed = beliefRevisionReviewSchema.safeParse(item.refs);

    if (parsed.success) {
      keys.add(reviewKey(parsed.data));
    }
  }

  return keys;
}

function hasFreshBeliefRevisionClaim(item: ReviewQueueItem, staleBefore: number): boolean {
  const parsed = beliefRevisionClaimSchema.safeParse(item.refs[CLAIM_REF_KEY]);

  return parsed.success && parsed.data.claimed_at >= staleBefore;
}

function isEscalatedBeliefRevision(item: ReviewQueueItem): boolean {
  return typeof item.refs[ESCALATED_AT_REF_KEY] === "number";
}

function listPendingRegradeItems(
  ctx: OfflineContext,
  batchSize: number,
  claimStaleSec: number,
): z.infer<typeof beliefRevisionRegradeItemSchema>[] {
  const staleBefore = ctx.clock.now() - claimStaleSec * 1_000;

  return ctx.reviewQueueRepository
    .list({
      kind: "belief_revision",
      openOnly: true,
    })
    .filter((item) => !hasFreshBeliefRevisionClaim(item, staleBefore) && !isEscalatedBeliefRevision(item))
    .slice(0, batchSize)
    .map((item) => ({
      review_id: item.id,
    }));
}

function survivingSupportEdgeIds(
  ctx: OfflineContext,
  targetId: SemanticNode["id"],
  dependencyPathEdgeIds: readonly SemanticEdge["id"][],
): SemanticEdge["id"][] {
  const pathEdges = new Set<string>(dependencyPathEdgeIds);

  return ctx.semanticEdgeRepository
    .listEdges({
      toId: targetId,
      relation: "supports",
    })
    .filter((edge) => !pathEdges.has(edge.id))
    .map((edge) => edge.id);
}

async function buildNodeReview(input: {
  ctx: OfflineContext;
  targetId: SemanticNode["id"];
  invalidatedEdgeId: SemanticEdge["id"];
  dependencyPathEdgeIds: readonly SemanticEdge["id"][];
  staleEvidenceEpisodeIds: readonly z.infer<typeof episodeIdSchema>[];
}): Promise<BeliefRevisionReview | null> {
  const node = await input.ctx.semanticNodeRepository.get(input.targetId);

  if (node === null) {
    return null;
  }
  const evidenceEpisodeIds = evidenceEpisodeIdsWithoutStaleEvidence(
    node.source_episode_ids,
    input.staleEvidenceEpisodeIds,
  );

  return beliefRevisionReviewSchema.parse({
    target_type: "semantic_node",
    target_id: node.id,
    invalidated_edge_id: input.invalidatedEdgeId,
    dependency_path_edge_ids: uniqueEdgeIds(input.dependencyPathEdgeIds),
    surviving_support_edge_ids: survivingSupportEdgeIds(
      input.ctx,
      node.id,
      input.dependencyPathEdgeIds,
    ),
    evidence_episode_ids: evidenceEpisodeIds,
    audience_entity_id: await inferAudienceForEpisodeIds(input.ctx, evidenceEpisodeIds),
  });
}

async function buildEdgeReview(input: {
  ctx: OfflineContext;
  targetId: SemanticEdge["id"];
  invalidatedEdgeId: SemanticEdge["id"];
  dependencyPathEdgeIds: readonly SemanticEdge["id"][];
  staleEvidenceEpisodeIds: readonly z.infer<typeof episodeIdSchema>[];
}): Promise<BeliefRevisionReview | null> {
  const edge = input.ctx.semanticEdgeRepository.getEdge(input.targetId);

  if (edge === null) {
    return null;
  }
  const evidenceEpisodeIds = evidenceEpisodeIdsWithoutStaleEvidence(
    edge.evidence_episode_ids,
    input.staleEvidenceEpisodeIds,
  );

  return beliefRevisionReviewSchema.parse({
    target_type: "semantic_edge",
    target_id: edge.id,
    invalidated_edge_id: input.invalidatedEdgeId,
    dependency_path_edge_ids: uniqueEdgeIds(input.dependencyPathEdgeIds),
    surviving_support_edge_ids: [],
    evidence_episode_ids: evidenceEpisodeIds,
    audience_entity_id: await inferAudienceForEpisodeIds(input.ctx, evidenceEpisodeIds),
  });
}

async function addReviewIfNew(
  collector: ReviewCollector,
  review: Promise<BeliefRevisionReview | null> | BeliefRevisionReview | null,
): Promise<AddReviewResult> {
  const resolved = await review;

  if (resolved === null) {
    return "missing";
  }

  const key = reviewKey(resolved);

  if (collector.reviewsByKey.has(key)) {
    return "duplicate";
  }

  if (collector.reviewsByKey.size >= collector.maxReviews) {
    collector.fanoutCapped = true;
    return "capped";
  }

  collector.reviewsByKey.set(key, resolved);
  return "added";
}

async function walkSupportDescendants(input: {
  ctx: OfflineContext;
  collector: ReviewCollector;
  invalidatedEdgeId: SemanticEdge["id"];
  fromNodeId: SemanticNode["id"];
  dependencyPathEdgeIds: readonly SemanticEdge["id"][];
  supportHopDepth: number;
  staleEvidenceEpisodeIds: readonly z.infer<typeof episodeIdSchema>[];
}): Promise<void> {
  if (input.supportHopDepth >= MAX_SUPPORT_DESCENDANT_HOPS) {
    return;
  }

  const supportEdges = input.ctx.semanticEdgeRepository.listEdges({
    fromId: input.fromNodeId,
    relation: "supports",
  });

  if (input.collector.reviewsByKey.size >= input.collector.maxReviews) {
    input.collector.fanoutCapped = input.collector.fanoutCapped || supportEdges.length > 0;
    return;
  }

  for (const [index, supportEdge] of supportEdges.entries()) {
    const nextPath = [...input.dependencyPathEdgeIds, supportEdge.id];
    const addResult = await addReviewIfNew(
      input.collector,
      buildNodeReview({
        ctx: input.ctx,
        targetId: supportEdge.to_node_id,
        invalidatedEdgeId: input.invalidatedEdgeId,
        dependencyPathEdgeIds: nextPath,
        staleEvidenceEpisodeIds: input.staleEvidenceEpisodeIds,
      }),
    );

    if (addResult === "capped") {
      return;
    }

    await walkSupportDescendants({
      ctx: input.ctx,
      collector: input.collector,
      invalidatedEdgeId: input.invalidatedEdgeId,
      fromNodeId: supportEdge.to_node_id,
      dependencyPathEdgeIds: nextPath,
      supportHopDepth: input.supportHopDepth + 1,
      staleEvidenceEpisodeIds: input.staleEvidenceEpisodeIds,
    });

    if (
      input.collector.reviewsByKey.size >= input.collector.maxReviews &&
      index < supportEdges.length - 1
    ) {
      input.collector.fanoutCapped = true;
      return;
    }
  }
}

async function buildReviewsForEvent(
  ctx: OfflineContext,
  event: InvalidationEvent,
  maxReviewsPerEvent: number,
): Promise<BuildReviewsResult> {
  const collector: ReviewCollector = {
    reviewsByKey: new Map<string, BeliefRevisionReview>(),
    maxReviews: maxReviewsPerEvent,
    fanoutCapped: false,
  };
  const invalidatedEdge = ctx.semanticEdgeRepository.getEdge(event.edge_id);
  const staleEvidenceEpisodeIds = invalidatedEdge?.evidence_episode_ids ?? [];
  const dependencies = ctx.semanticBeliefDependencyRepository.listBySourceEdge(event.edge_id);

  for (const [index, dependency] of dependencies.entries()) {
    const dependencyPathEdgeIds = [event.edge_id];

    if (dependency.target_type === "semantic_edge") {
      const addResult = await addReviewIfNew(
        collector,
        buildEdgeReview({
          ctx,
          targetId: dependency.target_id,
          invalidatedEdgeId: event.edge_id,
          dependencyPathEdgeIds,
          staleEvidenceEpisodeIds,
        }),
      );
      if (addResult === "capped") {
        break;
      }
      continue;
    }

    const addResult = await addReviewIfNew(
      collector,
      buildNodeReview({
        ctx,
        targetId: dependency.target_id,
        invalidatedEdgeId: event.edge_id,
        dependencyPathEdgeIds,
        staleEvidenceEpisodeIds,
      }),
    );

    if (addResult === "capped") {
      break;
    }

    if (dependency.dependency_kind !== "supports") {
      continue;
    }

    await walkSupportDescendants({
      ctx,
      collector,
      invalidatedEdgeId: event.edge_id,
      fromNodeId: dependency.target_id,
      dependencyPathEdgeIds,
      supportHopDepth: 0,
      staleEvidenceEpisodeIds,
    });

    if (collector.reviewsByKey.size >= collector.maxReviews && index < dependencies.length - 1) {
      collector.fanoutCapped = true;
      break;
    }
  }

  return {
    reviews: [...collector.reviewsByKey.values()],
    fanoutCapped: collector.fanoutCapped,
  };
}

function refsForReview(
  review: BeliefRevisionReview,
  autoConfidenceDrop?: AutoConfidenceDrop,
): ReviewQueueInsertInput["refs"] {
  const refs: ReviewQueueInsertInput["refs"] = {
    target_type: review.target_type,
    target_id: review.target_id,
    invalidated_edge_id: review.invalidated_edge_id,
    dependency_path_edge_ids: review.dependency_path_edge_ids,
    surviving_support_edge_ids: review.surviving_support_edge_ids,
    evidence_episode_ids: review.evidence_episode_ids,
    audience_entity_id: review.audience_entity_id,
  };

  if (autoConfidenceDrop !== undefined) {
    refs[AUTO_DROP_REF_KEY] = autoConfidenceDrop;
  }

  return refs;
}

export class BeliefReviserProcess implements OfflineProcess<BeliefReviserPlan> {
  readonly name = "belief-reviser" as const;
  private readonly maxReviewsPerEvent: number;
  private readonly confidenceDropMultiplier: number;
  private readonly confidenceFloor: number;
  private readonly regradeBatchSize: number;
  private readonly maxEventsPerRun: number;
  private readonly maxReviewsPerRun: number;
  private readonly claimStaleSec: number;
  private readonly maxParseFailures: number;
  private readonly budget: number;
  private readonly consecutiveParseFailureLimit: number;

  constructor(private readonly options: BeliefReviserProcessOptions) {
    this.maxReviewsPerEvent = normalizeMaxReviewsPerEvent(options.maxReviewsPerEvent);
    this.regradeBatchSize = normalizeRegradeBatchSize(options.regradeBatchSize);
    this.maxEventsPerRun = normalizePositiveIntegerOption(
      options.maxEventsPerRun,
      DEFAULT_MAX_EVENTS_PER_RUN,
      "maxEventsPerRun",
    );
    this.maxReviewsPerRun = normalizePositiveIntegerOption(
      options.maxReviewsPerRun,
      DEFAULT_MAX_REVIEWS_PER_RUN,
      "maxReviewsPerRun",
    );
    this.claimStaleSec = normalizePositiveNumberOption(
      options.claimStaleSec,
      DEFAULT_CLAIM_STALE_SEC,
      "claimStaleSec",
    );
    this.maxParseFailures = normalizePositiveIntegerOption(
      options.maxParseFailures,
      DEFAULT_MAX_PARSE_FAILURES,
      "maxParseFailures",
    );
    this.budget = normalizePositiveIntegerOption(options.budget, DEFAULT_BUDGET, "budget");
    this.consecutiveParseFailureLimit = normalizePositiveIntegerOption(
      options.consecutiveParseFailureLimit,
      DEFAULT_CONSECUTIVE_PARSE_FAILURE_LIMIT,
      "consecutiveParseFailureLimit",
    );
    this.confidenceDropMultiplier = normalizeUnitIntervalOption(
      options.confidenceDropMultiplier,
      DEFAULT_CONFIDENCE_DROP_MULTIPLIER,
      "confidenceDropMultiplier",
    );
    this.confidenceFloor = normalizeUnitIntervalOption(
      options.confidenceFloor,
      DEFAULT_CONFIDENCE_FLOOR,
      "confidenceFloor",
    );
  }

  private hasValidNonSupportIncomingEdges(
    ctx: OfflineContext,
    review: BeliefRevisionReview,
  ): boolean {
    if (review.target_type !== "semantic_node") {
      return false;
    }

    const dependencyPathEdgeIds = new Set(review.dependency_path_edge_ids);

    return ctx.semanticEdgeRepository
      .listEdges({
        toId: review.target_id,
      })
      .some((edge) => edge.relation !== "supports" && !dependencyPathEdgeIds.has(edge.id));
  }

  private maybeDropUnsupportedNodeConfidenceInSqlite(
    ctx: OfflineContext,
    review: BeliefRevisionReview,
  ): ConfidenceDrop | null {
    if (review.target_type !== "semantic_node" || review.surviving_support_edge_ids.length > 0) {
      return null;
    }

    if (this.hasValidNonSupportIncomingEdges(ctx, review)) {
      return null;
    }

    const current = this.options.db
      .prepare("SELECT confidence FROM semantic_nodes WHERE id = ?")
      .get(review.target_id) as { confidence: number } | undefined;

    if (current === undefined) {
      return null;
    }

    const currentConfidence = Number(current.confidence);
    const nextConfidence = Math.max(
      this.confidenceFloor,
      currentConfidence * this.confidenceDropMultiplier,
    );

    if (nextConfidence >= currentConfidence) {
      return null;
    }

    this.options.db
      .prepare("UPDATE semantic_nodes SET confidence = ?, updated_at = ? WHERE id = ?")
      .run(nextConfidence, ctx.clock.now(), review.target_id);

    return {
      targetId: review.target_id,
      previousConfidence: currentConfidence,
      nextConfidence,
    };
  }

  private async syncConfidenceDropToVectorStore(
    ctx: OfflineContext,
    drop: ConfidenceDrop,
  ): Promise<void> {
    const current = await ctx.semanticNodeRepository.get(drop.targetId);

    if (current === null || current.confidence === drop.nextConfidence) {
      return;
    }

    await ctx.semanticNodeRepository.update(drop.targetId, {
      confidence: drop.nextConfidence,
    });
  }

  private async syncNodeToVectorStore(ctx: OfflineContext, sync: NodeVectorSync): Promise<void> {
    const current = await ctx.semanticNodeRepository.get(sync.targetId);

    if (current === null) {
      return;
    }

    if (sync.kind === "confidence") {
      if (current.confidence === sync.confidence) {
        return;
      }

      await ctx.semanticNodeRepository.update(sync.targetId, {
        confidence: sync.confidence,
      });
      return;
    }

    if (current.archived) {
      return;
    }

    await ctx.semanticNodeRepository.update(sync.targetId, {
      archived: true,
    });
  }

  private claimReview(ctx: OfflineContext, reviewId: number): ReviewQueueItem | null {
    const claim = {
      run_id: String(ctx.runId),
      claimed_at: ctx.clock.now(),
    };
    const staleBefore = ctx.clock.now() - this.claimStaleSec * 1_000;
    const result = this.options.db
      .prepare(
        `
          UPDATE review_queue
          SET refs = json_set(refs, '$.${CLAIM_REF_KEY}', json(?))
          WHERE id = ?
            AND kind = 'belief_revision'
            AND resolved_at IS NULL
            AND json_extract(refs, '$.${ESCALATED_AT_REF_KEY}') IS NULL
            AND (
              json_extract(refs, '$.${CLAIM_REF_KEY}.claimed_at') IS NULL
              OR json_extract(refs, '$.${CLAIM_REF_KEY}.claimed_at') < ?
            )
        `,
      )
      .run(serializeJsonValue(claim), reviewId, staleBefore);

    if (result.changes === 0) {
      return null;
    }

    return ctx.reviewQueueRepository.get(reviewId);
  }

  private clearReviewClaim(
    ctx: OfflineContext,
    reviewId: number,
    options: { clearApplying?: boolean } = {},
  ): void {
    const transaction = this.options.db.transaction(() => {
      const current = ctx.reviewQueueRepository.get(reviewId);

      if (current === null || current.resolved_at !== null) {
        return;
      }

      const nextRefs = {
        ...current.refs,
      };
      delete nextRefs[CLAIM_REF_KEY];

      if (options.clearApplying !== false) {
        delete nextRefs[APPLYING_REF_KEY];
      }

      this.options.db
        .prepare("UPDATE review_queue SET refs = ? WHERE id = ? AND resolved_at IS NULL")
        .run(serializeJsonValue(nextRefs), reviewId);
    });

    transaction();
  }

  private recordParseFailure(
    ctx: OfflineContext,
    reviewId: number,
    message: string,
  ): { failureCount: number; escalated: boolean } | null {
    const transaction = this.options.db.transaction(() => {
      const current = ctx.reviewQueueRepository.get(reviewId);

      if (current === null || current.resolved_at !== null || current.kind !== "belief_revision") {
        return null;
      }

      const previousCount =
        typeof current.refs[FAILURE_COUNT_REF_KEY] === "number"
          ? Number(current.refs[FAILURE_COUNT_REF_KEY])
          : 0;
      const failureCount = previousCount + 1;
      const nextRefs: Record<string, unknown> = {
        ...current.refs,
        [FAILURE_COUNT_REF_KEY]: failureCount,
        [LAST_ERROR_REF_KEY]: message,
      };
      delete nextRefs[CLAIM_REF_KEY];
      delete nextRefs[APPLYING_REF_KEY];

      const escalated = failureCount >= this.maxParseFailures;

      if (escalated) {
        nextRefs[ESCALATED_AT_REF_KEY] = ctx.clock.now();
        nextRefs[LLM_REVIEW_REF_KEY] = {
          verdict: "manual_review",
          original_verdict: "parse_failure",
          rationale: message,
          confidence_delta: null,
          applied_at: ctx.clock.now(),
        };
      }

      this.options.db
        .prepare("UPDATE review_queue SET refs = ? WHERE id = ? AND resolved_at IS NULL")
        .run(serializeJsonValue(nextRefs), reviewId);

      return {
        failureCount,
        escalated,
      };
    });

    return transaction();
  }

  private async buildLlmInput(
    ctx: OfflineContext,
    item: ReviewQueueItem,
  ): Promise<BeliefRevisionLlmInput> {
    const refs = beliefRevisionReviewSchema.parse(item.refs);
    const invalidatedEdge = ctx.semanticEdgeRepository.getEdge(refs.invalidated_edge_id);
    const survivingSupports = refs.surviving_support_edge_ids
      .map((edgeId) => ctx.semanticEdgeRepository.getEdge(edgeId))
      .filter((edge): edge is SemanticEdge => edge !== null);
    const evidenceEpisodes = (await ctx.episodicRepository.getMany(refs.evidence_episode_ids)).filter(
      (episode): episode is Episode => episode !== null,
    );
    const audienceEntityId = refs.audience_entity_id ?? inferBeliefRevisionAudience(evidenceEpisodes);
    const visibleEpisodes = visibleBeliefRevisionEpisodes(evidenceEpisodes, audienceEntityId);

    if (refs.target_type === "semantic_node") {
      const target = await ctx.semanticNodeRepository.get(refs.target_id);

      if (target === null) {
        throw new Error(`Belief revision target node not found: ${refs.target_id}`);
      }
      const visibleEpisodeIds = await visibleEpisodeIdsForAudience(
        ctx,
        [...collectEpisodeIds({ target, invalidatedEdge, survivingSupports, evidenceEpisodes })],
        audienceEntityId,
      );

      return {
        review_id: item.id,
        audience_entity_id: audienceEntityId,
        visible_episode_ids: visibleEpisodeIds,
        target: {
          target_type: "semantic_node",
          record: target,
        },
        invalidated_edge: invalidatedEdge,
        surviving_supports: survivingSupports,
        evidence_episodes: visibleEpisodes,
      };
    }

    const target = ctx.semanticEdgeRepository.getEdge(refs.target_id);

    if (target === null) {
      throw new Error(`Belief revision target edge not found: ${refs.target_id}`);
    }
    const visibleEpisodeIds = await visibleEpisodeIdsForAudience(
      ctx,
      [...collectEpisodeIds({ target, invalidatedEdge, survivingSupports, evidenceEpisodes })],
      audienceEntityId,
    );

    return {
      review_id: item.id,
      audience_entity_id: audienceEntityId,
      visible_episode_ids: visibleEpisodeIds,
      target: {
        target_type: "semantic_edge",
        record: target,
      },
      invalidated_edge: invalidatedEdge,
      surviving_supports: survivingSupports,
      evidence_episodes: visibleEpisodes,
    };
  }

  private confidenceFromSqlite(targetId: SemanticNode["id"]): number | null {
    const row = this.options.db
      .prepare("SELECT confidence FROM semantic_nodes WHERE id = ?")
      .get(targetId) as { confidence: number } | undefined;

    return row === undefined ? null : Number(row.confidence);
  }

  private nodeSyncFromApplyingState(state: z.infer<typeof beliefRevisionApplyingSchema>): NodeVectorSync {
    if (state.archived === true) {
      return {
        kind: "archive",
        targetId: state.target_id,
      };
    }

    if (state.confidence !== undefined) {
      return {
        kind: "confidence",
        targetId: state.target_id,
        confidence: state.confidence,
      };
    }

    throw new Error("Belief revision applying state is missing node patch");
  }

  private prepareNodeVectorSync(
    ctx: OfflineContext,
    item: ReviewQueueItem,
    verdict: BeliefRevisionVerdict,
  ): PreparedVerdict {
    const transaction = this.options.db.transaction(() => {
      const current = ctx.reviewQueueRepository.get(item.id);

      if (current === null || current.resolved_at !== null || current.kind !== "belief_revision") {
        return {
          verdict,
          nodeSyncs: [],
        } satisfies PreparedVerdict;
      }

      const claim = beliefRevisionClaimSchema.safeParse(current.refs[CLAIM_REF_KEY]);

      if (!claim.success || claim.data.run_id !== String(ctx.runId)) {
        return {
          verdict,
          nodeSyncs: [],
        } satisfies PreparedVerdict;
      }

      const existing = beliefRevisionApplyingSchema.safeParse(current.refs[APPLYING_REF_KEY]);

      if (existing.success) {
        return {
          verdict,
          nodeSyncs: [this.nodeSyncFromApplyingState(existing.data)],
        } satisfies PreparedVerdict;
      }

      const refs = beliefRevisionReviewSchema.parse(current.refs);

      if (refs.target_type !== "semantic_node") {
        return {
          verdict,
          nodeSyncs: [],
        } satisfies PreparedVerdict;
      }

      if (verdict.verdict === "keep") {
        const parsedDrop = autoConfidenceDropSchema.safeParse(current.refs[AUTO_DROP_REF_KEY]);

        if (!parsedDrop.success) {
          return {
            verdict,
            nodeSyncs: [],
          } satisfies PreparedVerdict;
        }

        const applying = {
          verdict: "keep",
          target_id: refs.target_id,
          confidence: parsedDrop.data.previous_confidence,
        } satisfies z.infer<typeof beliefRevisionApplyingSchema>;
        this.options.db
          .prepare("UPDATE review_queue SET refs = ? WHERE id = ? AND resolved_at IS NULL")
          .run(
            serializeJsonValue({
              ...current.refs,
              [APPLYING_REF_KEY]: applying,
            }),
            current.id,
          );

        return {
          verdict,
          nodeSyncs: [this.nodeSyncFromApplyingState(applying)],
        } satisfies PreparedVerdict;
      }

      if (verdict.verdict === "weaken") {
        const currentConfidence = this.confidenceFromSqlite(refs.target_id);

        if (currentConfidence === null) {
          return {
            verdict,
            nodeSyncs: [],
          } satisfies PreparedVerdict;
        }

        const applying = {
          verdict: "weaken",
          target_id: refs.target_id,
          confidence: Math.max(this.confidenceFloor, currentConfidence + (verdict.confidence_delta ?? 0)),
        } satisfies z.infer<typeof beliefRevisionApplyingSchema>;
        this.options.db
          .prepare("UPDATE review_queue SET refs = ? WHERE id = ? AND resolved_at IS NULL")
          .run(
            serializeJsonValue({
              ...current.refs,
              [APPLYING_REF_KEY]: applying,
            }),
            current.id,
          );

        return {
          verdict,
          nodeSyncs: [this.nodeSyncFromApplyingState(applying)],
        } satisfies PreparedVerdict;
      }

      if (verdict.verdict === "archive_node") {
        const applying = {
          verdict: "archive_node",
          target_id: refs.target_id,
          archived: true,
        } satisfies z.infer<typeof beliefRevisionApplyingSchema>;
        this.options.db
          .prepare("UPDATE review_queue SET refs = ? WHERE id = ? AND resolved_at IS NULL")
          .run(
            serializeJsonValue({
              ...current.refs,
              [APPLYING_REF_KEY]: applying,
            }),
            current.id,
          );

        return {
          verdict,
          nodeSyncs: [this.nodeSyncFromApplyingState(applying)],
        } satisfies PreparedVerdict;
      }

      return {
        verdict,
        nodeSyncs: [],
      } satisfies PreparedVerdict;
    });

    return transaction();
  }

  private finalizeManualReview(input: {
    item: ReviewQueueItem;
    refs: BeliefRevisionReview;
    verdict: BeliefRevisionVerdict;
    originalVerdict: BeliefRevisionVerdict["verdict"];
    now: number;
  }): ApplyRegradeResult {
    const nextRefs: Record<string, unknown> = {
      ...input.item.refs,
      [LLM_REVIEW_REF_KEY]: {
        verdict: "manual_review",
        original_verdict: input.originalVerdict,
        rationale: input.verdict.rationale,
        confidence_delta: input.verdict.confidence_delta ?? null,
        applied_at: input.now,
      },
      [ESCALATED_AT_REF_KEY]: input.now,
    };
    delete nextRefs[CLAIM_REF_KEY];
    delete nextRefs[APPLYING_REF_KEY];

    this.options.db
      .prepare("UPDATE review_queue SET refs = ? WHERE id = ? AND resolved_at IS NULL")
      .run(serializeJsonValue(nextRefs), input.item.id);

    return {
      applied: true,
      resolution: "manual_review",
      nodeSyncs: [],
      change: buildRegradeChange({
        reviewId: input.item.id,
        verdict: "manual_review",
        targetType: input.refs.target_type,
        targetId: input.refs.target_id,
      }),
    };
  }

  private applyVerdict(
    ctx: OfflineContext,
    item: ReviewQueueItem,
    verdict: BeliefRevisionVerdict,
  ): ApplyRegradeResult {
    const transaction = this.options.db.transaction(() => {
      const current = ctx.reviewQueueRepository.get(item.id);
      const now = ctx.clock.now();

      if (current === null || current.resolved_at !== null || current.kind !== "belief_revision") {
        return {
          applied: false,
          resolution: "manual_review",
          nodeSyncs: [],
          change: null,
        } satisfies ApplyRegradeResult;
      }

      const claim = beliefRevisionClaimSchema.safeParse(current.refs[CLAIM_REF_KEY]);

      if (!claim.success || claim.data.run_id !== String(ctx.runId)) {
        return {
          applied: false,
          resolution: "manual_review",
          nodeSyncs: [],
          change: null,
        } satisfies ApplyRegradeResult;
      }

      const refs = beliefRevisionReviewSchema.parse(current.refs);
      const originalVerdict = verdict.verdict;

      if (
        verdict.verdict === "manual_review" ||
        (verdict.verdict === "invalidate_edge" && refs.target_type !== "semantic_edge") ||
        (verdict.verdict === "archive_node" && refs.target_type !== "semantic_node")
      ) {
        return this.finalizeManualReview({
          item: current,
          refs,
          verdict,
          originalVerdict,
          now,
        });
      }

      const nodeSyncs: NodeVectorSync[] = [];
      const nextRefs: Record<string, unknown> = {
        ...current.refs,
        [LLM_REVIEW_REF_KEY]: {
          verdict: verdict.verdict,
          original_verdict: originalVerdict,
          rationale: verdict.rationale,
          confidence_delta: verdict.confidence_delta ?? null,
          applied_at: now,
        },
      };
      delete nextRefs[CLAIM_REF_KEY];
      delete nextRefs[APPLYING_REF_KEY];

      if (verdict.verdict === "weaken") {
        const delta = verdict.confidence_delta ?? 0;

        if (refs.target_type === "semantic_edge") {
          const edge = ctx.semanticEdgeRepository.getEdge(refs.target_id);

          if (edge !== null) {
            ctx.semanticEdgeRepository.updateConfidence(
              refs.target_id,
              Math.max(this.confidenceFloor, edge.confidence + delta),
              now,
            );
          }
        }
      }

      if (verdict.verdict === "invalidate_edge" && refs.target_type === "semantic_edge") {
        ctx.semanticEdgeRepository.invalidateEdge(refs.target_id, {
          at: now,
          by_process: "review",
          by_review_id: current.id,
          reason: verdict.rationale,
        });
      }

      this.options.db
        .prepare(
          "UPDATE review_queue SET refs = ?, resolved_at = ?, resolution = ? WHERE id = ? AND resolved_at IS NULL",
        )
        .run(serializeJsonValue(nextRefs), now, verdict.verdict, current.id);

      return {
        applied: true,
        resolution: verdict.verdict,
        nodeSyncs,
        change: buildRegradeChange({
          reviewId: current.id,
          verdict: verdict.verdict,
          targetType: refs.target_type,
          targetId: refs.target_id,
        }),
      } satisfies ApplyRegradeResult;
    });

    return transaction();
  }

  async plan(ctx: OfflineContext): Promise<BeliefReviserPlan> {
    const items: BeliefReviserPlan["items"] = [];
    const events = listUnprocessedEvents(this.options.db);
    let plannedReviews = 0;
    let runCapped = false;

    for (const event of events) {
      if (items.length >= this.maxEventsPerRun || plannedReviews >= this.maxReviewsPerRun) {
        runCapped = true;
        break;
      }

      const remainingReviewSlots = this.maxReviewsPerRun - plannedReviews;
      const eventReviewCap = Math.min(this.maxReviewsPerEvent, remainingReviewSlots);
      const result = await buildReviewsForEvent(ctx, event, eventReviewCap);
      items.push({
        event_id: event.id,
        invalidated_edge_id: event.edge_id,
        reviews: result.reviews,
        fanout_capped: result.fanoutCapped,
        review_cap: eventReviewCap,
      });
      plannedReviews += result.reviews.length;

      if (plannedReviews >= this.maxReviewsPerRun) {
        runCapped = true;

        if (events.length > items.length) {
          break;
        }
      }
    }

    return beliefReviserPlanSchema.parse({
      process: this.name,
      items,
      regrade_items: listPendingRegradeItems(ctx, this.regradeBatchSize, this.claimStaleSec),
      run_capped: runCapped || events.length > items.length,
      event_cap: this.maxEventsPerRun,
      review_run_cap: this.maxReviewsPerRun,
      pending_event_count: Math.max(0, events.length - items.length),
      errors: [],
      tokens_used: 0,
      budget_exhausted: false,
    });
  }

  preview(plan: BeliefReviserPlan): OfflineResult {
    const parsed = beliefReviserPlanSchema.parse(plan);

    return {
      process: this.name,
      dryRun: true,
      changes: [
        ...parsed.items.flatMap((item) =>
          item.reviews.map((review) => buildChange(review, item.event_id)),
        ),
        ...parsed.regrade_items.map((item) => ({
          process: "belief-reviser" as const,
          action: "regrade_belief_revision",
          targets: {
            review_id: item.review_id,
          },
          preview: {},
        })),
      ],
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: BeliefReviserPlan): Promise<OfflineResult> {
    const plan = beliefReviserPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];
    const errors = [...plan.errors];
    let tokensUsed = plan.tokens_used;

    for (const item of plan.items) {
      const applyEvent = this.options.db.transaction(() => {
        const latest = this.options.db
          .prepare("SELECT processed_at FROM semantic_edge_invalidation_events WHERE id = ?")
          .get(item.event_id) as { processed_at: number | null } | undefined;

        if (latest === undefined || latest.processed_at !== null) {
          return {
            enqueued: [],
            confidenceDrops: [],
            processed: false,
          } satisfies ApplyEventResult;
        }

        const openKeys = collectOpenBeliefRevisionKeys(ctx);
        const enqueued: BeliefRevisionReview[] = [];
        const confidenceDrops: ConfidenceDrop[] = [];

        for (const review of item.reviews) {
          const key = reviewKey(review);

          if (openKeys.has(key)) {
            continue;
          }

          const drop = this.maybeDropUnsupportedNodeConfidenceInSqlite(ctx, review);

          const autoConfidenceDrop =
            drop === null
              ? undefined
              : {
                  previous_confidence: drop.previousConfidence,
                  next_confidence: drop.nextConfidence,
                  applied_at: ctx.clock.now(),
                };

          ctx.reviewQueueRepository.enqueue({
            kind: "belief_revision",
            refs: refsForReview(review, autoConfidenceDrop),
            reason: BELIEF_REVISION_REASON,
          });
          openKeys.add(key);
          enqueued.push(review);

          if (drop !== null) {
            confidenceDrops.push(drop);
          }
        }

        this.options.db
          .prepare(
            "UPDATE semantic_edge_invalidation_events SET processed_at = ? WHERE id = ? AND processed_at IS NULL",
          )
          .run(ctx.clock.now(), item.event_id);

        return {
          enqueued,
          confidenceDrops,
          processed: true,
        } satisfies ApplyEventResult;
      });

      const result = applyEvent();

      for (const review of result.enqueued) {
        changes.push(buildChange(review, item.event_id));
      }

      if (result.processed) {
        for (const drop of result.confidenceDrops) {
          try {
            await this.syncConfidenceDropToVectorStore(ctx, drop);
          } catch (error) {
            errors.push({
              process: this.name,
              message: error instanceof Error ? error.message : String(error),
              code: "belief_reviser_confidence_drop_vector_sync_failed",
            });
          }

          changes.push(buildConfidenceDropChange(drop, item.event_id));

          try {
            await ctx.streamWriter.append({
              kind: "internal_event",
              content: {
                hook: "belief_reviser_confidence_dropped",
                event_id: item.event_id,
                target_type: "semantic_node",
                target_id: drop.targetId,
                previous_confidence: drop.previousConfidence,
                next_confidence: drop.nextConfidence,
              },
            });
          } catch (error) {
            errors.push({
              process: this.name,
              message: error instanceof Error ? error.message : String(error),
              code: "belief_reviser_confidence_drop_log_failed",
            });
          }
        }
      }

      if (result.processed && item.fanout_capped) {
        try {
          await ctx.streamWriter.append({
            kind: "internal_event",
            content: {
              hook: "belief_reviser_fanout_capped",
              event_id: item.event_id,
              invalidated_edge_id: item.invalidated_edge_id,
              review_cap: item.review_cap,
              planned_reviews: item.reviews.length,
            },
          });
        } catch (error) {
          errors.push({
            process: this.name,
            message: error instanceof Error ? error.message : String(error),
            code: "belief_reviser_fanout_cap_log_failed",
          });
        }
      }
    }

    if (plan.run_capped) {
      try {
        await ctx.streamWriter.append({
          kind: "internal_event",
          content: {
            hook: "belief_reviser_run_capped",
            planned_events: plan.items.length,
            planned_reviews: plan.items.reduce((sum, item) => sum + item.reviews.length, 0),
            pending_event_count: plan.pending_event_count,
            event_cap: plan.event_cap,
            review_run_cap: plan.review_run_cap,
          },
        });
      } catch (error) {
        errors.push({
          process: this.name,
          message: error instanceof Error ? error.message : String(error),
          code: "belief_reviser_run_cap_log_failed",
        });
      }
    }

    let llmCalls = 0;
    let consecutiveParseFailures = 0;
    let budgetExhausted = plan.budget_exhausted;
    for (const regradeItem of plan.regrade_items) {
      if (llmCalls >= this.budget) {
        budgetExhausted = true;
        break;
      }

      if (consecutiveParseFailures >= this.consecutiveParseFailureLimit) {
        errors.push({
          process: this.name,
          message: "belief revision parse-failure circuit breaker opened",
          code: "belief_reviser_parse_failure_circuit_breaker",
        });
        break;
      }

      const claimed = this.claimReview(ctx, regradeItem.review_id);

      if (claimed === null) {
        continue;
      }
      let nodeSyncCompleted = false;

      try {
        const input = await this.buildLlmInput(ctx, claimed);
        llmCalls += 1;
        const result = await evaluateBeliefRevision({
          llm: ctx.llm.background,
          model: ctx.config.anthropic.models.background,
          input,
          timeoutMs: this.options.llmTimeoutMs,
        });
        tokensUsed += result.tokensUsed;
        const prepared = this.prepareNodeVectorSync(ctx, claimed, result.verdict);

        for (const sync of prepared.nodeSyncs) {
          try {
            await this.syncNodeToVectorStore(ctx, sync);
            nodeSyncCompleted = true;
          } catch (error) {
            errors.push({
              process: this.name,
              message: error instanceof Error ? error.message : String(error),
              code: "belief_reviser_regrade_vector_sync_failed",
            });
            throw error;
          }
        }
        const applied = this.applyVerdict(ctx, claimed, prepared.verdict);

        if (applied.change !== null) {
          changes.push(applied.change);
        }
        consecutiveParseFailures = 0;
      } catch (error) {
        const parseFailure =
          error instanceof BeliefRevisionParseError
            ? this.recordParseFailure(ctx, regradeItem.review_id, error.message)
            : null;

        if (parseFailure === null) {
          this.clearReviewClaim(ctx, regradeItem.review_id, {
            clearApplying: !nodeSyncCompleted,
          });
          consecutiveParseFailures = 0;
        } else {
          consecutiveParseFailures += 1;
        }
        const message = error instanceof Error ? error.message : String(error);
        errors.push({
          process: this.name,
          message,
          code: "belief_reviser_regrade_failed",
        });

        try {
          await ctx.streamWriter.append({
            kind: "internal_event",
            content: {
              hook: "belief_reviser_regrade_failed",
              review_id: regradeItem.review_id,
              error: message,
            },
          });
        } catch (logError) {
          errors.push({
            process: this.name,
            message: logError instanceof Error ? logError.message : String(logError),
            code: "belief_reviser_regrade_failure_log_failed",
          });
        }
      }
    }

    return {
      process: this.name,
      dryRun: false,
      changes,
      tokens_used: tokensUsed,
      errors,
      budget_exhausted: budgetExhausted,
    };
  }

  async run(ctx: OfflineContext, opts: OfflineProcessRunOptions = {}): Promise<OfflineResult> {
    const plan = await this.plan(ctx);
    return opts.dryRun === true ? this.preview(plan) : this.apply(ctx, plan);
  }
}
