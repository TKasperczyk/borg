import { z } from "zod";

import { episodeIdSchema } from "../../memory/episodic/index.js";
import {
  semanticEdgeIdSchema,
  semanticNodeIdSchema,
  type ReviewQueueInsertInput,
  type SemanticEdge,
  type SemanticNode,
} from "../../memory/semantic/index.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";

import type {
  OfflineChange,
  OfflineContext,
  OfflineProcess,
  OfflineProcessRunOptions,
  OfflineResult,
} from "../types.js";

const DEFAULT_MAX_REVIEWS_PER_EVENT = 64;
const MAX_SUPPORT_DESCENDANT_HOPS = 2;
const BELIEF_REVISION_REASON =
  "Supporting semantic edge was invalidated; target needs re-evaluation";

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
  }),
);

const beliefReviserPlanItemSchema = z.object({
  event_id: z.number().int().positive(),
  invalidated_edge_id: semanticEdgeIdSchema,
  reviews: z.array(beliefRevisionReviewSchema),
  fanout_capped: z.boolean().default(false),
  review_cap: z.number().int().positive().default(DEFAULT_MAX_REVIEWS_PER_EVENT),
});

export const beliefReviserPlanSchema = z.object({
  process: z.literal("belief-reviser"),
  items: z.array(beliefReviserPlanItemSchema),
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

function uniqueEpisodeIds(ids: readonly z.infer<typeof episodeIdSchema>[]) {
  return [...new Set(ids)];
}

function uniqueEdgeIds(ids: readonly z.infer<typeof semanticEdgeIdSchema>[]) {
  return [...new Set(ids)];
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
    evidence_episode_ids: evidenceEpisodeIdsWithoutStaleEvidence(
      node.source_episode_ids,
      input.staleEvidenceEpisodeIds,
    ),
  });
}

function buildEdgeReview(input: {
  ctx: OfflineContext;
  targetId: SemanticEdge["id"];
  invalidatedEdgeId: SemanticEdge["id"];
  dependencyPathEdgeIds: readonly SemanticEdge["id"][];
  staleEvidenceEpisodeIds: readonly z.infer<typeof episodeIdSchema>[];
}): BeliefRevisionReview | null {
  const edge = input.ctx.semanticEdgeRepository.getEdge(input.targetId);

  if (edge === null) {
    return null;
  }

  return beliefRevisionReviewSchema.parse({
    target_type: "semantic_edge",
    target_id: edge.id,
    invalidated_edge_id: input.invalidatedEdgeId,
    dependency_path_edge_ids: uniqueEdgeIds(input.dependencyPathEdgeIds),
    surviving_support_edge_ids: [],
    evidence_episode_ids: evidenceEpisodeIdsWithoutStaleEvidence(
      edge.evidence_episode_ids,
      input.staleEvidenceEpisodeIds,
    ),
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

function refsForReview(review: BeliefRevisionReview): ReviewQueueInsertInput["refs"] {
  return {
    target_type: review.target_type,
    target_id: review.target_id,
    invalidated_edge_id: review.invalidated_edge_id,
    dependency_path_edge_ids: review.dependency_path_edge_ids,
    surviving_support_edge_ids: review.surviving_support_edge_ids,
    evidence_episode_ids: review.evidence_episode_ids,
  };
}

export class BeliefReviserProcess implements OfflineProcess<BeliefReviserPlan> {
  readonly name = "belief-reviser" as const;
  private readonly maxReviewsPerEvent: number;

  constructor(private readonly options: BeliefReviserProcessOptions) {
    this.maxReviewsPerEvent = normalizeMaxReviewsPerEvent(options.maxReviewsPerEvent);
  }

  async plan(ctx: OfflineContext): Promise<BeliefReviserPlan> {
    const items: BeliefReviserPlan["items"] = [];

    for (const event of listUnprocessedEvents(this.options.db)) {
      const result = await buildReviewsForEvent(ctx, event, this.maxReviewsPerEvent);
      items.push({
        event_id: event.id,
        invalidated_edge_id: event.edge_id,
        reviews: result.reviews,
        fanout_capped: result.fanoutCapped,
        review_cap: this.maxReviewsPerEvent,
      });
    }

    return beliefReviserPlanSchema.parse({
      process: this.name,
      items,
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
      changes: parsed.items.flatMap((item) =>
        item.reviews.map((review) => buildChange(review, item.event_id)),
      ),
      tokens_used: parsed.tokens_used,
      errors: parsed.errors,
      budget_exhausted: parsed.budget_exhausted,
    };
  }

  async apply(ctx: OfflineContext, rawPlan: BeliefReviserPlan): Promise<OfflineResult> {
    const plan = beliefReviserPlanSchema.parse(rawPlan);
    const changes: OfflineChange[] = [];
    const errors = [...plan.errors];

    for (const item of plan.items) {
      const applyEvent = this.options.db.transaction(() => {
        const current = this.options.db
          .prepare("SELECT processed_at FROM semantic_edge_invalidation_events WHERE id = ?")
          .get(item.event_id) as { processed_at: number | null } | undefined;

        if (current === undefined || current.processed_at !== null) {
          return [] as BeliefRevisionReview[];
        }

        const openKeys = collectOpenBeliefRevisionKeys(ctx);
        const enqueued: BeliefRevisionReview[] = [];

        for (const review of item.reviews) {
          const key = reviewKey(review);

          if (openKeys.has(key)) {
            continue;
          }

          ctx.reviewQueueRepository.enqueue({
            kind: "belief_revision",
            refs: refsForReview(review),
            reason: BELIEF_REVISION_REASON,
          });
          openKeys.add(key);
          enqueued.push(review);
        }

        this.options.db
          .prepare(
            "UPDATE semantic_edge_invalidation_events SET processed_at = ? WHERE id = ? AND processed_at IS NULL",
          )
          .run(ctx.clock.now(), item.event_id);

        return enqueued;
      });

      for (const review of applyEvent()) {
        changes.push(buildChange(review, item.event_id));
      }

      if (item.fanout_capped) {
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

    return {
      process: this.name,
      dryRun: false,
      changes,
      tokens_used: plan.tokens_used,
      errors,
      budget_exhausted: plan.budget_exhausted,
    };
  }

  async run(ctx: OfflineContext, opts: OfflineProcessRunOptions = {}): Promise<OfflineResult> {
    const plan = await this.plan(ctx);
    return opts.dryRun === true ? this.preview(plan) : this.apply(ctx, plan);
  }
}
