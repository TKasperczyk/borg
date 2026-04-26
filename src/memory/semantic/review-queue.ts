import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { SemanticError } from "../../util/errors.js";
import { serializeJsonValue, type JsonValue } from "../../util/json-value.js";
import { parseReviewProvenance, provenanceSchema, type Provenance } from "../common/provenance.js";
import { EpisodicRepository, episodeIdSchema, episodePatchSchema } from "../episodic/index.js";
import { type IdentityEventRepository, type IdentityService } from "../identity/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
  autobiographicalPeriodIdSchema,
  autobiographicalPeriodSchema,
  goalIdSchema,
  traitIdSchema,
  valueIdSchema,
} from "../self/index.js";
import { CommitmentRepository, commitmentIdSchema } from "../commitments/index.js";
import type { SemanticEdgeRepository, SemanticNodeRepository } from "./repository.js";
import {
  semanticEdgeIdSchema,
  semanticNodeIdSchema,
  semanticNodeSchema,
  type SemanticEdge,
  type SemanticNode,
} from "./types.js";

export const REVIEW_KINDS = [
  "contradiction",
  "duplicate",
  "stale",
  "new_insight",
  "misattribution",
  "temporal_drift",
  "identity_inconsistency",
  "correction",
] as const;
export const REVIEW_RESOLUTIONS = [
  "keep_both",
  "supersede",
  "invalidate",
  "dismiss",
  "accept",
  "reject",
] as const;

export const reviewKindSchema = z.enum(REVIEW_KINDS);
export const reviewResolutionSchema = z.enum(REVIEW_RESOLUTIONS);
export const reviewResolutionInputSchema = z.union([
  reviewResolutionSchema,
  z
    .object({
      decision: reviewResolutionSchema,
      winner_node_id: semanticNodeIdSchema.optional(),
    })
    .strict(),
]);

export const reviewQueueItemSchema = z.object({
  id: z.number().int().positive(),
  kind: reviewKindSchema,
  refs: z.record(z.string(), z.unknown()),
  reason: z.string().min(1),
  created_at: z.number().finite(),
  resolved_at: z.number().finite().nullable(),
  resolution: reviewResolutionSchema.nullable(),
});

export type ReviewQueueItem = z.infer<typeof reviewQueueItemSchema>;
export type ReviewKind = z.infer<typeof reviewKindSchema>;
export type ReviewResolution = z.infer<typeof reviewResolutionSchema>;
export type ReviewResolutionInput = z.infer<typeof reviewResolutionInputSchema>;
type ResolvedReviewDecision = {
  decision: ReviewResolution;
  winner_node_id?: z.infer<typeof semanticNodeIdSchema>;
};

export type ReviewQueueInsertInput = {
  kind: ReviewKind;
  refs: Record<string, unknown>;
  reason: string;
};

export type ReviewQueueRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  episodicRepository?: EpisodicRepository;
  semanticNodeRepository?: SemanticNodeRepository;
  semanticEdgeRepository?: SemanticEdgeRepository;
  valuesRepository?: ValuesRepository;
  goalsRepository?: GoalsRepository;
  traitsRepository?: TraitsRepository;
  autobiographicalRepository?: AutobiographicalRepository;
  commitmentRepository?: CommitmentRepository;
  identityService?: IdentityService;
  identityEventRepository?: IdentityEventRepository;
  applyCorrection?: (item: ReviewQueueItem) => Promise<void> | void;
  onEnqueue?: (item: ReviewQueueItem, input: ReviewQueueInsertInput) => void;
  onEnqueueError?: (error: unknown, item: ReviewQueueItem, input: ReviewQueueInsertInput) => void;
};

const SEMANTIC_REVIEW_RESOLUTIONS = new Set<ReviewResolution>([
  "keep_both",
  "supersede",
  "invalidate",
  "dismiss",
]);
const NEW_INSIGHT_REVIEW_RESOLUTIONS = new Set<ReviewResolution>([
  "accept",
  "invalidate",
  "dismiss",
]);
const LIFECYCLE_REVIEW_RESOLUTIONS = new Set<ReviewResolution>(["accept", "reject", "dismiss"]);
const CORRECTION_REVIEW_RESOLUTIONS = new Set<ReviewResolution>(["accept", "reject"]);

const misattributionEpisodePatchSchema = episodePatchSchema
  .pick({
    participants: true,
    audience_entity_id: true,
    narrative: true,
    tags: true,
  })
  .strict();

const semanticNodeMisattributionPatchSchema = z
  .object({
    label: z.string().min(1).optional(),
    aliases: z.array(z.string().min(1)).optional(),
    description: z.string().min(1).optional(),
    source_episode_ids: z.array(episodeIdSchema).min(1).optional(),
  })
  .strict();

const misattributionRefsSchema = z.discriminatedUnion("target_type", [
  z.object({
    target_type: z.literal("episode"),
    target_id: episodeIdSchema,
    patch: misattributionEpisodePatchSchema,
    proposed_provenance: provenanceSchema.optional(),
  }),
  z.object({
    target_type: z.literal("semantic_node"),
    target_id: semanticNodeIdSchema,
    patch: semanticNodeMisattributionPatchSchema,
    proposed_provenance: provenanceSchema.optional(),
  }),
]);

const temporalDriftRefsSchema = z.discriminatedUnion("target_type", [
  z.object({
    target_type: z.literal("episode"),
    target_id: episodeIdSchema,
    corrected_start_time: z.number().finite().optional(),
    corrected_end_time: z.number().finite().optional(),
    patch_description: z.string().min(1).optional(),
    proposed_provenance: provenanceSchema.optional(),
  }),
  z.object({
    target_type: z.literal("semantic_node"),
    target_id: semanticNodeIdSchema,
    patch_description: z.string().min(1).optional(),
    proposed_provenance: provenanceSchema.optional(),
  }),
  z.object({
    target_type: z.literal("semantic_edge"),
    target_kind: z.literal("semantic_edge").optional(),
    target_id: semanticEdgeIdSchema,
    suggested_valid_to: z.number().finite().optional(),
    by_edge_id: semanticEdgeIdSchema.optional(),
    reason: z.string().min(1).optional(),
    proposed_provenance: provenanceSchema.optional(),
  }),
]);

const semanticEdgeReviewClosureRefsSchema = z
  .object({
    target_type: z.literal("semantic_edge").optional(),
    target_kind: z.literal("semantic_edge").optional(),
    target_id: semanticEdgeIdSchema.optional(),
    loser_edge_id: semanticEdgeIdSchema.optional(),
    suggested_valid_to: z.number().finite().optional(),
    by_edge_id: semanticEdgeIdSchema.optional(),
    winner_edge_id: semanticEdgeIdSchema.optional(),
    reason: z.string().min(1).optional(),
  })
  .passthrough();

const reviewSemanticNodePayloadSchema = z.object({
  id: semanticNodeIdSchema,
  kind: z.enum(["concept", "entity", "proposition"]),
  label: z.string().min(1),
  description: z.string().min(1),
  domain: z.string().min(1).nullable().default(null),
  aliases: z.array(z.string().min(1)),
  confidence: z.number().min(0).max(1),
  source_episode_ids: z.array(episodeIdSchema).min(1),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
  last_verified_at: z.number().finite(),
  embedding: z.array(z.number().finite()),
  archived: z.boolean(),
  superseded_by: semanticNodeIdSchema.nullable(),
});

const pendingReflectorTargetSchema = z.discriminatedUnion("mode", [
  z.object({
    mode: z.literal("insert"),
    node: reviewSemanticNodePayloadSchema,
  }),
  z.object({
    mode: z.literal("update"),
    node_id: semanticNodeIdSchema,
    patch: z.object({
      description: z.string().min(1),
      confidence: z.number().min(0).max(1),
      source_episode_ids: z.array(episodeIdSchema).min(1),
      last_verified_at: z.number().finite(),
      embedding: z.array(z.number().finite()),
      archived: z.boolean(),
    }),
  }),
]);

const pendingReflectorSupportEdgeSchema = z.object({
  id: semanticEdgeIdSchema,
  insight_node_id: semanticNodeIdSchema,
  target_node_id: semanticNodeIdSchema,
  source_episode_ids: z.array(episodeIdSchema).min(1),
  confidence: z.number().min(0).max(1),
});

const pendingReflectorInsightSchema = z
  .object({
    target: pendingReflectorTargetSchema,
    candidate_support_edges: z.array(pendingReflectorSupportEdgeSchema).default([]),
    evidence_cluster: z.object({
      key: z.string().min(1),
      episode_ids: z.array(episodeIdSchema).min(1),
      size: z.number().int().positive(),
    }),
  })
  .strict();

const identityInconsistencyTargetTypeSchema = z.enum([
  "trait",
  "value",
  "commitment",
  "goal",
  "autobiographical_period",
]);
const identityRepairOpSchema = z.enum(["reinforce", "contradict", "patch"]);
const REVIEW_APPLYING_REF_KEY = "__borg_resolution_applying";
const reviewApplyingStateSchema = z
  .object({
    decision: reviewResolutionSchema,
    winner_node_id: semanticNodeIdSchema.nullable().optional(),
    started_at: z.number().finite(),
    semantic_node_patch: z
      .object({
        node_id: semanticNodeIdSchema,
        confidence: z.number().min(0).max(1).optional(),
        last_verified_at: z.number().finite().optional(),
      })
      .strict()
      .optional(),
  })
  .strict();

type ReviewApplyingState = z.infer<typeof reviewApplyingStateSchema>;
type SemanticEdgeClosureRefs = {
  edgeId: SemanticEdge["id"];
  validTo: number;
  byEdgeId?: SemanticEdge["id"];
  reason: string;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function deserializeReviewSemanticNode(
  node: z.infer<typeof reviewSemanticNodePayloadSchema>,
): SemanticNode {
  return semanticNodeSchema.parse({
    ...node,
    embedding: Float32Array.from(node.embedding),
  });
}

function targetNodeId(target: z.infer<typeof pendingReflectorTargetSchema>): SemanticNode["id"] {
  return target.mode === "insert" ? target.node.id : target.node_id;
}

function parseEvidenceProvenance(refs: Record<string, unknown>): Provenance | null {
  const parsed = z.array(episodeIdSchema).safeParse(refs.evidence_episode_ids);

  if (!parsed.success || parsed.data.length === 0) {
    return null;
  }

  return {
    kind: "episodes",
    episode_ids: parsed.data,
  };
}

function throwLegacyRepairRefsError(kind: ReviewKind): never {
  // Accept must fail loudly for legacy under-specified repair rows; silently resolving them
  // would recreate the exact epistemic theater Sprint 14 was meant to remove.
  throw new SemanticError("cannot apply accept on legacy review row -- structured patch required", {
    code: "REVIEW_QUEUE_REPAIR_REQUIRES_STRUCTURED_REFS",
    cause: { kind },
  });
}

function throwMalformedPairRefsError(kind: ReviewKind): never {
  throw new SemanticError("cannot apply pair resolution on malformed review row", {
    code: "REVIEW_QUEUE_MALFORMED_PAIR_REFS",
    cause: { kind },
  });
}

function isResolutionCompatible(kind: ReviewKind, resolution: ReviewResolution): boolean {
  switch (kind) {
    case "correction":
      return CORRECTION_REVIEW_RESOLUTIONS.has(resolution);
    case "contradiction":
    case "duplicate":
      return SEMANTIC_REVIEW_RESOLUTIONS.has(resolution);
    case "new_insight":
      return NEW_INSIGHT_REVIEW_RESOLUTIONS.has(resolution);
    case "stale":
    case "misattribution":
    case "temporal_drift":
    case "identity_inconsistency":
      return LIFECYCLE_REVIEW_RESOLUTIONS.has(resolution);
  }
}

function parseRefs(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value) as unknown;

    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new TypeError("refs must be an object");
    }

    return parsed as Record<string, unknown>;
  } catch (error) {
    throw new SemanticError("Failed to parse review queue refs", {
      cause: error,
      code: "REVIEW_QUEUE_INVALID",
    });
  }
}

function mapReviewRow(row: Record<string, unknown>): ReviewQueueItem {
  const parsed = reviewQueueItemSchema.safeParse({
    id: Number(row.id),
    kind: row.kind,
    refs: parseRefs(String(row.refs ?? "{}")),
    reason: row.reason,
    created_at: Number(row.created_at),
    resolved_at:
      row.resolved_at === null || row.resolved_at === undefined ? null : Number(row.resolved_at),
    resolution:
      row.resolution === null || row.resolution === undefined ? null : String(row.resolution),
  });

  if (!parsed.success) {
    throw new SemanticError("Review queue row failed validation", {
      cause: parsed.error,
      code: "REVIEW_QUEUE_INVALID",
    });
  }

  return parsed.data;
}

export class ReviewQueueRepository {
  private readonly clock: Clock;

  constructor(private readonly options: ReviewQueueRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  enqueue(input: ReviewQueueInsertInput): ReviewQueueItem {
    const parsed = reviewKindSchema.parse(input.kind);
    const timestamp = this.clock.now();
    const result = this.db
      .prepare(
        `
          INSERT INTO review_queue (kind, refs, reason, created_at, resolved_at, resolution)
          VALUES (?, ?, ?, ?, NULL, NULL)
        `,
      )
      .run(parsed, serializeJsonValue(input.refs), input.reason, timestamp);

    const row = this.db
      .prepare("SELECT * FROM review_queue WHERE id = ?")
      .get(result.lastInsertRowid) as Record<string, unknown> | undefined;

    if (row === undefined) {
      throw new SemanticError("Failed to read back queued review item", {
        code: "REVIEW_QUEUE_INSERT_FAILED",
      });
    }

    const item = mapReviewRow(row);

    try {
      this.options.onEnqueue?.(item, input);
    } catch (error) {
      try {
        this.options.onEnqueueError?.(error, item, input);
      } catch {
        // Best-effort hook error reporting only.
      }
    }

    return item;
  }

  list(options: { kind?: ReviewKind; openOnly?: boolean } = {}): ReviewQueueItem[] {
    if (options.kind !== undefined) {
      reviewKindSchema.parse(options.kind);
    }

    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.kind !== undefined) {
      filters.push("kind = ?");
      values.push(options.kind);
    }

    if (options.openOnly === true) {
      filters.push("resolved_at IS NULL");
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT id, kind, refs, reason, created_at, resolved_at, resolution
          FROM review_queue
          ${whereClause}
          ORDER BY created_at DESC, id DESC
        `,
      )
      .all(...values) as Record<string, unknown>[];

    return rows.map((row) => mapReviewRow(row));
  }

  getOpen(): ReviewQueueItem[] {
    return this.list({
      openOnly: true,
    });
  }

  delete(itemId: number): boolean {
    const result = this.db.prepare("DELETE FROM review_queue WHERE id = ?").run(itemId);
    return result.changes > 0;
  }

  private getApplyingState(item: ReviewQueueItem): ReviewApplyingState | null {
    const candidate = item.refs[REVIEW_APPLYING_REF_KEY];
    const parsed = reviewApplyingStateSchema.safeParse(candidate);
    return parsed.success ? parsed.data : null;
  }

  private resolutionMatchesApplyingState(
    state: ReviewApplyingState,
    resolution: ResolvedReviewDecision,
  ): boolean {
    return (
      state.decision === resolution.decision &&
      (state.winner_node_id ?? null) === (resolution.winner_node_id ?? null)
    );
  }

  private refsWithoutApplyingState(refs: Record<string, unknown>): Record<string, unknown> {
    const next = { ...refs };
    delete next[REVIEW_APPLYING_REF_KEY];
    return next;
  }

  private parseSemanticEdgeClosureRefs(item: ReviewQueueItem): SemanticEdgeClosureRefs | null {
    const parsed = semanticEdgeReviewClosureRefsSchema.safeParse(item.refs);

    if (!parsed.success) {
      return null;
    }

    const refs = parsed.data;
    const describesSemanticEdge =
      refs.loser_edge_id !== undefined ||
      refs.target_type === "semantic_edge" ||
      refs.target_kind === "semantic_edge";

    if (!describesSemanticEdge) {
      return null;
    }

    const edgeId = refs.loser_edge_id ?? refs.target_id;

    if (edgeId === undefined) {
      throw new SemanticError("semantic edge review row is missing target edge id", {
        code: "REVIEW_QUEUE_EDGE_TARGET_REQUIRED",
        cause: { itemId: item.id, kind: item.kind },
      });
    }

    return {
      edgeId,
      validTo: refs.suggested_valid_to ?? this.clock.now(),
      byEdgeId: refs.by_edge_id ?? refs.winner_edge_id,
      reason: refs.reason ?? item.reason,
    };
  }

  private recordSemanticEdgeInvalidationAudit(input: {
    item: ReviewQueueItem;
    previous: SemanticEdge;
    next: SemanticEdge;
  }): void {
    const identityEventRepository = this.options.identityEventRepository;

    if (identityEventRepository === undefined) {
      return;
    }

    const existing = identityEventRepository.findByReviewKey({
      reviewItemId: input.item.id,
      recordType: "semantic_edge",
      recordId: input.next.id,
      action: "edge_invalidate",
    });

    if (existing !== null) {
      return;
    }

    const auditShape = {
      edge_id: input.next.id,
      prior_valid_to: input.previous.valid_to,
      new_valid_to: input.next.valid_to,
      by_process: input.next.invalidated_by_process,
      by_review_id: input.next.invalidated_by_review_id,
      reason: input.next.invalidated_reason,
      by_edge_id: input.next.invalidated_by_edge_id,
    } satisfies JsonValue;

    identityEventRepository.record({
      record_type: "semantic_edge",
      record_id: input.next.id,
      action: "edge_invalidate",
      old_value: {
        edge_id: input.previous.id,
        prior_valid_to: input.previous.valid_to,
      },
      new_value: auditShape,
      reason: input.next.invalidated_reason,
      provenance: {
        kind: "manual",
      },
      review_item_id: input.item.id,
    });
  }

  private closeSemanticEdgeFromReview(item: ReviewQueueItem, refs: SemanticEdgeClosureRefs): void {
    if (this.options.semanticEdgeRepository === undefined) {
      throw new SemanticError("Semantic edge repository is required for edge review repair", {
        code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
      });
    }

    const current = this.options.semanticEdgeRepository.getEdge(refs.edgeId);

    if (current === null) {
      throw new SemanticError(`Unknown semantic edge id for review repair: ${refs.edgeId}`, {
        code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
      });
    }

    const invalidated = this.options.semanticEdgeRepository.invalidateEdge(refs.edgeId, {
      at: refs.validTo,
      by_edge_id: refs.byEdgeId,
      by_process: "review",
      by_review_id: item.id,
      reason: refs.reason,
    });

    if (invalidated === null) {
      throw new SemanticError(`Unknown semantic edge id for review repair: ${refs.edgeId}`, {
        code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
      });
    }

    if (current.valid_to === null) {
      this.recordSemanticEdgeInvalidationAudit({
        item,
        previous: current,
        next: invalidated,
      });
    }
  }

  private targetLooksCrossStore(targetId: unknown): boolean {
    return (
      typeof targetId === "string" && (targetId.startsWith("ep_") || targetId.startsWith("semn_"))
    );
  }

  private resolutionTouchesCrossStore(
    item: ReviewQueueItem,
    resolution: ResolvedReviewDecision,
  ): boolean {
    if (resolution.decision === "dismiss") {
      return false;
    }

    if (item.kind === "correction") {
      return resolution.decision === "accept" && this.targetLooksCrossStore(item.refs.target_id);
    }

    if (item.kind === "contradiction" || item.kind === "duplicate") {
      return (
        (resolution.decision === "supersede" || resolution.decision === "invalidate") &&
        this.parseSemanticEdgeClosureRefs(item) === null
      );
    }

    if (item.kind === "new_insight") {
      return resolution.decision === "accept" || resolution.decision === "invalidate";
    }

    if (item.kind === "stale") {
      return resolution.decision === "accept";
    }

    if (item.kind === "misattribution" || item.kind === "temporal_drift") {
      return (
        resolution.decision === "accept" &&
        (item.kind !== "temporal_drift" || this.parseSemanticEdgeClosureRefs(item) === null)
      );
    }

    return false;
  }

  private markResolved(
    itemId: number,
    resolution: ResolvedReviewDecision,
    resolvedAt: number,
    refs: Record<string, unknown>,
  ): void {
    this.db
      .prepare("UPDATE review_queue SET refs = ?, resolved_at = ?, resolution = ? WHERE id = ?")
      .run(
        serializeJsonValue(this.refsWithoutApplyingState(refs)),
        resolvedAt,
        resolution.decision,
        itemId,
      );
  }

  private async buildApplyingState(
    item: ReviewQueueItem,
    resolution: ResolvedReviewDecision,
  ): Promise<ReviewApplyingState> {
    const state: ReviewApplyingState = {
      decision: resolution.decision,
      winner_node_id: resolution.winner_node_id ?? null,
      started_at: this.clock.now(),
    };

    if (
      this.options.semanticNodeRepository === undefined ||
      (item.kind !== "new_insight" && item.kind !== "stale") ||
      resolution.decision !== "accept"
    ) {
      return state;
    }

    const rawNodeId =
      item.kind === "new_insight"
        ? Array.isArray(item.refs.node_ids)
          ? item.refs.node_ids[0]
          : undefined
        : item.refs.target_type === "semantic_node"
          ? item.refs.target_id
          : item.refs.node_id;

    if (typeof rawNodeId !== "string") {
      return state;
    }

    const nodeId = semanticNodeIdSchema.parse(rawNodeId);
    const current = await this.options.semanticNodeRepository.get(nodeId);

    if (current === null) {
      return state;
    }

    state.semantic_node_patch = {
      node_id: nodeId,
      confidence: clamp(current.confidence + (item.kind === "new_insight" ? 0.1 : -0.05), 0, 1),
      last_verified_at: this.clock.now(),
    };
    return state;
  }

  private async ensureApplyingState(
    item: ReviewQueueItem,
    resolution: ResolvedReviewDecision,
  ): Promise<ReviewQueueItem> {
    const existing = this.getApplyingState(item);

    if (existing !== null) {
      if (!this.resolutionMatchesApplyingState(existing, resolution)) {
        throw new SemanticError("Review item is already applying a different resolution", {
          code: "REVIEW_QUEUE_RESOLUTION_IN_PROGRESS",
          cause: { itemId: item.id, decision: existing.decision },
        });
      }

      return item;
    }

    const applyingState = await this.buildApplyingState(item, resolution);
    const refs = {
      ...item.refs,
      [REVIEW_APPLYING_REF_KEY]: applyingState,
    };

    this.db
      .prepare("UPDATE review_queue SET refs = ? WHERE id = ? AND resolved_at IS NULL")
      .run(serializeJsonValue(refs), item.id);

    return {
      ...item,
      refs,
    };
  }

  private async resolveInSqliteTransaction(
    item: ReviewQueueItem,
    resolution: ResolvedReviewDecision,
  ): Promise<ReviewQueueItem> {
    const resolvedAt = this.clock.now();

    this.db.exec("BEGIN IMMEDIATE");

    try {
      await this.applyResolution(item, resolution);
      this.markResolved(item.id, resolution, resolvedAt, item.refs);
      this.db.exec("COMMIT");
    } catch (error) {
      try {
        this.db.exec("ROLLBACK");
      } catch {
        // Keep the original failure.
      }

      throw error;
    }

    return {
      ...item,
      resolved_at: resolvedAt,
      resolution: resolution.decision,
    };
  }

  private async resolveWithApplyingState(
    item: ReviewQueueItem,
    resolution: ResolvedReviewDecision,
  ): Promise<ReviewQueueItem> {
    const applyingItem = await this.ensureApplyingState(item, resolution);

    await this.applyResolution(applyingItem, resolution);
    const resolvedAt = this.clock.now();
    this.markResolved(applyingItem.id, resolution, resolvedAt, applyingItem.refs);

    return {
      ...applyingItem,
      refs: this.refsWithoutApplyingState(applyingItem.refs),
      resolved_at: resolvedAt,
      resolution: resolution.decision,
    };
  }

  async resolve(itemId: number, decision: ReviewResolutionInput): Promise<ReviewQueueItem | null> {
    const parsedResolution = reviewResolutionInputSchema.parse(decision);
    const resolution: ResolvedReviewDecision =
      typeof parsedResolution === "string" ? { decision: parsedResolution } : parsedResolution;
    const row = this.db.prepare("SELECT * FROM review_queue WHERE id = ?").get(itemId) as
      | Record<string, unknown>
      | undefined;

    if (row === undefined) {
      return null;
    }

    const item = mapReviewRow(row);

    if (item.resolved_at !== null) {
      return item;
    }

    if (!isResolutionCompatible(item.kind, resolution.decision)) {
      throw new SemanticError(
        `Resolution "${resolution.decision}" is incompatible with review kind "${item.kind}"`,
        {
          code: "REVIEW_QUEUE_RESOLUTION_INVALID",
        },
      );
    }

    return this.resolutionTouchesCrossStore(item, resolution)
      ? this.resolveWithApplyingState(item, resolution)
      : this.resolveInSqliteTransaction(item, resolution);
  }

  private async applyResolution(
    item: ReviewQueueItem,
    decision: ResolvedReviewDecision,
  ): Promise<void> {
    if (item.kind === "correction") {
      if (decision.decision === "accept") {
        if (this.options.applyCorrection === undefined) {
          throw new SemanticError("No correction applier configured for review queue", {
            code: "REVIEW_QUEUE_CORRECTION_UNSUPPORTED",
          });
        }

        await this.options.applyCorrection(item);
      }

      return;
    }

    switch (item.kind) {
      case "contradiction":
      case "duplicate":
        await this.applySemanticPairResolution(item, decision);
        return;
      case "new_insight":
        await this.applyNewInsightResolution(item, decision);
        return;
      case "stale":
        await this.applyStaleResolution(item, decision);
        return;
      case "misattribution":
        await this.applyMisattributionResolution(item, decision);
        return;
      case "temporal_drift":
        await this.applyTemporalDriftResolution(item, decision);
        return;
      case "identity_inconsistency":
        await this.applyIdentityInconsistencyResolution(item, decision);
        return;
    }
  }

  private async applySemanticPairResolution(
    item: ReviewQueueItem,
    decision: ResolvedReviewDecision,
  ): Promise<void> {
    if (decision.decision !== "supersede" && decision.decision !== "invalidate") {
      return;
    }

    const edgeClosureRefs = this.parseSemanticEdgeClosureRefs(item);

    if (edgeClosureRefs !== null) {
      this.closeSemanticEdgeFromReview(item, edgeClosureRefs);
      return;
    }

    if (this.options.semanticNodeRepository === undefined) {
      return;
    }

    const rawNodeIds = item.refs.node_ids;

    if (!Array.isArray(rawNodeIds) || rawNodeIds.length < 2) {
      throwMalformedPairRefsError(item.kind);
    }

    const parsedNodeIds = rawNodeIds.map((value) => semanticNodeIdSchema.parse(value));
    const winnerNodeId = decision.winner_node_id;

    if (winnerNodeId === undefined) {
      throw new SemanticError("winner_node_id is required for supersede/invalidate", {
        code: "REVIEW_QUEUE_WINNER_REQUIRED",
        cause: { itemId: item.id, kind: item.kind },
      });
    }

    if (!parsedNodeIds.includes(winnerNodeId)) {
      throw new SemanticError("winner_node_id must reference a node in the review item", {
        code: "REVIEW_QUEUE_WINNER_INVALID",
        cause: { itemId: item.id, kind: item.kind, winner_node_id: winnerNodeId },
      });
    }

    const nodes = await this.options.semanticNodeRepository.getMany(parsedNodeIds, {
      includeArchived: true,
    });
    const first = nodes[0];
    const second = nodes[1];

    if (first === null || first === undefined || second === null || second === undefined) {
      return;
    }

    const winner = first.id === winnerNodeId ? first : second;
    const loser = winner.id === first.id ? second : first;

    if (decision.decision === "supersede") {
      await this.options.semanticNodeRepository.update(loser.id, {
        superseded_by: winner.id,
        archived: true,
      });
      return;
    }

    await this.options.semanticNodeRepository.update(loser.id, {
      confidence: 0,
      archived: true,
    });
  }

  private async applyNewInsightResolution(
    item: ReviewQueueItem,
    decision: ResolvedReviewDecision,
  ): Promise<void> {
    const pendingReflectorInsight = pendingReflectorInsightSchema.safeParse(
      item.refs.reflector_pending_insight,
    );

    if (pendingReflectorInsight.success) {
      if (decision.decision !== "accept") {
        return;
      }

      if (this.options.semanticNodeRepository === undefined) {
        throw new SemanticError("Semantic node repository is required for pending insight review", {
          code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
        });
      }

      const target = pendingReflectorInsight.data.target;
      const candidateSupportEdges = pendingReflectorInsight.data.candidate_support_edges;
      const insightNodeId = targetNodeId(target);

      if (target.mode === "insert") {
        await this.options.semanticNodeRepository.insert(
          deserializeReviewSemanticNode(target.node),
        );
      } else {
        const updated = await this.options.semanticNodeRepository.update(target.node_id, {
          description: target.patch.description,
          confidence: target.patch.confidence,
          source_episode_ids: target.patch.source_episode_ids,
          last_verified_at: target.patch.last_verified_at,
          embedding: Float32Array.from(target.patch.embedding),
          archived: target.patch.archived,
        });

        if (updated === null) {
          throw new SemanticError(
            `Unknown semantic node id for pending insight: ${target.node_id}`,
            {
              code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
            },
          );
        }
      }

      if (candidateSupportEdges.length > 0) {
        if (this.options.semanticEdgeRepository === undefined) {
          throw new SemanticError(
            "Semantic edge repository is required for pending insight review",
            {
              code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
            },
          );
        }

        for (const edge of candidateSupportEdges) {
          if (edge.insight_node_id !== insightNodeId) {
            throw new SemanticError(
              "Pending insight support edge points at the wrong insight node",
              {
                code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
                cause: { itemId: item.id, insightNodeId, edgeInsightNodeId: edge.insight_node_id },
              },
            );
          }

          const duplicate = this.options.semanticEdgeRepository.listEdges({
            fromId: edge.insight_node_id,
            toId: edge.target_node_id,
            relation: "supports",
          });

          if (duplicate.length > 0) {
            continue;
          }

          this.options.semanticEdgeRepository.addEdge({
            id: edge.id,
            from_node_id: edge.insight_node_id,
            to_node_id: edge.target_node_id,
            relation: "supports",
            confidence: edge.confidence,
            evidence_episode_ids: edge.source_episode_ids,
            created_at: this.clock.now(),
            last_verified_at: this.clock.now(),
          });
        }
      }

      return;
    }

    if (
      this.options.semanticNodeRepository === undefined ||
      (decision.decision !== "invalidate" && decision.decision !== "accept")
    ) {
      return;
    }

    const rawNodeIds = item.refs.node_ids;

    if (!Array.isArray(rawNodeIds) || rawNodeIds.length < 1) {
      return;
    }

    const nodeId = semanticNodeIdSchema.parse(rawNodeIds[0]);

    if (decision.decision === "accept") {
      const applyingPatch = this.getApplyingState(item)?.semantic_node_patch;

      if (applyingPatch?.node_id === nodeId && applyingPatch.confidence !== undefined) {
        await this.options.semanticNodeRepository.update(nodeId, {
          confidence: applyingPatch.confidence,
          last_verified_at: applyingPatch.last_verified_at ?? this.clock.now(),
          archived: false,
        });
        return;
      }

      const current = await this.options.semanticNodeRepository.get(nodeId);

      if (current === null) {
        return;
      }

      await this.options.semanticNodeRepository.update(nodeId, {
        confidence: clamp(current.confidence + 0.1, 0, 1),
        last_verified_at: this.clock.now(),
        archived: false,
      });
      return;
    }

    await this.options.semanticNodeRepository.update(nodeId, {
      archived: true,
    });
  }

  private async applyStaleResolution(
    item: ReviewQueueItem,
    decision: ResolvedReviewDecision,
  ): Promise<void> {
    if (decision.decision !== "accept" || this.options.semanticNodeRepository === undefined) {
      return;
    }

    const rawTargetId =
      item.refs.target_type === "semantic_node" ? item.refs.target_id : item.refs.node_id;

    if (typeof rawTargetId !== "string") {
      return;
    }

    const targetId = semanticNodeIdSchema.parse(rawTargetId);
    const applyingPatch = this.getApplyingState(item)?.semantic_node_patch;

    if (applyingPatch?.node_id === targetId && applyingPatch.confidence !== undefined) {
      await this.options.semanticNodeRepository.update(targetId, {
        last_verified_at: applyingPatch.last_verified_at ?? this.clock.now(),
        confidence: applyingPatch.confidence,
      });
      return;
    }

    const current = await this.options.semanticNodeRepository.get(targetId);

    if (current === null) {
      return;
    }

    await this.options.semanticNodeRepository.update(targetId, {
      last_verified_at: this.clock.now(),
      confidence: clamp(current.confidence - 0.05, 0, 1),
    });
  }

  private async applyMisattributionResolution(
    item: ReviewQueueItem,
    decision: ResolvedReviewDecision,
  ): Promise<void> {
    if (decision.decision !== "accept") {
      return;
    }

    const parsed = misattributionRefsSchema.safeParse(item.refs);

    if (!parsed.success) {
      throwLegacyRepairRefsError(item.kind);
    }

    const refs = parsed.data;

    if (Object.keys(refs.patch).length === 0) {
      throwLegacyRepairRefsError(item.kind);
    }

    if (refs.target_type === "episode") {
      if (this.options.episodicRepository === undefined) {
        throw new SemanticError("Episode repository is required for misattribution repair", {
          code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
        });
      }

      const updated = await this.options.episodicRepository.update(refs.target_id, refs.patch);

      if (updated === null) {
        throw new SemanticError(`Unknown episode id for misattribution repair: ${refs.target_id}`, {
          code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
        });
      }

      return;
    }

    if (this.options.semanticNodeRepository === undefined) {
      throw new SemanticError("Semantic node repository is required for misattribution repair", {
        code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
      });
    }

    const updated = await this.options.semanticNodeRepository.update(refs.target_id, {
      ...refs.patch,
      ...(refs.patch.aliases === undefined ? {} : { replace_aliases: true }),
      ...(refs.patch.source_episode_ids === undefined ? {} : { replace_source_episode_ids: true }),
    });

    if (updated === null) {
      throw new SemanticError(
        `Unknown semantic node id for misattribution repair: ${refs.target_id}`,
        {
          code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
        },
      );
    }
  }

  private async applyTemporalDriftResolution(
    item: ReviewQueueItem,
    decision: ResolvedReviewDecision,
  ): Promise<void> {
    if (decision.decision !== "accept") {
      return;
    }

    const parsed = temporalDriftRefsSchema.safeParse(item.refs);

    if (!parsed.success) {
      throwLegacyRepairRefsError(item.kind);
    }

    const refs = parsed.data;

    if (refs.target_type === "semantic_edge") {
      this.closeSemanticEdgeFromReview(item, {
        edgeId: refs.target_id,
        validTo: refs.suggested_valid_to ?? this.clock.now(),
        byEdgeId: refs.by_edge_id,
        reason: refs.reason ?? item.reason,
      });
      return;
    }

    if (refs.target_type === "episode") {
      if (this.options.episodicRepository === undefined) {
        throw new SemanticError("Episode repository is required for temporal drift repair", {
          code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
        });
      }

      if (
        refs.corrected_start_time === undefined &&
        refs.corrected_end_time === undefined &&
        refs.patch_description === undefined
      ) {
        throwLegacyRepairRefsError(item.kind);
      }

      const patch = episodePatchSchema.parse({
        ...(refs.corrected_start_time === undefined
          ? {}
          : { start_time: refs.corrected_start_time }),
        ...(refs.corrected_end_time === undefined ? {} : { end_time: refs.corrected_end_time }),
        ...(refs.patch_description === undefined ? {} : { narrative: refs.patch_description }),
      });

      if (Object.keys(patch).length === 0) {
        throwLegacyRepairRefsError(item.kind);
      }

      const updated = await this.options.episodicRepository.update(refs.target_id, patch);

      if (updated === null) {
        throw new SemanticError(`Unknown episode id for temporal drift repair: ${refs.target_id}`, {
          code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
        });
      }

      return;
    }

    if (this.options.semanticNodeRepository === undefined) {
      throw new SemanticError("Semantic node repository is required for temporal drift repair", {
        code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
      });
    }

    if (refs.patch_description === undefined) {
      throwLegacyRepairRefsError(item.kind);
    }

    const updated = await this.options.semanticNodeRepository.update(refs.target_id, {
      ...(refs.patch_description === undefined ? {} : { description: refs.patch_description }),
      last_verified_at: this.clock.now(),
    });

    if (updated === null) {
      throw new SemanticError(
        `Unknown semantic node id for temporal drift repair: ${refs.target_id}`,
        {
          code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
        },
      );
    }
  }

  private async applyIdentityInconsistencyResolution(
    item: ReviewQueueItem,
    decision: ResolvedReviewDecision,
  ): Promise<void> {
    if (decision.decision !== "accept") {
      return;
    }

    const edgeClosureRefs = this.parseSemanticEdgeClosureRefs(item);

    if (edgeClosureRefs !== null) {
      this.closeSemanticEdgeFromReview(item, edgeClosureRefs);
      return;
    }

    const targetTypeResult = identityInconsistencyTargetTypeSchema.safeParse(item.refs.target_type);
    const repairOpResult = identityRepairOpSchema.safeParse(item.refs.repair_op);

    if (!targetTypeResult.success || !repairOpResult.success) {
      throwLegacyRepairRefsError(item.kind);
    }

    const targetType = targetTypeResult.data;
    const repairOp = repairOpResult.data;
    const proposedProvenance = parseReviewProvenance(item.refs);
    const evidenceProvenance = parseEvidenceProvenance(item.refs);
    const patch =
      item.refs.patch !== undefined && item.refs.patch !== null ? (item.refs.patch as unknown) : {};

    switch (targetType) {
      case "value": {
        const targetId = valueIdSchema.parse(item.refs.target_id);

        if (repairOp === "reinforce") {
          if (evidenceProvenance === null) {
            throwLegacyRepairRefsError(item.kind);
          }

          if (this.options.valuesRepository === undefined) {
            throw new SemanticError(
              "Values repository is required for value reinforcement repair",
              {
                code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
              },
            );
          }

          this.options.valuesRepository.reinforce(targetId, evidenceProvenance, this.clock.now());
          return;
        }

        if (repairOp === "contradict") {
          if (evidenceProvenance === null) {
            throwLegacyRepairRefsError(item.kind);
          }

          if (this.options.valuesRepository === undefined) {
            throw new SemanticError(
              "Values repository is required for value contradiction repair",
              {
                code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
              },
            );
          }

          this.options.valuesRepository.recordContradiction({
            valueId: targetId,
            provenance: evidenceProvenance,
            timestamp: this.clock.now(),
          });
          return;
        }

        if (this.options.identityService === undefined) {
          throw new SemanticError("Identity service is required for identity patch repair", {
            code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
          });
        }

        if (
          item.refs.patch === undefined ||
          item.refs.patch === null ||
          typeof item.refs.patch !== "object" ||
          Array.isArray(item.refs.patch) ||
          Object.keys(item.refs.patch as Record<string, unknown>).length === 0
        ) {
          throwLegacyRepairRefsError(item.kind);
        }

        const result = this.options.identityService.updateValue(
          targetId,
          patch,
          proposedProvenance,
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );

        if (result.status !== "applied") {
          throw new SemanticError(`Identity patch for value ${targetId} still requires review`, {
            code: "IDENTITY_REVIEW_REQUIRED",
          });
        }
        return;
      }
      case "trait": {
        const targetId = traitIdSchema.parse(item.refs.target_id);

        if (repairOp === "reinforce") {
          if (evidenceProvenance === null) {
            throwLegacyRepairRefsError(item.kind);
          }

          if (this.options.traitsRepository === undefined) {
            throw new SemanticError(
              "Traits repository is required for trait reinforcement repair",
              {
                code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
              },
            );
          }

          const current = this.options.traitsRepository.get(targetId);

          if (current === null) {
            throw new SemanticError(`Unknown trait id for reinforcement repair: ${targetId}`, {
              code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
            });
          }

          this.options.traitsRepository.reinforce({
            label: current.label,
            delta: 0.05,
            provenance: evidenceProvenance,
            timestamp: this.clock.now(),
          });
          return;
        }

        if (repairOp === "contradict") {
          if (evidenceProvenance === null) {
            throwLegacyRepairRefsError(item.kind);
          }

          if (this.options.traitsRepository === undefined) {
            throw new SemanticError(
              "Traits repository is required for trait contradiction repair",
              {
                code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
              },
            );
          }

          const current = this.options.traitsRepository.get(targetId);

          if (current === null) {
            throw new SemanticError(`Unknown trait id for contradiction repair: ${targetId}`, {
              code: "REVIEW_QUEUE_TARGET_NOT_FOUND",
            });
          }

          this.options.traitsRepository.recordContradiction({
            label: current.label,
            provenance: evidenceProvenance,
            timestamp: this.clock.now(),
          });
          return;
        }

        if (this.options.identityService === undefined) {
          throw new SemanticError("Identity service is required for identity patch repair", {
            code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
          });
        }

        if (
          item.refs.patch === undefined ||
          item.refs.patch === null ||
          typeof item.refs.patch !== "object" ||
          Array.isArray(item.refs.patch) ||
          Object.keys(item.refs.patch as Record<string, unknown>).length === 0
        ) {
          throwLegacyRepairRefsError(item.kind);
        }

        const result = this.options.identityService.updateTrait(
          targetId,
          patch,
          proposedProvenance,
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );

        if (result.status !== "applied") {
          throw new SemanticError(`Identity patch for trait ${targetId} still requires review`, {
            code: "IDENTITY_REVIEW_REQUIRED",
          });
        }
        return;
      }
      case "commitment": {
        const targetId = commitmentIdSchema.parse(item.refs.target_id);

        if (repairOp !== "patch") {
          throw new SemanticError(
            `Repair op "${repairOp}" is unsupported for commitment identity inconsistencies`,
            {
              code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
            },
          );
        }

        if (this.options.identityService === undefined) {
          throw new SemanticError("Identity service is required for identity patch repair", {
            code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
          });
        }

        if (
          item.refs.patch === undefined ||
          item.refs.patch === null ||
          typeof item.refs.patch !== "object" ||
          Array.isArray(item.refs.patch) ||
          Object.keys(item.refs.patch as Record<string, unknown>).length === 0
        ) {
          throwLegacyRepairRefsError(item.kind);
        }

        const result = this.options.identityService.updateCommitment(
          targetId,
          patch,
          proposedProvenance,
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );

        if (result.status !== "applied") {
          throw new SemanticError(
            `Identity patch for commitment ${targetId} still requires review`,
            {
              code: "IDENTITY_REVIEW_REQUIRED",
            },
          );
        }
        return;
      }
      case "goal": {
        const targetId = goalIdSchema.parse(item.refs.target_id);

        if (repairOp !== "patch") {
          throw new SemanticError(
            `Repair op "${repairOp}" is unsupported for goal identity inconsistencies`,
            {
              code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
            },
          );
        }

        if (this.options.identityService === undefined) {
          throw new SemanticError("Identity service is required for identity patch repair", {
            code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
          });
        }

        if (
          item.refs.patch === undefined ||
          item.refs.patch === null ||
          typeof item.refs.patch !== "object" ||
          Array.isArray(item.refs.patch) ||
          Object.keys(item.refs.patch as Record<string, unknown>).length === 0
        ) {
          throwLegacyRepairRefsError(item.kind);
        }

        const result = this.options.identityService.updateGoal(
          targetId,
          patch,
          proposedProvenance,
          {
            throughReview: true,
            reason: item.reason,
            reviewItemId: item.id,
          },
        );

        if (result.status !== "applied") {
          throw new SemanticError(`Identity patch for goal ${targetId} still requires review`, {
            code: "IDENTITY_REVIEW_REQUIRED",
          });
        }
        return;
      }
      case "autobiographical_period": {
        const targetId = autobiographicalPeriodIdSchema.parse(item.refs.target_id);

        if (repairOp !== "patch") {
          throw new SemanticError(
            `Repair op "${repairOp}" is unsupported for autobiographical period inconsistencies`,
            {
              code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
            },
          );
        }

        if (this.options.identityService === undefined) {
          throw new SemanticError("Identity service is required for identity patch repair", {
            code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
          });
        }

        if (
          item.refs.patch === undefined ||
          item.refs.patch === null ||
          typeof item.refs.patch !== "object" ||
          Array.isArray(item.refs.patch) ||
          Object.keys(item.refs.patch as Record<string, unknown>).length === 0
        ) {
          throwLegacyRepairRefsError(item.kind);
        }

        const applyPeriodPatch = () => {
          const result = this.options.identityService!.updatePeriod(
            targetId,
            patch,
            proposedProvenance,
            {
              throughReview: true,
              reason: item.reason,
              reviewItemId: item.id,
            },
          );

          if (result.status !== "applied") {
            throw new SemanticError(
              `Identity patch for autobiographical period ${targetId} still requires review`,
              {
                code: "IDENTITY_REVIEW_REQUIRED",
              },
            );
          }
        };
        const nextPeriodPayload =
          item.refs.next_period_open_payload === undefined
            ? null
            : autobiographicalPeriodSchema.parse(item.refs.next_period_open_payload);

        if (nextPeriodPayload === null) {
          applyPeriodPatch();
          return;
        }

        if (this.options.autobiographicalRepository === undefined) {
          throw new SemanticError(
            "Autobiographical repository is required for period rollover repair",
            {
              code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
            },
          );
        }

        this.options.autobiographicalRepository.runInTransaction(() => {
          applyPeriodPatch();
          this.options.autobiographicalRepository!.upsertPeriod(nextPeriodPayload);
        });
        return;
      }
    }
  }
}
