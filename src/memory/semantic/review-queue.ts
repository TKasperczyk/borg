import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { SemanticError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  parseReviewProvenance,
  provenanceSchema,
  type Provenance,
} from "../common/provenance.js";
import {
  EpisodicRepository,
  episodeIdSchema,
  episodePatchSchema,
} from "../episodic/index.js";
import { type IdentityService } from "../identity/index.js";
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
import type { SemanticNodeRepository } from "./repository.js";
import { semanticNodeIdSchema } from "./types.js";

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
  valuesRepository?: ValuesRepository;
  goalsRepository?: GoalsRepository;
  traitsRepository?: TraitsRepository;
  autobiographicalRepository?: AutobiographicalRepository;
  commitmentRepository?: CommitmentRepository;
  identityService?: IdentityService;
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
const NEW_INSIGHT_REVIEW_RESOLUTIONS = new Set<ReviewResolution>(["accept", "invalidate", "dismiss"]);
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
]);

const identityInconsistencyTargetTypeSchema = z.enum([
  "trait",
  "value",
  "commitment",
  "goal",
  "autobiographical_period",
]);
const identityRepairOpSchema = z.enum(["reinforce", "contradict", "patch"]);

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
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

  async resolve(itemId: number, decision: ReviewResolution): Promise<ReviewQueueItem | null> {
    reviewResolutionSchema.parse(decision);
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

    if (!isResolutionCompatible(item.kind, decision)) {
      throw new SemanticError(
        `Resolution "${decision}" is incompatible with review kind "${item.kind}"`,
        {
          code: "REVIEW_QUEUE_RESOLUTION_INVALID",
        },
      );
    }

    await this.applyResolution(item, decision);
    const resolvedAt = this.clock.now();
    this.db
      .prepare("UPDATE review_queue SET resolved_at = ?, resolution = ? WHERE id = ?")
      .run(resolvedAt, decision, itemId);

    return {
      ...item,
      resolved_at: resolvedAt,
      resolution: decision,
    };
  }

  private async applyResolution(item: ReviewQueueItem, decision: ReviewResolution): Promise<void> {
    if (item.kind === "correction") {
      if (decision === "accept") {
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
    decision: ReviewResolution,
  ): Promise<void> {
    if (
      this.options.semanticNodeRepository === undefined ||
      (decision !== "supersede" && decision !== "invalidate")
    ) {
      return;
    }

    const rawNodeIds = item.refs.node_ids;

    if (!Array.isArray(rawNodeIds) || rawNodeIds.length < 2) {
      return;
    }

    const parsedNodeIds = rawNodeIds.map((value) => semanticNodeIdSchema.parse(value));
    const nodes = await this.options.semanticNodeRepository.getMany(parsedNodeIds, {
      includeArchived: true,
    });
    const first = nodes[0];
    const second = nodes[1];

    if (first === null || first === undefined || second === null || second === undefined) {
      return;
    }

    const winner = first.confidence >= second.confidence ? first : second;
    const loser = winner.id === first.id ? second : first;

    if (decision === "supersede") {
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
    decision: ReviewResolution,
  ): Promise<void> {
    if (
      this.options.semanticNodeRepository === undefined ||
      (decision !== "invalidate" && decision !== "accept")
    ) {
      return;
    }

    const rawNodeIds = item.refs.node_ids;

    if (!Array.isArray(rawNodeIds) || rawNodeIds.length < 1) {
      return;
    }

    const nodeId = semanticNodeIdSchema.parse(rawNodeIds[0]);

    if (decision === "accept") {
      const current = await this.options.semanticNodeRepository.get(nodeId);

      if (current === null) {
        return;
      }

      await this.options.semanticNodeRepository.update(nodeId, {
        confidence: clamp(current.confidence + 0.1, 0, 1),
        last_verified_at: this.clock.now(),
      });
      return;
    }

    await this.options.semanticNodeRepository.update(nodeId, {
      archived: true,
    });
  }

  private async applyStaleResolution(
    item: ReviewQueueItem,
    decision: ReviewResolution,
  ): Promise<void> {
    if (decision !== "accept" || this.options.semanticNodeRepository === undefined) {
      return;
    }

    const rawTargetId =
      item.refs.target_type === "semantic_node" ? item.refs.target_id : item.refs.node_id;

    if (typeof rawTargetId !== "string") {
      return;
    }

    const targetId = semanticNodeIdSchema.parse(rawTargetId);
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
    decision: ReviewResolution,
  ): Promise<void> {
    if (decision !== "accept") {
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

    const updated = await this.options.semanticNodeRepository.update(refs.target_id, refs.patch);

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
    decision: ReviewResolution,
  ): Promise<void> {
    if (decision !== "accept") {
      return;
    }

    const parsed = temporalDriftRefsSchema.safeParse(item.refs);

    if (!parsed.success) {
      throwLegacyRepairRefsError(item.kind);
    }

    const refs = parsed.data;

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
    decision: ReviewResolution,
  ): Promise<void> {
    if (decision !== "accept") {
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
            throw new SemanticError("Values repository is required for value reinforcement repair", {
              code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
            });
          }

          this.options.valuesRepository.reinforce(targetId, evidenceProvenance, this.clock.now());
          return;
        }

        if (repairOp === "contradict") {
          if (evidenceProvenance === null) {
            throwLegacyRepairRefsError(item.kind);
          }

          if (this.options.valuesRepository === undefined) {
            throw new SemanticError("Values repository is required for value contradiction repair", {
              code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
            });
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

        const result = this.options.identityService.updateValue(targetId, patch, proposedProvenance, {
          throughReview: true,
          reason: item.reason,
          reviewItemId: item.id,
        });

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
            throw new SemanticError("Traits repository is required for trait reinforcement repair", {
              code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
            });
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
            throw new SemanticError("Traits repository is required for trait contradiction repair", {
              code: "REVIEW_QUEUE_REPAIR_UNSUPPORTED",
            });
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

        const result = this.options.identityService.updateTrait(targetId, patch, proposedProvenance, {
          throughReview: true,
          reason: item.reason,
          reviewItemId: item.id,
        });

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
          throw new SemanticError(`Identity patch for commitment ${targetId} still requires review`, {
            code: "IDENTITY_REVIEW_REQUIRED",
          });
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

        const result = this.options.identityService.updateGoal(targetId, patch, proposedProvenance, {
          throughReview: true,
          reason: item.reason,
          reviewItemId: item.id,
        });

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
