import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { SemanticError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  EpisodicRepository,
  episodeIdSchema,
  isEpisodeVisibleToAudience,
  type Episode,
} from "../episodic/index.js";
import { type IdentityEventRepository, type IdentityService } from "../identity/index.js";
import {
  AutobiographicalRepository,
  GoalsRepository,
  TraitsRepository,
  ValuesRepository,
} from "../self/index.js";
import { CommitmentRepository, entityIdSchema } from "../commitments/index.js";
import { type EntityId } from "../../util/ids.js";
import type { SemanticEdgeRepository, SemanticNodeRepository } from "./repository.js";
import {
  semanticEdgeIdSchema,
  semanticNodeIdSchema,
  type SemanticEdge,
  type SemanticNode,
} from "./types.js";

export const REVIEW_KINDS = [
  "contradiction",
  "duplicate",
  "new_insight",
  "misattribution",
  "temporal_drift",
  "identity_inconsistency",
  "correction",
  "belief_revision",
  "skill_split",
] as const;
export const REVIEW_RESOLUTIONS = [
  "keep_both",
  "supersede",
  "invalidate",
  "dismiss",
  "accept",
  "reject",
  "keep",
  "weaken",
  "archive_node",
  "invalidate_edge",
] as const;

export const reviewKindSchema = z.enum(REVIEW_KINDS);
export const reviewResolutionSchema = z.enum(REVIEW_RESOLUTIONS);
export const reviewResolutionInputSchema = z.union([
  reviewResolutionSchema,
  z
    .object({
      decision: reviewResolutionSchema,
      winner_node_id: semanticNodeIdSchema.optional(),
      reason: z.string().min(1).optional(),
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
export type ReviewApplyDecision = {
  decision: ReviewResolution;
  winner_node_id?: z.infer<typeof semanticNodeIdSchema>;
  reason?: string;
};
type ResolvedReviewDecision = ReviewApplyDecision;

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
  handlers?: ReviewQueueHandlerRegistry;
  onEnqueue?: (item: ReviewQueueItem, input: ReviewQueueInsertInput) => void | Promise<void>;
  onEnqueueError?: (
    error: unknown,
    item: ReviewQueueItem,
    input: ReviewQueueInsertInput,
  ) => void | Promise<void>;
};

const beliefRevisionRefsSchema = z.discriminatedUnion("target_type", [
  z
    .object({
      target_type: z.literal("semantic_node"),
      target_id: semanticNodeIdSchema,
      invalidated_edge_id: semanticEdgeIdSchema,
      dependency_path_edge_ids: z.array(semanticEdgeIdSchema),
      surviving_support_edge_ids: z.array(semanticEdgeIdSchema),
      evidence_episode_ids: z.array(episodeIdSchema),
      audience_entity_id: entityIdSchema.nullable().optional(),
    })
    .passthrough(),
  z
    .object({
      target_type: z.literal("semantic_edge"),
      target_id: semanticEdgeIdSchema,
      invalidated_edge_id: semanticEdgeIdSchema,
      dependency_path_edge_ids: z.array(semanticEdgeIdSchema),
      surviving_support_edge_ids: z.array(semanticEdgeIdSchema),
      evidence_episode_ids: z.array(episodeIdSchema),
      audience_entity_id: entityIdSchema.nullable().optional(),
    })
    .passthrough(),
]);

export type ReviewTransactionScope = "sqlite" | "cross_store_applying_state" | "external";

export type ReviewApplyOutcome = {
  finalResolution?: ReviewApplyDecision;
  refs?: Record<string, unknown>;
};

export type ReviewHandlerContext = {
  db: SqliteDatabase;
  clock: Clock;
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
};

export type ReviewApplyingStateSpec<TRefs, TState> = {
  key?: string;
  schema: z.ZodType<TState>;
  prepare: (input: {
    item: ReviewQueueItem;
    refs: TRefs;
    resolution: ReviewApplyDecision;
    ctx: ReviewHandlerContext;
  }) => Promise<TState> | TState;
  matches: (state: TState, resolution: ReviewApplyDecision) => boolean;
};

export type ReviewQueueHandler<K extends ReviewKind, TRefs, TState = never> = {
  kind: K;
  refsSchema: z.ZodType<TRefs>;
  allowedResolutions: ReadonlySet<ReviewResolution>;
  transactionScope: (input: {
    item: ReviewQueueItem;
    refs: TRefs;
    resolution: ReviewApplyDecision;
    ctx: ReviewHandlerContext;
  }) => ReviewTransactionScope;
  applyingState?: ReviewApplyingStateSpec<TRefs, TState>;
  apply: (input: {
    item: ReviewQueueItem;
    refs: TRefs;
    resolution: ReviewApplyDecision;
    applyingState: TState | null;
    ctx: ReviewHandlerContext;
  }) => Promise<ReviewApplyOutcome | void> | ReviewApplyOutcome | void;
};

export class ReviewQueueHandlerRegistry {
  private readonly handlers = new Map<
    ReviewKind,
    ReviewQueueHandler<ReviewKind, unknown, unknown>
  >();

  register<K extends ReviewKind, TRefs, TState>(
    handler: ReviewQueueHandler<K, TRefs, TState>,
  ): void {
    this.handlers.set(
      handler.kind,
      handler as unknown as ReviewQueueHandler<ReviewKind, unknown, unknown>,
    );
  }

  get(kind: ReviewKind): ReviewQueueHandler<ReviewKind, unknown, unknown> | null {
    return this.handlers.get(kind) ?? null;
  }
}

const REVIEW_APPLYING_REF_KEY = "__borg_resolution_applying";
type BeliefRevisionRefs = z.infer<typeof beliefRevisionRefsSchema>;
export type BeliefRevisionReasonCode =
  | "evidence_invalidated"
  | "support_chain_collapsed"
  | "manual_review";
export type BeliefRevisionTarget =
  | {
      target_type: "semantic_node";
      target_id: SemanticNode["id"];
    }
  | {
      target_type: "semantic_edge";
      target_id: SemanticEdge["id"];
    };
export type OpenBeliefRevisionStatus = BeliefRevisionTarget & {
  review_id: ReviewQueueItem["id"];
  reason: string;
  reason_code: BeliefRevisionReasonCode;
  created_at: number;
  invalidated_edge_id: SemanticEdge["id"];
  evidence_episode_ids: Episode["id"][];
};
export type BeliefRevisionVisibilityOptions = {
  audienceEntityId?: EntityId | null;
  crossAudience?: boolean;
  episodicRepository?: Pick<EpisodicRepository, "getMany">;
};
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

function beliefRevisionTargetKey(target: BeliefRevisionTarget): string {
  return JSON.stringify([target.target_type, target.target_id]);
}

function beliefRevisionReasonCode(refs: BeliefRevisionRefs): BeliefRevisionReasonCode {
  if (refs.target_type === "semantic_node" && refs.surviving_support_edge_ids.length === 0) {
    return "support_chain_collapsed";
  }

  if (refs.invalidated_edge_id !== undefined) {
    return "evidence_invalidated";
  }

  return "manual_review";
}

function uniqueEpisodeIds(ids: readonly Episode["id"][]): Episode["id"][] {
  return [...new Set(ids)];
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
  private readonly handlers: ReviewQueueHandlerRegistry;
  private readonly pendingEnqueueHooks = new Set<Promise<void>>();

  constructor(private readonly options: ReviewQueueRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.handlers = options.handlers ?? new ReviewQueueHandlerRegistry();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private get handlerContext(): ReviewHandlerContext {
    return {
      db: this.db,
      clock: this.clock,
      episodicRepository: this.options.episodicRepository,
      semanticNodeRepository: this.options.semanticNodeRepository,
      semanticEdgeRepository: this.options.semanticEdgeRepository,
      valuesRepository: this.options.valuesRepository,
      goalsRepository: this.options.goalsRepository,
      traitsRepository: this.options.traitsRepository,
      autobiographicalRepository: this.options.autobiographicalRepository,
      commitmentRepository: this.options.commitmentRepository,
      identityService: this.options.identityService,
      identityEventRepository: this.options.identityEventRepository,
    };
  }

  private reportEnqueueHookError(
    error: unknown,
    item: ReviewQueueItem,
    input: ReviewQueueInsertInput,
  ): void {
    try {
      void Promise.resolve(this.options.onEnqueueError?.(error, item, input)).catch(() => {
        // Best-effort hook error reporting only.
      });
    } catch {
      // Best-effort hook error reporting only.
    }
  }

  private trackEnqueueHook(task: Promise<void>): void {
    this.pendingEnqueueHooks.add(task);
    void task.finally(() => {
      this.pendingEnqueueHooks.delete(task);
    });
  }

  async flushEnqueueHooks(): Promise<void> {
    // Online turns let review-derived open-question extraction run in the
    // background. Maintenance drains after each process, and lifecycle close
    // drains before SQLite closes so pending hook writes survive shutdown.
    while (this.pendingEnqueueHooks.size > 0) {
      await Promise.all([...this.pendingEnqueueHooks]);
    }
  }

  registerHandler<K extends ReviewKind, TRefs, TState>(
    handler: ReviewQueueHandler<K, TRefs, TState>,
  ): void {
    this.handlers.register(handler);
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
      const hookResult = this.options.onEnqueue?.(item, input);

      if (hookResult !== undefined) {
        this.trackEnqueueHook(
          Promise.resolve(hookResult).catch((error) => {
            this.reportEnqueueHookError(error, item, input);
          }),
        );
      }
    } catch (error) {
      this.reportEnqueueHookError(error, item, input);
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

  get(itemId: number): ReviewQueueItem | null {
    const row = this.db.prepare("SELECT * FROM review_queue WHERE id = ?").get(itemId) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapReviewRow(row);
  }

  private beliefRevisionEvidenceEpisodeIds(refs: BeliefRevisionRefs): Episode["id"][] {
    const invalidatedEdge = this.options.semanticEdgeRepository?.getEdge(refs.invalidated_edge_id);

    return uniqueEpisodeIds([
      ...refs.evidence_episode_ids,
      ...(invalidatedEdge?.evidence_episode_ids ?? []),
    ]);
  }

  private async isBeliefRevisionVisible(
    refs: BeliefRevisionRefs,
    status: Pick<OpenBeliefRevisionStatus, "evidence_episode_ids">,
    options: BeliefRevisionVisibilityOptions,
  ): Promise<boolean> {
    if (options.crossAudience === true) {
      return true;
    }

    if (refs.audience_entity_id !== undefined && refs.audience_entity_id !== null) {
      return refs.audience_entity_id === (options.audienceEntityId ?? null);
    }

    if (status.evidence_episode_ids.length === 0) {
      return true;
    }

    const episodicRepository = options.episodicRepository ?? this.options.episodicRepository;

    if (episodicRepository === undefined) {
      return false;
    }

    const episodes = (await episodicRepository.getMany(status.evidence_episode_ids)).filter(
      (episode): episode is Episode => episode !== null,
    );

    return (
      episodes.length === status.evidence_episode_ids.length &&
      episodes.every((episode) =>
        isEpisodeVisibleToAudience(episode, options.audienceEntityId, {
          crossAudience: false,
        }),
      )
    );
  }

  async listOpenBeliefRevisionsByTarget(
    targets: readonly BeliefRevisionTarget[],
    options: BeliefRevisionVisibilityOptions = {},
  ): Promise<Map<string, OpenBeliefRevisionStatus>> {
    if (targets.length === 0) {
      return new Map();
    }

    const targetsByKey = new Map(
      targets.map((target) => [beliefRevisionTargetKey(target), target]),
    );
    const whereTargets = [...targetsByKey.values()];
    const targetFilter = whereTargets
      .map(
        () => "(json_extract(refs, '$.target_type') = ? AND json_extract(refs, '$.target_id') = ?)",
      )
      .join(" OR ");
    const targetValues = whereTargets.flatMap((target) => [target.target_type, target.target_id]);
    const rows = this.db
      .prepare(
        `
          SELECT id, kind, refs, reason, created_at, resolved_at, resolution
          FROM review_queue
          WHERE kind = 'belief_revision'
            AND resolved_at IS NULL
            AND (${targetFilter})
          ORDER BY created_at DESC, id DESC
        `,
      )
      .all(...targetValues) as Record<string, unknown>[];
    const results = new Map<string, OpenBeliefRevisionStatus>();

    for (const item of rows.map((row) => mapReviewRow(row))) {
      const parsed = beliefRevisionRefsSchema.safeParse(item.refs);

      if (!parsed.success) {
        continue;
      }

      const refs: BeliefRevisionRefs = parsed.data;
      const target = {
        target_type: refs.target_type,
        target_id: refs.target_id,
      } as BeliefRevisionTarget;
      const key = beliefRevisionTargetKey(target);

      if (!targetsByKey.has(key) || results.has(key)) {
        continue;
      }

      const status: OpenBeliefRevisionStatus = {
        ...target,
        review_id: item.id,
        reason: item.reason,
        reason_code: beliefRevisionReasonCode(refs),
        created_at: item.created_at,
        invalidated_edge_id: refs.invalidated_edge_id,
        evidence_episode_ids: this.beliefRevisionEvidenceEpisodeIds(refs),
      };

      if (await this.isBeliefRevisionVisible(refs, status, options)) {
        results.set(key, status);
      }
    }

    return results;
  }

  async getOpenBeliefRevisionForTarget(
    target: BeliefRevisionTarget,
    options: BeliefRevisionVisibilityOptions = {},
  ): Promise<OpenBeliefRevisionStatus | null> {
    const results = await this.listOpenBeliefRevisionsByTarget([target], options);

    return results.get(beliefRevisionTargetKey(target)) ?? null;
  }

  delete(itemId: number): boolean {
    const result = this.db.prepare("DELETE FROM review_queue WHERE id = ?").run(itemId);
    return result.changes > 0;
  }

  private refsWithoutApplyingState(
    refs: Record<string, unknown>,
    key = REVIEW_APPLYING_REF_KEY,
  ): Record<string, unknown> {
    const next = { ...refs };
    delete next[key];
    return next;
  }

  private markResolved(
    itemId: number,
    resolution: ResolvedReviewDecision,
    resolvedAt: number,
    refs: Record<string, unknown>,
    applyingStateKey = REVIEW_APPLYING_REF_KEY,
  ): void {
    this.db
      .prepare("UPDATE review_queue SET refs = ?, resolved_at = ?, resolution = ? WHERE id = ?")
      .run(
        serializeJsonValue(this.refsWithoutApplyingState(refs, applyingStateKey)),
        resolvedAt,
        resolution.decision,
        itemId,
      );
  }

  private handlerApplyingStateKey(
    handler: ReviewQueueHandler<ReviewKind, unknown, unknown>,
  ): string {
    return handler.applyingState?.key ?? REVIEW_APPLYING_REF_KEY;
  }

  private parseHandlerRefs(
    handler: ReviewQueueHandler<ReviewKind, unknown, unknown>,
    item: ReviewQueueItem,
  ): unknown {
    return handler.refsSchema.parse(
      this.refsWithoutApplyingState(item.refs, this.handlerApplyingStateKey(handler)),
    );
  }

  private getHandlerApplyingState(
    handler: ReviewQueueHandler<ReviewKind, unknown, unknown>,
    item: ReviewQueueItem,
  ): unknown | null {
    if (handler.applyingState === undefined) {
      return null;
    }

    const key = this.handlerApplyingStateKey(handler);

    if (!Object.hasOwn(item.refs, key)) {
      return null;
    }

    return handler.applyingState.schema.parse(item.refs[key]);
  }

  private async ensureHandlerApplyingState(
    handler: ReviewQueueHandler<ReviewKind, unknown, unknown>,
    item: ReviewQueueItem,
    refs: unknown,
    resolution: ResolvedReviewDecision,
  ): Promise<{ item: ReviewQueueItem; applyingState: unknown }> {
    if (handler.applyingState === undefined) {
      throw new SemanticError("Review handler did not declare applying state", {
        code: "REVIEW_QUEUE_APPLYING_STATE_UNSUPPORTED",
        cause: { itemId: item.id, kind: item.kind },
      });
    }

    const existing = this.getHandlerApplyingState(handler, item);

    if (existing !== null) {
      if (!handler.applyingState.matches(existing, resolution)) {
        throw new SemanticError("Review item is already applying a different resolution", {
          code: "REVIEW_QUEUE_RESOLUTION_IN_PROGRESS",
          cause: { itemId: item.id },
        });
      }

      return { item, applyingState: existing };
    }

    const applyingState = await handler.applyingState.prepare({
      item,
      refs,
      resolution,
      ctx: this.handlerContext,
    });
    const key = this.handlerApplyingStateKey(handler);
    const nextRefs = {
      ...item.refs,
      [key]: applyingState,
    };

    this.db
      .prepare("UPDATE review_queue SET refs = ? WHERE id = ? AND resolved_at IS NULL")
      .run(serializeJsonValue(nextRefs), item.id);

    return {
      item: {
        ...item,
        refs: nextRefs,
      },
      applyingState,
    };
  }

  private async resolveWithHandlerSqliteTransaction(
    handler: ReviewQueueHandler<ReviewKind, unknown, unknown>,
    item: ReviewQueueItem,
    refs: unknown,
    resolution: ResolvedReviewDecision,
  ): Promise<ReviewQueueItem> {
    const resolvedAt = this.clock.now();
    const applyingStateKey = this.handlerApplyingStateKey(handler);

    this.db.exec("BEGIN IMMEDIATE");

    try {
      const outcome = await handler.apply({
        item,
        refs,
        resolution,
        applyingState: null,
        ctx: this.handlerContext,
      });
      const finalResolution = outcome?.finalResolution ?? resolution;
      const nextRefs = outcome?.refs ?? item.refs;

      this.markResolved(item.id, finalResolution, resolvedAt, nextRefs, applyingStateKey);
      this.db.exec("COMMIT");

      return {
        ...item,
        refs: this.refsWithoutApplyingState(nextRefs, applyingStateKey),
        resolved_at: resolvedAt,
        resolution: finalResolution.decision,
      };
    } catch (error) {
      try {
        this.db.exec("ROLLBACK");
      } catch {
        // Keep the original failure.
      }

      throw error;
    }
  }

  private async resolveWithHandlerApplyingState(
    handler: ReviewQueueHandler<ReviewKind, unknown, unknown>,
    item: ReviewQueueItem,
    refs: unknown,
    resolution: ResolvedReviewDecision,
  ): Promise<ReviewQueueItem> {
    const { item: applyingItem, applyingState } = await this.ensureHandlerApplyingState(
      handler,
      item,
      refs,
      resolution,
    );
    const outcome = await handler.apply({
      item: applyingItem,
      refs,
      resolution,
      applyingState,
      ctx: this.handlerContext,
    });
    const resolvedAt = this.clock.now();
    const finalResolution = outcome?.finalResolution ?? resolution;
    const nextRefs = outcome?.refs ?? applyingItem.refs;
    const applyingStateKey = this.handlerApplyingStateKey(handler);

    this.markResolved(applyingItem.id, finalResolution, resolvedAt, nextRefs, applyingStateKey);

    return {
      ...applyingItem,
      refs: this.refsWithoutApplyingState(nextRefs, applyingStateKey),
      resolved_at: resolvedAt,
      resolution: finalResolution.decision,
    };
  }

  private async resolveWithExternalHandler(
    handler: ReviewQueueHandler<ReviewKind, unknown, unknown>,
    item: ReviewQueueItem,
    refs: unknown,
    resolution: ResolvedReviewDecision,
  ): Promise<ReviewQueueItem> {
    const resolvedAt = this.clock.now();
    const outcome = await handler.apply({
      item,
      refs,
      resolution,
      applyingState: null,
      ctx: this.handlerContext,
    });
    const finalResolution = outcome?.finalResolution ?? resolution;
    const nextRefs = outcome?.refs ?? item.refs;
    const applyingStateKey = this.handlerApplyingStateKey(handler);

    this.markResolved(item.id, finalResolution, resolvedAt, nextRefs, applyingStateKey);

    return {
      ...item,
      refs: this.refsWithoutApplyingState(nextRefs, applyingStateKey),
      resolved_at: resolvedAt,
      resolution: finalResolution.decision,
    };
  }

  private async resolveWithRegisteredHandler(
    handler: ReviewQueueHandler<ReviewKind, unknown, unknown>,
    item: ReviewQueueItem,
    resolution: ResolvedReviewDecision,
  ): Promise<ReviewQueueItem> {
    if (!handler.allowedResolutions.has(resolution.decision)) {
      throw new SemanticError(
        `Resolution "${resolution.decision}" is incompatible with review kind "${item.kind}"`,
        {
          code: "REVIEW_QUEUE_RESOLUTION_INVALID",
        },
      );
    }

    const refs = this.parseHandlerRefs(handler, item);
    const scope = handler.transactionScope({
      item,
      refs,
      resolution,
      ctx: this.handlerContext,
    });

    if (scope === "sqlite") {
      return this.resolveWithHandlerSqliteTransaction(handler, item, refs, resolution);
    }

    if (scope === "cross_store_applying_state") {
      return this.resolveWithHandlerApplyingState(handler, item, refs, resolution);
    }

    return this.resolveWithExternalHandler(handler, item, refs, resolution);
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

    const handler = this.handlers.get(item.kind);

    if (handler === null) {
      throw new SemanticError(`No review queue handler registered for kind "${item.kind}"`, {
        code: "REVIEW_QUEUE_HANDLER_UNREGISTERED",
        cause: { kind: item.kind },
      });
    }

    return this.resolveWithRegisteredHandler(handler, item, resolution);
  }
}
