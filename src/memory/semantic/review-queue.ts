import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { SemanticError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
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
  semanticNodeRepository?: SemanticNodeRepository;
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
const LIFECYCLE_REVIEW_RESOLUTIONS = new Set<ReviewResolution>(["accept", "reject", "dismiss"]);
const CORRECTION_REVIEW_RESOLUTIONS = new Set<ReviewResolution>(["accept", "reject"]);

function isResolutionCompatible(kind: ReviewKind, resolution: ReviewResolution): boolean {
  switch (kind) {
    case "correction":
      return CORRECTION_REVIEW_RESOLUTIONS.has(resolution);
    case "contradiction":
    case "duplicate":
    case "new_insight":
      return SEMANTIC_REVIEW_RESOLUTIONS.has(resolution);
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

    if (
      this.options.semanticNodeRepository === undefined ||
      (decision !== "supersede" && decision !== "invalidate")
    ) {
      return;
    }

    const refs = item.refs;
    const rawNodeIds = refs.node_ids;

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
}
