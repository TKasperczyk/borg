import { createHash } from "node:crypto";

import { z } from "zod";

import { parseJsonArray, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { sortStrings } from "../../util/collections.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  PROMPT_SURFACE_BLOCKS,
  PROMPT_SURFACES,
  type PromptSurface,
  type PromptSurfaceBlock,
} from "./prompt-surface-registry.js";

export const PROMPT_SURFACE_CHANGES_DEFAULT_LIMIT = 10;
export const PROMPT_SURFACE_CHANGES_MAX_LIMIT = 20;

const PROMPT_SURFACE_VALUES = Object.values(PROMPT_SURFACES) as [
  PromptSurface,
  ...PromptSurface[],
];

const PROMPT_SURFACE_HISTORY_JSON_ARRAY_CODEC = {
  errorCode: "PROMPT_SURFACE_HISTORY_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse prompt surface history ${label}`,
} satisfies JsonArrayCodecOptions;

export const promptSurfacePlacementRecordSchema = z
  .object({
    block_id: z.string().min(1),
    surface: z.enum(PROMPT_SURFACE_VALUES),
    order: z.number().int().finite(),
  })
  .strict();

export const promptSurfaceProjectionSchema = z
  .object({
    block_ids: z.array(z.string().min(1)),
    surface_placements: z.array(promptSurfacePlacementRecordSchema),
  })
  .strict();

export const promptSurfaceSnapshotSchema = promptSurfaceProjectionSchema
  .extend({
    hash: z.string().length(64),
    observed_at: z.number().int().finite(),
  })
  .strict();

export const promptSurfaceCurrentSchema = promptSurfaceProjectionSchema
  .extend({
    hash: z.string().length(64),
    observed_at: z.number().int().finite().nullable(),
  })
  .strict();

export const promptSurfaceChangeRecordSchema = z
  .object({
    observed_at: z.number().int().finite(),
    from_hash: z.string().length(64).nullable(),
    to_hash: z.string().length(64),
    added_block_ids: z.array(z.string().min(1)),
    removed_block_ids: z.array(z.string().min(1)),
    added_surface_placements: z.array(promptSurfacePlacementRecordSchema),
    removed_surface_placements: z.array(promptSurfacePlacementRecordSchema),
  })
  .strict();

export type PromptSurfacePlacementRecord = z.infer<typeof promptSurfacePlacementRecordSchema>;
export type PromptSurfaceProjection = z.infer<typeof promptSurfaceProjectionSchema>;
export type PromptSurfaceSnapshot = z.infer<typeof promptSurfaceSnapshotSchema>;
export type PromptSurfaceCurrent = z.infer<typeof promptSurfaceCurrentSchema>;
export type PromptSurfaceChangeRecord = z.infer<typeof promptSurfaceChangeRecordSchema>;

export type PromptSurfaceObservationResult = {
  snapshot: PromptSurfaceSnapshot;
  change: PromptSurfaceChangeRecord | null;
  inserted: boolean;
};

export type PromptSurfaceHistoryListOptions = {
  limit?: number;
  /**
   * Exclusive cursor matching a prior change row's to_hash. Unknown cursors
   * return an empty change list rather than falling back to recent history.
   */
  sinceVersion?: string;
};

export type PromptSurfaceHistoryRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  blocks?: readonly PromptSurfaceBlock[];
};

type PromptSurfaceSnapshotRow = {
  hash: string;
  observed_at: number;
  block_ids: string;
  surface_placements: string;
};

type PromptSurfaceChangeRow = {
  id: number;
  observed_at: number;
  from_hash: string | null;
  to_hash: string;
  added_block_ids: string;
  removed_block_ids: string;
  added_surface_placements: string;
  removed_surface_placements: string;
};

function parseStringArray(value: string, label: string): string[] {
  const parsed = parseJsonArray<unknown>(
    value,
    label,
    PROMPT_SURFACE_HISTORY_JSON_ARRAY_CODEC,
  );
  const result = z.array(z.string().min(1)).safeParse(parsed);

  if (!result.success) {
    throw new StorageError(`Prompt surface history ${label} failed validation`, {
      cause: result.error,
      code: "PROMPT_SURFACE_HISTORY_ROW_INVALID",
    });
  }

  return result.data;
}

function parsePlacementArray(value: string, label: string): PromptSurfacePlacementRecord[] {
  const parsed = parseJsonArray<unknown>(
    value,
    label,
    PROMPT_SURFACE_HISTORY_JSON_ARRAY_CODEC,
  );
  const result = z.array(promptSurfacePlacementRecordSchema).safeParse(parsed);

  if (!result.success) {
    throw new StorageError(`Prompt surface history ${label} failed validation`, {
      cause: result.error,
      code: "PROMPT_SURFACE_HISTORY_ROW_INVALID",
    });
  }

  return result.data;
}

function mapSnapshotRow(row: PromptSurfaceSnapshotRow): PromptSurfaceSnapshot {
  const parsed = promptSurfaceSnapshotSchema.safeParse({
    hash: row.hash,
    observed_at: Number(row.observed_at),
    block_ids: parseStringArray(row.block_ids, "block_ids"),
    surface_placements: parsePlacementArray(row.surface_placements, "surface_placements"),
  });

  if (!parsed.success) {
    throw new StorageError("Prompt surface snapshot row failed validation", {
      cause: parsed.error,
      code: "PROMPT_SURFACE_HISTORY_ROW_INVALID",
    });
  }

  return parsed.data;
}

function mapChangeRow(row: PromptSurfaceChangeRow): PromptSurfaceChangeRecord {
  const parsed = promptSurfaceChangeRecordSchema.safeParse({
    observed_at: Number(row.observed_at),
    from_hash: row.from_hash,
    to_hash: row.to_hash,
    added_block_ids: parseStringArray(row.added_block_ids, "added_block_ids"),
    removed_block_ids: parseStringArray(row.removed_block_ids, "removed_block_ids"),
    added_surface_placements: parsePlacementArray(
      row.added_surface_placements,
      "added_surface_placements",
    ),
    removed_surface_placements: parsePlacementArray(
      row.removed_surface_placements,
      "removed_surface_placements",
    ),
  });

  if (!parsed.success) {
    throw new StorageError("Prompt surface change row failed validation", {
      cause: parsed.error,
      code: "PROMPT_SURFACE_HISTORY_ROW_INVALID",
    });
  }

  return parsed.data;
}

function placementKey(placement: PromptSurfacePlacementRecord): string {
  return `${placement.block_id}\u0000${placement.surface}\u0000${placement.order}`;
}

function sortPlacements(
  placements: readonly PromptSurfacePlacementRecord[],
): PromptSurfacePlacementRecord[] {
  return [...placements].sort((left, right) => {
    const surfaceComparison = left.surface.localeCompare(right.surface);

    if (surfaceComparison !== 0) {
      return surfaceComparison;
    }

    const orderComparison = left.order - right.order;

    if (orderComparison !== 0) {
      return orderComparison;
    }

    return left.block_id.localeCompare(right.block_id);
  });
}

function diffStrings(left: readonly string[], right: readonly string[]): string[] {
  const rightSet = new Set(right);
  return sortStrings(left.filter((value) => !rightSet.has(value)));
}

function diffPlacements(
  left: readonly PromptSurfacePlacementRecord[],
  right: readonly PromptSurfacePlacementRecord[],
): PromptSurfacePlacementRecord[] {
  const rightKeys = new Set(right.map(placementKey));
  return sortPlacements(left.filter((placement) => !rightKeys.has(placementKey(placement))));
}

function normalizeLimit(limit: number | undefined): number {
  if (limit === undefined) {
    return PROMPT_SURFACE_CHANGES_DEFAULT_LIMIT;
  }

  if (!Number.isInteger(limit) || limit <= 0) {
    throw new StorageError("Prompt surface changes limit must be a positive integer", {
      code: "PROMPT_SURFACE_HISTORY_LIMIT_INVALID",
    });
  }

  return Math.min(limit, PROMPT_SURFACE_CHANGES_MAX_LIMIT);
}

export function buildPromptSurfaceProjection(
  blocks: readonly PromptSurfaceBlock[] = PROMPT_SURFACE_BLOCKS,
): PromptSurfaceProjection {
  const projection = {
    block_ids: sortStrings(blocks.map((block) => block.id)),
    surface_placements: sortPlacements(
      blocks.flatMap((block) =>
        block.surfaces.map((placement) => ({
          block_id: block.id,
          surface: placement.surface,
          order: placement.order,
        })),
      ),
    ),
  };

  return promptSurfaceProjectionSchema.parse(projection);
}

export function hashPromptSurfaceProjection(projection: PromptSurfaceProjection): string {
  return createHash("sha256").update(serializeJsonValue(projection)).digest("hex");
}

function buildChangeRecord(input: {
  observedAt: number;
  prior: PromptSurfaceSnapshot | null;
  snapshot: PromptSurfaceSnapshot;
}): PromptSurfaceChangeRecord {
  if (input.prior === null) {
    return {
      observed_at: input.observedAt,
      from_hash: null,
      to_hash: input.snapshot.hash,
      added_block_ids: [],
      removed_block_ids: [],
      added_surface_placements: [],
      removed_surface_placements: [],
    };
  }

  return {
    observed_at: input.observedAt,
    from_hash: input.prior.hash,
    to_hash: input.snapshot.hash,
    added_block_ids: diffStrings(input.snapshot.block_ids, input.prior.block_ids),
    removed_block_ids: diffStrings(input.prior.block_ids, input.snapshot.block_ids),
    added_surface_placements: diffPlacements(
      input.snapshot.surface_placements,
      input.prior.surface_placements,
    ),
    removed_surface_placements: diffPlacements(
      input.prior.surface_placements,
      input.snapshot.surface_placements,
    ),
  };
}

export class PromptSurfaceHistoryRepository {
  private readonly clock: Clock;
  private readonly blocks: readonly PromptSurfaceBlock[];

  constructor(private readonly options: PromptSurfaceHistoryRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.blocks = options.blocks ?? PROMPT_SURFACE_BLOCKS;
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  observeCurrent(): PromptSurfaceObservationResult {
    const projection = buildPromptSurfaceProjection(this.blocks);
    const hash = hashPromptSurfaceProjection(projection);
    const observedAt = this.clock.now();
    const snapshot = promptSurfaceSnapshotSchema.parse({
      hash,
      observed_at: observedAt,
      block_ids: projection.block_ids,
      surface_placements: projection.surface_placements,
    });
    const change = this.db.raw.transaction((): PromptSurfaceChangeRecord | null => {
      const insertedSnapshot = this.insertSnapshotIfNew(snapshot);

      if (!insertedSnapshot) {
        return null;
      }

      const prior = this.latestSnapshot();
      const newChange = buildChangeRecord({
        observedAt,
        prior,
        snapshot,
      });

      this.insertChange(newChange);
      return newChange;
    })();

    if (change === null) {
      const stored = this.getSnapshot(hash);

      if (stored === null) {
        throw new StorageError("Prompt surface snapshot insert was ignored but no row exists", {
          code: "PROMPT_SURFACE_HISTORY_OBSERVE_FAILED",
        });
      }

      return {
        snapshot: stored,
        change: null,
        inserted: false,
      };
    }

    return {
      snapshot,
      change,
      inserted: true,
    };
  }

  current(): PromptSurfaceCurrent {
    const projection = buildPromptSurfaceProjection(this.blocks);
    const hash = hashPromptSurfaceProjection(projection);
    const snapshot = this.getSnapshot(hash);

    if (snapshot !== null) {
      return snapshot;
    }

    return promptSurfaceCurrentSchema.parse({
      hash,
      observed_at: null,
      block_ids: projection.block_ids,
      surface_placements: projection.surface_placements,
    });
  }

  getSnapshot(hash: string): PromptSurfaceSnapshot | null {
    const row = this.db
      .prepare(
        `
          SELECT hash, observed_at, block_ids, surface_placements
          FROM prompt_surface_snapshots
          WHERE hash = ?
        `,
      )
      .get(hash) as PromptSurfaceSnapshotRow | undefined;

    return row === undefined ? null : mapSnapshotRow(row);
  }

  latestSnapshot(): PromptSurfaceSnapshot | null {
    const row = this.db
      .prepare(
        `
          SELECT snapshot.hash, snapshot.observed_at, snapshot.block_ids, snapshot.surface_placements
          FROM prompt_surface_changes AS change
          JOIN prompt_surface_snapshots AS snapshot
            ON snapshot.hash = change.to_hash
          ORDER BY change.id DESC
          LIMIT 1
        `,
      )
      .get() as PromptSurfaceSnapshotRow | undefined;

    return row === undefined ? null : mapSnapshotRow(row);
  }

  listChanges(options: PromptSurfaceHistoryListOptions = {}): PromptSurfaceChangeRecord[] {
    const limit = normalizeLimit(options.limit);

    if (options.sinceVersion !== undefined) {
      const sinceRow = this.db
        .prepare("SELECT id FROM prompt_surface_changes WHERE to_hash = ?")
        .get(options.sinceVersion) as { id: number } | undefined;

      if (sinceRow !== undefined) {
        const rows = this.db
          .prepare(
            `
              SELECT
                id, observed_at, from_hash, to_hash,
                added_block_ids, removed_block_ids,
                added_surface_placements, removed_surface_placements
              FROM prompt_surface_changes
              WHERE id > ?
              ORDER BY id ASC
              LIMIT ?
            `,
          )
          .all(Number(sinceRow.id), limit) as PromptSurfaceChangeRow[];

        return rows.map(mapChangeRow);
      }

      return [];
    }

    const rows = this.db
      .prepare(
        `
          SELECT
            id, observed_at, from_hash, to_hash,
            added_block_ids, removed_block_ids,
            added_surface_placements, removed_surface_placements
          FROM prompt_surface_changes
          ORDER BY id DESC
          LIMIT ?
        `,
      )
      .all(limit) as PromptSurfaceChangeRow[];

    return rows.reverse().map(mapChangeRow);
  }

  countSnapshots(): number {
    const row = this.db
      .prepare("SELECT COUNT(*) AS count FROM prompt_surface_snapshots")
      .get() as { count: number };

    return Number(row.count);
  }

  countChanges(): number {
    const row = this.db
      .prepare("SELECT COUNT(*) AS count FROM prompt_surface_changes")
      .get() as { count: number };

    return Number(row.count);
  }

  private insertSnapshotIfNew(snapshot: PromptSurfaceSnapshot): boolean {
    const result = this.db
      .prepare(
        `
          INSERT OR IGNORE INTO prompt_surface_snapshots (
            hash, observed_at, block_ids, surface_placements
          ) VALUES (?, ?, ?, ?)
        `,
      )
      .run(
        snapshot.hash,
        snapshot.observed_at,
        serializeJsonValue(snapshot.block_ids),
        serializeJsonValue(snapshot.surface_placements),
      );

    return result.changes > 0;
  }

  private insertChange(change: PromptSurfaceChangeRecord): void {
    this.db
      .prepare(
        `
          INSERT INTO prompt_surface_changes (
            observed_at, from_hash, to_hash,
            added_block_ids, removed_block_ids,
            added_surface_placements, removed_surface_placements
          ) VALUES (?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        change.observed_at,
        change.from_hash,
        change.to_hash,
        serializeJsonValue(change.added_block_ids),
        serializeJsonValue(change.removed_block_ids),
        serializeJsonValue(change.added_surface_placements),
        serializeJsonValue(change.removed_surface_placements),
      );
  }
}
