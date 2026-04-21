import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import {
  createGrowthMarkerId,
  growthMarkerIdHelpers,
  type GrowthMarkerId,
} from "../../util/ids.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { episodeIdSchema } from "../episodic/types.js";

export const GROWTH_MARKER_CATEGORIES = [
  "skill",
  "value",
  "habit",
  "relationship",
  "understanding",
] as const;

export const growthMarkerIdSchema = z
  .string()
  .refine((value) => growthMarkerIdHelpers.is(value), {
    message: "Invalid growth marker id",
  })
  .transform((value) => value as GrowthMarkerId);

export const growthMarkerCategorySchema = z.enum(GROWTH_MARKER_CATEGORIES);

export const growthMarkerSchema = z.object({
  id: growthMarkerIdSchema,
  ts: z.number().finite(),
  category: growthMarkerCategorySchema,
  what_changed: z.string().min(1),
  before_description: z.string().nullable(),
  after_description: z.string().nullable(),
  evidence_episode_ids: z.array(episodeIdSchema).min(1),
  confidence: z.number().min(0).max(1),
  source_process: z.string().min(1),
  created_at: z.number().finite(),
});

export type GrowthMarker = z.infer<typeof growthMarkerSchema>;
export type GrowthMarkerCategory = z.infer<typeof growthMarkerCategorySchema>;

export type GrowthMarkersSummary = {
  counts: Record<GrowthMarkerCategory, number>;
  top_changes: string[];
};

export type GrowthMarkersRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function parseEpisodeIds(value: string) {
  let parsed: unknown;

  try {
    parsed = JSON.parse(value) as unknown;
  } catch (error) {
    throw new StorageError("Failed to parse growth marker evidence_episode_ids", {
      cause: error,
      code: "GROWTH_MARKER_INVALID",
    });
  }

  const result = z.array(episodeIdSchema).safeParse(parsed);

  if (!result.success) {
    throw new StorageError("Invalid growth marker evidence_episode_ids", {
      cause: result.error,
      code: "GROWTH_MARKER_INVALID",
    });
  }

  return result.data;
}

function mapGrowthMarkerRow(row: Record<string, unknown>): GrowthMarker {
  const parsed = growthMarkerSchema.safeParse({
    id: row.id,
    ts: Number(row.ts),
    category: row.category,
    what_changed: row.what_changed,
    before_description:
      row.before_description === null || row.before_description === undefined
        ? null
        : String(row.before_description),
    after_description:
      row.after_description === null || row.after_description === undefined
        ? null
        : String(row.after_description),
    evidence_episode_ids: parseEpisodeIds(String(row.evidence_episode_ids ?? "[]")),
    confidence: Number(row.confidence),
    source_process: row.source_process,
    created_at: Number(row.created_at),
  });

  if (!parsed.success) {
    throw new StorageError("Growth marker row failed validation", {
      cause: parsed.error,
      code: "GROWTH_MARKER_INVALID",
    });
  }

  return parsed.data;
}

export class GrowthMarkersRepository {
  private readonly clock: Clock;

  constructor(private readonly options: GrowthMarkersRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  add(input: {
    id?: GrowthMarkerId;
    ts: number;
    category: GrowthMarkerCategory;
    what_changed: string;
    before_description?: string | null;
    after_description?: string | null;
    evidence_episode_ids: readonly z.infer<typeof episodeIdSchema>[];
    confidence: number;
    source_process: string;
    created_at?: number;
  }): GrowthMarker {
    const marker = growthMarkerSchema.parse({
      id: input.id ?? createGrowthMarkerId(),
      ts: input.ts,
      category: input.category,
      what_changed: input.what_changed,
      before_description: input.before_description ?? null,
      after_description: input.after_description ?? null,
      evidence_episode_ids: input.evidence_episode_ids,
      confidence: input.confidence,
      source_process: input.source_process,
      created_at: input.created_at ?? this.clock.now(),
    });

    this.db
      .prepare(
        `
          INSERT INTO growth_markers (
            id, ts, category, what_changed, before_description, after_description,
            evidence_episode_ids, confidence, source_process, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        marker.id,
        marker.ts,
        marker.category,
        marker.what_changed,
        marker.before_description,
        marker.after_description,
        serializeJsonValue(marker.evidence_episode_ids),
        marker.confidence,
        marker.source_process,
        marker.created_at,
      );

    return marker;
  }

  list(
    options: {
      sinceTs?: number;
      untilTs?: number;
      category?: GrowthMarkerCategory;
      limit?: number;
    } = {},
  ): GrowthMarker[] {
    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.sinceTs !== undefined) {
      filters.push("ts >= ?");
      values.push(options.sinceTs);
    }

    if (options.untilTs !== undefined) {
      filters.push("ts <= ?");
      values.push(options.untilTs);
    }

    if (options.category !== undefined) {
      growthMarkerCategorySchema.parse(options.category);
      filters.push("category = ?");
      values.push(options.category);
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const limit = options.limit ?? 50;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM growth_markers
          ${whereClause}
          ORDER BY ts DESC, created_at DESC
          LIMIT ?
        `,
      )
      .all(...values, limit) as Record<string, unknown>[];

    return rows.map((row) => mapGrowthMarkerRow(row));
  }

  get(id: GrowthMarkerId): GrowthMarker | null {
    const row = this.db.prepare("SELECT * FROM growth_markers WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapGrowthMarkerRow(row);
  }

  delete(id: GrowthMarkerId): boolean {
    const result = this.db.prepare("DELETE FROM growth_markers WHERE id = ?").run(id);
    return result.changes > 0;
  }

  summarize(
    options: {
      periodId?: string;
      fromTs?: number;
      toTs?: number;
    } = {},
  ): GrowthMarkersSummary {
    let fromTs = options.fromTs;
    let toTs = options.toTs;

    if (options.periodId !== undefined) {
      const periodRow = this.db
        .prepare("SELECT start_ts, end_ts FROM autobiographical_periods WHERE id = ?")
        .get(options.periodId) as Record<string, unknown> | undefined;

      if (periodRow === undefined) {
        throw new StorageError(`Unknown autobiographical period id: ${options.periodId}`, {
          code: "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND",
        });
      }

      fromTs = Number(periodRow.start_ts);
      toTs =
        periodRow.end_ts === null || periodRow.end_ts === undefined
          ? undefined
          : Number(periodRow.end_ts);
    }

    const markers = this.list({
      sinceTs: fromTs,
      untilTs: toTs,
      limit: 500,
    });
    const counts = Object.fromEntries(
      GROWTH_MARKER_CATEGORIES.map((category) => [category, 0]),
    ) as Record<GrowthMarkerCategory, number>;

    for (const marker of markers) {
      counts[marker.category] += 1;
    }

    const topChanges = [...new Set(markers.map((marker) => marker.what_changed))].slice(0, 5);

    return {
      counts,
      top_changes: topChanges,
    };
  }
}
