import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import {
  autobiographicalPeriodIdHelpers,
  createAutobiographicalPeriodId,
  type AutobiographicalPeriodId,
} from "../../util/ids.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { episodeIdSchema } from "../episodic/types.js";

export const autobiographicalPeriodIdSchema = z
  .string()
  .refine((value) => autobiographicalPeriodIdHelpers.is(value), {
    message: "Invalid autobiographical period id",
  })
  .transform((value) => value as AutobiographicalPeriodId);

export const autobiographicalPeriodSchema = z
  .object({
    id: autobiographicalPeriodIdSchema,
    label: z.string().min(1),
    start_ts: z.number().finite(),
    end_ts: z.number().finite().nullable(),
    narrative: z.string(),
    key_episode_ids: z.array(episodeIdSchema),
    themes: z.array(z.string().min(1)),
    created_at: z.number().finite(),
    last_updated: z.number().finite(),
  })
  .refine((value) => value.end_ts === null || value.end_ts >= value.start_ts, {
    message: "Autobiographical period end_ts must be after start_ts",
    path: ["end_ts"],
  });

export type AutobiographicalPeriod = z.infer<typeof autobiographicalPeriodSchema>;

export type AutobiographicalRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function parseStringArray<T>(value: string, schema: z.ZodType<T>, label: string): T[] {
  let parsed: unknown;

  try {
    parsed = JSON.parse(value) as unknown;
  } catch (error) {
    throw new StorageError(`Failed to parse ${label}`, {
      cause: error,
      code: "SELF_AUTOBIOGRAPHICAL_INVALID",
    });
  }

  const result = z.array(schema).safeParse(parsed);

  if (!result.success) {
    throw new StorageError(`Invalid ${label}`, {
      cause: result.error,
      code: "SELF_AUTOBIOGRAPHICAL_INVALID",
    });
  }

  return result.data;
}

function mapPeriodRow(row: Record<string, unknown>): AutobiographicalPeriod {
  const parsed = autobiographicalPeriodSchema.safeParse({
    id: row.id,
    label: row.label,
    start_ts: Number(row.start_ts),
    end_ts: row.end_ts === null || row.end_ts === undefined ? null : Number(row.end_ts),
    narrative: String(row.narrative ?? ""),
    key_episode_ids: parseStringArray(
      String(row.key_episode_ids ?? "[]"),
      episodeIdSchema,
      "autobiographical key_episode_ids",
    ),
    themes: parseStringArray(
      String(row.themes ?? "[]"),
      z.string().min(1),
      "autobiographical themes",
    ),
    created_at: Number(row.created_at),
    last_updated: Number(row.last_updated),
  });

  if (!parsed.success) {
    throw new StorageError("Autobiographical period row failed validation", {
      cause: parsed.error,
      code: "SELF_AUTOBIOGRAPHICAL_INVALID",
    });
  }

  return parsed.data;
}

export class AutobiographicalRepository {
  private readonly clock: Clock;

  constructor(private readonly options: AutobiographicalRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  runInTransaction<T>(callback: () => T): T {
    return this.db.raw.transaction(callback)();
  }

  upsertPeriod(input: {
    id?: AutobiographicalPeriodId;
    label: string;
    start_ts: number;
    end_ts?: number | null;
    narrative: string;
    key_episode_ids?: readonly z.infer<typeof episodeIdSchema>[];
    themes?: readonly string[];
    created_at?: number;
    last_updated?: number;
  }): AutobiographicalPeriod {
    const existing = input.id === undefined ? null : this.getPeriod(input.id);
    const nowMs = this.clock.now();
    const period = autobiographicalPeriodSchema.parse({
      id: input.id ?? createAutobiographicalPeriodId(),
      label: input.label,
      start_ts: input.start_ts,
      end_ts: input.end_ts ?? null,
      narrative: input.narrative,
      key_episode_ids: input.key_episode_ids ?? [],
      themes: input.themes ?? [],
      created_at: existing?.created_at ?? input.created_at ?? nowMs,
      last_updated: input.last_updated ?? nowMs,
    });

    this.runInTransaction(() => {
      if (period.end_ts === null) {
        this.db
          .prepare(
            `
              UPDATE autobiographical_periods
              SET end_ts = ?, last_updated = ?
              WHERE end_ts IS NULL AND id != ?
            `,
          )
          .run(period.start_ts, period.last_updated, period.id);
      }

      if (existing === null) {
        this.db
          .prepare(
            `
              INSERT INTO autobiographical_periods (
                id, label, start_ts, end_ts, narrative, key_episode_ids, themes, created_at, last_updated
              ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            `,
          )
          .run(
            period.id,
            period.label,
            period.start_ts,
            period.end_ts,
            period.narrative,
            serializeJsonValue(period.key_episode_ids),
            serializeJsonValue(period.themes),
            period.created_at,
            period.last_updated,
          );
        return;
      }

      this.db
        .prepare(
          `
            UPDATE autobiographical_periods
            SET label = ?, start_ts = ?, end_ts = ?, narrative = ?, key_episode_ids = ?, themes = ?, last_updated = ?
            WHERE id = ?
          `,
        )
        .run(
          period.label,
          period.start_ts,
          period.end_ts,
          period.narrative,
          serializeJsonValue(period.key_episode_ids),
          serializeJsonValue(period.themes),
          period.last_updated,
          period.id,
        );
    });

    return period;
  }

  getPeriod(id: AutobiographicalPeriodId): AutobiographicalPeriod | null {
    const row = this.db.prepare("SELECT * FROM autobiographical_periods WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapPeriodRow(row);
  }

  getByLabel(label: string): AutobiographicalPeriod | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM autobiographical_periods
          WHERE label = ?
          ORDER BY start_ts DESC, created_at DESC
          LIMIT 1
        `,
      )
      .get(label) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapPeriodRow(row);
  }

  listPeriods(
    options: {
      fromTs?: number;
      toTs?: number;
      limit?: number;
    } = {},
  ): AutobiographicalPeriod[] {
    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.fromTs !== undefined) {
      filters.push("(end_ts IS NULL OR end_ts >= ?)");
      values.push(options.fromTs);
    }

    if (options.toTs !== undefined) {
      filters.push("start_ts <= ?");
      values.push(options.toTs);
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const limit = options.limit ?? 50;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM autobiographical_periods
          ${whereClause}
          ORDER BY start_ts DESC, created_at DESC
          LIMIT ?
        `,
      )
      .all(...values, limit) as Record<string, unknown>[];

    return rows.map((row) => mapPeriodRow(row));
  }

  currentPeriod(): AutobiographicalPeriod | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM autobiographical_periods
          WHERE end_ts IS NULL
          ORDER BY start_ts DESC
          LIMIT 1
        `,
      )
      .get() as Record<string, unknown> | undefined;

    return row === undefined ? null : mapPeriodRow(row);
  }

  closePeriod(id: AutobiographicalPeriodId, endTs: number): void {
    const result = this.db
      .prepare(
        `
          UPDATE autobiographical_periods
          SET end_ts = ?, last_updated = ?
          WHERE id = ?
        `,
      )
      .run(endTs, this.clock.now(), id);

    if (result.changes === 0) {
      throw new StorageError(`Unknown autobiographical period id: ${id}`, {
        code: "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND",
      });
    }
  }

  updateNarrative(
    id: AutobiographicalPeriodId,
    narrative: string,
    keyEpisodeIds?: readonly z.infer<typeof episodeIdSchema>[],
    themes?: readonly string[],
  ): AutobiographicalPeriod {
    const existing = this.getPeriod(id);

    if (existing === null) {
      throw new StorageError(`Unknown autobiographical period id: ${id}`, {
        code: "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND",
      });
    }

    return this.upsertPeriod({
      ...existing,
      narrative,
      key_episode_ids: keyEpisodeIds ?? existing.key_episode_ids,
      themes: themes ?? existing.themes,
      last_updated: this.clock.now(),
    });
  }

  deletePeriod(id: AutobiographicalPeriodId): boolean {
    const result = this.db.prepare("DELETE FROM autobiographical_periods WHERE id = ?").run(id);
    return result.changes > 0;
  }
}
