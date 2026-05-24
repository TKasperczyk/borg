import { z } from "zod";

import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { ProvenanceError, StorageError } from "../../util/errors.js";
import {
  createAutobiographicalPeriodId,
  type AutobiographicalPeriodId,
  type EpisodeId,
} from "../../util/ids.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  assertIdentityCasUpdated,
  expectedRecordVersion,
  nextRecordVersion,
  type IdentityCasOptions,
} from "../common/cas.js";
import {
  parseStoredProvenance,
  toStoredProvenance,
  type Provenance,
} from "../common/provenance.js";
import {
  autobiographicalPeriodIdSchema,
  autobiographicalPeriodPatchSchema,
  autobiographicalPeriodSchema,
  valueSourceEpisodeIdSchema,
  type AutobiographicalPeriod,
  type AutobiographicalPeriodPatch,
} from "./types.js";

export {
  autobiographicalPeriodIdSchema,
  autobiographicalPeriodPatchSchema,
  autobiographicalPeriodSchema,
};
export type { AutobiographicalPeriod, AutobiographicalPeriodPatch };

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
    record_version: Number(row.record_version ?? 1),
    label: row.label,
    start_ts: Number(row.start_ts),
    end_ts: row.end_ts === null || row.end_ts === undefined ? null : Number(row.end_ts),
    narrative: String(row.narrative ?? ""),
    key_episode_ids: parseStringArray(
      String(row.key_episode_ids ?? "[]"),
      valueSourceEpisodeIdSchema,
      "autobiographical key_episode_ids",
    ),
    themes: parseStringArray(
      String(row.themes ?? "[]"),
      z.string().min(1),
      "autobiographical themes",
    ),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
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

  upsertPeriod(
    input: {
      id?: AutobiographicalPeriodId;
      label: string;
      start_ts: number;
      end_ts?: number | null;
      narrative: string;
      key_episode_ids?: readonly EpisodeId[];
      themes?: readonly string[];
      provenance: Provenance;
      created_at?: number;
      last_updated?: number;
    },
    options: IdentityCasOptions & {
      expectedOpenPeriod?: { id: AutobiographicalPeriodId; expectedVersion: number } | null;
    } = {},
  ): AutobiographicalPeriod {
    if (input.provenance === undefined) {
      throw new ProvenanceError("Autobiographical period requires provenance", {
        code: "PROVENANCE_REQUIRED",
      });
    }

    const existing = input.id === undefined ? null : this.getPeriod(input.id);
    const expectedExistingVersion =
      existing === null ? null : expectedRecordVersion(existing, options);
    const openPeriod =
      input.end_ts === undefined || input.end_ts === null ? this.currentPeriod() : null;
    const expectedOpenPeriod =
      options.expectedOpenPeriod ??
      (openPeriod !== null && openPeriod.id !== input.id
        ? { id: openPeriod.id, expectedVersion: expectedRecordVersion(openPeriod) }
        : null);
    const nowMs = this.clock.now();
    const period = autobiographicalPeriodSchema.parse({
      id: input.id ?? createAutobiographicalPeriodId(),
      record_version:
        existing === null || expectedExistingVersion === null
          ? 1
          : nextRecordVersion(expectedExistingVersion),
      label: input.label,
      start_ts: input.start_ts,
      end_ts: input.end_ts ?? null,
      narrative: input.narrative,
      key_episode_ids: input.key_episode_ids ?? [],
      themes: input.themes ?? [],
      provenance: input.provenance,
      created_at: existing?.created_at ?? input.created_at ?? nowMs,
      last_updated: input.last_updated ?? nowMs,
    });
    const storedProvenance = toStoredProvenance(period.provenance);

    this.runInTransaction(() => {
      if (period.end_ts === null && expectedOpenPeriod !== null) {
        const closeResult = this.db
          .prepare(
            `
              UPDATE autobiographical_periods
              SET end_ts = ?, last_updated = ?, record_version = record_version + 1
              WHERE id = ? AND end_ts IS NULL AND record_version = ?
            `,
          )
          .run(
            period.start_ts,
            period.last_updated,
            expectedOpenPeriod.id,
            expectedOpenPeriod.expectedVersion,
          );
        assertIdentityCasUpdated({
          result: closeResult,
          recordType: "autobiographical_period",
          recordId: expectedOpenPeriod.id,
          expectedVersion: expectedOpenPeriod.expectedVersion,
        });
      }

      if (existing === null) {
        this.db
          .prepare(
            `
              INSERT INTO autobiographical_periods (
                id, label, start_ts, end_ts, narrative, key_episode_ids, themes, provenance_kind,
                provenance_episode_ids, provenance_process, created_at, last_updated
              ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            storedProvenance.provenance_kind,
            storedProvenance.provenance_episode_ids,
            storedProvenance.provenance_process,
            period.created_at,
            period.last_updated,
          );
        return;
      }

      const result = this.db
        .prepare(
          `
            UPDATE autobiographical_periods
            SET label = ?, start_ts = ?, end_ts = ?, narrative = ?, key_episode_ids = ?, themes = ?,
                provenance_kind = ?, provenance_episode_ids = ?, provenance_process = ?, last_updated = ?,
                record_version = record_version + 1
            WHERE id = ? AND record_version = ?
          `,
        )
        .run(
          period.label,
          period.start_ts,
          period.end_ts,
          period.narrative,
          serializeJsonValue(period.key_episode_ids),
          serializeJsonValue(period.themes),
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          period.last_updated,
          period.id,
          expectedExistingVersion,
        );
      assertIdentityCasUpdated({
        result,
        recordType: "autobiographical_period",
        recordId: period.id,
        expectedVersion: expectedExistingVersion ?? 0,
      });
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

  closePeriod(id: AutobiographicalPeriodId, endTs: number, options: IdentityCasOptions = {}): void {
    const existing = this.getPeriod(id);

    if (existing === null) {
      throw new StorageError(`Unknown autobiographical period id: ${id}`, {
        code: "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND",
      });
    }

    const expectedVersion = expectedRecordVersion(existing, options);
    const result = this.db
      .prepare(
        `
          UPDATE autobiographical_periods
          SET end_ts = ?, last_updated = ?, record_version = record_version + 1
          WHERE id = ? AND record_version = ?
        `,
      )
      .run(endTs, this.clock.now(), id, expectedVersion);

    if (result.changes === 0) {
      assertIdentityCasUpdated({
        result,
        recordType: "autobiographical_period",
        recordId: id,
        expectedVersion,
      });
    }
  }

  updateNarrative(
    id: AutobiographicalPeriodId,
    narrative: string,
    keyEpisodeIds?: readonly EpisodeId[],
    themes?: readonly string[],
    provenance?: Provenance,
  ): AutobiographicalPeriod {
    const existing = this.getPeriod(id);

    if (existing === null) {
      throw new StorageError(`Unknown autobiographical period id: ${id}`, {
        code: "AUTOBIOGRAPHICAL_PERIOD_NOT_FOUND",
      });
    }

    return this.upsertPeriod(
      {
        ...existing,
        narrative,
        key_episode_ids: keyEpisodeIds ?? existing.key_episode_ids,
        themes: themes ?? existing.themes,
        provenance: provenance ?? existing.provenance,
        last_updated: this.clock.now(),
      },
      {
        expectedVersion: expectedRecordVersion(existing),
      },
    );
  }

  deletePeriod(id: AutobiographicalPeriodId): boolean {
    // Current callers are offline audit reversers for periods created by the
    // audited action. AuditLog.revert runs reversers inside BEGIN IMMEDIATE, so
    // no concurrent writer can interleave with this cleanup delete.
    const result = this.db.prepare("DELETE FROM autobiographical_periods WHERE id = ?").run(id);
    return result.changes > 0;
  }
}
