import { z } from "zod";

import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import {
  createLivedExperienceDaySummaryId,
  livedExperienceDaySummaryIdHelpers,
  episodeIdHelpers,
  entityIdHelpers,
  maintenanceRunIdHelpers,
  streamEntryIdHelpers,
  type EntityId,
  type EpisodeId,
  type LivedExperienceDaySummaryId,
  type MaintenanceRunId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  assertJsonValue,
  jsonValueSchema,
  serializeJsonValue,
  type JsonValue,
} from "../../util/json-value.js";
import { timestampFromUtcDayKey, utcDayKey } from "../../util/utc-day.js";
import {
  memoryDisclosureLabelSchema,
  parseMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../common/disclosure-label.js";
import {
  parseStoredProvenance,
  provenanceSchema,
  toStoredProvenance,
  type Provenance,
} from "../common/provenance.js";

const livedExperienceDaySummaryIdSchema = z
  .string()
  .refine((value) => livedExperienceDaySummaryIdHelpers.is(value), {
    message: "Invalid lived experience day summary id",
  })
  .transform((value) => value as LivedExperienceDaySummaryId);

const entityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid lived experience summary entity id",
  })
  .transform((value) => value as EntityId);

const episodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid lived experience summary episode id",
  })
  .transform((value) => value as EpisodeId);

const streamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid lived experience summary stream entry id",
  })
  .transform((value) => value as StreamEntryId);

const maintenanceRunIdSchema = z
  .string()
  .refine((value) => maintenanceRunIdHelpers.is(value), {
    message: "Invalid lived experience summary run id",
  })
  .transform((value) => value as MaintenanceRunId);

const utcDaySchema = z.string().refine(
  (value) => {
    const timestamp = timestampFromUtcDayKey(value);

    return Number.isFinite(timestamp) && utcDayKey(timestamp) === value;
  },
  {
    message: "Invalid UTC day key",
  },
);

export const livedExperienceDaySummarySchema = z
  .object({
    id: livedExperienceDaySummaryIdSchema,
    self_entity_id: entityIdSchema,
    utc_day: utcDaySchema,
    day_start_ms: z.number().int().finite(),
    day_end_ms: z.number().int().finite(),
    gist: z.string().min(1),
    salience: z.number().min(0).max(1),
    counts_snapshot: jsonValueSchema,
    source_episode_ids: z.array(episodeIdSchema),
    source_stream_entry_ids: z.array(streamEntryIdSchema),
    disclosure_label: memoryDisclosureLabelSchema,
    provenance: provenanceSchema,
    source_run_id: maintenanceRunIdSchema.nullable(),
    created_at: z.number().int().finite(),
    updated_at: z.number().int().finite(),
  })
  .strict()
  .refine((value) => value.day_end_ms >= value.day_start_ms, {
    message: "Day summary end must be after start",
    path: ["day_end_ms"],
  })
  .refine((value) => value.utc_day === utcDayKey(value.day_start_ms), {
    message: "utc_day must match day_start_ms",
    path: ["utc_day"],
  });

export type LivedExperienceDaySummary = z.infer<typeof livedExperienceDaySummarySchema>;

export type LivedExperienceDaySummaryInput = {
  id?: LivedExperienceDaySummaryId;
  selfEntityId: EntityId;
  utcDay: string;
  dayStartMs: number;
  dayEndMs: number;
  gist: string;
  salience?: number;
  countsSnapshot: JsonValue;
  sourceEpisodeIds?: readonly EpisodeId[];
  sourceStreamEntryIds?: readonly StreamEntryId[];
  disclosureLabel?: MemoryDisclosureLabel;
  provenance: Provenance;
  sourceRunId?: MaintenanceRunId | null;
  createdAt?: number;
  updatedAt?: number;
};

export type LivedExperienceDaySummaryListOptions = {
  selfEntityId?: EntityId;
  fromMs: number;
  toMs: number;
  limit?: number;
};

export type LivedExperienceDaySummaryRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function parseJson(value: unknown, label: string): JsonValue {
  if (typeof value !== "string") {
    throw new StorageError(`Failed to parse ${label}`, {
      code: "LIVED_EXPERIENCE_DAY_SUMMARY_INVALID",
    });
  }

  try {
    const parsed = JSON.parse(value) as unknown;
    assertJsonValue(parsed);
    return parsed;
  } catch (error) {
    throw new StorageError(`Failed to parse ${label}`, {
      cause: error,
      code: "LIVED_EXPERIENCE_DAY_SUMMARY_INVALID",
    });
  }
}

function parseStringArray<T>(value: unknown, schema: z.ZodType<T>, label: string): T[] {
  const parsed = parseJson(value, label);
  const result = z.array(schema).safeParse(parsed);

  if (!result.success) {
    throw new StorageError(`Invalid ${label}`, {
      cause: result.error,
      code: "LIVED_EXPERIENCE_DAY_SUMMARY_INVALID",
    });
  }

  return result.data;
}

function mapDaySummaryRow(row: Record<string, unknown>): LivedExperienceDaySummary {
  const parsed = livedExperienceDaySummarySchema.safeParse({
    id: row.id,
    self_entity_id: row.self_entity_id,
    utc_day: row.utc_day,
    day_start_ms: Number(row.day_start_ms),
    day_end_ms: Number(row.day_end_ms),
    gist: String(row.gist ?? ""),
    salience: Number(row.salience ?? 0),
    counts_snapshot: parseJson(row.counts_snapshot, "lived experience counts_snapshot"),
    source_episode_ids: parseStringArray(
      row.source_episode_ids,
      episodeIdSchema,
      "lived experience source_episode_ids",
    ),
    source_stream_entry_ids: parseStringArray(
      row.source_stream_entry_ids,
      streamEntryIdSchema,
      "lived experience source_stream_entry_ids",
    ),
    disclosure_label: parseMemoryDisclosureLabel(row.disclosure_label),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
    source_run_id:
      row.source_run_id === null || row.source_run_id === undefined ? null : row.source_run_id,
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Lived experience day summary row failed validation", {
      cause: parsed.error,
      code: "LIVED_EXPERIENCE_DAY_SUMMARY_INVALID",
    });
  }

  return parsed.data;
}

export class LivedExperienceDaySummaryRepository {
  private readonly clock: Clock;

  constructor(private readonly options: LivedExperienceDaySummaryRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  upsert(input: LivedExperienceDaySummaryInput): LivedExperienceDaySummary {
    const existing = this.getByDay(input.selfEntityId, input.utcDay);
    const now = this.clock.now();
    const summary = livedExperienceDaySummarySchema.parse({
      id: existing?.id ?? input.id ?? createLivedExperienceDaySummaryId(),
      self_entity_id: input.selfEntityId,
      utc_day: input.utcDay,
      day_start_ms: input.dayStartMs,
      day_end_ms: input.dayEndMs,
      gist: input.gist,
      salience: input.salience ?? 0.5,
      counts_snapshot: input.countsSnapshot,
      source_episode_ids: input.sourceEpisodeIds ?? [],
      source_stream_entry_ids: input.sourceStreamEntryIds ?? [],
      disclosure_label:
        input.disclosureLabel ?? existing?.disclosure_label ?? unknownMemoryDisclosureLabel(),
      provenance: input.provenance,
      source_run_id: input.sourceRunId ?? null,
      created_at: existing?.created_at ?? input.createdAt ?? now,
      updated_at: input.updatedAt ?? now,
    });
    const storedProvenance = toStoredProvenance(summary.provenance);

    this.db
      .prepare(
        `
          INSERT INTO lived_experience_day_summaries (
            id, self_entity_id, utc_day, day_start_ms, day_end_ms, gist, salience,
            counts_snapshot, source_episode_ids, source_stream_entry_ids, disclosure_label,
            provenance_kind, provenance_episode_ids, provenance_process, source_run_id,
            created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(self_entity_id, utc_day) DO UPDATE SET
            day_start_ms = excluded.day_start_ms,
            day_end_ms = excluded.day_end_ms,
            gist = excluded.gist,
            salience = excluded.salience,
            counts_snapshot = excluded.counts_snapshot,
            source_episode_ids = excluded.source_episode_ids,
            source_stream_entry_ids = excluded.source_stream_entry_ids,
            disclosure_label = excluded.disclosure_label,
            provenance_kind = excluded.provenance_kind,
            provenance_episode_ids = excluded.provenance_episode_ids,
            provenance_process = excluded.provenance_process,
            source_run_id = excluded.source_run_id,
            updated_at = excluded.updated_at
        `,
      )
      .run(
        summary.id,
        summary.self_entity_id,
        summary.utc_day,
        summary.day_start_ms,
        summary.day_end_ms,
        summary.gist,
        summary.salience,
        serializeJsonValue(summary.counts_snapshot),
        serializeJsonValue(summary.source_episode_ids),
        serializeJsonValue(summary.source_stream_entry_ids),
        serializeJsonValue(summary.disclosure_label),
        storedProvenance.provenance_kind,
        storedProvenance.provenance_episode_ids,
        storedProvenance.provenance_process,
        summary.source_run_id,
        summary.created_at,
        summary.updated_at,
      );

    const stored = this.getByDay(summary.self_entity_id, summary.utc_day);

    if (stored === null) {
      throw new StorageError(`Lived experience day summary ${summary.id} was not stored`, {
        code: "LIVED_EXPERIENCE_DAY_SUMMARY_STORE_FAILED",
      });
    }

    return stored;
  }

  get(id: LivedExperienceDaySummaryId): LivedExperienceDaySummary | null {
    const row = this.db
      .prepare("SELECT * FROM lived_experience_day_summaries WHERE id = ?")
      .get(id) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapDaySummaryRow(row);
  }

  getByDay(selfEntityId: EntityId, utcDay: string): LivedExperienceDaySummary | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM lived_experience_day_summaries
          WHERE self_entity_id = ? AND utc_day = ?
          LIMIT 1
        `,
      )
      .get(selfEntityId, utcDay) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapDaySummaryRow(row);
  }

  listForWindow(options: LivedExperienceDaySummaryListOptions): LivedExperienceDaySummary[] {
    const filters = ["day_end_ms >= ?", "day_start_ms <= ?"];
    const values: unknown[] = [options.fromMs, options.toMs];

    if (options.selfEntityId !== undefined) {
      filters.push("self_entity_id = ?");
      values.push(options.selfEntityId);
    }

    values.push(Math.max(1, Math.floor(options.limit ?? 50)));

    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM lived_experience_day_summaries
          WHERE ${filters.join(" AND ")}
          ORDER BY day_start_ms ASC, created_at ASC
          LIMIT ?
        `,
      )
      .all(...values) as Record<string, unknown>[];

    return rows.map(mapDaySummaryRow);
  }

  delete(id: LivedExperienceDaySummaryId): boolean {
    const result = this.db
      .prepare("DELETE FROM lived_experience_day_summaries WHERE id = ?")
      .run(id);

    return result.changes > 0;
  }
}
