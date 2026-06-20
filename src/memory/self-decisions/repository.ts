import { parseJsonArray, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createSelfDecisionEventId,
  parseSelfDecisionEventId,
  type SelfDecisionEventId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { timestampFromUtcDayKey } from "../../util/utc-day.js";
import {
  selfDecisionEventSchema,
  type SelfDecisionEvent,
  type SelfDecisionTriggerType,
} from "./types.js";

const SELF_DECISION_JSON_ARRAY_CODEC = {
  errorCode: "SELF_DECISION_EVENT_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse self decision event ${label}`,
} satisfies JsonArrayCodecOptions;

export type SelfDecisionEventRecordInput = {
  id?: SelfDecisionEventId;
  occurredAt: number;
  sessionId: SessionId;
  triggerName: string;
  triggerType: SelfDecisionTriggerType;
  sourceEventId: string;
  fireEventId: StreamEntryId;
  decisionSummary: string;
  decisionRationale?: string | null;
  turnResultId?: string | null;
  sourceStreamEntryIds: readonly StreamEntryId[];
  now?: number;
};

export type SelfDecisionProjectionSourceEvent = {
  occurredAt: number;
  triggerName: string;
  triggerType: SelfDecisionTriggerType;
  decisionSummary: string;
  decisionRationale: string | null;
  sourceStreamEntryIds: readonly StreamEntryId[];
};

export type SelfDecisionDailyDensityRow = {
  dayKey: string;
  dayStartMs: number;
  decisionCount: number;
  distinctDecisionShapeCount: number;
  firstOccurredAt: number;
  lastOccurredAt: number;
};

export type SelfDecisionRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function parseStreamEntryIds(value: string, label: string): StreamEntryId[] {
  return parseJsonArray<StreamEntryId>(value, label, SELF_DECISION_JSON_ARRAY_CODEC);
}

function uniqueStreamEntryIds(values: readonly StreamEntryId[]): StreamEntryId[] {
  return dedupePreservingOrder(values);
}

function mapSelfDecisionRow(row: Record<string, unknown>): SelfDecisionEvent {
  const parsed = selfDecisionEventSchema.safeParse({
    id: row.id,
    occurred_at: Number(row.occurred_at),
    session_id: row.session_id,
    trigger_name: row.trigger_name,
    trigger_type: row.trigger_type,
    source_event_id: row.source_event_id,
    fire_event_id: row.fire_event_id,
    origin: row.origin,
    decision_summary: row.decision_summary,
    decision_rationale:
      row.decision_rationale === null || row.decision_rationale === undefined
        ? null
        : row.decision_rationale,
    turn_result_id:
      row.turn_result_id === null || row.turn_result_id === undefined ? null : row.turn_result_id,
    source_stream_entry_ids: parseStreamEntryIds(
      String(row.source_stream_entry_ids ?? "[]"),
      "source_stream_entry_ids",
    ),
    disclosure_class: row.disclosure_class,
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Self decision event row failed validation", {
      cause: parsed.error,
      code: "SELF_DECISION_EVENT_ROW_INVALID",
    });
  }

  return parsed.data;
}

function mapProjectionRow(row: Record<string, unknown>): SelfDecisionProjectionSourceEvent {
  return {
    occurredAt: Number(row.occurred_at),
    triggerName: String(row.trigger_name),
    triggerType: row.trigger_type as SelfDecisionTriggerType,
    decisionSummary: String(row.decision_summary ?? ""),
    decisionRationale:
      row.decision_rationale === null || row.decision_rationale === undefined
        ? null
        : String(row.decision_rationale),
    sourceStreamEntryIds: parseStreamEntryIds(
      String(row.source_stream_entry_ids ?? "[]"),
      "source_stream_entry_ids",
    ),
  };
}

function mapDailyDensityRow(row: Record<string, unknown>): SelfDecisionDailyDensityRow {
  const dayKey = String(row.day_key);

  return {
    dayKey,
    dayStartMs: timestampFromUtcDayKey(dayKey),
    decisionCount: Number(row.decision_count),
    distinctDecisionShapeCount: Number(row.distinct_decision_shape_count ?? 0),
    firstOccurredAt: Number(row.first_occurred_at),
    lastOccurredAt: Number(row.last_occurred_at),
  };
}

export class SelfDecisionRepository {
  private readonly clock: Clock;

  constructor(private readonly options: SelfDecisionRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  record(input: SelfDecisionEventRecordInput): SelfDecisionEvent {
    const sourceStreamEntryIds = uniqueStreamEntryIds(input.sourceStreamEntryIds);

    if (sourceStreamEntryIds.length === 0) {
      throw new StorageError("Self decision event requires at least one source stream entry id", {
        code: "SELF_DECISION_EVENT_SOURCE_REQUIRED",
      });
    }

    const now = input.now ?? this.clock.now();
    const id = input.id ?? createSelfDecisionEventId();

    this.db
      .prepare(
        `
          INSERT INTO self_decision_events (
            id, occurred_at, session_id, trigger_name, trigger_type, source_event_id,
            fire_event_id, origin, decision_summary, decision_rationale, turn_result_id,
            source_stream_entry_ids, disclosure_class, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, 'autonomous', ?, ?, ?, ?, 'self_private', ?, ?)
          ON CONFLICT(fire_event_id) DO NOTHING
        `,
      )
      .run(
        id,
        input.occurredAt,
        input.sessionId,
        input.triggerName,
        input.triggerType,
        input.sourceEventId,
        input.fireEventId,
        input.decisionSummary,
        input.decisionRationale ?? null,
        input.turnResultId ?? null,
        serializeJsonValue(sourceStreamEntryIds),
        now,
        now,
      );

    const stored = this.getByFireEvent(input.fireEventId);

    if (stored === null) {
      throw new StorageError(`Self decision event ${id} was not stored`, {
        code: "SELF_DECISION_EVENT_STORE_FAILED",
      });
    }

    return stored;
  }

  get(id: SelfDecisionEventId): SelfDecisionEvent | null {
    const row = this.db
      .prepare(
        `
          SELECT
            id, occurred_at, session_id, trigger_name, trigger_type, source_event_id, origin,
            fire_event_id, decision_summary, decision_rationale, turn_result_id,
            source_stream_entry_ids,
            disclosure_class, created_at, updated_at
          FROM self_decision_events
          WHERE id = ?
        `,
      )
      .get(parseSelfDecisionEventId(id)) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapSelfDecisionRow(row);
  }

  getByFireEvent(fireEventId: StreamEntryId): SelfDecisionEvent | null {
    const row = this.db
      .prepare(
        `
          SELECT
            id, occurred_at, session_id, trigger_name, trigger_type, source_event_id,
            fire_event_id, origin, decision_summary, decision_rationale, turn_result_id,
            source_stream_entry_ids,
            disclosure_class, created_at, updated_at
          FROM self_decision_events
          WHERE fire_event_id = ?
        `,
      )
      .get(fireEventId) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapSelfDecisionRow(row);
  }

  listRecentAutonomousSelfPrivate(input: {
    sinceMs: number;
    limit: number;
  }): SelfDecisionProjectionSourceEvent[] {
    const rows = this.db
      .prepare(
        `
          SELECT
            occurred_at, trigger_name, trigger_type, decision_summary, decision_rationale,
            source_stream_entry_ids
          FROM self_decision_events
          WHERE
            origin = 'autonomous'
            AND disclosure_class = 'self_private'
            AND occurred_at >= ?
          ORDER BY occurred_at DESC, id DESC
          LIMIT ?
        `,
      )
      .all(input.sinceMs, input.limit) as Record<string, unknown>[];

    return rows.map(mapProjectionRow);
  }

  listDailyAutonomousSelfPrivateDensity(input: {
    sinceMs: number;
    untilMs?: number;
    limit: number;
  }): SelfDecisionDailyDensityRow[] {
    const filters = [
      "origin = 'autonomous'",
      "disclosure_class = 'self_private'",
      "occurred_at >= ?",
    ];
    const values: unknown[] = [input.sinceMs];

    if (input.untilMs !== undefined) {
      filters.push("occurred_at <= ?");
      values.push(input.untilMs);
    }

    values.push(Math.max(1, Math.floor(input.limit)));

    const rows = this.db
      .prepare(
        `
          SELECT
            strftime('%Y-%m-%d', occurred_at / 1000, 'unixepoch') AS day_key,
            COUNT(*) AS decision_count,
            COUNT(DISTINCT (
              session_id || '|' ||
              trigger_type || '|' ||
              CASE WHEN decision_rationale IS NULL OR decision_rationale = '' THEN 'r0' ELSE 'r1' END || '|' ||
              CASE WHEN turn_result_id IS NULL THEN 't0' ELSE 't1' END
            )) AS distinct_decision_shape_count,
            MIN(occurred_at) AS first_occurred_at,
            MAX(occurred_at) AS last_occurred_at
          FROM self_decision_events
          WHERE ${filters.join(" AND ")}
          GROUP BY day_key
          ORDER BY last_occurred_at DESC
          LIMIT ?
        `,
      )
      .all(...values) as Record<string, unknown>[];

    return rows.map(mapDailyDensityRow);
  }
}
