import { parseJsonArray, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createActivityEventId,
  parseActivityEventId,
  type ActivityEventId,
  type EntityId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { timestampFromUtcDayKey } from "../../util/utc-day.js";
import {
  activityEventSchema,
  type ActivityEvent,
  type ActivityEventKind,
  type ActivityEventStatus,
} from "./types.js";

const ACTIVITY_JSON_ARRAY_CODEC = {
  errorCode: "ACTIVITY_EVENT_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse activity event ${label}`,
} satisfies JsonArrayCodecOptions;

export type ActivityEventRecordInput = {
  id?: ActivityEventId;
  kind: ActivityEventKind;
  occurredAt: number;
  sessionId: SessionId;
  turnId?: string | null;
  speakerEntityId?: EntityId | null;
  actorEntityId?: EntityId | null;
  audienceEntityId?: EntityId | null;
  participantEntityIds?: readonly EntityId[];
  sourceStreamEntryIds: readonly StreamEntryId[];
  status?: ActivityEventStatus;
  now?: number;
};

export type ActivityProjectionSourceEvent = {
  kind: ActivityEventKind;
  occurredAt: number;
  sessionId: SessionId;
  participantLabel: string;
  audienceEntityId: EntityId | null;
  sourceStreamEntryIds: readonly StreamEntryId[];
};

export type ActivityAutobiographicalSourceEvent = {
  id: ActivityEventId;
  kind: ActivityEventKind;
  occurredAt: number;
  sessionId: SessionId;
  sessionSourceType: string;
  sessionAudienceRole: string;
  sessionLabel: string;
  participantLabel: string;
  audienceEntityId: EntityId | null;
  participantEntityIds: readonly EntityId[];
  sourceStreamEntryIds: readonly StreamEntryId[];
};

export type ActivityEventKindCounts = {
  userContact: number;
  borgReplied: number;
  turnCompleted: number;
};

export type ActivityDailyDensityRow = {
  dayKey: string;
  dayStartMs: number;
  sessionId: SessionId;
  sessionLabel: string;
  audienceLabel: string;
  audienceEntityId: EntityId | null;
  eventCount: number;
  conversationTurnCount: number;
  kindCounts: ActivityEventKindCounts;
  firstOccurredAt: number;
  lastOccurredAt: number;
};

export type ActivityGlobalDailyDensityRow = {
  dayKey: string;
  dayStartMs: number;
  eventCount: number;
  conversationTurnCount: number;
  distinctSessionCount: number;
  kindCounts: ActivityEventKindCounts;
  firstOccurredAt: number;
  lastOccurredAt: number;
};

export type ActivityRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function parseEntityIds(value: string, label: string): EntityId[] {
  return parseJsonArray<EntityId>(value, label, ACTIVITY_JSON_ARRAY_CODEC);
}

function parseStreamEntryIds(value: string, label: string): StreamEntryId[] {
  return parseJsonArray<StreamEntryId>(value, label, ACTIVITY_JSON_ARRAY_CODEC);
}

function uniqueEntityIds(values: readonly (EntityId | null | undefined)[]): EntityId[] {
  return dedupePreservingOrder(
    values.filter((value): value is EntityId => value !== null && value !== undefined),
  );
}

function uniqueStreamEntryIds(values: readonly StreamEntryId[]): StreamEntryId[] {
  return dedupePreservingOrder(values);
}

function mapActivityRow(row: Record<string, unknown>): ActivityEvent {
  const parsed = activityEventSchema.safeParse({
    id: row.id,
    kind: row.kind,
    occurred_at: Number(row.occurred_at),
    session_id: row.session_id,
    turn_id: row.turn_id === null || row.turn_id === undefined ? null : row.turn_id,
    speaker_entity_id:
      row.speaker_entity_id === null || row.speaker_entity_id === undefined
        ? null
        : row.speaker_entity_id,
    actor_entity_id:
      row.actor_entity_id === null || row.actor_entity_id === undefined
        ? null
        : row.actor_entity_id,
    audience_entity_id:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : row.audience_entity_id,
    participant_entity_ids: parseEntityIds(
      String(row.participant_entity_ids ?? "[]"),
      "participant_entity_ids",
    ),
    source_stream_entry_ids: parseStreamEntryIds(
      String(row.source_stream_entry_ids ?? "[]"),
      "source_stream_entry_ids",
    ),
    status: row.status,
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Activity event row failed validation", {
      cause: parsed.error,
      code: "ACTIVITY_EVENT_ROW_INVALID",
    });
  }

  return parsed.data;
}

function mapProjectionRow(row: Record<string, unknown>): ActivityProjectionSourceEvent {
  return {
    kind: row.kind as ActivityEventKind,
    occurredAt: Number(row.occurred_at),
    sessionId: row.session_id as SessionId,
    participantLabel: String(row.participant_label ?? "A participant"),
    audienceEntityId:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : (row.audience_entity_id as EntityId),
    sourceStreamEntryIds: parseStreamEntryIds(
      String(row.source_stream_entry_ids ?? "[]"),
      "source_stream_entry_ids",
    ),
  };
}

function mapAutobiographicalRow(row: Record<string, unknown>): ActivityAutobiographicalSourceEvent {
  return {
    id: row.id as ActivityEventId,
    kind: row.kind as ActivityEventKind,
    occurredAt: Number(row.occurred_at),
    sessionId: row.session_id as SessionId,
    sessionSourceType: String(row.session_source_type ?? "unknown"),
    sessionAudienceRole: String(row.session_audience_role ?? "participant"),
    sessionLabel: String(row.session_label ?? "session"),
    participantLabel: String(row.participant_label ?? "A participant"),
    audienceEntityId:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : (row.audience_entity_id as EntityId),
    participantEntityIds: parseEntityIds(
      String(row.participant_entity_ids ?? "[]"),
      "participant_entity_ids",
    ),
    sourceStreamEntryIds: parseStreamEntryIds(
      String(row.source_stream_entry_ids ?? "[]"),
      "source_stream_entry_ids",
    ),
  };
}

function mapDailyDensityRow(row: Record<string, unknown>): ActivityDailyDensityRow {
  const dayKey = String(row.day_key);

  return {
    dayKey,
    dayStartMs: timestampFromUtcDayKey(dayKey),
    sessionId: row.session_id as SessionId,
    sessionLabel: String(row.session_label ?? "session"),
    audienceLabel: String(row.audience_label ?? "A participant"),
    audienceEntityId:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : (row.audience_entity_id as EntityId),
    eventCount: Number(row.event_count),
    conversationTurnCount: Number(row.conversation_turn_count ?? 0),
    kindCounts: {
      userContact: Number(row.user_contact_count ?? 0),
      borgReplied: Number(row.borg_replied_count ?? 0),
      turnCompleted: Number(row.turn_completed_count ?? 0),
    },
    firstOccurredAt: Number(row.first_occurred_at),
    lastOccurredAt: Number(row.last_occurred_at),
  };
}

function mapGlobalDailyDensityRow(row: Record<string, unknown>): ActivityGlobalDailyDensityRow {
  const dayKey = String(row.day_key);

  return {
    dayKey,
    dayStartMs: timestampFromUtcDayKey(dayKey),
    eventCount: Number(row.event_count),
    conversationTurnCount: Number(row.conversation_turn_count ?? 0),
    distinctSessionCount: Number(row.distinct_session_count ?? 0),
    kindCounts: {
      userContact: Number(row.user_contact_count ?? 0),
      borgReplied: Number(row.borg_replied_count ?? 0),
      turnCompleted: Number(row.turn_completed_count ?? 0),
    },
    firstOccurredAt: Number(row.first_occurred_at),
    lastOccurredAt: Number(row.last_occurred_at),
  };
}

export class ActivityRepository {
  private readonly clock: Clock;

  constructor(private readonly options: ActivityRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  record(input: ActivityEventRecordInput): ActivityEvent {
    const sourceStreamEntryIds = uniqueStreamEntryIds(input.sourceStreamEntryIds);

    if (sourceStreamEntryIds.length === 0) {
      throw new StorageError("Activity event requires at least one source stream entry id", {
        code: "ACTIVITY_EVENT_SOURCE_REQUIRED",
      });
    }

    const now = input.now ?? this.clock.now();
    const id = input.id ?? createActivityEventId();
    const participantEntityIds = uniqueEntityIds([
      ...(input.participantEntityIds ?? []),
      input.speakerEntityId ?? null,
      input.actorEntityId ?? null,
      input.audienceEntityId ?? null,
    ]);
    const participantEntityIdsJson = serializeJsonValue(participantEntityIds);
    const sourceStreamEntryIdsJson = serializeJsonValue(sourceStreamEntryIds);

    this.db
      .prepare(
        `
          INSERT INTO activity_events (
            id, kind, occurred_at, session_id, turn_id, speaker_entity_id, actor_entity_id,
            audience_entity_id, participant_entity_ids, source_stream_entry_ids, status,
            created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(kind, source_stream_entry_ids) DO UPDATE SET
            occurred_at = excluded.occurred_at,
            session_id = excluded.session_id,
            turn_id = excluded.turn_id,
            speaker_entity_id = excluded.speaker_entity_id,
            actor_entity_id = excluded.actor_entity_id,
            audience_entity_id = excluded.audience_entity_id,
            participant_entity_ids = excluded.participant_entity_ids,
            status = excluded.status,
            updated_at = excluded.updated_at
        `,
      )
      .run(
        id,
        input.kind,
        input.occurredAt,
        input.sessionId,
        input.turnId ?? null,
        input.speakerEntityId ?? null,
        input.actorEntityId ?? null,
        input.audienceEntityId ?? null,
        participantEntityIdsJson,
        sourceStreamEntryIdsJson,
        input.status ?? "active",
        now,
        now,
      );

    const stored = this.getByKindAndSource(input.kind, sourceStreamEntryIds);

    if (stored === null) {
      throw new StorageError(`Activity event ${id} was not stored`, {
        code: "ACTIVITY_EVENT_STORE_FAILED",
      });
    }

    return stored;
  }

  get(id: ActivityEventId): ActivityEvent | null {
    const row = this.db
      .prepare(
        `
          SELECT
            id, kind, occurred_at, session_id, turn_id, speaker_entity_id, actor_entity_id,
            audience_entity_id, participant_entity_ids, source_stream_entry_ids, status,
            created_at, updated_at
          FROM activity_events
          WHERE id = ?
        `,
      )
      .get(parseActivityEventId(id)) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapActivityRow(row);
  }

  getByKindAndSource(
    kind: ActivityEventKind,
    sourceStreamEntryIds: readonly StreamEntryId[],
  ): ActivityEvent | null {
    const row = this.db
      .prepare(
        `
          SELECT
            id, kind, occurred_at, session_id, turn_id, speaker_entity_id, actor_entity_id,
            audience_entity_id, participant_entity_ids, source_stream_entry_ids, status,
            created_at, updated_at
          FROM activity_events
          WHERE kind = ? AND source_stream_entry_ids = ?
        `,
      )
      .get(kind, serializeJsonValue(uniqueStreamEntryIds(sourceStreamEntryIds))) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapActivityRow(row);
  }

  listRecentOtherActiveSessionEvents(input: {
    currentSessionId: SessionId;
    sinceMs: number;
    limit: number;
  }): ActivityProjectionSourceEvent[] {
    const rows = this.db
      .prepare(
        `
          SELECT
            e.kind,
            e.occurred_at,
            e.session_id,
            e.audience_entity_id,
            e.source_stream_entry_ids,
            COALESCE(speaker.canonical_name, audience.canonical_name, s.audience_label)
              AS participant_label
          FROM activity_events e
          INNER JOIN sessions s ON s.session_id = e.session_id
          LEFT JOIN entities speaker ON speaker.id = e.speaker_entity_id
          LEFT JOIN entities audience ON audience.id = e.audience_entity_id
          WHERE
            e.status = 'active'
            AND s.status = 'active'
            AND e.session_id <> ?
            AND e.occurred_at >= ?
            AND (
              e.kind IN ('user_contact', 'borg_replied')
              OR EXISTS (
                SELECT 1
                FROM activity_events engaged
                WHERE
                  engaged.status = 'active'
                  AND engaged.session_id = e.session_id
                  AND engaged.kind IN ('user_contact', 'borg_replied')
                  AND strftime('%Y-%m-%d', engaged.occurred_at / 1000, 'unixepoch') =
                    strftime('%Y-%m-%d', e.occurred_at / 1000, 'unixepoch')
              )
            )
          ORDER BY
            CASE e.kind
              WHEN 'user_contact' THEN 0
              WHEN 'borg_replied' THEN 1
              ELSE 2
            END ASC,
            e.occurred_at DESC,
            e.id ASC
          LIMIT ?
        `,
      )
      .all(input.currentSessionId, input.sinceMs, input.limit) as Record<string, unknown>[];

    return rows.map(mapProjectionRow);
  }

  getMostRecentOtherActiveSessionEventOccurredAt(input: {
    currentSessionId: SessionId;
    sinceMs: number;
  }): number | null {
    const row = this.db
      .prepare(
        `
          SELECT MAX(e.occurred_at) AS occurred_at
          FROM activity_events e
          INNER JOIN sessions s ON s.session_id = e.session_id
          WHERE
            e.status = 'active'
            AND s.status = 'active'
            AND e.session_id <> ?
            AND e.occurred_at >= ?
        `,
      )
      .get(input.currentSessionId, input.sinceMs) as { occurred_at: number | null } | undefined;

    return row === undefined || row.occurred_at === null ? null : Number(row.occurred_at);
  }

  listDailyOtherActiveSessionDensity(input: {
    currentSessionId: SessionId;
    sinceMs: number;
    untilMs?: number;
    limit: number;
  }): ActivityDailyDensityRow[] {
    const filters = [
      "e.status = 'active'",
      "s.status = 'active'",
      "e.session_id <> ?",
      "e.occurred_at >= ?",
    ];
    const values: unknown[] = [input.currentSessionId, input.sinceMs];

    if (input.untilMs !== undefined) {
      filters.push("e.occurred_at <= ?");
      values.push(input.untilMs);
    }

    values.push(Math.max(1, Math.floor(input.limit)));

    const rows = this.db
      .prepare(
        `
          SELECT
            strftime('%Y-%m-%d', e.occurred_at / 1000, 'unixepoch') AS day_key,
            e.session_id,
            s.label AS session_label,
            s.audience_label,
            s.audience_entity_id,
            COUNT(*) AS event_count,
            SUM(CASE WHEN e.kind = 'user_contact' THEN 1 ELSE 0 END) AS user_contact_count,
            SUM(CASE WHEN e.kind = 'borg_replied' THEN 1 ELSE 0 END) AS borg_replied_count,
            SUM(CASE WHEN e.kind = 'turn_completed' THEN 1 ELSE 0 END) AS turn_completed_count,
            COUNT(DISTINCT CASE
              WHEN e.kind IN ('user_contact', 'borg_replied') THEN COALESCE(e.turn_id, e.id)
              ELSE NULL
            END) AS conversation_turn_count,
            MIN(e.occurred_at) AS first_occurred_at,
            MAX(e.occurred_at) AS last_occurred_at
          FROM activity_events e
          INNER JOIN sessions s ON s.session_id = e.session_id
          WHERE ${filters.join(" AND ")}
          GROUP BY day_key, e.session_id
          HAVING user_contact_count > 0 OR borg_replied_count > 0
          ORDER BY last_occurred_at DESC, e.session_id ASC
          LIMIT ?
        `,
      )
      .all(...values) as Record<string, unknown>[];

    return rows.map(mapDailyDensityRow);
  }

  listDailyGlobalActiveDensity(input: {
    sinceMs: number;
    untilMs?: number;
    limit: number;
  }): ActivityGlobalDailyDensityRow[] {
    const filters = ["e.status = 'active'", "s.status = 'active'", "e.occurred_at >= ?"];
    const values: unknown[] = [input.sinceMs];

    if (input.untilMs !== undefined) {
      filters.push("e.occurred_at <= ?");
      values.push(input.untilMs);
    }

    values.push(Math.max(1, Math.floor(input.limit)));

    const rows = this.db
      .prepare(
        `
          SELECT
            strftime('%Y-%m-%d', e.occurred_at / 1000, 'unixepoch') AS day_key,
            COUNT(*) AS event_count,
            COUNT(DISTINCT e.session_id) AS distinct_session_count,
            SUM(CASE WHEN e.kind = 'user_contact' THEN 1 ELSE 0 END) AS user_contact_count,
            SUM(CASE WHEN e.kind = 'borg_replied' THEN 1 ELSE 0 END) AS borg_replied_count,
            SUM(CASE WHEN e.kind = 'turn_completed' THEN 1 ELSE 0 END) AS turn_completed_count,
            COUNT(DISTINCT CASE
              WHEN e.kind IN ('user_contact', 'borg_replied') THEN
                e.session_id || '|' || COALESCE(e.turn_id, e.id)
              ELSE NULL
            END) AS conversation_turn_count,
            MIN(e.occurred_at) AS first_occurred_at,
            MAX(e.occurred_at) AS last_occurred_at
          FROM activity_events e
          INNER JOIN sessions s ON s.session_id = e.session_id
          WHERE ${filters.join(" AND ")}
          GROUP BY day_key
          HAVING user_contact_count > 0 OR borg_replied_count > 0
          ORDER BY last_occurred_at DESC
          LIMIT ?
        `,
      )
      .all(...values) as Record<string, unknown>[];

    return rows.map(mapGlobalDailyDensityRow);
  }

  listRecentGlobalEvents(input: {
    sinceMs: number;
    untilMs?: number;
    limit: number;
  }): ActivityAutobiographicalSourceEvent[] {
    const filters = ["e.status = 'active'", "e.occurred_at >= ?"];
    const values: unknown[] = [input.sinceMs];

    if (input.untilMs !== undefined) {
      filters.push("e.occurred_at <= ?");
      values.push(input.untilMs);
    }

    values.push(Math.max(1, Math.floor(input.limit)));

    const rows = this.db
      .prepare(
        `
          SELECT
            e.id,
            e.kind,
            e.occurred_at,
            e.session_id,
            e.audience_entity_id,
            e.participant_entity_ids,
            e.source_stream_entry_ids,
            s.source_type AS session_source_type,
            s.audience_role AS session_audience_role,
            s.label AS session_label,
            COALESCE(speaker.canonical_name, audience.canonical_name, s.audience_label)
              AS participant_label
          FROM activity_events e
          INNER JOIN sessions s ON s.session_id = e.session_id
          LEFT JOIN entities speaker ON speaker.id = e.speaker_entity_id
          LEFT JOIN entities audience ON audience.id = e.audience_entity_id
          WHERE ${filters.join(" AND ")}
          ORDER BY e.occurred_at DESC, e.id ASC
          LIMIT ?
        `,
      )
      .all(...values) as Record<string, unknown>[];

    return rows.map(mapAutobiographicalRow);
  }
}
