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
  participantLabel: string;
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
    participantLabel: String(row.participant_label ?? "A participant"),
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
            s.audience_label AS participant_label
          FROM activity_events e
          INNER JOIN sessions s ON s.session_id = e.session_id
          WHERE
            e.status = 'active'
            AND s.status = 'active'
            AND e.session_id <> ?
            AND e.occurred_at >= ?
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
}
