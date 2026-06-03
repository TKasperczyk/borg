import { parseJsonArray, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createObservedEventId,
  parseObservedEventId,
  type EntityId,
  type ObservedEventId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  observedEventSchema,
  type ObservedEvent,
  type ObservedEventDisclosureClass,
} from "./types.js";

const OBSERVED_EVENT_JSON_ARRAY_CODEC = {
  errorCode: "OBSERVED_EVENT_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse observed event ${label}`,
} satisfies JsonArrayCodecOptions;

export type ObservedEventRecordInput = {
  id?: ObservedEventId;
  occurredAt: number;
  sessionId: SessionId;
  stance: string;
  taint: string;
  beliefEffect: string;
  classificationKind: string;
  disclosureClass: ObservedEventDisclosureClass;
  interactionText: string;
  recurrenceKey: string;
  fireDedupKey?: string;
  speakerEntityId?: EntityId | null;
  audienceEntityId?: EntityId | null;
  sourceEntityId?: EntityId | null;
  sourceStreamEntryIds: readonly StreamEntryId[];
  now?: number;
};

export type ObservedEventProjectionSourceEvent = {
  occurredAt: number;
  lastSeenAt: number;
  stance: string;
  taint: string;
  beliefEffect: string;
  disclosureClass: ObservedEventDisclosureClass;
  interactionText: string;
  recurrenceCount: number;
  speakerEntityId: EntityId | null;
  audienceEntityId: EntityId | null;
};

export type ObservedEventRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function parseStreamEntryIds(value: string, label: string): StreamEntryId[] {
  return parseJsonArray<StreamEntryId>(value, label, OBSERVED_EVENT_JSON_ARRAY_CODEC);
}

function uniqueStreamEntryIds(values: readonly StreamEntryId[]): StreamEntryId[] {
  return dedupePreservingOrder(values);
}

function nullableRowValue(value: unknown): unknown {
  return value === null || value === undefined ? null : value;
}

function mapObservedEventRow(row: Record<string, unknown>): ObservedEvent {
  const parsed = observedEventSchema.safeParse({
    id: row.id,
    occurred_at: Number(row.occurred_at),
    session_id: row.session_id,
    stance: row.stance,
    taint: row.taint,
    belief_effect: row.belief_effect,
    classification_kind: row.classification_kind,
    disclosure_class: row.disclosure_class,
    interaction_text: row.interaction_text,
    recurrence_key: row.recurrence_key,
    fire_dedup_key: nullableRowValue(row.fire_dedup_key),
    recurrence_count: Number(row.recurrence_count),
    last_seen_at: Number(row.last_seen_at),
    speaker_entity_id: nullableRowValue(row.speaker_entity_id),
    audience_entity_id: nullableRowValue(row.audience_entity_id),
    source_entity_id: nullableRowValue(row.source_entity_id),
    source_stream_entry_ids: parseStreamEntryIds(
      String(row.source_stream_entry_ids ?? "[]"),
      "source_stream_entry_ids",
    ),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Observed event row failed validation", {
      cause: parsed.error,
      code: "OBSERVED_EVENT_ROW_INVALID",
    });
  }

  return parsed.data;
}

function mapProjectionRow(row: Record<string, unknown>): ObservedEventProjectionSourceEvent {
  return {
    occurredAt: Number(row.occurred_at),
    lastSeenAt: Number(row.last_seen_at),
    stance: String(row.stance),
    taint: String(row.taint),
    beliefEffect: String(row.belief_effect),
    disclosureClass: row.disclosure_class as ObservedEventDisclosureClass,
    interactionText: String(row.interaction_text ?? ""),
    recurrenceCount: Number(row.recurrence_count),
    speakerEntityId: nullableRowValue(row.speaker_entity_id) as EntityId | null,
    audienceEntityId: nullableRowValue(row.audience_entity_id) as EntityId | null,
  };
}

export class ObservedEventRepository {
  private readonly clock: Clock;

  constructor(private readonly options: ObservedEventRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  record(input: ObservedEventRecordInput): ObservedEvent {
    if (input.fireDedupKey !== undefined) {
      const duplicate = this.getByFireDedupKey(input.fireDedupKey);

      if (duplicate !== null) {
        return duplicate;
      }
    }

    const sourceStreamEntryIds = uniqueStreamEntryIds(input.sourceStreamEntryIds);

    if (sourceStreamEntryIds.length === 0) {
      throw new StorageError("Observed event requires at least one source stream entry id", {
        code: "OBSERVED_EVENT_SOURCE_REQUIRED",
      });
    }

    const now = input.now ?? this.clock.now();
    const id = input.id ?? createObservedEventId();

    this.db
      .prepare(
        `
          INSERT INTO observed_events (
            id, occurred_at, session_id, stance, taint, belief_effect, classification_kind,
            disclosure_class, interaction_text, recurrence_key, fire_dedup_key, recurrence_count,
            last_seen_at, speaker_entity_id, audience_entity_id, source_entity_id,
            source_stream_entry_ids, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT(recurrence_key) DO UPDATE SET
            recurrence_count = observed_events.recurrence_count + 1,
            fire_dedup_key = COALESCE(excluded.fire_dedup_key, observed_events.fire_dedup_key),
            last_seen_at = excluded.last_seen_at,
            updated_at = excluded.updated_at
        `,
      )
      .run(
        id,
        input.occurredAt,
        input.sessionId,
        input.stance,
        input.taint,
        input.beliefEffect,
        input.classificationKind,
        input.disclosureClass,
        input.interactionText,
        input.recurrenceKey,
        input.fireDedupKey ?? null,
        input.occurredAt,
        input.speakerEntityId ?? null,
        input.audienceEntityId ?? null,
        input.sourceEntityId ?? null,
        serializeJsonValue(sourceStreamEntryIds),
        now,
        now,
      );

    const stored = this.getByRecurrenceKey(input.recurrenceKey);

    if (stored === null) {
      throw new StorageError(`Observed event ${id} was not stored`, {
        code: "OBSERVED_EVENT_STORE_FAILED",
      });
    }

    return stored;
  }

  get(id: ObservedEventId): ObservedEvent | null {
    const row = this.db
      .prepare(
        `
          SELECT
            id, occurred_at, session_id, stance, taint, belief_effect, classification_kind,
            disclosure_class, interaction_text, recurrence_key, fire_dedup_key, recurrence_count,
            last_seen_at, speaker_entity_id, audience_entity_id, source_entity_id,
            source_stream_entry_ids, created_at, updated_at
          FROM observed_events
          WHERE id = ?
        `,
      )
      .get(parseObservedEventId(id)) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapObservedEventRow(row);
  }

  getByRecurrenceKey(recurrenceKey: string): ObservedEvent | null {
    const row = this.db
      .prepare(
        `
          SELECT
            id, occurred_at, session_id, stance, taint, belief_effect, classification_kind,
            disclosure_class, interaction_text, recurrence_key, fire_dedup_key, recurrence_count,
            last_seen_at, speaker_entity_id, audience_entity_id, source_entity_id,
            source_stream_entry_ids, created_at, updated_at
          FROM observed_events
          WHERE recurrence_key = ?
        `,
      )
      .get(recurrenceKey) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapObservedEventRow(row);
  }

  getByFireDedupKey(fireDedupKey: string): ObservedEvent | null {
    const row = this.db
      .prepare(
        `
          SELECT
            id, occurred_at, session_id, stance, taint, belief_effect, classification_kind,
            disclosure_class, interaction_text, recurrence_key, fire_dedup_key, recurrence_count,
            last_seen_at, speaker_entity_id, audience_entity_id, source_entity_id,
            source_stream_entry_ids, created_at, updated_at
          FROM observed_events
          WHERE fire_dedup_key = ?
        `,
      )
      .get(fireDedupKey) as Record<string, unknown> | undefined;

    return row === undefined ? null : mapObservedEventRow(row);
  }

  listRecentForSession(input: {
    sessionId: SessionId;
    disclosureClass: ObservedEventDisclosureClass;
    sinceMs: number;
    limit: number;
  }): ObservedEventProjectionSourceEvent[] {
    const rows = this.db
      .prepare(
        `
          SELECT
            occurred_at, last_seen_at, stance, taint, belief_effect, disclosure_class,
            interaction_text, recurrence_count, speaker_entity_id, audience_entity_id
          FROM observed_events
          WHERE
            session_id = ?
            AND disclosure_class = ?
            AND last_seen_at >= ?
          ORDER BY last_seen_at DESC, id DESC
          LIMIT ?
        `,
      )
      .all(input.sessionId, input.disclosureClass, input.sinceMs, input.limit) as Record<
      string,
      unknown
    >[];

    return rows.map(mapProjectionRow);
  }

  listRecentBySpeakers(input: {
    speakerEntityIds: readonly EntityId[];
    disclosureClass: ObservedEventDisclosureClass;
    sinceMs: number;
    limit: number;
  }): ObservedEventProjectionSourceEvent[] {
    if (input.speakerEntityIds.length === 0) {
      return [];
    }

    const placeholders = input.speakerEntityIds.map(() => "?").join(", ");
    const rows = this.db
      .prepare(
        `
          SELECT
            occurred_at, last_seen_at, stance, taint, belief_effect, disclosure_class,
            interaction_text, recurrence_count, speaker_entity_id, audience_entity_id
          FROM observed_events
          WHERE
            speaker_entity_id IN (${placeholders})
            AND disclosure_class = ?
            AND last_seen_at >= ?
          ORDER BY last_seen_at DESC, id DESC
          LIMIT ?
        `,
      )
      .all(...input.speakerEntityIds, input.disclosureClass, input.sinceMs, input.limit) as Record<
      string,
      unknown
    >[];

    return rows.map(mapProjectionRow);
  }
}
