import type { EmbeddingClient } from "../../embeddings/index.js";
import { z } from "zod";
import {
  parseJsonArray,
  quoteSqlString,
  toFloat32Array,
  type Float32ArrayCodecOptions,
  type JsonArrayCodecOptions,
} from "../../storage/codecs.js";
import {
  LanceDbTable,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import { getDistance, toSimilarity } from "../../storage/lancedb/vector-results.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createObservedEventId,
  observedEventIdHelpers,
  parseObservedEventId,
  type EntityId,
  type ObservedEventId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  observedEventBeliefEffectInputSchema,
  observedEventSchema,
  observedEventStanceInputSchema,
  observedEventTaintInputSchema,
  type ObservedEvent,
  type ObservedEventBeliefEffect,
  type ObservedEventDisclosureClass,
  type ObservedEventStance,
  type ObservedEventTaint,
} from "./types.js";

const OBSERVED_EVENT_JSON_ARRAY_CODEC = {
  errorCode: "OBSERVED_EVENT_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse observed event ${label}`,
} satisfies JsonArrayCodecOptions;

export type ObservedEventRecordInput = {
  id?: ObservedEventId;
  occurredAt: number;
  sessionId: SessionId;
  stance: ObservedEventStance;
  taint: ObservedEventTaint;
  beliefEffect: ObservedEventBeliefEffect;
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

const observedEventRecordDimensionsSchema = z
  .object({
    stance: observedEventStanceInputSchema,
    taint: observedEventTaintInputSchema,
    beliefEffect: observedEventBeliefEffectInputSchema,
  })
  .strict();

export type ObservedEventProjectionSourceEvent = {
  id: ObservedEventId;
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
  sourceStreamEntryIds: readonly StreamEntryId[];
};

export type ObservedEventSearchCandidate = {
  event: ObservedEventProjectionSourceEvent;
  similarity: number;
};

export type ObservedEventEmbeddingBackfillReport = {
  scanned: number;
  embedded: number;
  skipped: number;
  failed: number;
};

export type ObservedEventEmbeddingFailureDetails = {
  operation: "insert" | "metadata_sync" | "backfill";
  eventId: ObservedEventId;
  interactionText: string;
};

export type ObservedEventRepositoryOptions = {
  db: SqliteDatabase;
  table?: LanceDbTable;
  embeddingClient?: EmbeddingClient;
  clock?: Clock;
  onEmbeddingFailure?: (
    error: unknown,
    details: ObservedEventEmbeddingFailureDetails,
  ) => void | Promise<void>;
};

type ObservedEventVectorRow = {
  id: string;
  disclosure_class: string;
  stance: string;
  taint: string;
  belief_effect: string;
  classification_kind: string;
  interaction_text: string;
  speaker_entity_id: string | null;
  audience_entity_id: string | null;
  recurrence_count: number;
  last_seen_at: number;
  embedding: number[];
  _distance?: number;
};

const OBSERVED_EVENT_VECTOR_CODEC = {
  arrayLikeErrorMessage: "Observed event embedding must be array-like",
  nonFiniteErrorMessage: "Observed event embedding contains a non-finite value",
  errorCode: "OBSERVED_EVENT_ROW_INVALID",
} satisfies Float32ArrayCodecOptions;

export function createObservedEventsTableSchema(dimensions: number) {
  return schema([
    utf8Field("id"),
    utf8Field("disclosure_class"),
    utf8Field("stance"),
    utf8Field("taint"),
    utf8Field("belief_effect"),
    utf8Field("classification_kind"),
    utf8Field("interaction_text"),
    utf8Field("speaker_entity_id", true),
    utf8Field("audience_entity_id", true),
    float64Field("recurrence_count"),
    float64Field("last_seen_at"),
    vectorField("embedding", dimensions),
  ]);
}

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
    id: parseObservedEventId(String(row.id)),
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
    sourceStreamEntryIds: parseStreamEntryIds(
      String(row.source_stream_entry_ids ?? "[]"),
      "source_stream_entry_ids",
    ),
  };
}

function projectionEventFromObservedEvent(
  event: ObservedEvent,
): ObservedEventProjectionSourceEvent {
  return {
    id: event.id,
    occurredAt: event.occurred_at,
    lastSeenAt: event.last_seen_at,
    stance: event.stance,
    taint: event.taint,
    beliefEffect: event.belief_effect,
    disclosureClass: event.disclosure_class,
    interactionText: event.interaction_text,
    recurrenceCount: event.recurrence_count,
    speakerEntityId: event.speaker_entity_id,
    audienceEntityId: event.audience_entity_id,
    sourceStreamEntryIds: event.source_stream_entry_ids,
  };
}

function vectorRowFromObservedEvent(
  event: ObservedEvent,
  embedding: Float32Array,
): ObservedEventVectorRow {
  return {
    id: event.id,
    disclosure_class: event.disclosure_class,
    stance: event.stance,
    taint: event.taint,
    belief_effect: event.belief_effect,
    classification_kind: event.classification_kind,
    interaction_text: event.interaction_text,
    speaker_entity_id: event.speaker_entity_id,
    audience_entity_id: event.audience_entity_id,
    recurrence_count: event.recurrence_count,
    last_seen_at: event.last_seen_at,
    embedding: Array.from(embedding),
  };
}

export class ObservedEventRepository {
  private readonly clock: Clock;
  private readonly pendingEmbeddingTasks = new Set<Promise<void>>();

  constructor(private readonly options: ObservedEventRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private get table(): LanceDbTable | undefined {
    return this.options.table;
  }

  private get embeddingClient(): EmbeddingClient | undefined {
    return this.options.embeddingClient;
  }

  private hasVectorStorage(): boolean {
    return this.table !== undefined && this.embeddingClient !== undefined;
  }

  private enqueueEmbeddingTask(task: Promise<void>): void {
    this.pendingEmbeddingTasks.add(task);
    void task.finally(() => {
      this.pendingEmbeddingTasks.delete(task);
    });
  }

  private reportEmbeddingFailure(
    error: unknown,
    details: ObservedEventEmbeddingFailureDetails,
  ): void {
    try {
      void Promise.resolve(this.options.onEmbeddingFailure?.(error, details)).catch(() => {
        // Best-effort failure reporting only.
      });
    } catch {
      // Best-effort failure reporting only.
    }
  }

  private async readStoredEmbedding(id: ObservedEventId): Promise<Float32Array | null> {
    const table = this.table;

    if (table === undefined) {
      return null;
    }

    const [row] = await table.list({
      where: `id = ${quoteSqlString(id)}`,
      limit: 1,
      columns: ["embedding"],
    });

    if (row === undefined) {
      return null;
    }

    return toFloat32Array(row.embedding, OBSERVED_EVENT_VECTOR_CODEC);
  }

  private async upsertEventVector(
    event: ObservedEvent,
    operation: ObservedEventEmbeddingFailureDetails["operation"],
    options: { forceEmbed?: boolean; skipIfMissing?: boolean } = {},
  ): Promise<void> {
    const table = this.table;
    const embeddingClient = this.embeddingClient;

    if (table === undefined || embeddingClient === undefined) {
      return;
    }

    try {
      const storedEmbedding =
        options.forceEmbed === true ? null : await this.readStoredEmbedding(event.id);

      if (options.skipIfMissing === true && storedEmbedding === null) {
        return;
      }

      const embedding = storedEmbedding ?? (await embeddingClient.embed(event.interaction_text));

      await table.upsert([vectorRowFromObservedEvent(event, embedding)], {
        on: "id",
      });
    } catch (error) {
      this.reportEmbeddingFailure(error, {
        operation,
        eventId: event.id,
        interactionText: event.interaction_text,
      });
    }
  }

  private scheduleEventVectorUpsert(
    event: ObservedEvent,
    operation: ObservedEventEmbeddingFailureDetails["operation"],
    options: { forceEmbed?: boolean; skipIfMissing?: boolean } = {},
  ): void {
    if (!this.hasVectorStorage()) {
      return;
    }

    this.enqueueEmbeddingTask(this.upsertEventVector(event, operation, options));
  }

  async waitForPendingEmbeddings(): Promise<void> {
    await Promise.allSettled([...this.pendingEmbeddingTasks]);
  }

  async getEmbeddedEventIds(ids: readonly ObservedEventId[]): Promise<Set<ObservedEventId>> {
    const table = this.table;
    const uniqueIds = [...new Set(ids)];

    if (table === undefined || uniqueIds.length === 0) {
      return new Set();
    }

    const rows = await table.list({
      where: `id IN (${uniqueIds.map((id) => quoteSqlString(id)).join(", ")})`,
      columns: ["id"],
    });

    return new Set(
      rows
        .map((row) => String(row.id))
        .filter((id) => observedEventIdHelpers.is(id))
        .map((id) => parseObservedEventId(id)),
    );
  }

  async backfillMissingEmbeddings(
    options: { limit?: number } = {},
  ): Promise<ObservedEventEmbeddingBackfillReport> {
    const table = this.table;
    const embeddingClient = this.embeddingClient;
    const report: ObservedEventEmbeddingBackfillReport = {
      scanned: 0,
      embedded: 0,
      skipped: 0,
      failed: 0,
    };

    if (table === undefined || embeddingClient === undefined) {
      return report;
    }

    const limitClause = options.limit === undefined ? "" : "LIMIT ?";
    const rows = this.db
      .prepare(
        `
          SELECT
            id, occurred_at, session_id, stance, taint, belief_effect, classification_kind,
            disclosure_class, interaction_text, recurrence_key, fire_dedup_key, recurrence_count,
            last_seen_at, speaker_entity_id, audience_entity_id, source_entity_id,
            source_stream_entry_ids, created_at, updated_at
          FROM observed_events
          ORDER BY created_at ASC, id ASC
          ${limitClause}
        `,
      )
      .all(...(options.limit === undefined ? [] : [Math.max(1, options.limit)])) as Record<
      string,
      unknown
    >[];
    const events = rows.map((row) => mapObservedEventRow(row));
    const existingIds = await this.getEmbeddedEventIds(events.map((event) => event.id));

    for (const event of events) {
      report.scanned += 1;

      if (existingIds.has(event.id)) {
        report.skipped += 1;
        continue;
      }

      try {
        const embedding = await embeddingClient.embed(event.interaction_text);
        await table.upsert([vectorRowFromObservedEvent(event, embedding)], {
          on: "id",
        });
        report.embedded += 1;
      } catch (error) {
        report.failed += 1;
        this.reportEmbeddingFailure(error, {
          operation: "backfill",
          eventId: event.id,
          interactionText: event.interaction_text,
        });
      }
    }

    return report;
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
    const dimensions = observedEventRecordDimensionsSchema.safeParse({
      stance: input.stance,
      taint: input.taint,
      beliefEffect: input.beliefEffect,
    });

    if (!dimensions.success) {
      throw new StorageError("Observed event dimensions failed validation", {
        cause: dimensions.error,
        code: "OBSERVED_EVENT_INVALID",
      });
    }

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
        dimensions.data.stance,
        dimensions.data.taint,
        dimensions.data.beliefEffect,
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

    this.scheduleEventVectorUpsert(stored, "insert");

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

  listRecentGlobal(input: {
    disclosureClass: ObservedEventDisclosureClass;
    sinceMs: number;
    limit: number;
  }): ObservedEventProjectionSourceEvent[] {
    const rows = this.db
      .prepare(
        `
          SELECT
            id, occurred_at, last_seen_at, stance, taint, belief_effect, disclosure_class,
            interaction_text, recurrence_count, speaker_entity_id, audience_entity_id,
            source_stream_entry_ids
          FROM observed_events
          WHERE
            disclosure_class = ?
            AND last_seen_at >= ?
          ORDER BY last_seen_at DESC, id DESC
          LIMIT ?
        `,
      )
      .all(input.disclosureClass, input.sinceMs, input.limit) as Record<string, unknown>[];

    return rows.map(mapProjectionRow);
  }

  listRecurringGlobal(input: {
    disclosureClass: ObservedEventDisclosureClass;
    sinceMs: number;
    limit: number;
  }): ObservedEventProjectionSourceEvent[] {
    const rows = this.db
      .prepare(
        `
          SELECT
            id, occurred_at, last_seen_at, stance, taint, belief_effect, disclosure_class,
            interaction_text, recurrence_count, speaker_entity_id, audience_entity_id,
            source_stream_entry_ids
          FROM observed_events
          WHERE
            disclosure_class = ?
            AND last_seen_at >= ?
            AND recurrence_count > 1
          ORDER BY recurrence_count DESC, last_seen_at DESC, id DESC
          LIMIT ?
        `,
      )
      .all(input.disclosureClass, input.sinceMs, input.limit) as Record<string, unknown>[];

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
            id, occurred_at, last_seen_at, stance, taint, belief_effect, disclosure_class,
            interaction_text, recurrence_count, speaker_entity_id, audience_entity_id,
            source_stream_entry_ids
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

  async searchByVector(
    vector: Float32Array,
    options: {
      limit?: number;
      minSimilarity?: number;
    } = {},
  ): Promise<ObservedEventSearchCandidate[]> {
    const table = this.table;

    if (table === undefined) {
      return [];
    }

    const limit = Math.max(1, options.limit ?? 10);
    const rows = (await table.search(Array.from(vector), {
      limit: Math.max(limit * 5, 20),
      vectorColumn: "embedding",
      distanceType: "cosine",
    })) as ObservedEventVectorRow[];
    const results: ObservedEventSearchCandidate[] = [];

    for (const row of rows) {
      const id = String(row.id);

      if (!observedEventIdHelpers.is(id)) {
        continue;
      }

      const event = this.get(parseObservedEventId(id));

      if (event === null) {
        continue;
      }

      const similarity = toSimilarity(getDistance(row));

      if (options.minSimilarity !== undefined && similarity < options.minSimilarity) {
        continue;
      }

      results.push({
        event: projectionEventFromObservedEvent(event),
        similarity,
      });

      if (results.length >= limit) {
        break;
      }
    }

    return results;
  }
}
