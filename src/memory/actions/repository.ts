import { parseJsonArray, quoteSqlString, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import {
  LanceDbTable,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createActionId,
  parseActionId,
  type ActionId,
  type EntityId,
  type EpisodeId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  actionRecordPatchSchema,
  actionRecordSchema,
  actionStateSchema,
  type ActionActor,
  type ActionRecord,
  type ActionRecordPatch,
  type ActionState,
} from "./types.js";

const ACTION_JSON_ARRAY_CODEC = {
  errorCode: "ACTION_RECORD_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse action record ${label}`,
} satisfies JsonArrayCodecOptions;

function toSimilarity(distance: number | undefined): number {
  if (distance === undefined) {
    return 0;
  }

  return Math.max(0, Math.min(1, 1 - distance));
}

function getDistance(row: Record<string, unknown>): number | undefined {
  const value = row._distance;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function mapActionRow(row: Record<string, unknown>): ActionRecord {
  const parsed = actionRecordSchema.safeParse({
    id: row.id,
    description: row.description,
    actor: row.actor,
    audience_entity_id:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : row.audience_entity_id,
    state: row.state,
    confidence: Number(row.confidence),
    provenance_episode_ids: parseJsonArray<EpisodeId>(
      String(row.provenance_episode_ids ?? "[]"),
      "provenance_episode_ids",
      ACTION_JSON_ARRAY_CODEC,
    ),
    provenance_stream_entry_ids: parseJsonArray<StreamEntryId>(
      String(row.provenance_stream_entry_ids ?? "[]"),
      "provenance_stream_entry_ids",
      ACTION_JSON_ARRAY_CODEC,
    ),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
    considering_at:
      row.considering_at === null || row.considering_at === undefined
        ? null
        : Number(row.considering_at),
    committed_at:
      row.committed_at === null || row.committed_at === undefined ? null : Number(row.committed_at),
    scheduled_at:
      row.scheduled_at === null || row.scheduled_at === undefined ? null : Number(row.scheduled_at),
    completed_at:
      row.completed_at === null || row.completed_at === undefined ? null : Number(row.completed_at),
    not_done_at:
      row.not_done_at === null || row.not_done_at === undefined ? null : Number(row.not_done_at),
    unknown_at:
      row.unknown_at === null || row.unknown_at === undefined ? null : Number(row.unknown_at),
  });

  if (!parsed.success) {
    throw new StorageError("Action record row failed validation", {
      cause: parsed.error,
      code: "ACTION_RECORD_ROW_INVALID",
    });
  }

  return parsed.data;
}

function vectorRowFromAction(record: ActionRecord, embedding: Float32Array) {
  return {
    id: record.id,
    description: record.description,
    actor: record.actor,
    state: record.state,
    audience_entity_id: record.audience_entity_id,
    updated_at: record.updated_at,
    embedding: Array.from(embedding),
  };
}

type ActionStateTimestampField =
  | "considering_at"
  | "committed_at"
  | "scheduled_at"
  | "completed_at"
  | "not_done_at"
  | "unknown_at";

function stateTimestampField(state: ActionState): ActionStateTimestampField {
  switch (state) {
    case "considering":
      return "considering_at";
    case "committed_to_do":
      return "committed_at";
    case "scheduled":
      return "scheduled_at";
    case "completed":
      return "completed_at";
    case "not_done":
      return "not_done_at";
    case "unknown":
      return "unknown_at";
  }
}

export type ActionRecordListFilter = {
  state?: ActionState;
  actor?: ActionActor;
  audienceEntityId?: EntityId | null;
  limit?: number;
};

export type ActionRepositoryOptions = {
  db: SqliteDatabase;
  table?: LanceDbTable;
  embeddingClient?: EmbeddingClient;
  clock?: Clock;
};

export function createActionRecordsTableSchema(dimensions: number) {
  return schema([
    utf8Field("id"),
    utf8Field("description"),
    utf8Field("actor"),
    utf8Field("state"),
    utf8Field("audience_entity_id", true),
    float64Field("updated_at"),
    vectorField("embedding", dimensions),
  ]);
}

export class ActionRepository {
  private readonly clock: Clock;
  private readonly pendingEmbeddingTasks = new Set<Promise<void>>();

  constructor(private readonly options: ActionRepositoryOptions) {
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

  private enqueueEmbeddingTask(task: Promise<void>): void {
    this.pendingEmbeddingTasks.add(task);
    void task.finally(() => {
      this.pendingEmbeddingTasks.delete(task);
    });
  }

  private scheduleVectorUpsert(record: ActionRecord): void {
    const table = this.table;
    const embeddingClient = this.embeddingClient;

    if (table === undefined || embeddingClient === undefined) {
      return;
    }

    this.enqueueEmbeddingTask(
      (async () => {
        try {
          const embedding = await embeddingClient.embed(record.description);
          await table.upsert([vectorRowFromAction(record, embedding)], {
            on: "id",
          });
        } catch {
          // SQL is the source of truth; vector refresh can retry on a later update.
        }
      })(),
    );
  }

  async waitForPendingEmbeddings(): Promise<void> {
    await Promise.allSettled([...this.pendingEmbeddingTasks]);
  }

  private upsertSqlRow(record: ActionRecord): void {
    this.db
      .prepare(
        `
          INSERT INTO action_records (
            id, description, actor, audience_entity_id, state, confidence,
            provenance_episode_ids, provenance_stream_entry_ids, created_at, updated_at,
            considering_at, committed_at, scheduled_at, completed_at, not_done_at, unknown_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (id) DO UPDATE SET
            description = excluded.description,
            actor = excluded.actor,
            audience_entity_id = excluded.audience_entity_id,
            state = excluded.state,
            confidence = excluded.confidence,
            provenance_episode_ids = excluded.provenance_episode_ids,
            provenance_stream_entry_ids = excluded.provenance_stream_entry_ids,
            updated_at = excluded.updated_at,
            considering_at = excluded.considering_at,
            committed_at = excluded.committed_at,
            scheduled_at = excluded.scheduled_at,
            completed_at = excluded.completed_at,
            not_done_at = excluded.not_done_at,
            unknown_at = excluded.unknown_at
        `,
      )
      .run(
        record.id,
        record.description,
        record.actor,
        record.audience_entity_id,
        record.state,
        record.confidence,
        serializeJsonValue(record.provenance_episode_ids),
        serializeJsonValue(record.provenance_stream_entry_ids),
        record.created_at,
        record.updated_at,
        record.considering_at,
        record.committed_at,
        record.scheduled_at,
        record.completed_at,
        record.not_done_at,
        record.unknown_at,
      );
  }

  add(record: ActionRecord): void {
    const parsed = actionRecordSchema.parse({
      ...record,
      id: record.id ?? createActionId(),
    });

    this.upsertSqlRow(parsed);
    this.scheduleVectorUpsert(parsed);
  }

  update(id: ActionId, patch: ActionRecordPatch): void {
    const current = this.get(id);

    if (current === null) {
      throw new StorageError(`Unknown action record id: ${id}`, {
        code: "ACTION_RECORD_NOT_FOUND",
      });
    }

    const parsedPatch = actionRecordPatchSchema.parse(patch);
    const nextState = parsedPatch.state ?? current.state;
    const nowMs = parsedPatch.updated_at ?? this.clock.now();
    const timestampField = stateTimestampField(nextState);
    const next = actionRecordSchema.parse({
      ...current,
      ...parsedPatch,
      state: nextState,
      updated_at: nowMs,
      ...(parsedPatch.state === undefined || parsedPatch[timestampField] !== undefined
        ? {}
        : { [timestampField]: nowMs }),
    });

    this.upsertSqlRow(next);
    this.scheduleVectorUpsert(next);
  }

  get(id: ActionId): ActionRecord | null {
    const row = this.db.prepare("SELECT * FROM action_records WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapActionRow(row);
  }

  private getMany(ids: readonly ActionId[]): Array<ActionRecord | null> {
    if (ids.length === 0) {
      return [];
    }

    const rows = this.db
      .prepare(`SELECT * FROM action_records WHERE id IN (${ids.map(() => "?").join(", ")})`)
      .all(...ids) as Record<string, unknown>[];
    const byId = new Map(rows.map((row) => [String(row.id), mapActionRow(row)]));

    return ids.map((id) => byId.get(id) ?? null);
  }

  list(filter: ActionRecordListFilter = {}): ActionRecord[] {
    const clauses: string[] = [];
    const values: unknown[] = [];

    if (filter.state !== undefined) {
      clauses.push("state = ?");
      values.push(actionStateSchema.parse(filter.state));
    }

    if (filter.actor !== undefined) {
      clauses.push("actor = ?");
      values.push(filter.actor);
    }

    if ("audienceEntityId" in filter) {
      if (filter.audienceEntityId === null) {
        clauses.push("audience_entity_id IS NULL");
      } else if (filter.audienceEntityId !== undefined) {
        clauses.push("audience_entity_id = ?");
        values.push(filter.audienceEntityId);
      }
    }

    const limit = filter.limit === undefined ? null : Math.max(1, Math.floor(filter.limit));
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM action_records
          ${clauses.length === 0 ? "" : `WHERE ${clauses.join(" AND ")}`}
          ORDER BY updated_at DESC, id ASC
          ${limit === null ? "" : "LIMIT ?"}
        `,
      )
      .all(...values, ...(limit === null ? [] : [limit])) as Record<string, unknown>[];

    return rows.map((row) => mapActionRow(row));
  }

  async findByDescription(description: string, limit: number): Promise<ActionRecord[]> {
    const text = description.trim();
    const table = this.table;
    const embeddingClient = this.embeddingClient;

    if (text.length === 0 || table === undefined || embeddingClient === undefined) {
      return [];
    }

    const searchLimit = Math.max(Math.max(1, limit) * 5, 20);
    const embedding = await embeddingClient.embed(text);
    const rows = await table.search(Array.from(embedding), {
      limit: searchLimit,
      vectorColumn: "embedding",
      distanceType: "cosine",
    });
    const ids = rows
      .map((row) => row.id)
      .filter((value): value is string => typeof value === "string")
      .map((value) => parseActionId(value));
    const records = this.getMany(ids);

    return rows
      .map((row, index) => {
        const record = records[index];

        if (record === null) {
          return null;
        }

        return {
          record,
          similarity: toSimilarity(getDistance(row)),
        };
      })
      .filter((item): item is { record: ActionRecord; similarity: number } => item !== null)
      .sort((left, right) => right.similarity - left.similarity)
      .slice(0, Math.max(1, limit))
      .map((item) => item.record);
  }

  async delete(id: ActionId): Promise<boolean> {
    const result = this.db.prepare("DELETE FROM action_records WHERE id = ?").run(id);

    if (result.changes > 0 && this.table !== undefined) {
      await this.table.remove(`id = ${quoteSqlString(id)}`);
    }

    return result.changes > 0;
  }
}
