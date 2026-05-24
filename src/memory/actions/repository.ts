import {
  parseJsonArray,
  quoteSqlString,
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
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createActionId,
  parseActionId,
  type ActionId,
  type EntityId,
  type EpisodeId,
  type GoalId,
  type OpenQuestionId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  ACTION_STATES,
  actionRecordPatchSchema,
  actionRecordSchema,
  actionStateSchema,
  type ActionActor,
  type ActionRecord,
  type ActionRecordPatch,
  type ActionSessionScope,
  type ActionState,
} from "./types.js";

const ACTION_JSON_ARRAY_CODEC = {
  errorCode: "ACTION_RECORD_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse action record ${label}`,
} satisfies JsonArrayCodecOptions;

function mapActionRow(row: Record<string, unknown>): ActionRecord {
  const parsed = actionRecordSchema.safeParse({
    id: row.id,
    description: row.description,
    actor: row.actor,
    audience_entity_id:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : row.audience_entity_id,
    goal_id: row.goal_id === null || row.goal_id === undefined ? null : row.goal_id,
    open_question_id:
      row.open_question_id === null || row.open_question_id === undefined
        ? null
        : row.open_question_id,
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
    expired_at:
      row.expired_at === null || row.expired_at === undefined ? null : Number(row.expired_at),
    archived_at:
      row.archived_at === null || row.archived_at === undefined ? null : Number(row.archived_at),
    unknown_at:
      row.unknown_at === null || row.unknown_at === undefined ? null : Number(row.unknown_at),
    canonicalized_by_artifact_entry_id:
      row.canonicalized_by_artifact_entry_id === null ||
      row.canonicalized_by_artifact_entry_id === undefined
        ? null
        : String(row.canonicalized_by_artifact_entry_id),
    session_scope:
      row.session_scope === null || row.session_scope === undefined ? null : row.session_scope,
    session_anchor_id:
      row.session_anchor_id === null || row.session_anchor_id === undefined
        ? null
        : row.session_anchor_id,
    last_referenced_at_ms:
      row.last_referenced_at_ms === null || row.last_referenced_at_ms === undefined
        ? null
        : Number(row.last_referenced_at_ms),
    last_referenced_turn_counter:
      row.last_referenced_turn_counter === null || row.last_referenced_turn_counter === undefined
        ? null
        : Number(row.last_referenced_turn_counter),
    last_referenced_turn_global:
      row.last_referenced_turn_global === null || row.last_referenced_turn_global === undefined
        ? null
        : Number(row.last_referenced_turn_global),
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

function embeddingFromRow(row: Record<string, unknown>): Float32Array | null {
  const value = row.embedding;

  if (value instanceof Float32Array) {
    return value;
  }

  if (Array.isArray(value) && value.every((item) => typeof item === "number")) {
    return Float32Array.from(value);
  }

  if (ArrayBuffer.isView(value) && "length" in value) {
    return Float32Array.from(Array.from(value as unknown as ArrayLike<number>));
  }

  if (
    typeof value === "object" &&
    value !== null &&
    "toArray" in value &&
    typeof value.toArray === "function"
  ) {
    const array = value.toArray() as unknown;

    if (Array.isArray(array) && array.every((item) => typeof item === "number")) {
      return Float32Array.from(array);
    }

    if (ArrayBuffer.isView(array) && "length" in array) {
      return Float32Array.from(Array.from(array as unknown as ArrayLike<number>));
    }
  }

  return null;
}

type ActionStateTimestampField =
  | "considering_at"
  | "committed_at"
  | "scheduled_at"
  | "completed_at"
  | "not_done_at"
  | "expired_at"
  | "archived_at"
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
    case "expired":
      return "expired_at";
    case "archived":
      return "archived_at";
    case "unknown":
      return "unknown_at";
  }
}

export type ActionRecordListFilter = {
  state?: ActionState;
  states?: readonly ActionState[];
  actor?: ActionActor;
  sessionScope?: ActionSessionScope | null;
  sessionAnchorId?: SessionId | null;
  audienceEntityId?: EntityId | null;
  goalId?: GoalId;
  openQuestionId?: OpenQuestionId;
  limit?: number;
};

export type ActionRepositoryOptions = {
  db: SqliteDatabase;
  table?: LanceDbTable;
  embeddingClient?: EmbeddingClient;
  clock?: Clock;
  onCompleted?: (record: ActionRecord, previous: ActionRecord | null) => void;
};

export type ActionCountByState = Record<ActionState, number>;
export type ActionRecordCreationSource = "extractor" | "reflector" | "api" | "unknown";
export type ActionCreationCountsBySource = Record<ActionRecordCreationSource, number>;
export type ActionDescriptionSimilarityPair = {
  leftId: ActionId;
  rightId: ActionId;
  similarity: number;
};
export type ActionAddOptions = {
  creationSource?: ActionRecordCreationSource;
};
export type ActionUpdateOptions = {
  skipSideEffects?: boolean;
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
  private readonly creationCountsBySource: ActionCreationCountsBySource = {
    extractor: 0,
    reflector: 0,
    api: 0,
    unknown: 0,
  };

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

  nextLifecycleTurnGlobal(): number {
    return this.db.transaction(() => {
      this.db
        .prepare(
          `
            INSERT OR IGNORE INTO action_lifecycle_turn_counter (id, value)
              VALUES ('global', 0)
          `,
        )
        .run();
      const row = this.db
        .prepare("SELECT value FROM action_lifecycle_turn_counter WHERE id = 'global'")
        .get() as { value: number } | undefined;
      const next = Number(row?.value ?? 0) + 1;

      this.db
        .prepare("UPDATE action_lifecycle_turn_counter SET value = ? WHERE id = 'global'")
        .run(next);

      return next;
    })();
  }

  ensureLifecycleTurnGlobal(value: number): number {
    const normalized = Math.max(0, Math.floor(value));

    if (!Number.isFinite(normalized)) {
      throw new StorageError("Invalid action lifecycle global turn counter", {
        code: "ACTION_LIFECYCLE_TURN_COUNTER_INVALID",
      });
    }

    return this.db.transaction(() => {
      this.db
        .prepare(
          `
            INSERT INTO action_lifecycle_turn_counter (id, value)
              VALUES ('global', ?)
            ON CONFLICT (id) DO UPDATE SET
              value = CASE
                WHEN excluded.value > action_lifecycle_turn_counter.value
                THEN excluded.value
                ELSE action_lifecycle_turn_counter.value
              END
          `,
        )
        .run(normalized);

      const row = this.db
        .prepare("SELECT value FROM action_lifecycle_turn_counter WHERE id = 'global'")
        .get() as { value: number } | undefined;

      return Number(row?.value ?? normalized);
    })();
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
            id, description, actor, audience_entity_id, goal_id, open_question_id, state, confidence,
            provenance_episode_ids, provenance_stream_entry_ids, created_at, updated_at,
            considering_at, committed_at, scheduled_at, completed_at, not_done_at, expired_at,
            archived_at, unknown_at, canonicalized_by_artifact_entry_id, session_scope,
            session_anchor_id, last_referenced_at_ms, last_referenced_turn_counter,
            last_referenced_turn_global
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (id) DO UPDATE SET
            description = excluded.description,
            actor = excluded.actor,
            audience_entity_id = excluded.audience_entity_id,
            goal_id = excluded.goal_id,
            open_question_id = excluded.open_question_id,
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
            expired_at = excluded.expired_at,
            archived_at = excluded.archived_at,
            unknown_at = excluded.unknown_at,
            canonicalized_by_artifact_entry_id = excluded.canonicalized_by_artifact_entry_id,
            session_scope = excluded.session_scope,
            session_anchor_id = excluded.session_anchor_id,
            last_referenced_at_ms = excluded.last_referenced_at_ms,
            last_referenced_turn_counter = excluded.last_referenced_turn_counter,
            last_referenced_turn_global = excluded.last_referenced_turn_global
        `,
      )
      .run(
        record.id,
        record.description,
        record.actor,
        record.audience_entity_id,
        record.goal_id,
        record.open_question_id,
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
        record.expired_at,
        record.archived_at,
        record.unknown_at,
        record.canonicalized_by_artifact_entry_id ?? null,
        record.session_scope ?? null,
        record.session_anchor_id ?? null,
        record.last_referenced_at_ms ?? null,
        record.last_referenced_turn_counter ?? null,
        record.last_referenced_turn_global ?? null,
      );
  }

  add(record: ActionRecord, options: ActionAddOptions = {}): void {
    const parsed = actionRecordSchema.parse({
      ...record,
      id: record.id ?? createActionId(),
    });
    const existing = this.db.prepare("SELECT 1 FROM action_records WHERE id = ?").get(parsed.id) as
      | { 1: number }
      | undefined;

    this.upsertSqlRow(parsed);
    this.scheduleVectorUpsert(parsed);
    if (existing === undefined) {
      this.creationCountsBySource[options.creationSource ?? "unknown"] += 1;
    }
    if (parsed.state === "completed") {
      this.options.onCompleted?.(parsed, null);
    }
  }

  update(id: ActionId, patch: ActionRecordPatch, options: ActionUpdateOptions = {}): void {
    const current = this.get(id);

    if (current === null) {
      throw new StorageError(`Unknown action record id: ${id}`, {
        code: "ACTION_RECORD_NOT_FOUND",
      });
    }

    const parsedPatch = actionRecordPatchSchema.parse({
      ...patch,
      ...("goal_id" in patch ? {} : { goal_id: current.goal_id }),
      ...("open_question_id" in patch ? {} : { open_question_id: current.open_question_id }),
    });
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
    if (
      options.skipSideEffects !== true &&
      current.state !== "completed" &&
      next.state === "completed"
    ) {
      this.options.onCompleted?.(next, current);
    }
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

    if (filter.states !== undefined) {
      const states = [...new Set(filter.states.map((state) => actionStateSchema.parse(state)))];

      if (states.length === 0) {
        return [];
      }

      clauses.push(`state IN (${states.map(() => "?").join(", ")})`);
      values.push(...states);
    }

    if (filter.actor !== undefined) {
      clauses.push("actor = ?");
      values.push(filter.actor);
    }

    if ("sessionScope" in filter) {
      if (filter.sessionScope === null) {
        clauses.push("session_scope IS NULL");
      } else if (filter.sessionScope !== undefined) {
        clauses.push("session_scope = ?");
        values.push(filter.sessionScope);
      }
    }

    if ("sessionAnchorId" in filter) {
      if (filter.sessionAnchorId === null) {
        clauses.push("session_anchor_id IS NULL");
      } else if (filter.sessionAnchorId !== undefined) {
        clauses.push("session_anchor_id = ?");
        values.push(filter.sessionAnchorId);
      }
    }

    if ("audienceEntityId" in filter) {
      if (filter.audienceEntityId === null) {
        clauses.push("audience_entity_id IS NULL");
      } else if (filter.audienceEntityId !== undefined) {
        clauses.push("audience_entity_id = ?");
        values.push(filter.audienceEntityId);
      }
    }

    if (filter.goalId !== undefined) {
      clauses.push("goal_id = ?");
      values.push(filter.goalId);
    }

    if (filter.openQuestionId !== undefined) {
      clauses.push("open_question_id = ?");
      values.push(filter.openQuestionId);
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

  count(): number {
    const row = this.db.prepare("SELECT COUNT(*) AS count FROM action_records").get() as
      | { count: number }
      | undefined;

    return Number(row?.count ?? 0);
  }

  countByState(): ActionCountByState {
    const counts = Object.fromEntries(
      ACTION_STATES.map((state) => [state, 0]),
    ) as ActionCountByState;
    const rows = this.db
      .prepare(
        `
          SELECT state, COUNT(*) AS count
          FROM action_records
          GROUP BY state
        `,
      )
      .all() as Array<{ state: string; count: number }>;

    for (const row of rows) {
      const state = actionStateSchema.parse(row.state);
      counts[state] = Number(row.count);
    }

    return counts;
  }

  countCanonicalized(): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM action_records
          WHERE canonicalized_by_artifact_entry_id IS NOT NULL
        `,
      )
      .get() as { count: number } | undefined;

    return Number(row?.count ?? 0);
  }

  countActive(): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM action_records
          -- Active means non-terminal: still pressure-creating for
          -- canonicalization and durable action bloat observability.
          WHERE state IN ('considering', 'committed_to_do', 'scheduled', 'unknown')
        `,
      )
      .get() as { count: number } | undefined;

    return Number(row?.count ?? 0);
  }

  getCreationCountsBySource(): ActionCreationCountsBySource {
    return { ...this.creationCountsBySource };
  }

  countCompletedSince(timestampMs: number): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM action_records
          WHERE state = 'completed'
            AND completed_at IS NOT NULL
            AND completed_at >= ?
        `,
      )
      .get(timestampMs) as { count: number } | undefined;

    return Number(row?.count ?? 0);
  }

  listCompletedIds(): ActionId[] {
    const rows = this.db
      .prepare(
        `
          SELECT id
          FROM action_records
          WHERE state = 'completed'
            AND completed_at IS NOT NULL
          ORDER BY completed_at ASC, id ASC
        `,
      )
      .all() as Array<{ id: string }>;

    return rows.map((row) => parseActionId(row.id));
  }

  latestCompletedAt(): number | null {
    const row = this.db
      .prepare(
        `
          SELECT MAX(completed_at) AS completed_at
          FROM action_records
          WHERE state = 'completed'
            AND completed_at IS NOT NULL
        `,
      )
      .get() as { completed_at: number | null } | undefined;

    return row?.completed_at ?? null;
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

  async findSimilarDescriptionPairs(
    records: readonly ActionRecord[],
    threshold: number,
  ): Promise<ActionDescriptionSimilarityPair[]> {
    const uniqueRecords = [...new Map(records.map((record) => [record.id, record])).values()];

    if (uniqueRecords.length < 2) {
      return [];
    }

    const vectorsById = await this.loadActionVectors(uniqueRecords);
    const pairs: ActionDescriptionSimilarityPair[] = [];

    for (let leftIndex = 0; leftIndex < uniqueRecords.length; leftIndex += 1) {
      const left = uniqueRecords[leftIndex];
      const leftVector = left === undefined ? undefined : vectorsById.get(left.id);

      if (left === undefined || leftVector === undefined) {
        continue;
      }

      for (let rightIndex = leftIndex + 1; rightIndex < uniqueRecords.length; rightIndex += 1) {
        const right = uniqueRecords[rightIndex];
        const rightVector = right === undefined ? undefined : vectorsById.get(right.id);

        if (right === undefined || rightVector === undefined) {
          continue;
        }

        const similarity = cosineSimilarity(leftVector, rightVector);

        if (similarity >= threshold) {
          pairs.push({
            leftId: left.id,
            rightId: right.id,
            similarity,
          });
        }
      }
    }

    return pairs;
  }

  private async loadActionVectors(
    records: readonly ActionRecord[],
  ): Promise<Map<ActionId, Float32Array>> {
    const vectors = new Map<ActionId, Float32Array>();
    const table = this.table;

    if (table !== undefined && records.length > 0) {
      const rows = await table.list({
        where: `id IN (${records.map((record) => quoteSqlString(record.id)).join(", ")})`,
        columns: ["id", "embedding"],
        limit: records.length,
      });

      for (const row of rows) {
        if (typeof row.id !== "string") {
          continue;
        }

        const embedding = embeddingFromRow(row);

        if (embedding !== null) {
          vectors.set(parseActionId(row.id), embedding);
        }
      }
    }

    const missing = records.filter((record) => !vectors.has(record.id));
    const embeddingClient = this.embeddingClient;

    if (missing.length === 0 || embeddingClient === undefined) {
      return vectors;
    }

    const embedded = await embeddingClient.embedBatch(missing.map((record) => record.description));

    for (const [index, record] of missing.entries()) {
      const vector = embedded[index];

      if (vector !== undefined) {
        vectors.set(record.id, vector);
      }
    }

    return vectors;
  }

  async delete(id: ActionId): Promise<boolean> {
    const result = this.db.prepare("DELETE FROM action_records WHERE id = ?").run(id);

    if (result.changes > 0 && this.table !== undefined) {
      await this.table.remove(`id = ${quoteSqlString(id)}`);
    }

    return result.changes > 0;
  }
}
