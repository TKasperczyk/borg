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
import { actionActivityCutoff, actionActivityTier } from "./activity.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { actionMemoryDisclosureLabel } from "../common/disclosure-serializers.js";
import {
  combineMemoryDisclosureLabels,
  parseMemoryDisclosureLabel,
} from "../common/disclosure-label.js";
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
  type SharedStateEntryId,
} from "../../util/ids.js";
import {
  ACTIVE_ACTION_STATES,
  ACTION_STATE_METADATA,
  ACTION_STATES,
  actionRecordPatchSchema,
  actionRecordSchema,
  actionStateSchema,
  actionSelectionAuditSchema,
  actionRecordUpdateSchema,
  actionExtractionDecisionSchema,
  type ActionExtractionDecision,
  isActionTerminalClosureGuardState,
  type ActionMatchRejection,
  type ActionSelectionAudit,
  type ActionUpdateHistory,
  type ActionActor,
  type ActionRecord,
  type ActionRecordPatch,
  type ActionSessionScope,
  type ActionState,
  type ActionStateTimestampField,
} from "./types.js";

const ACTION_JSON_ARRAY_CODEC = {
  errorCode: "ACTION_RECORD_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse action record ${label}`,
} satisfies JsonArrayCodecOptions;

export const ACTION_ACTOR_RESERVED_RECORDS = 4;
export const STALE_OPEN_DRAW_LIMIT = 16;
const ACTION_ACTIVITY_SQL = "MAX(updated_at, COALESCE(last_referenced_at_ms, updated_at))";

function parseSelectionAudit(value: unknown): ActionSelectionAudit | null {
  if (value == null) return null;
  try {
    return actionSelectionAuditSchema.parse(JSON.parse(String(value)));
  } catch (cause) {
    throw new StorageError("Invalid action selection audit", {
      code: "ACTION_RECORD_ROW_INVALID",
      cause,
    });
  }
}

export function mapActionRow(row: Record<string, unknown>): ActionRecord {
  const parsed = actionRecordSchema.safeParse({
    id: row.id,
    description: row.description,
    actor: row.actor,
    audience_entity_id:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : row.audience_entity_id,
    goal_id: row.goal_id === null || row.goal_id === undefined ? null : row.goal_id,
    ...(row.goal_link_basis == null ? {} : { goal_link_basis: row.goal_link_basis }),
    ...(row.origin_stream_entry_ids == null
      ? {}
      : {
          origin_stream_entry_ids: parseJsonArray<StreamEntryId>(
            String(row.origin_stream_entry_ids),
            "origin_stream_entry_ids",
            ACTION_JSON_ARRAY_CODEC,
          ),
        }),
    ...(row.selection_audit == null
      ? {}
      : { selection_audit: parseSelectionAudit(row.selection_audit) }),
    ...(row.disclosure_label == null
      ? {}
      : { disclosure_label: parseMemoryDisclosureLabel(row.disclosure_label) }),
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
    declined_at: row.declined_at == null ? null : Number(row.declined_at),
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

export function mapActionUpdateRow(row: Record<string, unknown>) {
  return actionRecordUpdateSchema.parse({
    ...row,
    source_stream_entry_ids: parseJsonArray<StreamEntryId>(
      String(row.source_stream_entry_ids),
      "source_stream_entry_ids",
      ACTION_JSON_ARRAY_CODEC,
    ),
    selection_audit: parseSelectionAudit(row.selection_audit),
    disclosure_label: parseMemoryDisclosureLabel(row.disclosure_label),
  });
}

function monotonic(
  current: number | null | undefined,
  incoming: number | null | undefined,
): number | null {
  return current == null
    ? (incoming ?? null)
    : incoming == null
      ? current
      : Math.max(current, incoming);
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

function stateTimestampField(state: ActionState): ActionStateTimestampField {
  return ACTION_STATE_METADATA[state].timestamp_field;
}

export type ActionRecordListFilter = {
  state?: ActionState;
  states?: readonly ActionState[];
  actor?: ActionActor;
  sessionScope?: ActionSessionScope | null;
  sessionAnchorId?: SessionId | null;
  audienceEntityId?: EntityId | null;
  recallAllAudiences?: boolean;
  rankAudienceEntityIds?: readonly EntityId[];
  rankActorEntityIds?: readonly EntityId[];
  prioritizeActive?: boolean;
  /** Page draw: live open, recent other states, stale open, then older closed. */
  activityNowMs?: number;
  reserveActors?: readonly ActionActor[];
  /** Boost this audience's live/recent rows before the global page fill. */
  reserveAudienceEntityId?: EntityId | null;
  goalId?: GoalId;
  openQuestionId?: OpenQuestionId;
  createdSinceMs?: number;
  createdUntilMs?: number;
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
export type ActionRecordCreationSource = "extractor" | "reflector" | "api" | "tool" | "unknown";
export type ActionCreationCountsBySource = Record<ActionRecordCreationSource, number>;
export type ActionDescriptionSimilarityPair = {
  leftId: ActionId;
  rightId: ActionId;
  similarity: number;
};
export type ActionCompletionFailure = { actionId: ActionId; error: unknown };

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
    tool: 0,
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

  /** Offline audited actor repair: preserve the existing vector and all other metadata.
   * The caller journals and preflights both stores, then commits the SQL repair first.
   * Unlike the ordinary background refresh this awaited write propagates failures.
   */
  async syncRepairedActorMirror(id: ActionId, previousActor: ActionActor): Promise<void> {
    const record = this.get(id);
    const table = this.table;
    if (record === null || table === undefined) {
      throw new StorageError(`Missing action or mirror for ${id}`, {
        code: "ACTION_REPAIR_MISSING",
      });
    }
    const rows = await table.list({ where: `id = ${quoteSqlString(id)}` });
    if (rows.length !== 1) {
      throw new StorageError(`Expected one action mirror for ${id}`, {
        code: "ACTION_REPAIR_MISSING",
      });
    }
    const row = rows[0]!;
    if (row.actor === record.actor) return;
    if (row.actor !== previousActor) {
      throw new StorageError(`Action mirror actor changed for ${id}`, {
        code: "ACTION_REPAIR_STALE",
      });
    }
    // Retain the mirror's own updated_at and metadata even when SQL has newer values.
    await table.upsert([{ ...row, actor: record.actor }], {
      on: "id",
    });
  }

  private upsertSqlRow(record: ActionRecord): void {
    this.db
      .prepare(
        `
          INSERT INTO action_records (
            id, description, actor, audience_entity_id, goal_id, open_question_id, state, confidence,
            provenance_episode_ids, provenance_stream_entry_ids, created_at, updated_at,
            considering_at, committed_at, scheduled_at, completed_at, not_done_at, declined_at, expired_at,
            archived_at, unknown_at, canonicalized_by_artifact_entry_id, session_scope,
            session_anchor_id, last_referenced_at_ms, last_referenced_turn_counter,
            last_referenced_turn_global, goal_link_basis, selection_audit, disclosure_label, origin_stream_entry_ids
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (id) DO UPDATE SET
            description = excluded.description,
            actor = excluded.actor,
            audience_entity_id = excluded.audience_entity_id,
            goal_id = excluded.goal_id,
            goal_link_basis = excluded.goal_link_basis,
            selection_audit = excluded.selection_audit,
            disclosure_label = excluded.disclosure_label,
            origin_stream_entry_ids = excluded.origin_stream_entry_ids,
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
            declined_at = excluded.declined_at,
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
        record.declined_at,
        record.expired_at,
        record.archived_at,
        record.unknown_at,
        record.canonicalized_by_artifact_entry_id ?? null,
        record.session_scope ?? null,
        record.session_anchor_id ?? null,
        record.last_referenced_at_ms ?? null,
        record.last_referenced_turn_counter ?? null,
        record.last_referenced_turn_global ?? null,
        record.goal_link_basis ?? null,
        record.selection_audit == null ? null : serializeJsonValue(record.selection_audit),
        record.disclosure_label == null ? null : serializeJsonValue(record.disclosure_label),
        record.origin_stream_entry_ids == null
          ? null
          : serializeJsonValue(record.origin_stream_entry_ids),
      );
  }

  // Opus 5.0: in-scope because identity is the model's choice; this transaction
  // checks handles, actor and lifecycle only, and keeps both the assertion and its audit.
  appendExtractionUpdate(
    id: ActionId,
    candidate: ActionRecord,
    turnId: string | null,
    options: {
      skipSideEffects?: boolean;
      canonicalizedByArtifactEntryId?: SharedStateEntryId;
    } = {},
  ):
    | {
        status: "applied";
        record: ActionRecord;
        completionFailures: readonly ActionCompletionFailure[];
      }
    | { status: "replayed"; record: ActionRecord }
    | { status: "rejected"; reason: ActionMatchRejection } {
    const incoming = actionRecordSchema.parse(candidate);
    const audit = actionSelectionAuditSchema.parse(incoming.selection_audit);
    const result = this.db.transaction(() =>
      this.persistContinuation(id, incoming, audit, turnId, options.canonicalizedByArtifactEntryId),
    )();
    if (result.status !== "applied") return result;
    // Completion-hook suppression does not suppress the deferred, best-effort mirror refresh.
    this.scheduleVectorUpsert(result.record);
    return {
      ...result,
      completionFailures: options.skipSideEffects
        ? []
        : this.runCompletionHook(result.record, result.previous),
    };
  }

  private runCompletionHook(
    record: ActionRecord,
    previous: ActionRecord | null,
  ): ActionCompletionFailure[] {
    if (record.state !== "completed") return [];
    try {
      this.options.onCompleted?.(record, previous);
      return [];
    } catch (error) {
      // SQL has committed. Return the failure separately so callers retain every committed ID.
      return [{ actionId: record.id, error }];
    }
  }

  // Called only inside a transaction; callers schedule effects after their outer commit.
  private persistContinuation(
    id: ActionId,
    incoming: ActionRecord,
    audit: ActionSelectionAudit,
    turnId: string | null,
    canonicalizedByArtifactEntryId?: SharedStateEntryId,
  ) {
    if (!audit.candidate_ids.includes(id))
      return { status: "rejected", reason: "not_offered" } as const;
    const current = this.get(id);
    if (current === null) return { status: "rejected", reason: "missing" } as const;
    if (
      audit.shared_state_entry_id !== undefined &&
      this.db
        .prepare(
          "SELECT 1 FROM action_record_updates WHERE action_id = ? AND json_extract(selection_audit, '$.shared_state_entry_id') = ? LIMIT 1",
        )
        .get(id, audit.shared_state_entry_id)
    )
      return { status: "replayed", record: current } as const;
    // The same five protected states as completeAction / markActionNotDone / declineAction.
    if (isActionTerminalClosureGuardState(current.state))
      return { status: "rejected", reason: "terminal" } as const;
    if (current.actor !== incoming.actor)
      return { status: "rejected", reason: "actor_mismatch" } as const;
    const next = actionRecordSchema.parse({
      ...current,
      // Overlapping non-terminal decisions are last-finisher-wins; history keeps both.
      state: incoming.state,
      origin_stream_entry_ids:
        current.origin_stream_entry_ids ?? current.provenance_stream_entry_ids,
      ...(canonicalizedByArtifactEntryId === undefined
        ? {}
        : { canonicalized_by_artifact_entry_id: canonicalizedByArtifactEntryId }),
      confidence: Math.max(current.confidence, incoming.confidence),
      disclosure_label: combineMemoryDisclosureLabels([
        actionMemoryDisclosureLabel(current),
        actionMemoryDisclosureLabel(incoming),
      ]),
      provenance_episode_ids: [
        ...new Set([...current.provenance_episode_ids, ...incoming.provenance_episode_ids]),
      ],
      provenance_stream_entry_ids: [
        ...new Set([
          ...current.provenance_stream_entry_ids,
          ...incoming.provenance_stream_entry_ids,
        ]),
      ],
      updated_at: incoming.updated_at,
      last_referenced_at_ms: monotonic(
        current.last_referenced_at_ms,
        incoming.last_referenced_at_ms,
      ),
      last_referenced_turn_counter: monotonic(
        current.last_referenced_turn_counter,
        incoming.last_referenced_turn_counter,
      ),
      last_referenced_turn_global: monotonic(
        current.last_referenced_turn_global,
        incoming.last_referenced_turn_global,
      ),
      ...(current.state === incoming.state
        ? {}
        : { [stateTimestampField(incoming.state)]: incoming.updated_at }),
    });
    this.db
      .prepare(
        `
      INSERT INTO action_record_updates
        (action_id, text, from_state, to_state, turn_id, source_stream_entry_ids, timestamp, selection_audit, disclosure_label)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `,
      )
      .run(
        id,
        incoming.description,
        current.state,
        incoming.state,
        turnId,
        serializeJsonValue(incoming.provenance_stream_entry_ids),
        incoming.updated_at,
        serializeJsonValue(audit),
        serializeJsonValue(actionMemoryDisclosureLabel(incoming)),
      );
    this.upsertSqlRow(next);
    return { status: "applied", record: next, previous: current } as const;
  }

  applyExtractionDecision(input: {
    records: readonly ActionRecord[];
    audit: ActionExtractionDecision;
    nowMs: number;
    turnCounter: number | null;
  }) {
    const records = input.records.map((record) => actionRecordSchema.parse(record));
    const audit = actionExtractionDecisionSchema.parse(input.audit);
    const inserted: ActionRecord[] = [];
    const matched: Array<{
      record: ActionRecord;
      decision: ActionRecord;
      previous: ActionRecord | null;
    }> = [];
    const touched: ActionRecord[] = [];
    this.db.transaction(() => {
      const offered = new Set(audit.candidate_ids);
      for (const id of new Set(audit.referenced_action_ids)) {
        const current = offered.has(id) ? this.get(id) : null;
        if (current === null) {
          audit.ignored_reference_ids.push(id);
          continue;
        }
        const next = {
          ...current,
          updated_at: Math.max(current.updated_at, input.nowMs),
          last_referenced_at_ms: monotonic(current.last_referenced_at_ms, input.nowMs),
          last_referenced_turn_counter: monotonic(
            current.last_referenced_turn_counter,
            input.turnCounter,
          ),
          last_referenced_turn_global: monotonic(
            current.last_referenced_turn_global,
            input.turnCounter,
          ),
        };
        this.upsertSqlRow(next);
        touched.push(next);
        audit.applied_reference_ids.push(id);
      }
      for (const record of records) {
        const decision = actionSelectionAuditSchema.parse({
          ...record.selection_audit,
          referenced_action_ids: audit.referenced_action_ids,
          applied_reference_ids: audit.applied_reference_ids,
        });
        record.selection_audit = decision;
        const id = decision.chosen_action_id;
        if (id !== null) {
          const previous = this.get(id);
          const result = this.persistContinuation(id, record, decision, audit.turn_id);
          if (result.status !== "rejected") {
            matched.push({ record: result.record, decision: record, previous });
            audit.decisions.push(decision);
            continue;
          }
          decision.rejection_reason = result.reason;
        }
        this.upsertSqlRow(record);
        inserted.push(record);
        audit.decisions.push(decision);
      }
      this.db
        .prepare("INSERT INTO action_extraction_decisions(timestamp, audit) VALUES (?, ?)")
        .run(input.nowMs, serializeJsonValue(audit));
    })();
    // No vector writes, completion callbacks or creation counters escape a rolled-back decision.
    this.creationCountsBySource.extractor += inserted.length;
    for (const record of new Map(
      [...touched, ...inserted, ...matched.map((item) => item.record)].map((record) => [
        record.id,
        record,
      ]),
    ).values())
      this.scheduleVectorUpsert(record);
    const completionFailures = [
      ...inserted.flatMap((record) => this.runCompletionHook(record, null)),
      ...matched.flatMap((item) => this.runCompletionHook(item.record, item.previous)),
    ];
    return { inserted, matched, audit, completionFailures };
  }

  readUpdateHistory(id: ActionId, limit: number): ActionUpdateHistory {
    // Like identity events, this append-only audit survives an explicit record deletion.
    const total = Number(
      (
        this.db
          .prepare("SELECT COUNT(*) AS n FROM action_record_updates WHERE action_id = ?")
          .get(id) as { n: number }
      ).n,
    );
    const rows = this.db
      .prepare(
        `SELECT * FROM action_record_updates WHERE action_id = ? ORDER BY timestamp DESC, id DESC LIMIT ?`,
      )
      .all(id, Math.max(0, Math.floor(limit))) as Record<string, unknown>[];
    return {
      total,
      updates: rows.map(mapActionUpdateRow),
    };
  }

  // Operator/admin write: bypasses semantic continuation checks and history.
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

  // Operator/admin write: bypasses semantic continuation checks and history.
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
    const limitStaleOpen = filter.activityNowMs !== undefined && filter.limit !== undefined;
    const reserveAudience =
      filter.reserveAudienceEntityId !== undefined && filter.activityNowMs !== undefined;
    if ((filter.reserveActors?.length || reserveAudience) && filter.limit !== undefined) {
      const limit = Math.max(1, Math.floor(filter.limit));
      const unreserved = {
        ...filter,
        reserveActors: undefined,
        reserveAudienceEntityId: undefined,
      };
      const actors = [...new Set(filter.reserveActors ?? [])].filter(
        (actor) => filter.actor === undefined || actor === filter.actor,
      );
      const openStates = ACTIVE_ACTION_STATES.filter(
        (state) =>
          (filter.state === undefined || state === filter.state) &&
          (filter.states === undefined || filter.states.includes(state)),
      );
      const groups = actors.map((actor) =>
        this.listUnreserved({
          ...unreserved,
          actor,
          state: undefined,
          states: openStates,
          rankAudienceEntityIds: undefined,
          rankActorEntityIds: undefined,
          limit: ACTION_ACTOR_RESERVED_RECORDS,
        }),
      );
      const records = new Map<ActionId, ActionRecord>();
      for (let index = 0; index < ACTION_ACTOR_RESERVED_RECORDS; index++) {
        for (const group of groups) {
          const row = group[index];
          if (row !== undefined && records.size < limit) records.set(row.id, row);
        }
      }
      if (
        reserveAudience &&
        (filter.audienceEntityId === undefined ||
          filter.audienceEntityId === filter.reserveAudienceEntityId)
      ) {
        // This is a reserved prefix, not a recall gate: the global draw fills all
        // remaining capacity below, and other actors' reservations survive it.
        const audienceRows = this.listUnreserved({
          ...unreserved,
          recallAllAudiences: false,
          audienceEntityId: filter.reserveAudienceEntityId,
        });
        for (const row of audienceRows) {
          if (records.size >= limit) break;
          if (actionActivityTier(row, filter.activityNowMs!) <= 1) records.set(row.id, row);
        }
      }
      if (records.size < limit) {
        for (const row of this.listUnreserved(
          { ...unreserved, limit: limit - records.size },
          { excludeIds: [...records.keys()], limitStaleOpen },
        )) {
          records.set(row.id, row);
        }
      }
      return [...records.values()];
    }
    return this.listUnreserved(filter, { limitStaleOpen });
  }

  private listUnreserved(
    filter: ActionRecordListFilter,
    options: { excludeIds?: readonly ActionId[]; limitStaleOpen?: boolean } = {},
  ): ActionRecord[] {
    const clauses: string[] = [];
    const values: unknown[] = [];
    const orderBy: string[] = [];
    const orderValues: unknown[] = [];

    if (options.excludeIds?.length) {
      clauses.push(`id NOT IN (${options.excludeIds.map(() => "?").join(", ")})`);
      values.push(...options.excludeIds);
    }

    if (filter.activityNowMs !== undefined) {
      const openSql = `state IN (${ACTIVE_ACTION_STATES.map(() => "?").join(", ")})`;
      orderBy.push(
        `CASE WHEN ${ACTION_ACTIVITY_SQL} >= ? THEN CASE WHEN ${openSql} THEN 0 ELSE 1 END ELSE CASE WHEN ${openSql} THEN 2 ELSE 3 END END ASC`,
        `${ACTION_ACTIVITY_SQL} DESC`,
      );
      orderValues.push(
        actionActivityCutoff(filter.activityNowMs),
        ...ACTIVE_ACTION_STATES,
        ...ACTIVE_ACTION_STATES,
      );
    } else if (filter.prioritizeActive === true) {
      orderBy.push(
        `CASE WHEN state IN (${ACTIVE_ACTION_STATES.map(() => "?").join(", ")}) THEN 0 ELSE 1 END ASC`,
      );
      orderValues.push(...ACTIVE_ACTION_STATES);
    }

    if (filter.recallAllAudiences === true && "audienceEntityId" in filter) {
      throw new StorageError("Action recallAllAudiences cannot be combined with audienceEntityId", {
        code: "ACTION_RECORD_FILTER_INVALID",
      });
    }

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

    if (filter.recallAllAudiences !== true && "audienceEntityId" in filter) {
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

    if (filter.createdSinceMs !== undefined) {
      clauses.push("created_at >= ?");
      values.push(filter.createdSinceMs);
    }

    if (filter.createdUntilMs !== undefined) {
      clauses.push("created_at <= ?");
      values.push(filter.createdUntilMs);
    }

    if (filter.rankAudienceEntityIds !== undefined) {
      const audienceRankIds = [...new Set(filter.rankAudienceEntityIds)];

      if (audienceRankIds.length > 0) {
        orderBy.push(
          `CASE WHEN audience_entity_id IN (${audienceRankIds.map(() => "?").join(", ")}) THEN 0 ELSE 1 END ASC`,
        );
        orderValues.push(...audienceRankIds);
      }
    }

    if (filter.rankActorEntityIds !== undefined) {
      const actorRankIds = [...new Set(filter.rankActorEntityIds)];

      if (actorRankIds.length > 0) {
        orderBy.push(
          `CASE WHEN actor IN (${actorRankIds.map(() => "?").join(", ")}) THEN 0 ELSE 1 END ASC`,
        );
        orderValues.push(...actorRankIds);
      }
    }

    const ordering = [...orderBy, "updated_at DESC", "id ASC"].join(", ");
    if (options.limitStaleOpen && filter.activityNowMs !== undefined) {
      const stale = `state IN (${ACTIVE_ACTION_STATES.map(() => "?").join(", ")}) AND ${ACTION_ACTIVITY_SQL} < ?`;
      const staleValues = [...ACTIVE_ACTION_STATES, actionActivityCutoff(filter.activityNowMs)];
      const staleClauses = [...clauses, stale];
      const staleSelectionValues = [
        ...values,
        ...staleValues,
        ...orderValues,
        STALE_OPEN_DRAW_LIMIT,
      ];
      // Cap stale rows before LIMIT, so older closed rows can reach the renderer.
      // Actor reservations are excluded from this fill and do not consume its cap.
      clauses.push(`(NOT (${stale}) OR id IN (
        SELECT id FROM action_records WHERE ${staleClauses.join(" AND ")}
        ORDER BY ${ordering} LIMIT ?
      ))`);
      values.push(...staleValues, ...staleSelectionValues);
    }

    const limit = filter.limit === undefined ? null : Math.max(1, Math.floor(filter.limit));
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM action_records
          ${clauses.length === 0 ? "" : `WHERE ${clauses.join(" AND ")}`}
          ORDER BY ${ordering}
          ${limit === null ? "" : "LIMIT ?"}
        `,
      )
      .all(...values, ...orderValues, ...(limit === null ? [] : [limit])) as Record<
      string,
      unknown
    >[];

    return rows.map((row) => mapActionRow(row));
  }

  count(): number {
    const row = this.db.prepare("SELECT COUNT(*) AS count FROM action_records").get() as
      | { count: number }
      | undefined;

    return Number(row?.count ?? 0);
  }

  countStaleOpen(nowMs: number): number {
    const row = this.db
      .prepare(
        `
      SELECT COUNT(*) AS count FROM action_records
      WHERE state IN (${ACTIVE_ACTION_STATES.map(() => "?").join(", ")})
        AND ${ACTION_ACTIVITY_SQL} < ?
    `,
      )
      .get(...ACTIVE_ACTION_STATES, actionActivityCutoff(nowMs)) as { count: number };
    return row.count;
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

  countActive(filter: Pick<ActionRecordListFilter, "audienceEntityId"> = {}): number {
    if (ACTIVE_ACTION_STATES.length === 0) {
      return 0;
    }

    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM action_records
          WHERE state IN (${ACTIVE_ACTION_STATES.map(() => "?").join(", ")})
          ${!("audienceEntityId" in filter) || filter.audienceEntityId === undefined ? "" : filter.audienceEntityId === null ? "AND audience_entity_id IS NULL" : "AND audience_entity_id = ?"}
        `,
      )
      .get(
        ...ACTIVE_ACTION_STATES,
        ...(filter.audienceEntityId == null ? [] : [filter.audienceEntityId]),
      ) as { count: number } | undefined;

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
