import { parseJsonArray, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createSharedStateEntryId,
  parseSharedStateEntryId,
  type SharedStateEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  assertIdentityCasUpdated,
  expectedRecordVersion,
  nextRecordVersion,
  type IdentityCasOptions,
} from "../common/cas.js";
import {
  sharedStateEntrySchema,
  sharedStateCanonicalizesSchema,
  sharedStateArtifactSchema,
  allowAllSharedStateSourceTrustValidator,
  type SharedStateArtifact,
  type SharedStateCanonicalizes,
  type SharedStateEntry,
  type SharedStateEntryKind,
  type SharedStateSourceTrustValidator,
} from "./types.js";

const SHARED_STATE_JSON_ARRAY_CODEC = {
  errorCode: "SHARED_STATE_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse shared state ${label}`,
} satisfies JsonArrayCodecOptions;

const EMPTY_SHARED_STATE_CANONICALIZES: SharedStateCanonicalizes = {
  goal_ids: [],
  commitment_ids: [],
  action_ids: [],
  open_question_ids: [],
};

export type SharedStateAddOperation = {
  type: "add";
  id?: SharedStateEntryId;
  state_key: string;
  kind: SharedStateEntryKind;
  text: string;
  owner_entity_id?: EntityId | null;
  provenance_stream_entry_ids: readonly StreamEntryId[];
  last_updated_stream_entry_ids?: readonly StreamEntryId[];
  created_at?: number;
  last_updated_at?: number;
  last_updated_turn_global?: number | null;
  rank?: number;
  canonicalizes?: SharedStateCanonicalizes;
};

export type SharedStateUpdateOperation = {
  type: "update";
  id: SharedStateEntryId;
  state_key: string;
  kind?: SharedStateEntryKind;
  text?: string;
  owner_entity_id?: EntityId | null;
  add_provenance_stream_entry_ids?: readonly StreamEntryId[];
  last_updated_stream_entry_ids: readonly StreamEntryId[];
  last_updated_at?: number;
  last_updated_turn_global?: number | null;
  rank?: number;
  canonicalizes?: SharedStateCanonicalizes;
};

export type SharedStateSupersedeOperation = {
  type: "supersede";
  id: SharedStateEntryId;
  replacement: Omit<SharedStateAddOperation, "type">;
  last_updated_stream_entry_ids: readonly StreamEntryId[];
  last_updated_at?: number;
  last_updated_turn_global?: number | null;
};

export type SharedStatePruneOperation = {
  type: "prune";
  id: SharedStateEntryId;
};

export type SharedStateKindTransitionOperation = {
  type: "transition_kind";
  id: SharedStateEntryId;
  kind: SharedStateEntryKind;
};

export type SharedStateOperation =
  | SharedStateAddOperation
  | SharedStateUpdateOperation
  | SharedStateSupersedeOperation
  | SharedStatePruneOperation
  | SharedStateKindTransitionOperation;

export type SharedStateUpsertOptions = IdentityCasOptions & {
  now?: number;
  lastUpdatedTurnGlobal?: number | null;
  lastCompiledAt?: number | null;
  lastCompiledStreamEntryId?: StreamEntryId | null;
  sourceTrustValidator?: SharedStateSourceTrustValidator;
};

export type SharedStateRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  sourceTrustValidator?: SharedStateSourceTrustValidator;
};

function parseStreamEntryIds(value: string, label: string): StreamEntryId[] {
  return parseJsonArray<StreamEntryId>(value, label, SHARED_STATE_JSON_ARRAY_CODEC);
}

function uniqueStreamEntryIds(values: readonly StreamEntryId[]): StreamEntryId[] {
  return dedupePreservingOrder(values);
}

function emptyCanonicalizes(): SharedStateCanonicalizes {
  return {
    goal_ids: [],
    commitment_ids: [],
    action_ids: [],
    open_question_ids: [],
  };
}

function uniqueCanonicalizes(
  value: SharedStateCanonicalizes | undefined,
): SharedStateCanonicalizes {
  const source = value ?? EMPTY_SHARED_STATE_CANONICALIZES;

  return sharedStateCanonicalizesSchema.parse({
    goal_ids: dedupePreservingOrder(source.goal_ids),
    commitment_ids: dedupePreservingOrder(source.commitment_ids),
    action_ids: dedupePreservingOrder(source.action_ids),
    open_question_ids: dedupePreservingOrder(source.open_question_ids),
  });
}

function mergeCanonicalizes(
  current: SharedStateCanonicalizes,
  next: SharedStateCanonicalizes | undefined,
): SharedStateCanonicalizes {
  if (next === undefined) {
    return current;
  }

  return uniqueCanonicalizes({
    goal_ids: [...current.goal_ids, ...next.goal_ids],
    commitment_ids: [...current.commitment_ids, ...next.commitment_ids],
    action_ids: [...current.action_ids, ...next.action_ids],
    open_question_ids: [...current.open_question_ids, ...next.open_question_ids],
  });
}

function parseCanonicalizes(value: unknown): SharedStateCanonicalizes {
  if (value === null || value === undefined) {
    return emptyCanonicalizes();
  }

  try {
    const parsed = JSON.parse(String(value)) as unknown;
    const result = sharedStateCanonicalizesSchema.safeParse(parsed);

    if (!result.success) {
      throw result.error;
    }

    return uniqueCanonicalizes(result.data);
  } catch (error) {
    throw new StorageError("Failed to parse shared state canonicalizes", {
      cause: error,
      code: "SHARED_STATE_ROW_INVALID",
    });
  }
}

function requiredWriteStateKey(value: unknown, operationType: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new StorageError(`Shared state ${operationType} operation requires state_key`, {
      code: "SHARED_STATE_STATE_KEY_REQUIRED",
    });
  }

  return value;
}

function mapEntryRow(row: Record<string, unknown>): SharedStateEntry {
  const parsed = sharedStateEntrySchema.safeParse({
    id: row.id,
    audience_entity_id: row.audience_entity_id,
    state_key: row.state_key === null || row.state_key === undefined ? null : row.state_key,
    kind: row.kind,
    text: row.text,
    owner_entity_id:
      row.owner_entity_id === null || row.owner_entity_id === undefined
        ? null
        : row.owner_entity_id,
    provenance_stream_entry_ids: parseStreamEntryIds(
      String(row.provenance_stream_entry_ids ?? "[]"),
      "provenance_stream_entry_ids",
    ),
    last_updated_stream_entry_ids: parseStreamEntryIds(
      String(row.last_updated_stream_entry_ids ?? "[]"),
      "last_updated_stream_entry_ids",
    ),
    created_at: Number(row.created_at),
    last_updated_at: Number(row.last_updated_at),
    last_updated_turn_global:
      row.last_updated_turn_global === null || row.last_updated_turn_global === undefined
        ? null
        : Number(row.last_updated_turn_global),
    superseded_by_id:
      row.superseded_by_id === null || row.superseded_by_id === undefined
        ? null
        : row.superseded_by_id,
    rank: Number(row.rank ?? 0),
    canonicalizes: parseCanonicalizes(row.canonicalizes),
  });

  if (!parsed.success) {
    throw new StorageError("Shared state entry row failed validation", {
      cause: parsed.error,
      code: "SHARED_STATE_ROW_INVALID",
    });
  }

  return parsed.data;
}

function mapArtifactRow(
  row: Record<string, unknown>,
  entries: readonly SharedStateEntry[],
): SharedStateArtifact {
  const parsed = sharedStateArtifactSchema.safeParse({
    audience_entity_id: row.audience_entity_id,
    record_version: Number(row.record_version ?? 1),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
    last_compiled_at:
      row.last_compiled_at === null || row.last_compiled_at === undefined
        ? null
        : Number(row.last_compiled_at),
    last_compiled_stream_entry_id:
      row.last_compiled_stream_entry_id === null || row.last_compiled_stream_entry_id === undefined
        ? null
        : row.last_compiled_stream_entry_id,
    entries,
  });

  if (!parsed.success) {
    throw new StorageError("Shared state row failed validation", {
      cause: parsed.error,
      code: "SHARED_STATE_ROW_INVALID",
    });
  }

  return parsed.data;
}

export class SharedStateRepository {
  private readonly clock: Clock;

  constructor(private readonly options: SharedStateRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private listEntries(audienceEntityId: EntityId): SharedStateEntry[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM shared_state_entries
          WHERE audience_entity_id = ?
          ORDER BY
            CASE WHEN superseded_by_id IS NULL THEN 0 ELSE 1 END ASC,
            rank ASC,
            created_at ASC,
            id ASC
        `,
      )
      .all(audienceEntityId) as Record<string, unknown>[];

    return rows.map((row) => mapEntryRow(row));
  }

  get(audienceEntityId: EntityId): SharedStateArtifact | null {
    const row = this.db
      .prepare("SELECT * FROM shared_state_artifacts WHERE audience_entity_id = ?")
      .get(audienceEntityId) as Record<string, unknown> | undefined;

    if (row === undefined) {
      return null;
    }

    return mapArtifactRow(row, this.listEntries(audienceEntityId));
  }

  listRecentEntriesForCognition(
    input: {
      excludeAudienceEntityId?: EntityId | null;
      limit?: number;
    } = {},
  ): SharedStateEntry[] {
    const limit = Math.max(0, Math.floor(input.limit ?? 16));

    if (limit === 0) {
      return [];
    }

    const excludeAudienceEntityId = input.excludeAudienceEntityId ?? null;
    const audiencePredicate =
      excludeAudienceEntityId === null ? "" : "AND audience_entity_id != ?";
    const args =
      excludeAudienceEntityId === null ? [limit] : [excludeAudienceEntityId, limit];
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM shared_state_entries
          WHERE superseded_by_id IS NULL
            ${audiencePredicate}
          ORDER BY
            last_updated_at DESC,
            rank ASC,
            created_at DESC,
            id ASC
          LIMIT ?
        `,
      )
      .all(...args) as Record<string, unknown>[];

    return rows.map((row) => mapEntryRow(row));
  }

  private getEntry(id: SharedStateEntryId, audienceEntityId: EntityId): SharedStateEntry {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM shared_state_entries
          WHERE id = ? AND audience_entity_id = ?
        `,
      )
      .get(id, audienceEntityId) as Record<string, unknown> | undefined;

    if (row === undefined) {
      throw new StorageError(`Unknown shared state entry id: ${id}`, {
        code: "SHARED_STATE_ENTRY_NOT_FOUND",
      });
    }

    return mapEntryRow(row);
  }

  private insertParent(input: {
    audienceEntityId: EntityId;
    nowMs: number;
    lastCompiledAt: number | null;
    lastCompiledStreamEntryId: StreamEntryId | null;
  }): void {
    this.db
      .prepare(
        `
          INSERT INTO shared_state_artifacts (
            audience_entity_id, record_version, created_at, updated_at,
            last_compiled_at, last_compiled_stream_entry_id
          ) VALUES (?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        input.audienceEntityId,
        1,
        input.nowMs,
        input.nowMs,
        input.lastCompiledAt,
        input.lastCompiledStreamEntryId,
      );
  }

  private bumpParent(input: {
    current: SharedStateArtifact;
    expectedVersion: number;
    nowMs: number;
    lastCompiledAt: number | null;
    lastCompiledStreamEntryId: StreamEntryId | null;
  }): void {
    const result = this.db
      .prepare(
        `
          UPDATE shared_state_artifacts
          SET updated_at = ?,
              last_compiled_at = ?,
              last_compiled_stream_entry_id = ?,
              record_version = record_version + 1
          WHERE audience_entity_id = ? AND record_version = ?
        `,
      )
      .run(
        input.nowMs,
        input.lastCompiledAt,
        input.lastCompiledStreamEntryId,
        input.current.audience_entity_id,
        input.expectedVersion,
      );

    assertIdentityCasUpdated({
      result,
      recordType: "decision_artifact",
      recordId: input.current.audience_entity_id,
      expectedVersion: input.expectedVersion,
    });
  }

  private updateCompileMarker(input: {
    current: SharedStateArtifact;
    expectedVersion: number;
    nowMs: number;
    lastCompiledAt: number | null;
    lastCompiledStreamEntryId: StreamEntryId | null;
  }): void {
    const result = this.db
      .prepare(
        `
          UPDATE shared_state_artifacts
          SET updated_at = ?,
              last_compiled_at = ?,
              last_compiled_stream_entry_id = ?,
              record_version = record_version + 1
          WHERE audience_entity_id = ? AND record_version = ?
        `,
      )
      .run(
        input.nowMs,
        input.lastCompiledAt,
        input.lastCompiledStreamEntryId,
        input.current.audience_entity_id,
        input.expectedVersion,
      );

    assertIdentityCasUpdated({
      result,
      recordType: "decision_artifact",
      recordId: input.current.audience_entity_id,
      expectedVersion: input.expectedVersion,
    });
  }

  private insertEntry(entry: SharedStateEntry): void {
    const parsed = sharedStateEntrySchema.parse(entry);

    this.db
      .prepare(
        `
          INSERT INTO shared_state_entries (
            id, audience_entity_id, state_key, kind, text, owner_entity_id,
            provenance_stream_entry_ids, last_updated_stream_entry_ids,
            created_at, last_updated_at, last_updated_turn_global, superseded_by_id, rank,
            canonicalizes
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        parsed.id,
        parsed.audience_entity_id,
        parsed.state_key,
        parsed.kind,
        parsed.text,
        parsed.owner_entity_id,
        serializeJsonValue(parsed.provenance_stream_entry_ids),
        serializeJsonValue(parsed.last_updated_stream_entry_ids),
        parsed.created_at,
        parsed.last_updated_at,
        parsed.last_updated_turn_global,
        parsed.superseded_by_id,
        parsed.rank,
        serializeJsonValue(parsed.canonicalizes),
      );
  }

  private sourceTrustValidator(
    override: SharedStateSourceTrustValidator | undefined,
  ): SharedStateSourceTrustValidator {
    return override ?? this.options.sourceTrustValidator ?? allowAllSharedStateSourceTrustValidator;
  }

  private assertTrustedSourceStreamIds(
    streamEntryIds: readonly StreamEntryId[],
    field: "provenance_stream_entry_ids" | "last_updated_stream_entry_ids",
    validator: SharedStateSourceTrustValidator,
  ): void {
    for (const streamEntryId of streamEntryIds) {
      const trust = validator(streamEntryId);

      if (trust.allowed) {
        continue;
      }

      throw new StorageError(
        `Shared state ${field} contains a non-source-eligible stream entry: ${streamEntryId}`,
        {
          code: "SHARED_STATE_SOURCE_NOT_TRUSTED",
          cause: {
            streamEntryId,
            field,
            reason: trust.reason ?? "inactive",
          },
        },
      );
    }
  }

  private addEntry(
    audienceEntityId: EntityId,
    operation: Omit<SharedStateAddOperation, "type">,
    nowMs: number,
    lastUpdatedTurnGlobal: number | null,
    sourceTrustValidator: SharedStateSourceTrustValidator,
  ): SharedStateEntry {
    const provenanceStreamEntryIds = uniqueStreamEntryIds([
      ...operation.provenance_stream_entry_ids,
    ]);
    const lastUpdatedStreamEntryIds = uniqueStreamEntryIds([
      ...(operation.last_updated_stream_entry_ids ?? operation.provenance_stream_entry_ids),
    ]);

    this.assertTrustedSourceStreamIds(
      provenanceStreamEntryIds,
      "provenance_stream_entry_ids",
      sourceTrustValidator,
    );
    this.assertTrustedSourceStreamIds(
      lastUpdatedStreamEntryIds,
      "last_updated_stream_entry_ids",
      sourceTrustValidator,
    );
    const stateKey = requiredWriteStateKey(operation.state_key, "add");

    const entry = sharedStateEntrySchema.parse({
      id: operation.id ?? createSharedStateEntryId(),
      audience_entity_id: audienceEntityId,
      state_key: stateKey,
      kind: operation.kind,
      text: operation.text,
      owner_entity_id: operation.owner_entity_id ?? null,
      provenance_stream_entry_ids: provenanceStreamEntryIds,
      last_updated_stream_entry_ids: lastUpdatedStreamEntryIds,
      created_at: operation.created_at ?? nowMs,
      last_updated_at: operation.last_updated_at ?? operation.created_at ?? nowMs,
      last_updated_turn_global: operation.last_updated_turn_global ?? lastUpdatedTurnGlobal,
      superseded_by_id: null,
      rank: operation.rank ?? 0,
      canonicalizes: uniqueCanonicalizes(operation.canonicalizes),
    });

    this.insertEntry(entry);
    return entry;
  }

  private updateEntry(
    audienceEntityId: EntityId,
    operation: SharedStateUpdateOperation,
    nowMs: number,
    lastUpdatedTurnGlobal: number | null,
    sourceTrustValidator: SharedStateSourceTrustValidator,
  ): void {
    const current = this.getEntry(operation.id, audienceEntityId);
    const addProvenance = operation.add_provenance_stream_entry_ids ?? [];
    const provenanceStreamEntryIds = uniqueStreamEntryIds([
      ...current.provenance_stream_entry_ids,
      ...addProvenance,
    ]);
    const lastUpdatedStreamEntryIds = uniqueStreamEntryIds([
      ...operation.last_updated_stream_entry_ids,
    ]);

    this.assertTrustedSourceStreamIds(
      provenanceStreamEntryIds,
      "provenance_stream_entry_ids",
      sourceTrustValidator,
    );
    this.assertTrustedSourceStreamIds(
      lastUpdatedStreamEntryIds,
      "last_updated_stream_entry_ids",
      sourceTrustValidator,
    );
    const stateKey = requiredWriteStateKey(operation.state_key, "update");

    const next = sharedStateEntrySchema.parse({
      ...current,
      state_key: stateKey,
      kind: operation.kind ?? current.kind,
      text: operation.text ?? current.text,
      owner_entity_id:
        operation.owner_entity_id === undefined
          ? current.owner_entity_id
          : operation.owner_entity_id,
      provenance_stream_entry_ids: provenanceStreamEntryIds,
      last_updated_stream_entry_ids: lastUpdatedStreamEntryIds,
      last_updated_at: operation.last_updated_at ?? nowMs,
      last_updated_turn_global: operation.last_updated_turn_global ?? lastUpdatedTurnGlobal,
      rank: operation.rank ?? current.rank,
      canonicalizes: mergeCanonicalizes(current.canonicalizes, operation.canonicalizes),
    });

    this.db
      .prepare(
        `
          UPDATE shared_state_entries
          SET state_key = ?,
              kind = ?,
              text = ?,
              owner_entity_id = ?,
              provenance_stream_entry_ids = ?,
              last_updated_stream_entry_ids = ?,
              last_updated_at = ?,
              last_updated_turn_global = ?,
              rank = ?,
              canonicalizes = ?
          WHERE id = ? AND audience_entity_id = ?
        `,
      )
      .run(
        next.state_key,
        next.kind,
        next.text,
        next.owner_entity_id,
        serializeJsonValue(next.provenance_stream_entry_ids),
        serializeJsonValue(next.last_updated_stream_entry_ids),
        next.last_updated_at,
        next.last_updated_turn_global,
        next.rank,
        serializeJsonValue(next.canonicalizes),
        next.id,
        next.audience_entity_id,
      );
  }

  private supersedeEntry(
    audienceEntityId: EntityId,
    operation: SharedStateSupersedeOperation,
    nowMs: number,
    lastUpdatedTurnGlobal: number | null,
    sourceTrustValidator: SharedStateSourceTrustValidator,
  ): void {
    const current = this.getEntry(operation.id, audienceEntityId);
    const replacementId = operation.replacement.id ?? createSharedStateEntryId();

    if (replacementId === current.id) {
      throw new StorageError("Shared state replacement id must differ from superseded id", {
        code: "SHARED_STATE_INVALID_OPERATION",
      });
    }

    const replacement = this.addEntry(
      audienceEntityId,
      {
        ...operation.replacement,
        id: parseSharedStateEntryId(replacementId),
      },
      nowMs,
      operation.replacement.last_updated_turn_global ?? lastUpdatedTurnGlobal,
      sourceTrustValidator,
    );
    const lastUpdatedAt = operation.last_updated_at ?? nowMs;
    const lastUpdatedStreamEntryIds = uniqueStreamEntryIds([
      ...operation.last_updated_stream_entry_ids,
    ]);

    this.assertTrustedSourceStreamIds(
      lastUpdatedStreamEntryIds,
      "last_updated_stream_entry_ids",
      sourceTrustValidator,
    );

    sharedStateEntrySchema.parse({
      ...current,
      superseded_by_id: replacement.id,
      last_updated_at: lastUpdatedAt,
      last_updated_stream_entry_ids: lastUpdatedStreamEntryIds,
      last_updated_turn_global: operation.last_updated_turn_global ?? lastUpdatedTurnGlobal,
    });

    this.db
      .prepare(
        `
          UPDATE shared_state_entries
          SET superseded_by_id = ?,
              last_updated_at = ?,
              last_updated_stream_entry_ids = ?,
              last_updated_turn_global = ?
          WHERE id = ? AND audience_entity_id = ?
        `,
      )
      .run(
        replacement.id,
        lastUpdatedAt,
        serializeJsonValue(lastUpdatedStreamEntryIds),
        operation.last_updated_turn_global ?? lastUpdatedTurnGlobal,
        current.id,
        audienceEntityId,
      );
  }

  private pruneEntry(audienceEntityId: EntityId, operation: SharedStatePruneOperation): void {
    const result = this.db
      .prepare(
        `
          DELETE FROM shared_state_entries
          WHERE id = ? AND audience_entity_id = ?
        `,
      )
      .run(operation.id, audienceEntityId);

    if (result.changes === 0) {
      throw new StorageError(`Unknown shared state entry id: ${operation.id}`, {
        code: "SHARED_STATE_ENTRY_NOT_FOUND",
      });
    }
  }

  private transitionEntryKind(
    audienceEntityId: EntityId,
    operation: SharedStateKindTransitionOperation,
  ): void {
    const current = this.getEntry(operation.id, audienceEntityId);
    const next = sharedStateEntrySchema.parse({
      ...current,
      kind: operation.kind,
    });

    const result = this.db
      .prepare(
        `
          UPDATE shared_state_entries
          SET kind = ?
          WHERE id = ? AND audience_entity_id = ?
        `,
      )
      .run(next.kind, next.id, next.audience_entity_id);

    if (result.changes === 0) {
      throw new StorageError(`Unknown shared state entry id: ${operation.id}`, {
        code: "SHARED_STATE_ENTRY_NOT_FOUND",
      });
    }
  }

  upsert(
    audienceEntityId: EntityId,
    operations: readonly SharedStateOperation[],
    options: SharedStateUpsertOptions = {},
  ): SharedStateArtifact | null {
    const current = this.get(audienceEntityId);

    if (operations.length === 0) {
      if (options.lastCompiledAt === undefined && options.lastCompiledStreamEntryId === undefined) {
        return current;
      }

      const nowMs = options.now ?? this.clock.now();
      const requestedCompiledStreamEntryId = options.lastCompiledStreamEntryId;

      if (current === null) {
        if (options.expectedVersion !== undefined) {
          assertIdentityCasUpdated({
            result: { changes: 0 },
            recordType: "decision_artifact",
            recordId: audienceEntityId,
            expectedVersion: options.expectedVersion,
          });
        }

        if (
          requestedCompiledStreamEntryId === undefined ||
          requestedCompiledStreamEntryId === null
        ) {
          return null;
        }

        this.insertParent({
          audienceEntityId,
          nowMs,
          lastCompiledAt: options.lastCompiledAt ?? nowMs,
          lastCompiledStreamEntryId: requestedCompiledStreamEntryId,
        });

        return this.get(audienceEntityId);
      }

      const expectedVersion = expectedRecordVersion(current, options);
      const lastCompiledAt =
        options.lastCompiledAt === undefined
          ? (current.last_compiled_at ?? nowMs)
          : options.lastCompiledAt;
      const lastCompiledStreamEntryId =
        options.lastCompiledStreamEntryId === undefined
          ? current.last_compiled_stream_entry_id
          : options.lastCompiledStreamEntryId;

      this.updateCompileMarker({
        current,
        expectedVersion,
        nowMs,
        lastCompiledAt,
        lastCompiledStreamEntryId,
      });

      return this.get(audienceEntityId);
    }

    if (current === null && options.expectedVersion !== undefined) {
      assertIdentityCasUpdated({
        result: { changes: 0 },
        recordType: "decision_artifact",
        recordId: audienceEntityId,
        expectedVersion: options.expectedVersion,
      });
    }

    const nowMs = options.now ?? this.clock.now();
    const lastUpdatedTurnGlobal = options.lastUpdatedTurnGlobal ?? null;
    const lastCompiledAt = options.lastCompiledAt ?? nowMs;
    const lastCompiledStreamEntryId = options.lastCompiledStreamEntryId ?? null;
    const sourceTrustValidator = this.sourceTrustValidator(options.sourceTrustValidator);
    const write = this.db.transaction(() => {
      if (current === null) {
        this.insertParent({
          audienceEntityId,
          nowMs,
          lastCompiledAt,
          lastCompiledStreamEntryId,
        });
      } else {
        const expectedVersion = expectedRecordVersion(current, options);
        this.bumpParent({
          current,
          expectedVersion,
          nowMs,
          lastCompiledAt,
          lastCompiledStreamEntryId,
        });
      }

      for (const operation of operations) {
        switch (operation.type) {
          case "add":
            this.addEntry(
              audienceEntityId,
              operation,
              nowMs,
              lastUpdatedTurnGlobal,
              sourceTrustValidator,
            );
            break;
          case "update":
            this.updateEntry(
              audienceEntityId,
              operation,
              nowMs,
              lastUpdatedTurnGlobal,
              sourceTrustValidator,
            );
            break;
          case "supersede":
            this.supersedeEntry(
              audienceEntityId,
              operation,
              nowMs,
              lastUpdatedTurnGlobal,
              sourceTrustValidator,
            );
            break;
          case "prune":
            this.pruneEntry(audienceEntityId, operation);
            break;
          case "transition_kind":
            this.transitionEntryKind(audienceEntityId, operation);
            break;
        }
      }
    });

    write.immediate();
    const next = this.get(audienceEntityId);

    if (next === null) {
      return null;
    }

    if (current === null) {
      return next;
    }

    return {
      ...next,
      record_version: nextRecordVersion(expectedRecordVersion(current, options)),
    };
  }

  delete(audienceEntityId: EntityId): void {
    this.db
      .prepare("DELETE FROM shared_state_artifacts WHERE audience_entity_id = ?")
      .run(audienceEntityId);
  }
}
