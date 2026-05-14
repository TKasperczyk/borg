import { parseJsonArray, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createDecisionArtifactEntryId,
  parseDecisionArtifactEntryId,
  type DecisionArtifactEntryId,
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
  decisionArtifactEntrySchema,
  decisionArtifactCanonicalizesSchema,
  decisionArtifactSchema,
  type DecisionArtifact,
  type DecisionArtifactCanonicalizes,
  type DecisionArtifactEntry,
  type DecisionArtifactEntryKind,
} from "./types.js";

const DECISION_ARTIFACT_JSON_ARRAY_CODEC = {
  errorCode: "DECISION_ARTIFACT_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse decision artifact ${label}`,
} satisfies JsonArrayCodecOptions;

const EMPTY_DECISION_ARTIFACT_CANONICALIZES: DecisionArtifactCanonicalizes = {
  goal_ids: [],
  commitment_ids: [],
  action_ids: [],
  open_question_ids: [],
};

export type DecisionArtifactAddOperation = {
  type: "add";
  id?: DecisionArtifactEntryId;
  kind: DecisionArtifactEntryKind;
  text: string;
  owner_entity_id?: EntityId | null;
  provenance_stream_entry_ids: readonly StreamEntryId[];
  last_updated_stream_entry_ids?: readonly StreamEntryId[];
  created_at?: number;
  last_updated_at?: number;
  rank?: number;
  canonicalizes?: DecisionArtifactCanonicalizes;
};

export type DecisionArtifactUpdateOperation = {
  type: "update";
  id: DecisionArtifactEntryId;
  kind?: DecisionArtifactEntryKind;
  text?: string;
  owner_entity_id?: EntityId | null;
  add_provenance_stream_entry_ids?: readonly StreamEntryId[];
  last_updated_stream_entry_ids: readonly StreamEntryId[];
  last_updated_at?: number;
  rank?: number;
  canonicalizes?: DecisionArtifactCanonicalizes;
};

export type DecisionArtifactSupersedeOperation = {
  type: "supersede";
  id: DecisionArtifactEntryId;
  replacement: Omit<DecisionArtifactAddOperation, "type">;
  last_updated_stream_entry_ids: readonly StreamEntryId[];
  last_updated_at?: number;
};

export type DecisionArtifactPruneOperation = {
  type: "prune";
  id: DecisionArtifactEntryId;
};

export type DecisionArtifactOperation =
  | DecisionArtifactAddOperation
  | DecisionArtifactUpdateOperation
  | DecisionArtifactSupersedeOperation
  | DecisionArtifactPruneOperation;

export type DecisionArtifactUpsertOptions = IdentityCasOptions & {
  now?: number;
  lastCompiledAt?: number | null;
  lastCompiledStreamEntryId?: StreamEntryId | null;
};

export type DecisionArtifactRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function parseStreamEntryIds(value: string, label: string): StreamEntryId[] {
  return parseJsonArray<StreamEntryId>(value, label, DECISION_ARTIFACT_JSON_ARRAY_CODEC);
}

function uniqueStreamEntryIds(values: readonly StreamEntryId[]): StreamEntryId[] {
  return [...new Set(values)];
}

function emptyCanonicalizes(): DecisionArtifactCanonicalizes {
  return {
    goal_ids: [],
    commitment_ids: [],
    action_ids: [],
    open_question_ids: [],
  };
}

function uniqueCanonicalizes(
  value: DecisionArtifactCanonicalizes | undefined,
): DecisionArtifactCanonicalizes {
  const source = value ?? EMPTY_DECISION_ARTIFACT_CANONICALIZES;

  return decisionArtifactCanonicalizesSchema.parse({
    goal_ids: [...new Set(source.goal_ids)],
    commitment_ids: [...new Set(source.commitment_ids)],
    action_ids: [...new Set(source.action_ids)],
    open_question_ids: [...new Set(source.open_question_ids)],
  });
}

function mergeCanonicalizes(
  current: DecisionArtifactCanonicalizes,
  next: DecisionArtifactCanonicalizes | undefined,
): DecisionArtifactCanonicalizes {
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

function parseCanonicalizes(value: unknown): DecisionArtifactCanonicalizes {
  if (value === null || value === undefined) {
    return emptyCanonicalizes();
  }

  try {
    const parsed = JSON.parse(String(value)) as unknown;
    const result = decisionArtifactCanonicalizesSchema.safeParse(parsed);

    if (!result.success) {
      throw result.error;
    }

    return uniqueCanonicalizes(result.data);
  } catch (error) {
    throw new StorageError("Failed to parse decision artifact canonicalizes", {
      cause: error,
      code: "DECISION_ARTIFACT_ROW_INVALID",
    });
  }
}

function mapEntryRow(row: Record<string, unknown>): DecisionArtifactEntry {
  const parsed = decisionArtifactEntrySchema.safeParse({
    id: row.id,
    audience_entity_id: row.audience_entity_id,
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
    superseded_by_id:
      row.superseded_by_id === null || row.superseded_by_id === undefined
        ? null
        : row.superseded_by_id,
    rank: Number(row.rank ?? 0),
    canonicalizes: parseCanonicalizes(row.canonicalizes),
  });

  if (!parsed.success) {
    throw new StorageError("Decision artifact entry row failed validation", {
      cause: parsed.error,
      code: "DECISION_ARTIFACT_ROW_INVALID",
    });
  }

  return parsed.data;
}

function mapArtifactRow(
  row: Record<string, unknown>,
  entries: readonly DecisionArtifactEntry[],
): DecisionArtifact {
  const parsed = decisionArtifactSchema.safeParse({
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
    throw new StorageError("Decision artifact row failed validation", {
      cause: parsed.error,
      code: "DECISION_ARTIFACT_ROW_INVALID",
    });
  }

  return parsed.data;
}

export class DecisionArtifactRepository {
  private readonly clock: Clock;

  constructor(private readonly options: DecisionArtifactRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private listEntries(audienceEntityId: EntityId): DecisionArtifactEntry[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM decision_artifact_entries
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

  get(audienceEntityId: EntityId): DecisionArtifact | null {
    const row = this.db
      .prepare("SELECT * FROM decision_artifacts WHERE audience_entity_id = ?")
      .get(audienceEntityId) as Record<string, unknown> | undefined;

    if (row === undefined) {
      return null;
    }

    return mapArtifactRow(row, this.listEntries(audienceEntityId));
  }

  private getEntry(id: DecisionArtifactEntryId, audienceEntityId: EntityId): DecisionArtifactEntry {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM decision_artifact_entries
          WHERE id = ? AND audience_entity_id = ?
        `,
      )
      .get(id, audienceEntityId) as Record<string, unknown> | undefined;

    if (row === undefined) {
      throw new StorageError(`Unknown decision artifact entry id: ${id}`, {
        code: "DECISION_ARTIFACT_ENTRY_NOT_FOUND",
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
          INSERT INTO decision_artifacts (
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
    current: DecisionArtifact;
    expectedVersion: number;
    nowMs: number;
    lastCompiledAt: number | null;
    lastCompiledStreamEntryId: StreamEntryId | null;
  }): void {
    const result = this.db
      .prepare(
        `
          UPDATE decision_artifacts
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
    current: DecisionArtifact;
    expectedVersion: number;
    nowMs: number;
    lastCompiledAt: number | null;
    lastCompiledStreamEntryId: StreamEntryId | null;
  }): void {
    const result = this.db
      .prepare(
        `
          UPDATE decision_artifacts
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

  private insertEntry(entry: DecisionArtifactEntry): void {
    const parsed = decisionArtifactEntrySchema.parse(entry);

    this.db
      .prepare(
        `
          INSERT INTO decision_artifact_entries (
            id, audience_entity_id, kind, text, owner_entity_id,
            provenance_stream_entry_ids, last_updated_stream_entry_ids,
            created_at, last_updated_at, superseded_by_id, rank, canonicalizes
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        parsed.id,
        parsed.audience_entity_id,
        parsed.kind,
        parsed.text,
        parsed.owner_entity_id,
        serializeJsonValue(parsed.provenance_stream_entry_ids),
        serializeJsonValue(parsed.last_updated_stream_entry_ids),
        parsed.created_at,
        parsed.last_updated_at,
        parsed.superseded_by_id,
        parsed.rank,
        serializeJsonValue(parsed.canonicalizes),
      );
  }

  private addEntry(
    audienceEntityId: EntityId,
    operation: Omit<DecisionArtifactAddOperation, "type">,
    nowMs: number,
  ): DecisionArtifactEntry {
    const provenanceStreamEntryIds = uniqueStreamEntryIds([
      ...operation.provenance_stream_entry_ids,
    ]);
    const lastUpdatedStreamEntryIds = uniqueStreamEntryIds([
      ...(operation.last_updated_stream_entry_ids ?? operation.provenance_stream_entry_ids),
    ]);
    const entry = decisionArtifactEntrySchema.parse({
      id: operation.id ?? createDecisionArtifactEntryId(),
      audience_entity_id: audienceEntityId,
      kind: operation.kind,
      text: operation.text,
      owner_entity_id: operation.owner_entity_id ?? null,
      provenance_stream_entry_ids: provenanceStreamEntryIds,
      last_updated_stream_entry_ids: lastUpdatedStreamEntryIds,
      created_at: operation.created_at ?? nowMs,
      last_updated_at: operation.last_updated_at ?? operation.created_at ?? nowMs,
      superseded_by_id: null,
      rank: operation.rank ?? 0,
      canonicalizes: uniqueCanonicalizes(operation.canonicalizes),
    });

    this.insertEntry(entry);
    return entry;
  }

  private updateEntry(
    audienceEntityId: EntityId,
    operation: DecisionArtifactUpdateOperation,
    nowMs: number,
  ): void {
    const current = this.getEntry(operation.id, audienceEntityId);
    const addProvenance = operation.add_provenance_stream_entry_ids ?? [];
    const next = decisionArtifactEntrySchema.parse({
      ...current,
      kind: operation.kind ?? current.kind,
      text: operation.text ?? current.text,
      owner_entity_id:
        operation.owner_entity_id === undefined
          ? current.owner_entity_id
          : operation.owner_entity_id,
      provenance_stream_entry_ids: uniqueStreamEntryIds([
        ...current.provenance_stream_entry_ids,
        ...addProvenance,
      ]),
      last_updated_stream_entry_ids: uniqueStreamEntryIds([
        ...operation.last_updated_stream_entry_ids,
      ]),
      last_updated_at: operation.last_updated_at ?? nowMs,
      rank: operation.rank ?? current.rank,
      canonicalizes: mergeCanonicalizes(current.canonicalizes, operation.canonicalizes),
    });

    this.db
      .prepare(
        `
          UPDATE decision_artifact_entries
          SET kind = ?,
              text = ?,
              owner_entity_id = ?,
              provenance_stream_entry_ids = ?,
              last_updated_stream_entry_ids = ?,
              last_updated_at = ?,
              rank = ?,
              canonicalizes = ?
          WHERE id = ? AND audience_entity_id = ?
        `,
      )
      .run(
        next.kind,
        next.text,
        next.owner_entity_id,
        serializeJsonValue(next.provenance_stream_entry_ids),
        serializeJsonValue(next.last_updated_stream_entry_ids),
        next.last_updated_at,
        next.rank,
        serializeJsonValue(next.canonicalizes),
        next.id,
        next.audience_entity_id,
      );
  }

  private supersedeEntry(
    audienceEntityId: EntityId,
    operation: DecisionArtifactSupersedeOperation,
    nowMs: number,
  ): void {
    const current = this.getEntry(operation.id, audienceEntityId);
    const replacementId = operation.replacement.id ?? createDecisionArtifactEntryId();

    if (replacementId === current.id) {
      throw new StorageError("Decision artifact replacement id must differ from superseded id", {
        code: "DECISION_ARTIFACT_INVALID_OPERATION",
      });
    }

    const replacement = this.addEntry(
      audienceEntityId,
      {
        ...operation.replacement,
        id: parseDecisionArtifactEntryId(replacementId),
      },
      nowMs,
    );
    const lastUpdatedAt = operation.last_updated_at ?? nowMs;
    const lastUpdatedStreamEntryIds = uniqueStreamEntryIds([
      ...operation.last_updated_stream_entry_ids,
    ]);

    decisionArtifactEntrySchema.parse({
      ...current,
      superseded_by_id: replacement.id,
      last_updated_at: lastUpdatedAt,
      last_updated_stream_entry_ids: lastUpdatedStreamEntryIds,
    });

    this.db
      .prepare(
        `
          UPDATE decision_artifact_entries
          SET superseded_by_id = ?,
              last_updated_at = ?,
              last_updated_stream_entry_ids = ?
          WHERE id = ? AND audience_entity_id = ?
        `,
      )
      .run(
        replacement.id,
        lastUpdatedAt,
        serializeJsonValue(lastUpdatedStreamEntryIds),
        current.id,
        audienceEntityId,
      );
  }

  private pruneEntry(audienceEntityId: EntityId, operation: DecisionArtifactPruneOperation): void {
    const result = this.db
      .prepare(
        `
          DELETE FROM decision_artifact_entries
          WHERE id = ? AND audience_entity_id = ?
        `,
      )
      .run(operation.id, audienceEntityId);

    if (result.changes === 0) {
      throw new StorageError(`Unknown decision artifact entry id: ${operation.id}`, {
        code: "DECISION_ARTIFACT_ENTRY_NOT_FOUND",
      });
    }
  }

  upsert(
    audienceEntityId: EntityId,
    operations: readonly DecisionArtifactOperation[],
    options: DecisionArtifactUpsertOptions = {},
  ): DecisionArtifact | null {
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
    const lastCompiledAt = options.lastCompiledAt ?? nowMs;
    const lastCompiledStreamEntryId = options.lastCompiledStreamEntryId ?? null;
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
            this.addEntry(audienceEntityId, operation, nowMs);
            break;
          case "update":
            this.updateEntry(audienceEntityId, operation, nowMs);
            break;
          case "supersede":
            this.supersedeEntry(audienceEntityId, operation, nowMs);
            break;
          case "prune":
            this.pruneEntry(audienceEntityId, operation);
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
      .prepare("DELETE FROM decision_artifacts WHERE audience_entity_id = ?")
      .run(audienceEntityId);
  }
}
