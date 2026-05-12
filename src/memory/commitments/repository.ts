import { SystemClock, type Clock } from "../../util/clock.js";
import { CommitmentError, ProvenanceError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createCommitmentId,
  createEntityId,
  parseCommitmentId,
  parseEntityId,
  type CommitmentId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import { parseJsonArray, type JsonArrayCodecOptions } from "../../storage/codecs.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import {
  isEpisodeProvenance,
  parseStoredProvenance,
  provenanceSchema,
  toStoredProvenance,
} from "../common/provenance.js";
import {
  assertIdentityCasUpdated,
  expectedRecordVersion,
  nextRecordVersion,
  type IdentityCasOptions,
} from "../common/cas.js";
import { IdentityEventRepository } from "../identity/repository.js";
import { runIdentityWrite } from "../self/shared/identity-events.js";
import {
  commitmentPatchSchema,
  commitmentSchema,
  entityKindSchema,
  entityRecordSchema,
  nameProvenanceSchema,
  normalizeDirectiveFamily,
  type CommitmentApplicableOptions,
  type CommitmentListOptions,
  type CommitmentPatch,
  type CommitmentRecord,
  type CommitmentType,
  type EntityKind,
  type EntityRecord,
  type NameProvenance,
} from "./types.js";

function normalizeName(value: string): string {
  return value.trim().toLowerCase();
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
}

const NAME_PROVENANCE_RANK: Record<NameProvenance, number> = {
  unknown: 0,
  assistant_seeded: 1,
  config_default_user: 2,
  transport_audience_label: 2,
  user_confirmed: 3,
  user_declared: 4,
};

function strongerNameProvenance(
  current: NameProvenance | undefined,
  next: NameProvenance,
): NameProvenance {
  const currentValue = current ?? "unknown";

  return NAME_PROVENANCE_RANK[next] > NAME_PROVENANCE_RANK[currentValue] ? next : currentValue;
}

const COMMITMENT_JSON_ARRAY_CODEC = {
  errorCode: "COMMITMENT_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse ${label}`,
  createError: (message, options) => new CommitmentError(message, options),
} satisfies JsonArrayCodecOptions;

function mapEntityRow(row: Record<string, unknown>): EntityRecord {
  return entityRecordSchema.parse({
    id: row.id,
    canonical_name: row.canonical_name,
    aliases: uniqueStrings(
      parseJsonArray<string>(String(row.aliases ?? "[]"), "aliases", COMMITMENT_JSON_ARRAY_CODEC),
    ),
    kind: row.kind === null || row.kind === undefined ? null : entityKindSchema.parse(row.kind),
    name_provenance: nameProvenanceSchema.parse(row.name_provenance ?? "unknown"),
    created_at: Number(row.created_at),
  });
}

function mapCommitmentRow(row: Record<string, unknown>): CommitmentRecord {
  const sourceStreamEntryIds =
    row.source_stream_entry_ids === null || row.source_stream_entry_ids === undefined
      ? undefined
      : parseJsonArray<StreamEntryId>(
          String(row.source_stream_entry_ids),
          "source_stream_entry_ids",
          COMMITMENT_JSON_ARRAY_CODEC,
        );
  const parsed = commitmentSchema.safeParse({
    id: row.id,
    record_version: Number(row.record_version ?? 1),
    type: row.type,
    directive_family: row.directive_family,
    closure_pressure_relevance: row.closure_pressure_relevance,
    directive: row.directive,
    priority: Number(row.priority),
    made_to_entity:
      row.made_to_entity === null || row.made_to_entity === undefined
        ? null
        : parseEntityId(String(row.made_to_entity)),
    restricted_audience:
      row.restricted_audience === null || row.restricted_audience === undefined
        ? null
        : parseEntityId(String(row.restricted_audience)),
    about_entity:
      row.about_entity === null || row.about_entity === undefined
        ? null
        : parseEntityId(String(row.about_entity)),
    committed_by_entity_id:
      row.committed_by_entity_id === null || row.committed_by_entity_id === undefined
        ? null
        : parseEntityId(String(row.committed_by_entity_id)),
    provenance: parseStoredProvenance({
      provenance_kind: row.provenance_kind,
      provenance_episode_ids: row.provenance_episode_ids,
      provenance_process: row.provenance_process,
    }),
    ...(sourceStreamEntryIds === undefined || sourceStreamEntryIds.length === 0
      ? {}
      : { source_stream_entry_ids: sourceStreamEntryIds }),
    created_at: Number(row.created_at),
    expires_at:
      row.expires_at === null || row.expires_at === undefined ? null : Number(row.expires_at),
    expired_at:
      row.expired_at === null || row.expired_at === undefined ? null : Number(row.expired_at),
    revoked_at:
      row.revoked_at === null || row.revoked_at === undefined ? null : Number(row.revoked_at),
    revoked_reason:
      row.revoked_reason === null || row.revoked_reason === undefined
        ? null
        : String(row.revoked_reason),
    revoke_provenance:
      row.revoke_provenance_kind === null || row.revoke_provenance_kind === undefined
        ? null
        : parseStoredProvenance({
            provenance_kind: row.revoke_provenance_kind,
            provenance_episode_ids: row.revoke_provenance_episode_ids,
            provenance_process: row.revoke_provenance_process,
          }),
    superseded_by:
      row.superseded_by === null || row.superseded_by === undefined
        ? null
        : parseCommitmentId(String(row.superseded_by)),
    last_reinforced_at: Number(row.last_reinforced_at),
  });

  if (!parsed.success) {
    throw new CommitmentError("Commitment row failed validation", {
      cause: parsed.error,
      code: "COMMITMENT_ROW_INVALID",
    });
  }

  return parsed.data;
}

export type EntityRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export type EntityListOptions = {
  kind?: EntityKind;
};

export type EntityAddInput = {
  id?: EntityId;
  canonicalName: string;
  aliases?: readonly string[];
  kind?: EntityKind;
  provenance?: NameProvenance;
  createdAt?: number;
};

export type EntityResolveOptions = {
  provenance?: NameProvenance;
  kind?: EntityKind;
};

export class EntityRepository {
  private readonly clock: Clock;

  constructor(private readonly options: EntityRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private listEntities(options: EntityListOptions = {}): EntityRecord[] {
    const kindFilter = options.kind;
    const rows =
      kindFilter === undefined
        ? (this.db
            .prepare(
              `
                SELECT id, canonical_name, aliases, kind, name_provenance, created_at
                FROM entities
                ORDER BY created_at ASC
              `,
            )
            .all() as Record<string, unknown>[])
        : (this.db
            .prepare(
              `
                SELECT id, canonical_name, aliases, kind, name_provenance, created_at
                FROM entities
                WHERE kind = ?
                ORDER BY created_at ASC
              `,
            )
            .all(kindFilter) as Record<string, unknown>[]);

    return rows.map((row) => mapEntityRow(row));
  }

  list(options: EntityListOptions = {}): EntityRecord[] {
    return this.listEntities(options);
  }

  add(input: EntityAddInput): EntityRecord {
    const canonicalName = input.canonicalName.trim();
    const provenance = input.provenance ?? "unknown";

    if (canonicalName.length === 0) {
      throw new CommitmentError("Entity name is required", {
        code: "ENTITY_NAME_REQUIRED",
      });
    }

    const entity = entityRecordSchema.parse({
      id: input.id ?? createEntityId(),
      canonical_name: canonicalName,
      aliases: uniqueStrings(input.aliases ?? []),
      kind: input.kind ?? "person",
      name_provenance: provenance,
      created_at: input.createdAt ?? this.clock.now(),
    });

    this.db
      .prepare(
        `
          INSERT INTO entities (id, canonical_name, aliases, kind, name_provenance, created_at)
          VALUES (?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        entity.id,
        entity.canonical_name,
        serializeJsonValue(entity.aliases),
        entity.kind,
        entity.name_provenance,
        entity.created_at,
      );

    return entity;
  }

  findByName(name: string, options: EntityListOptions = {}): EntityId | null {
    const normalized = normalizeName(name);

    if (normalized.length === 0) {
      return null;
    }

    for (const entity of this.listEntities(options)) {
      const names = [entity.canonical_name, ...entity.aliases].map((value) => normalizeName(value));

      if (names.includes(normalized)) {
        return entity.id;
      }
    }

    return null;
  }

  resolve(name: string, options: EntityResolveOptions = {}): EntityId {
    const normalized = normalizeName(name);
    const provenance = options.provenance ?? "unknown";

    if (normalized.length === 0) {
      throw new CommitmentError("Entity name is required", {
        code: "ENTITY_NAME_REQUIRED",
      });
    }

    const existing = this.findByName(
      name,
      options.kind === undefined ? {} : { kind: options.kind },
    );

    if (existing !== null) {
      const current = this.get(existing);
      const nextProvenance = strongerNameProvenance(current?.name_provenance, provenance);

      if (current !== null && nextProvenance !== (current.name_provenance ?? "unknown")) {
        this.db
          .prepare("UPDATE entities SET name_provenance = ? WHERE id = ?")
          .run(nextProvenance, existing);
      }

      return existing;
    }

    const entity = this.add({
      canonicalName: name,
      kind: options.kind,
      provenance,
    });

    return entity.id;
  }

  get(id: EntityId): EntityRecord | null {
    const row = this.db.prepare("SELECT * FROM entities WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapEntityRow(row);
  }

  rename(id: EntityId, canonicalName: string): EntityRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    const next = entityRecordSchema.parse({
      ...current,
      canonical_name: canonicalName.trim(),
    });

    this.db
      .prepare("UPDATE entities SET canonical_name = ?, aliases = ? WHERE id = ?")
      .run(next.canonical_name, serializeJsonValue(next.aliases), id);

    return next;
  }

  addAlias(id: EntityId, alias: string): EntityRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    const next = entityRecordSchema.parse({
      ...current,
      aliases: uniqueStrings([...current.aliases, alias]),
    });

    this.db
      .prepare("UPDATE entities SET aliases = ? WHERE id = ?")
      .run(serializeJsonValue(next.aliases), id);

    return next;
  }
}

export type CommitmentRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  identityEventRepository?: IdentityEventRepository;
};

export class CommitmentRepository {
  private readonly clock: Clock;

  constructor(private readonly options: CommitmentRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private get identityEventRepository(): IdentityEventRepository | undefined {
    return this.options.identityEventRepository;
  }

  private materializeExpiredCommitments(nowMs = this.clock.now()): void {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM commitments
          WHERE expired_at IS NULL
            AND expires_at IS NOT NULL
            AND expires_at <= ?
            AND revoked_at IS NULL
            AND superseded_by IS NULL
          ORDER BY expires_at ASC, created_at ASC
        `,
      )
      .all(nowMs) as Record<string, unknown>[];

    if (rows.length === 0) {
      return;
    }

    const update = this.db.prepare(
      "UPDATE commitments SET expired_at = ?, record_version = record_version + 1 WHERE id = ? AND record_version = ?",
    );

    runIdentityWrite(this.identityEventRepository, () => {
      for (const row of rows) {
        const current = mapCommitmentRow(row);
        const currentVersion = expectedRecordVersion(current);
        const expiredAt = current.expires_at ?? nowMs;
        const result = update.run(expiredAt, current.id, currentVersion);

        if (result.changes === 0) {
          continue;
        }

        const next: CommitmentRecord = {
          ...current,
          record_version: nextRecordVersion(currentVersion),
          expired_at: expiredAt,
        };
        this.identityEventRepository?.record({
          record_type: "commitment",
          record_id: current.id,
          action: "expire",
          old_value: current,
          new_value: next,
          provenance: current.provenance,
          ts: expiredAt,
        });
      }
    });
  }

  private findActiveDirectiveFamilyMatches(
    record: CommitmentRecord,
    nowMs: number,
  ): CommitmentRecord[] {
    if (record.directive_family.length === 0) {
      return [];
    }

    this.materializeExpiredCommitments(nowMs);

    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM commitments
          WHERE directive_family = ?
            AND revoked_at IS NULL
            AND superseded_by IS NULL
            AND expired_at IS NULL
            AND (expires_at IS NULL OR expires_at > ?)
          ORDER BY last_reinforced_at DESC, created_at DESC
        `,
      )
      .all(record.directive_family, nowMs) as Record<string, unknown>[];

    return rows
      .map((row) => mapCommitmentRow(row))
      .filter(
        (candidate) =>
          candidate.made_to_entity === record.made_to_entity &&
          candidate.restricted_audience === record.restricted_audience &&
          candidate.about_entity === record.about_entity &&
          candidate.committed_by_entity_id === record.committed_by_entity_id,
      );
  }

  private mergeDirectiveFamilyMatch(
    incoming: CommitmentRecord,
    matches: readonly CommitmentRecord[],
  ): CommitmentRecord {
    const [kept, ...superseded] = [...matches].sort(
      (left, right) =>
        right.last_reinforced_at - left.last_reinforced_at ||
        right.created_at - left.created_at ||
        right.id.localeCompare(left.id),
    );

    if (kept === undefined) {
      return incoming;
    }

    const sourceStreamEntryIds = uniqueStrings([
      ...(kept.source_stream_entry_ids ?? []),
      ...(incoming.source_stream_entry_ids ?? []),
    ]);
    const keptVersion = expectedRecordVersion(kept);
    const next = commitmentSchema.parse({
      ...kept,
      record_version: nextRecordVersion(keptVersion),
      priority: Math.max(kept.priority, incoming.priority),
      closure_pressure_relevance:
        kept.closure_pressure_relevance === "no_closure" ||
        incoming.closure_pressure_relevance === "no_closure"
          ? "no_closure"
          : incoming.closure_pressure_relevance,
      source_stream_entry_ids: sourceStreamEntryIds.length === 0 ? undefined : sourceStreamEntryIds,
      last_reinforced_at: Math.max(kept.last_reinforced_at, incoming.last_reinforced_at),
    });

    const keptResult = this.db
      .prepare(
        `
          UPDATE commitments
          SET priority = ?, closure_pressure_relevance = ?, source_stream_entry_ids = ?,
              last_reinforced_at = ?, record_version = record_version + 1
          WHERE id = ? AND record_version = ?
        `,
      )
      .run(
        next.priority,
        next.closure_pressure_relevance,
        next.source_stream_entry_ids === undefined
          ? null
          : serializeJsonValue(next.source_stream_entry_ids),
        next.last_reinforced_at,
        kept.id,
        keptVersion,
      );
    assertIdentityCasUpdated({
      result: keptResult,
      recordType: "commitment",
      recordId: kept.id,
      expectedVersion: keptVersion,
    });

    this.identityEventRepository?.record({
      record_type: "commitment",
      record_id: kept.id,
      action: "update",
      old_value: kept,
      new_value: next,
      reason: "directive_family_reinforced",
      provenance: incoming.provenance,
      ts: next.last_reinforced_at,
    });

    const supersede = this.db.prepare(
      "UPDATE commitments SET superseded_by = ?, record_version = record_version + 1 WHERE id = ? AND record_version = ?",
    );

    for (const current of superseded) {
      const currentVersion = expectedRecordVersion(current);
      const result = supersede.run(kept.id, current.id, currentVersion);
      assertIdentityCasUpdated({
        result,
        recordType: "commitment",
        recordId: current.id,
        expectedVersion: currentVersion,
      });
      this.identityEventRepository?.record({
        record_type: "commitment",
        record_id: current.id,
        action: "update",
        old_value: current,
        new_value: {
          ...current,
          record_version: nextRecordVersion(currentVersion),
          superseded_by: kept.id,
        },
        reason: "directive_family_duplicate",
        provenance: incoming.provenance,
        ts: next.last_reinforced_at,
      });
    }

    return next;
  }

  add(input: {
    id?: CommitmentId;
    type: CommitmentType;
    directiveFamily: string;
    directive: string;
    priority: number;
    closurePressureRelevance?: CommitmentRecord["closure_pressure_relevance"];
    madeToEntity?: EntityId | null;
    restrictedAudience?: EntityId | null;
    aboutEntity?: EntityId | null;
    committedByEntityId?: EntityId | null;
    provenance: CommitmentRecord["provenance"];
    sourceStreamEntryIds?: readonly StreamEntryId[];
    createdAt?: number;
    expiresAt?: number | null;
  }): CommitmentRecord {
    if (input.provenance === undefined) {
      throw new ProvenanceError("Commitment requires provenance", {
        code: "PROVENANCE_REQUIRED",
      });
    }

    const createdAt = input.createdAt ?? this.clock.now();
    const expiresAt = input.expiresAt ?? null;

    const record = commitmentSchema.parse({
      id: input.id ?? createCommitmentId(),
      record_version: 1,
      type: input.type,
      directive_family: normalizeDirectiveFamily(input.directiveFamily),
      closure_pressure_relevance: input.closurePressureRelevance ?? "neutral",
      directive: input.directive,
      priority: input.priority,
      made_to_entity: input.madeToEntity ?? null,
      restricted_audience: input.restrictedAudience ?? null,
      about_entity: input.aboutEntity ?? null,
      committed_by_entity_id: input.committedByEntityId ?? null,
      provenance: provenanceSchema.parse(input.provenance),
      ...(input.sourceStreamEntryIds === undefined || input.sourceStreamEntryIds.length === 0
        ? {}
        : { source_stream_entry_ids: [...input.sourceStreamEntryIds] }),
      created_at: createdAt,
      expires_at: expiresAt,
      expired_at: expiresAt !== null && expiresAt <= createdAt ? expiresAt : null,
      revoked_at: null,
      revoked_reason: null,
      revoke_provenance: null,
      superseded_by: null,
      last_reinforced_at: createdAt,
    });
    const storedProvenance = toStoredProvenance(record.provenance);
    const familyMatches = this.findActiveDirectiveFamilyMatches(record, createdAt);

    return runIdentityWrite(this.identityEventRepository, () => {
      if (familyMatches.length > 0) {
        return this.mergeDirectiveFamilyMatch(record, familyMatches);
      }

      this.db
        .prepare(
          `
            INSERT INTO commitments (
              id, type, directive_family, closure_pressure_relevance, directive, priority,
              made_to_entity, restricted_audience, about_entity, committed_by_entity_id,
              source_episode_ids, provenance_kind, provenance_episode_ids, provenance_process,
              source_stream_entry_ids, created_at, expires_at, expired_at, revoked_at, revoked_reason,
              revoke_provenance_kind, revoke_provenance_episode_ids, revoke_provenance_process,
              superseded_by, last_reinforced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          record.id,
          record.type,
          record.directive_family,
          record.closure_pressure_relevance,
          record.directive,
          record.priority,
          record.made_to_entity,
          record.restricted_audience,
          record.about_entity,
          record.committed_by_entity_id,
          serializeJsonValue(
            record.provenance.kind === "episodes" ? record.provenance.episode_ids : [],
          ),
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          record.source_stream_entry_ids === undefined
            ? null
            : serializeJsonValue(record.source_stream_entry_ids),
          record.created_at,
          record.expires_at,
          record.expired_at,
          record.revoked_at,
          record.revoked_reason,
          null,
          null,
          null,
          record.superseded_by,
          record.last_reinforced_at,
        );

      this.identityEventRepository?.record({
        record_type: "commitment",
        record_id: record.id,
        action: "create",
        old_value: null,
        new_value: record,
        provenance: record.provenance,
        ts: record.created_at,
      });

      return record;
    });
  }

  get(id: CommitmentId): CommitmentRecord | null {
    const row = this.db.prepare("SELECT * FROM commitments WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapCommitmentRow(row);
  }

  list(options: CommitmentListOptions = {}): CommitmentRecord[] {
    const nowMs = options.nowMs ?? this.clock.now();
    this.materializeExpiredCommitments(nowMs);
    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.activeOnly === true) {
      filters.push("revoked_at IS NULL");
      filters.push("superseded_by IS NULL");
      filters.push("expired_at IS NULL");
      filters.push("(expires_at IS NULL OR expires_at > ?)");
      values.push(nowMs);
    }

    if (options.audience !== undefined) {
      if (options.audience === null) {
        filters.push("restricted_audience IS NULL");
        filters.push("made_to_entity IS NULL");
      } else {
        filters.push(
          "((restricted_audience IS NULL AND (made_to_entity IS NULL OR made_to_entity = ?)) OR restricted_audience = ?)",
        );
        values.push(options.audience, options.audience);
      }
    }

    if (options.aboutEntity !== undefined && options.aboutEntity !== null) {
      filters.push("(about_entity IS NULL OR about_entity = ?)");
      values.push(options.aboutEntity);
    }

    if (options.committedByEntity !== undefined) {
      if (options.committedByEntity === null) {
        filters.push("committed_by_entity_id IS NULL");
      } else {
        filters.push("committed_by_entity_id = ?");
        values.push(options.committedByEntity);
      }
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM commitments
          ${whereClause}
          ORDER BY priority DESC, created_at ASC
        `,
      )
      .all(...values) as Record<string, unknown>[];

    return rows.map((row) => mapCommitmentRow(row));
  }

  countActive(nowMs = this.clock.now()): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM commitments
          WHERE revoked_at IS NULL
            AND superseded_by IS NULL
            AND expired_at IS NULL
            AND (expires_at IS NULL OR expires_at > ?)
        `,
      )
      .get(nowMs) as { count: number } | undefined;

    return Number(row?.count ?? 0);
  }

  countSuperseded(): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM commitments
          WHERE superseded_by IS NOT NULL
        `,
      )
      .get() as { count: number } | undefined;

    return Number(row?.count ?? 0);
  }

  revoke(
    id: CommitmentId,
    reason: string,
    provenance: CommitmentRecord["provenance"],
    timestamp = this.clock.now(),
    options: IdentityCasOptions = {},
  ): CommitmentRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    const parsedReason = reason.trim();
    const parsedProvenance = provenanceSchema.parse(provenance);
    const expectedVersion = expectedRecordVersion(current, options);
    const storedProvenance = toStoredProvenance(parsedProvenance);

    return runIdentityWrite(this.identityEventRepository, () => {
      const result = this.db
        .prepare(
          `
            UPDATE commitments
            SET revoked_at = ?, revoked_reason = ?, revoke_provenance_kind = ?,
                revoke_provenance_episode_ids = ?, revoke_provenance_process = ?,
                record_version = record_version + 1
            WHERE id = ? AND record_version = ?
          `,
        )
        .run(
          timestamp,
          parsedReason,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          id,
          expectedVersion,
        );
      assertIdentityCasUpdated({
        result,
        recordType: "commitment",
        recordId: id,
        expectedVersion,
      });
      const next = {
        ...current,
        record_version: nextRecordVersion(expectedVersion),
        revoked_at: timestamp,
        revoked_reason: parsedReason,
        revoke_provenance: parsedProvenance,
      };
      this.identityEventRepository?.record({
        record_type: "commitment",
        record_id: id,
        action: "revoke",
        old_value: current,
        new_value: next,
        reason: parsedReason,
        provenance: parsedProvenance,
        ts: timestamp,
      });
      return next;
    });
  }

  supersede(
    id: CommitmentId,
    nextId: CommitmentId,
    options: IdentityCasOptions = {},
  ): CommitmentRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    const expectedVersion = expectedRecordVersion(current, options);
    return runIdentityWrite(this.identityEventRepository, () => {
      const result = this.db
        .prepare(
          "UPDATE commitments SET superseded_by = ?, record_version = record_version + 1 WHERE id = ? AND record_version = ?",
        )
        .run(nextId, id, expectedVersion);
      assertIdentityCasUpdated({
        result,
        recordType: "commitment",
        recordId: id,
        expectedVersion,
      });
      const next = {
        ...current,
        record_version: nextRecordVersion(expectedVersion),
        superseded_by: nextId,
      };
      this.identityEventRepository?.record({
        record_type: "commitment",
        record_id: id,
        action: "update",
        old_value: current,
        new_value: next,
        provenance: current.provenance,
      });
      return next;
    });
  }

  findByEvidenceStreamEntryId(entryId: StreamEntryId): boolean {
    const rows = this.db
      .prepare(
        `
          SELECT source_stream_entry_ids
          FROM commitments
          WHERE provenance_kind = 'online'
            AND provenance_process = 'corrective-preference-extractor'
            AND source_stream_entry_ids IS NOT NULL
        `,
      )
      .all() as Record<string, unknown>[];

    return rows.some((row) =>
      parseJsonArray<StreamEntryId>(
        String(row.source_stream_entry_ids ?? "[]"),
        "source_stream_entry_ids",
        COMMITMENT_JSON_ARRAY_CODEC,
      ).includes(entryId),
    );
  }

  /**
   * @internal Prefer IdentityService.updateCommitment() so episode-backed
   * established records cannot bypass review gating.
   */
  update(
    id: CommitmentId,
    patch: CommitmentPatch,
    provenance: CommitmentRecord["provenance"],
    options: {
      reason?: string | null;
      reviewItemId?: number | null;
      overwriteWithoutReview?: boolean;
      expectedVersion?: number;
    } = {},
  ): CommitmentRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    const parsedPatch = commitmentPatchSchema.parse(patch);
    const parsedProvenance = provenanceSchema.parse(provenance);
    const expectedVersion = expectedRecordVersion(current, options);
    const next = commitmentSchema.parse({
      ...current,
      ...parsedPatch,
      record_version: nextRecordVersion(expectedVersion),
      provenance: parsedPatch.provenance ?? current.provenance,
      revoke_provenance: parsedPatch.revoke_provenance ?? current.revoke_provenance,
    });
    const storedProvenance = toStoredProvenance(next.provenance);
    const storedRevokeProvenance =
      next.revoke_provenance === null ? null : toStoredProvenance(next.revoke_provenance);

    return runIdentityWrite(this.identityEventRepository, () => {
      const result = this.db
        .prepare(
          `
            UPDATE commitments
            SET type = ?, directive_family = ?, closure_pressure_relevance = ?, directive = ?,
                priority = ?, made_to_entity = ?, restricted_audience = ?, about_entity = ?,
                committed_by_entity_id = ?, source_episode_ids = ?, provenance_kind = ?,
                provenance_episode_ids = ?,
                provenance_process = ?, source_stream_entry_ids = ?, expires_at = ?, expired_at = ?, revoked_at = ?, revoked_reason = ?,
                revoke_provenance_kind = ?, revoke_provenance_episode_ids = ?, revoke_provenance_process = ?,
                superseded_by = ?, last_reinforced_at = ?, record_version = record_version + 1
            WHERE id = ? AND record_version = ?
          `,
        )
        .run(
          next.type,
          next.directive_family,
          next.closure_pressure_relevance,
          next.directive,
          next.priority,
          next.made_to_entity,
          next.restricted_audience,
          next.about_entity,
          next.committed_by_entity_id,
          serializeJsonValue(
            isEpisodeProvenance(next.provenance) ? next.provenance.episode_ids : [],
          ),
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          next.source_stream_entry_ids === undefined
            ? null
            : serializeJsonValue(next.source_stream_entry_ids),
          next.expires_at,
          next.expired_at,
          next.revoked_at,
          next.revoked_reason,
          storedRevokeProvenance?.provenance_kind ?? null,
          storedRevokeProvenance?.provenance_episode_ids ?? null,
          storedRevokeProvenance?.provenance_process ?? null,
          next.superseded_by,
          next.last_reinforced_at,
          id,
          expectedVersion,
        );
      assertIdentityCasUpdated({
        result,
        recordType: "commitment",
        recordId: id,
        expectedVersion,
      });

      this.identityEventRepository?.record({
        record_type: "commitment",
        record_id: id,
        action:
          options.reviewItemId === null || options.reviewItemId === undefined
            ? "update"
            : "correction_apply",
        old_value: current,
        new_value: next,
        reason: options.reason ?? null,
        provenance: parsedProvenance,
        review_item_id: options.reviewItemId ?? null,
        overwrite_without_review: options.overwriteWithoutReview === true,
      });

      return next;
    });
  }

  getApplicable(options: CommitmentApplicableOptions = {}): CommitmentRecord[] {
    const nowMs = options.nowMs ?? this.clock.now();

    return this.list({
      activeOnly: true,
      audience: options.audience ?? null,
      aboutEntity: options.aboutEntity ?? null,
      nowMs,
    });
  }
}
