import { z } from "zod";

import { SystemClock, type Clock } from "../../util/clock.js";
import { CommitmentError, ProvenanceError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createCommitmentId,
  createEntityId,
  parseCommitmentId,
  parseEntityId,
  type CommitmentId,
  type SharedStateEntryId,
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
  COMMITMENT_KINDS,
  COMMITMENT_ENFORCEMENT_CLASSES,
  borgRoleSchema,
  commitmentCriticalDomainSchema,
  commitmentEnforcementClassSchema,
  commitmentKindSchema,
  commitmentPatchSchema,
  commitmentSchema,
  defaultCommitmentCriticalDomain,
  defaultCommitmentEnforcementClass,
  entityKindSchema,
  entityRecordSchema,
  nameProvenanceSchema,
  normalizeDirectiveFamily,
  type CommitmentApplicableOptions,
  type CommitmentCriticalDomain,
  type CommitmentEnforcementClass,
  type CommitmentKind,
  type CommitmentListOptions,
  type CommitmentPatch,
  type CommitmentRecord,
  type CommitmentType,
  type BorgRole,
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
  creator_directive: 2,
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
    borg_role:
      row.borg_role === null || row.borg_role === undefined
        ? null
        : borgRoleSchema.parse(row.borg_role),
    name_provenance: nameProvenanceSchema.parse(row.name_provenance ?? "unknown"),
    created_at: Number(row.created_at),
  });
}

function mapCommitmentRow(row: Record<string, unknown>): CommitmentRecord {
  const kind = commitmentKindSchema.parse(row.kind ?? "assistant_commitment");
  const enforcementClass =
    row.enforcement_class === null || row.enforcement_class === undefined
      ? defaultCommitmentEnforcementClass(kind)
      : commitmentEnforcementClassSchema.parse(row.enforcement_class);
  const criticalDomain =
    row.critical_domain === null || row.critical_domain === undefined
      ? defaultCommitmentCriticalDomain(kind, enforcementClass)
      : commitmentCriticalDomainSchema.parse(row.critical_domain);
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
    kind,
    enforcement_class: enforcementClass,
    critical_domain: enforcementClass === "critical" ? criticalDomain : null,
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
    canonicalized_by_artifact_entry_id:
      row.canonicalized_by_artifact_entry_id === null ||
      row.canonicalized_by_artifact_entry_id === undefined
        ? null
        : String(row.canonicalized_by_artifact_entry_id),
    last_reinforced_at:
      row.last_reinforced_at === null || row.last_reinforced_at === undefined
        ? Number(row.created_at)
        : Number(row.last_reinforced_at),
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
  borg_role?: BorgRole | null;
  provenance?: NameProvenance;
  createdAt?: number;
};

export type EntityResolveOptions = {
  provenance?: NameProvenance;
  kind?: EntityKind;
};

export type EntityEnsureSelfOptions = {
  provenance?: NameProvenance;
};

export type CommitmentRevokeOptions = IdentityCasOptions & {
  canonicalizedByArtifactEntryId?: SharedStateEntryId | null;
};

export type CommitmentReconciliationMergedFields = Pick<
  CommitmentRecord,
  | "enforcement_class"
  | "critical_domain"
  | "priority"
  | "closure_pressure_relevance"
  | "source_stream_entry_ids"
  | "last_reinforced_at"
>;

export type CommitmentReconciliationSupersedeInput = {
  survivorId: CommitmentId;
  expectedSurvivorVersion: number;
  superseded: readonly {
    id: CommitmentId;
    expectedVersion: number;
  }[];
  mergedFields: CommitmentReconciliationMergedFields;
  provenance?: CommitmentRecord["provenance"];
  timestamp?: number;
};

export type CommitmentReconciliationSupersedeResult = {
  survivor: {
    id: CommitmentId;
    record_version: number;
  };
  survivor_before: CommitmentReconciliationMergedFields;
  superseded: Array<{
    id: CommitmentId;
    record_version: number;
  }>;
};

class CommitmentReconciliationSupersedeAbort extends Error {}

const commitmentReconciliationMergedFieldsSchema = commitmentSchema
  .pick({
    enforcement_class: true,
    critical_domain: true,
    priority: true,
    closure_pressure_relevance: true,
    source_stream_entry_ids: true,
    last_reinforced_at: true,
  })
  .strict();

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
                SELECT id, canonical_name, aliases, kind, borg_role, name_provenance, created_at
                FROM entities
                ORDER BY created_at ASC
              `,
            )
            .all() as Record<string, unknown>[])
        : (this.db
            .prepare(
              `
                SELECT id, canonical_name, aliases, kind, borg_role, name_provenance, created_at
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
      borg_role: input.borg_role ?? null,
      name_provenance: provenance,
      created_at: input.createdAt ?? this.clock.now(),
    });

    const insert = this.db.prepare(
      `
        INSERT INTO entities (
          id, canonical_name, aliases, kind, borg_role, name_provenance, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
      `,
    );

    const insertEntity = () => {
      insert.run(
        entity.id,
        entity.canonical_name,
        serializeJsonValue(entity.aliases),
        entity.kind,
        entity.borg_role,
        entity.name_provenance,
        entity.created_at,
      );
    };

    if (entity.borg_role === "creator") {
      this.db.transaction(() => {
        this.db.prepare("UPDATE entities SET borg_role = NULL WHERE borg_role = 'creator'").run();
        insertEntity();
      })();
    } else {
      insertEntity();
    }

    return entity;
  }

  findAllByName(name: string, options: EntityListOptions = {}): EntityId[] {
    const normalized = normalizeName(name);

    if (normalized.length === 0) {
      return [];
    }

    const matches: EntityId[] = [];

    for (const entity of this.listEntities(options)) {
      const names = [entity.canonical_name, ...entity.aliases].map((value) => normalizeName(value));

      if (names.includes(normalized)) {
        matches.push(entity.id);
      }
    }

    return matches;
  }

  findByName(name: string, options: EntityListOptions = {}): EntityId | null {
    return this.findAllByName(name, options)[0] ?? null;
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

  getCreator(): EntityRecord | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM entities
          WHERE borg_role = 'creator'
          ORDER BY created_at ASC
          LIMIT 1
        `,
      )
      .get() as Record<string, unknown> | undefined;

    return row === undefined ? null : mapEntityRow(row);
  }

  getSelf(): EntityRecord | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM entities
          WHERE kind = 'self'
          ORDER BY created_at ASC
          LIMIT 1
        `,
      )
      .get() as Record<string, unknown> | undefined;

    return row === undefined ? null : mapEntityRow(row);
  }

  ensureSelf(canonicalName: string, options: EntityEnsureSelfOptions = {}): EntityRecord {
    const name = canonicalName.trim();

    if (name.length === 0) {
      throw new CommitmentError("Entity name is required", {
        code: "ENTITY_NAME_REQUIRED",
      });
    }

    const provenance = options.provenance ?? "config_default_user";
    const ensure = this.db.transaction(() => {
      const current = this.getSelf();

      if (current === null) {
        return this.add({
          canonicalName: name,
          aliases: ["self"],
          kind: "self",
          provenance,
        });
      }

      const canonical = normalizeName(name);
      const aliases: string[] = [];
      const seen = new Set<string>();
      for (const candidate of [...current.aliases, current.canonical_name, "self"]) {
        const normalized = normalizeName(candidate);
        if (
          normalized.length === 0 ||
          (normalized === canonical && normalized !== "self") ||
          seen.has(normalized)
        ) {
          continue;
        }
        seen.add(normalized);
        aliases.push(normalized === "self" ? "self" : candidate.trim());
      }

      const next = entityRecordSchema.parse({
        ...current,
        canonical_name: name,
        aliases,
        name_provenance:
          normalizeName(current.canonical_name) === canonical
            ? strongerNameProvenance(current.name_provenance, provenance)
            : provenance,
      });

      this.db
        .prepare(
          "UPDATE entities SET canonical_name = ?, aliases = ?, name_provenance = ? WHERE id = ?",
        )
        .run(next.canonical_name, serializeJsonValue(next.aliases), next.name_provenance, next.id);

      return next;
    });

    return ensure.immediate();
  }

  setBorgRole(id: EntityId, role: BorgRole | null): EntityRecord | null {
    const parsedId = parseEntityId(id);
    const parsedRole = role === null ? null : borgRoleSchema.parse(role);

    const update = this.db.transaction(() => {
      const current = this.get(parsedId);

      if (current === null) {
        return null;
      }

      if (parsedRole === "creator") {
        this.db
          .prepare("UPDATE entities SET borg_role = NULL WHERE borg_role = 'creator' AND id != ?")
          .run(parsedId);
      }

      this.db.prepare("UPDATE entities SET borg_role = ? WHERE id = ?").run(parsedRole, parsedId);

      return this.get(parsedId);
    });

    return update();
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

export type CommitmentExpiringReadOnlyOptions = {
  nowMs: number;
  limit: number;
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
          candidate.kind === record.kind &&
          candidate.type === record.type &&
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
      enforcement_class: incoming.enforcement_class,
      critical_domain: incoming.critical_domain,
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
          SET enforcement_class = ?, critical_domain = ?,
              priority = ?, closure_pressure_relevance = ?, source_stream_entry_ids = ?,
              last_reinforced_at = ?, record_version = record_version + 1
          WHERE id = ? AND record_version = ?
        `,
      )
      .run(
        next.enforcement_class,
        next.critical_domain,
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

  private isActiveCommitment(record: CommitmentRecord, nowMs: number): boolean {
    return (
      record.revoked_at === null &&
      record.superseded_by === null &&
      record.expired_at === null &&
      (record.expires_at === null || record.expires_at > nowMs)
    );
  }

  private reconciliationFieldSnapshot(
    record: CommitmentRecord,
  ): CommitmentReconciliationMergedFields {
    return {
      enforcement_class: record.enforcement_class,
      critical_domain: record.critical_domain,
      priority: record.priority,
      closure_pressure_relevance: record.closure_pressure_relevance,
      ...(record.source_stream_entry_ids === undefined
        ? {}
        : { source_stream_entry_ids: [...record.source_stream_entry_ids] }),
      last_reinforced_at: record.last_reinforced_at,
    };
  }

  reconcileSupersedeOntoSurvivor(
    input: CommitmentReconciliationSupersedeInput,
  ): CommitmentReconciliationSupersedeResult | null {
    const survivorId = parseCommitmentId(input.survivorId);
    const expectedSurvivorVersion = z
      .number()
      .int()
      .positive()
      .parse(input.expectedSurvivorVersion);
    const superseded = z
      .array(
        z
          .object({
            id: commitmentSchema.shape.id,
            expectedVersion: z.number().int().positive(),
          })
          .strict(),
      )
      .min(1)
      .parse(input.superseded);
    const mergedFields = commitmentReconciliationMergedFieldsSchema.parse(input.mergedFields);
    const provenance = provenanceSchema.parse(
      input.provenance ?? { kind: "offline", process: "commitment-reconciler" },
    );
    const timestamp = input.timestamp ?? this.clock.now();
    const supersededIds = new Set<CommitmentId>();

    for (const item of superseded) {
      if (item.id === survivorId || supersededIds.has(item.id)) {
        return null;
      }

      supersededIds.add(item.id);
    }

    this.materializeExpiredCommitments(timestamp);

    const run = this.db.transaction((): CommitmentReconciliationSupersedeResult | null => {
      const survivor = this.get(survivorId);

      if (
        survivor === null ||
        !this.isActiveCommitment(survivor, timestamp) ||
        survivor.record_version !== expectedSurvivorVersion
      ) {
        return null;
      }

      const currentSuperseded: CommitmentRecord[] = [];

      for (const item of superseded) {
        const current = this.get(item.id);

        if (
          current === null ||
          !this.isActiveCommitment(current, timestamp) ||
          current.record_version !== item.expectedVersion
        ) {
          return null;
        }

        currentSuperseded.push(current);
      }

      const survivorBefore = this.reconciliationFieldSnapshot(survivor);
      const nextSurvivor = commitmentSchema.parse({
        ...survivor,
        ...mergedFields,
        critical_domain:
          mergedFields.enforcement_class === "critical" ? mergedFields.critical_domain : null,
        record_version: nextRecordVersion(expectedSurvivorVersion),
      });

      const survivorResult = this.db
        .prepare(
          `
            UPDATE commitments
            SET enforcement_class = ?, critical_domain = ?,
                priority = ?, closure_pressure_relevance = ?, source_stream_entry_ids = ?,
                last_reinforced_at = ?, record_version = record_version + 1
            WHERE id = ?
              AND revoked_at IS NULL
              AND superseded_by IS NULL
              AND expired_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
              AND record_version = ?
          `,
        )
        .run(
          nextSurvivor.enforcement_class,
          nextSurvivor.critical_domain,
          nextSurvivor.priority,
          nextSurvivor.closure_pressure_relevance,
          nextSurvivor.source_stream_entry_ids === undefined
            ? null
            : serializeJsonValue(nextSurvivor.source_stream_entry_ids),
          nextSurvivor.last_reinforced_at,
          survivor.id,
          timestamp,
          expectedSurvivorVersion,
        );

      if (survivorResult.changes === 0) {
        throw new CommitmentReconciliationSupersedeAbort();
      }

      this.identityEventRepository?.record({
        record_type: "commitment",
        record_id: survivor.id,
        action: "update",
        old_value: survivor,
        new_value: nextSurvivor,
        reason: "commitment_reconciliation_survivor",
        provenance,
        ts: timestamp,
      });

      const supersededRows: CommitmentReconciliationSupersedeResult["superseded"] = [];

      for (const current of currentSuperseded) {
        const currentVersion = expectedRecordVersion(current);
        const result = this.db
          .prepare(
            `
              UPDATE commitments
              SET superseded_by = ?, record_version = record_version + 1
              WHERE id = ?
                AND revoked_at IS NULL
                AND superseded_by IS NULL
                AND expired_at IS NULL
                AND (expires_at IS NULL OR expires_at > ?)
                AND record_version = ?
            `,
          )
          .run(survivor.id, current.id, timestamp, currentVersion);

        if (result.changes === 0) {
          throw new CommitmentReconciliationSupersedeAbort();
        }

        const nextSuperseded = commitmentSchema.parse({
          ...current,
          record_version: nextRecordVersion(currentVersion),
          superseded_by: survivor.id,
        });

        this.identityEventRepository?.record({
          record_type: "commitment",
          record_id: current.id,
          action: "update",
          old_value: current,
          new_value: nextSuperseded,
          reason: "commitment_reconciliation_duplicate",
          provenance,
          ts: timestamp,
        });

        supersededRows.push({
          id: current.id,
          record_version: nextRecordVersion(currentVersion),
        });
      }

      return {
        survivor: {
          id: survivor.id,
          record_version: nextRecordVersion(expectedSurvivorVersion),
        },
        survivor_before: survivorBefore,
        superseded: supersededRows,
      };
    });

    try {
      return run();
    } catch (error) {
      if (error instanceof CommitmentReconciliationSupersedeAbort) {
        return null;
      }

      throw error;
    }
  }

  add(input: {
    id?: CommitmentId;
    type: CommitmentType;
    kind?: CommitmentKind;
    enforcementClass?: CommitmentEnforcementClass;
    criticalDomain?: CommitmentCriticalDomain | null;
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
    skipDirectiveFamilyMerge?: boolean;
  }): CommitmentRecord {
    if (input.provenance === undefined) {
      throw new ProvenanceError("Commitment requires provenance", {
        code: "PROVENANCE_REQUIRED",
      });
    }

    const createdAt = input.createdAt ?? this.clock.now();
    const expiresAt = input.expiresAt ?? null;
    const kind = input.kind ?? "assistant_commitment";
    const enforcementClass = input.enforcementClass ?? defaultCommitmentEnforcementClass(kind);
    const criticalDomain =
      enforcementClass === "critical"
        ? (input.criticalDomain ?? defaultCommitmentCriticalDomain(kind, enforcementClass))
        : null;

    const record = commitmentSchema.parse({
      id: input.id ?? createCommitmentId(),
      record_version: 1,
      type: input.type,
      kind,
      enforcement_class: enforcementClass,
      critical_domain: criticalDomain,
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
      canonicalized_by_artifact_entry_id: null,
      last_reinforced_at: createdAt,
    });
    const storedProvenance = toStoredProvenance(record.provenance);
    const familyMatches =
      input.skipDirectiveFamilyMerge === true
        ? []
        : this.findActiveDirectiveFamilyMatches(record, createdAt);

    return runIdentityWrite(this.identityEventRepository, () => {
      if (familyMatches.length > 0) {
        return this.mergeDirectiveFamilyMatch(record, familyMatches);
      }

      this.db
        .prepare(
          `
            INSERT INTO commitments (
              id, type, kind, enforcement_class, critical_domain,
              directive_family, closure_pressure_relevance, directive, priority,
              made_to_entity, restricted_audience, about_entity, committed_by_entity_id,
              source_episode_ids, provenance_kind, provenance_episode_ids, provenance_process,
              source_stream_entry_ids, created_at, expires_at, expired_at, revoked_at, revoked_reason,
              revoke_provenance_kind, revoke_provenance_episode_ids, revoke_provenance_process,
              superseded_by, canonicalized_by_artifact_entry_id, last_reinforced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          record.id,
          record.type,
          record.kind,
          record.enforcement_class,
          record.critical_domain,
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
          record.canonicalized_by_artifact_entry_id ?? null,
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

    // audience: undefined = all, null = global-scoped only, id = global + that audience.
    // aboutEntity: undefined or null = ignored, id = matches that entity or unscoped.

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

  listFutureExpiringReadOnly(options: CommitmentExpiringReadOnlyOptions): CommitmentRecord[] {
    const limit = Number.isFinite(options.limit) ? Math.max(0, Math.floor(options.limit)) : 0;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM commitments
          WHERE revoked_at IS NULL
            AND superseded_by IS NULL
            AND expired_at IS NULL
            AND expires_at IS NOT NULL
            AND expires_at > ?
          ORDER BY expires_at ASC, priority DESC, created_at ASC, id ASC
          LIMIT ?
        `,
      )
      .all(options.nowMs, limit) as Record<string, unknown>[];

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

  countActiveByKind(nowMs = this.clock.now()): Record<CommitmentKind, number> {
    const counts = Object.fromEntries(COMMITMENT_KINDS.map((kind) => [kind, 0])) as Record<
      CommitmentKind,
      number
    >;
    const rows = this.db
      .prepare(
        `
          SELECT kind, COUNT(*) AS count
          FROM commitments
          WHERE revoked_at IS NULL
            AND superseded_by IS NULL
            AND expired_at IS NULL
            AND (expires_at IS NULL OR expires_at > ?)
          GROUP BY kind
        `,
      )
      .all(nowMs) as Array<{ kind: CommitmentKind; count: number }>;

    for (const row of rows) {
      counts[row.kind] = Number(row.count);
    }

    return counts;
  }

  countActiveByEnforcementClass(
    nowMs = this.clock.now(),
  ): Record<CommitmentEnforcementClass, number> {
    const counts = Object.fromEntries(
      COMMITMENT_ENFORCEMENT_CLASSES.map((enforcementClass) => [enforcementClass, 0]),
    ) as Record<CommitmentEnforcementClass, number>;
    const rows = this.db
      .prepare(
        `
          SELECT
            CASE
              WHEN enforcement_class IN ('critical', 'advisory') THEN enforcement_class
              WHEN kind IN ('boundary', 'audience_rule') THEN 'critical'
              ELSE 'advisory'
            END AS enforcement_class,
            COUNT(*) AS count
          FROM commitments
          WHERE revoked_at IS NULL
            AND superseded_by IS NULL
            AND expired_at IS NULL
            AND (expires_at IS NULL OR expires_at > ?)
          GROUP BY
            CASE
              WHEN enforcement_class IN ('critical', 'advisory') THEN enforcement_class
              WHEN kind IN ('boundary', 'audience_rule') THEN 'critical'
              ELSE 'advisory'
            END
        `,
      )
      .all(nowMs) as Array<{ enforcement_class: CommitmentEnforcementClass; count: number }>;

    for (const row of rows) {
      counts[row.enforcement_class] = Number(row.count);
    }

    return counts;
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

  countRevoked(): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM commitments
          WHERE revoked_at IS NOT NULL
        `,
      )
      .get() as { count: number } | undefined;

    return Number(row?.count ?? 0);
  }

  countExpired(nowMs = this.clock.now()): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM commitments
          WHERE expired_at IS NOT NULL
             OR (expires_at IS NOT NULL AND expires_at <= ?)
        `,
      )
      .get(nowMs) as { count: number } | undefined;

    return Number(row?.count ?? 0);
  }

  countCanonicalized(): number {
    const row = this.db
      .prepare(
        `
          SELECT COUNT(*) AS count
          FROM commitments
          WHERE canonicalized_by_artifact_entry_id IS NOT NULL
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
    options: CommitmentRevokeOptions = {},
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
                canonicalized_by_artifact_entry_id = ?,
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
          options.canonicalizedByArtifactEntryId === undefined
            ? (current.canonicalized_by_artifact_entry_id ?? null)
            : options.canonicalizedByArtifactEntryId,
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
        canonicalized_by_artifact_entry_id:
          options.canonicalizedByArtifactEntryId === undefined
            ? (current.canonicalized_by_artifact_entry_id ?? null)
            : options.canonicalizedByArtifactEntryId,
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

  restoreReconciledSurvivor(
    id: CommitmentId,
    expectedRecordVersion: number,
    fields: CommitmentReconciliationMergedFields,
  ): CommitmentRecord | null {
    const parsedId = parseCommitmentId(id);
    const parsedRecordVersion = z.number().int().positive().parse(expectedRecordVersion);
    const restoredFields = commitmentReconciliationMergedFieldsSchema.parse(fields);
    const current = this.get(parsedId);
    const timestamp = this.clock.now();

    if (
      current === null ||
      !this.isActiveCommitment(current, timestamp) ||
      current.record_version !== parsedRecordVersion
    ) {
      return null;
    }

    const next = commitmentSchema.parse({
      ...current,
      ...restoredFields,
      critical_domain:
        restoredFields.enforcement_class === "critical" ? restoredFields.critical_domain : null,
      record_version: nextRecordVersion(parsedRecordVersion),
    });
    const result = this.db
      .prepare(
        `
          UPDATE commitments
          SET enforcement_class = ?, critical_domain = ?,
              priority = ?, closure_pressure_relevance = ?, source_stream_entry_ids = ?,
              last_reinforced_at = ?, record_version = record_version + 1
          WHERE id = ?
            AND revoked_at IS NULL
            AND superseded_by IS NULL
            AND expired_at IS NULL
            AND (expires_at IS NULL OR expires_at > ?)
            AND record_version = ?
        `,
      )
      .run(
        next.enforcement_class,
        next.critical_domain,
        next.priority,
        next.closure_pressure_relevance,
        next.source_stream_entry_ids === undefined
          ? null
          : serializeJsonValue(next.source_stream_entry_ids),
        next.last_reinforced_at,
        parsedId,
        timestamp,
        parsedRecordVersion,
      );

    if (result.changes === 0) {
      return null;
    }

    this.identityEventRepository?.record({
      record_type: "commitment",
      record_id: parsedId,
      action: "update",
      old_value: current,
      new_value: next,
      reason: "commitment_reconciliation_reversal_survivor",
      provenance: {
        kind: "offline",
        process: "commitment-reconciler",
      },
      ts: timestamp,
    });

    return next;
  }

  reverseSupersede(
    id: CommitmentId,
    expectedSupersededById: CommitmentId,
    expectedRecordVersion: number,
  ): CommitmentRecord | null {
    const parsedId = parseCommitmentId(id);
    const parsedSupersededById = parseCommitmentId(expectedSupersededById);
    const parsedRecordVersion = z.number().int().positive().parse(expectedRecordVersion);
    const current = this.get(parsedId);

    if (
      current === null ||
      current.superseded_by !== parsedSupersededById ||
      current.revoked_at !== null ||
      current.expired_at !== null ||
      current.record_version !== parsedRecordVersion
    ) {
      return null;
    }

    const timestamp = this.clock.now();

    if (current.expires_at !== null && current.expires_at <= timestamp) {
      return null;
    }

    const result = this.db
      .prepare(
        `
          UPDATE commitments
          SET superseded_by = NULL, record_version = record_version + 1
          WHERE id = ?
            AND superseded_by = ?
            AND revoked_at IS NULL
            AND expired_at IS NULL
            AND (expires_at IS NULL OR expires_at > ?)
            AND record_version = ?
        `,
      )
      .run(parsedId, parsedSupersededById, timestamp, parsedRecordVersion);

    if (result.changes === 0) {
      return null;
    }

    const next = commitmentSchema.parse({
      ...current,
      record_version: nextRecordVersion(parsedRecordVersion),
      superseded_by: null,
    });

    this.identityEventRepository?.record({
      record_type: "commitment",
      record_id: parsedId,
      action: "update",
      old_value: current,
      new_value: next,
      reason: "commitment_reconciliation_reversal_duplicate",
      provenance: {
        kind: "offline",
        process: "commitment-reconciler",
      },
      ts: timestamp,
    });

    return next;
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
            SET type = ?, kind = ?, enforcement_class = ?, critical_domain = ?,
                directive_family = ?, closure_pressure_relevance = ?, directive = ?,
                priority = ?, made_to_entity = ?, restricted_audience = ?, about_entity = ?,
                committed_by_entity_id = ?, source_episode_ids = ?, provenance_kind = ?,
                provenance_episode_ids = ?,
                provenance_process = ?, source_stream_entry_ids = ?, expires_at = ?, expired_at = ?, revoked_at = ?, revoked_reason = ?,
                revoke_provenance_kind = ?, revoke_provenance_episode_ids = ?, revoke_provenance_process = ?,
                superseded_by = ?, canonicalized_by_artifact_entry_id = ?,
                last_reinforced_at = ?, record_version = record_version + 1
            WHERE id = ? AND record_version = ?
          `,
        )
        .run(
          next.type,
          next.kind,
          next.enforcement_class,
          next.critical_domain,
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
          next.canonicalized_by_artifact_entry_id ?? null,
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
