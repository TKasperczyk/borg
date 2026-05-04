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
import { IdentityEventRepository } from "../identity/repository.js";
import { runIdentityWrite } from "../self/shared/identity-events.js";
import {
  commitmentPatchSchema,
  commitmentSchema,
  entityRecordSchema,
  normalizeDirectiveFamily,
  type CommitmentApplicableOptions,
  type CommitmentListOptions,
  type CommitmentPatch,
  type CommitmentRecord,
  type CommitmentType,
  type EntityRecord,
} from "./types.js";

function normalizeName(value: string): string {
  return value.trim().toLowerCase();
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
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
    type: row.type,
    directive_family: row.directive_family,
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

export class EntityRepository {
  private readonly clock: Clock;

  constructor(private readonly options: EntityRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private listEntities(): EntityRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT id, canonical_name, aliases, created_at
          FROM entities
          ORDER BY created_at ASC
        `,
      )
      .all() as Record<string, unknown>[];

    return rows.map((row) => mapEntityRow(row));
  }

  findByName(name: string): EntityId | null {
    const normalized = normalizeName(name);

    if (normalized.length === 0) {
      return null;
    }

    for (const entity of this.listEntities()) {
      const names = [entity.canonical_name, ...entity.aliases].map((value) => normalizeName(value));

      if (names.includes(normalized)) {
        return entity.id;
      }
    }

    return null;
  }

  resolve(name: string): EntityId {
    const normalized = normalizeName(name);

    if (normalized.length === 0) {
      throw new CommitmentError("Entity name is required", {
        code: "ENTITY_NAME_REQUIRED",
      });
    }

    const existing = this.findByName(name);

    if (existing !== null) {
      return existing;
    }

    const entity = entityRecordSchema.parse({
      id: createEntityId(),
      canonical_name: name.trim(),
      aliases: [],
      created_at: this.clock.now(),
    });

    this.db
      .prepare(
        `
          INSERT INTO entities (id, canonical_name, aliases, created_at)
          VALUES (?, ?, ?, ?)
        `,
      )
      .run(entity.id, entity.canonical_name, serializeJsonValue(entity.aliases), entity.created_at);

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

    const update = this.db.prepare("UPDATE commitments SET expired_at = ? WHERE id = ?");

    runIdentityWrite(this.identityEventRepository, () => {
      for (const row of rows) {
        const current = mapCommitmentRow(row);
        const expiredAt = current.expires_at ?? nowMs;
        update.run(expiredAt, current.id);
        const next: CommitmentRecord = {
          ...current,
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
          candidate.about_entity === record.about_entity,
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
    const next = commitmentSchema.parse({
      ...kept,
      priority: Math.max(kept.priority, incoming.priority),
      source_stream_entry_ids:
        sourceStreamEntryIds.length === 0 ? undefined : sourceStreamEntryIds,
      last_reinforced_at: Math.max(kept.last_reinforced_at, incoming.last_reinforced_at),
    });

    this.db
      .prepare(
        `
          UPDATE commitments
          SET priority = ?, source_stream_entry_ids = ?, last_reinforced_at = ?
          WHERE id = ?
        `,
      )
      .run(
        next.priority,
        next.source_stream_entry_ids === undefined
          ? null
          : serializeJsonValue(next.source_stream_entry_ids),
        next.last_reinforced_at,
        kept.id,
      );

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

    const supersede = this.db.prepare("UPDATE commitments SET superseded_by = ? WHERE id = ?");

    for (const current of superseded) {
      supersede.run(kept.id, current.id);
      this.identityEventRepository?.record({
        record_type: "commitment",
        record_id: current.id,
        action: "update",
        old_value: current,
        new_value: {
          ...current,
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
    madeToEntity?: EntityId | null;
    restrictedAudience?: EntityId | null;
    aboutEntity?: EntityId | null;
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
      type: input.type,
      directive_family: normalizeDirectiveFamily(input.directiveFamily),
      directive: input.directive,
      priority: input.priority,
      made_to_entity: input.madeToEntity ?? null,
      restricted_audience: input.restrictedAudience ?? null,
      about_entity: input.aboutEntity ?? null,
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
              id, type, directive_family, directive, priority, made_to_entity, restricted_audience, about_entity,
              source_episode_ids, provenance_kind, provenance_episode_ids, provenance_process,
              source_stream_entry_ids, created_at, expires_at, expired_at, revoked_at, revoked_reason,
              revoke_provenance_kind, revoke_provenance_episode_ids, revoke_provenance_process,
              superseded_by, last_reinforced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(
          record.id,
          record.type,
          record.directive_family,
          record.directive,
          record.priority,
          record.made_to_entity,
          record.restricted_audience,
          record.about_entity,
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

  revoke(
    id: CommitmentId,
    reason: string,
    provenance: CommitmentRecord["provenance"],
    timestamp = this.clock.now(),
  ): CommitmentRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    const parsedReason = reason.trim();
    const parsedProvenance = provenanceSchema.parse(provenance);
    const storedProvenance = toStoredProvenance(parsedProvenance);

    return runIdentityWrite(this.identityEventRepository, () => {
      this.db
        .prepare(
          `
            UPDATE commitments
            SET revoked_at = ?, revoked_reason = ?, revoke_provenance_kind = ?,
                revoke_provenance_episode_ids = ?, revoke_provenance_process = ?
            WHERE id = ?
          `,
        )
        .run(
          timestamp,
          parsedReason,
          storedProvenance.provenance_kind,
          storedProvenance.provenance_episode_ids,
          storedProvenance.provenance_process,
          id,
        );
      const next = {
        ...current,
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

  supersede(id: CommitmentId, nextId: CommitmentId): CommitmentRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    return runIdentityWrite(this.identityEventRepository, () => {
      this.db.prepare("UPDATE commitments SET superseded_by = ? WHERE id = ?").run(nextId, id);
      const next = {
        ...current,
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
    } = {},
  ): CommitmentRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    const parsedPatch = commitmentPatchSchema.parse(patch);
    const parsedProvenance = provenanceSchema.parse(provenance);
    const next = commitmentSchema.parse({
      ...current,
      ...parsedPatch,
      provenance: parsedPatch.provenance ?? current.provenance,
      revoke_provenance: parsedPatch.revoke_provenance ?? current.revoke_provenance,
    });
    const storedProvenance = toStoredProvenance(next.provenance);
    const storedRevokeProvenance =
      next.revoke_provenance === null ? null : toStoredProvenance(next.revoke_provenance);

    return runIdentityWrite(this.identityEventRepository, () => {
      this.db
        .prepare(
          `
            UPDATE commitments
            SET type = ?, directive_family = ?, directive = ?, priority = ?, made_to_entity = ?, restricted_audience = ?,
                about_entity = ?, source_episode_ids = ?, provenance_kind = ?, provenance_episode_ids = ?,
                provenance_process = ?, source_stream_entry_ids = ?, expires_at = ?, expired_at = ?, revoked_at = ?, revoked_reason = ?,
                revoke_provenance_kind = ?, revoke_provenance_episode_ids = ?, revoke_provenance_process = ?,
                superseded_by = ?, last_reinforced_at = ?
            WHERE id = ?
          `,
        )
        .run(
          next.type,
          next.directive_family,
          next.directive,
          next.priority,
          next.made_to_entity,
          next.restricted_audience,
          next.about_entity,
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
        );

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
