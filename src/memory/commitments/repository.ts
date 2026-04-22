import { SystemClock, type Clock } from "../../util/clock.js";
import { CommitmentError, ProvenanceError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createCommitmentId,
  createEntityId,
  parseCommitmentId,
  parseEntityId,
  parseEpisodeId,
  type EpisodeId,
  type CommitmentId,
  type EntityId,
} from "../../util/ids.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import {
  parseStoredProvenance,
  provenanceSchema,
  toStoredProvenance,
} from "../common/provenance.js";
import {
  commitmentPatchSchema,
  commitmentSchema,
  entityRecordSchema,
  type CommitmentApplicableOptions,
  type CommitmentListOptions,
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

function parseJsonArray<T>(value: string, label: string): T[] {
  try {
    const parsed = JSON.parse(value) as unknown;

    if (!Array.isArray(parsed)) {
      throw new TypeError(`${label} must be an array`);
    }

    return parsed as T[];
  } catch (error) {
    throw new CommitmentError(`Failed to parse ${label}`, {
      cause: error,
      code: "COMMITMENT_ROW_INVALID",
    });
  }
}

function mapEntityRow(row: Record<string, unknown>): EntityRecord {
  return entityRecordSchema.parse({
    id: row.id,
    canonical_name: row.canonical_name,
    aliases: uniqueStrings(parseJsonArray<string>(String(row.aliases ?? "[]"), "aliases")),
    created_at: Number(row.created_at),
  });
}

function mapCommitmentRow(row: Record<string, unknown>): CommitmentRecord {
  const parsed = commitmentSchema.safeParse({
    id: row.id,
    type: row.type,
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
    created_at: Number(row.created_at),
    expires_at:
      row.expires_at === null || row.expires_at === undefined ? null : Number(row.expires_at),
    revoked_at:
      row.revoked_at === null || row.revoked_at === undefined ? null : Number(row.revoked_at),
    superseded_by:
      row.superseded_by === null || row.superseded_by === undefined
        ? null
        : parseCommitmentId(String(row.superseded_by)),
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

  resolve(name: string): EntityId {
    const normalized = normalizeName(name);

    if (normalized.length === 0) {
      throw new CommitmentError("Entity name is required", {
        code: "ENTITY_NAME_REQUIRED",
      });
    }

    const rows = this.db
      .prepare(
        `
          SELECT id, canonical_name, aliases, created_at
          FROM entities
          ORDER BY created_at ASC
        `,
      )
      .all() as Record<string, unknown>[];

    for (const row of rows) {
      const entity = mapEntityRow(row);
      const names = [entity.canonical_name, ...entity.aliases].map((value) => normalizeName(value));

      if (names.includes(normalized)) {
        return entity.id;
      }
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
};

export class CommitmentRepository {
  private readonly clock: Clock;

  constructor(private readonly options: CommitmentRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  add(input: {
    id?: CommitmentId;
    type: CommitmentType;
    directive: string;
    priority: number;
    madeToEntity?: EntityId | null;
    restrictedAudience?: EntityId | null;
    aboutEntity?: EntityId | null;
    provenance: CommitmentRecord["provenance"];
    createdAt?: number;
    expiresAt?: number | null;
  }): CommitmentRecord {
    if (input.provenance === undefined) {
      throw new ProvenanceError("Commitment requires provenance", {
        code: "PROVENANCE_REQUIRED",
      });
    }

    const record = commitmentSchema.parse({
      id: input.id ?? createCommitmentId(),
      type: input.type,
      directive: input.directive,
      priority: input.priority,
      made_to_entity: input.madeToEntity ?? null,
      restricted_audience: input.restrictedAudience ?? null,
      about_entity: input.aboutEntity ?? null,
      provenance: provenanceSchema.parse(input.provenance),
      created_at: input.createdAt ?? this.clock.now(),
      expires_at: input.expiresAt ?? null,
      revoked_at: null,
      superseded_by: null,
    });
    const storedProvenance = toStoredProvenance(record.provenance);

    this.db
      .prepare(
        `
          INSERT INTO commitments (
            id, type, directive, priority, made_to_entity, restricted_audience, about_entity,
            source_episode_ids, provenance_kind, provenance_episode_ids, provenance_process,
            created_at, expires_at, revoked_at, superseded_by
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        record.id,
        record.type,
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
        record.created_at,
        record.expires_at,
        record.revoked_at,
        record.superseded_by,
      );

    return record;
  }

  get(id: CommitmentId): CommitmentRecord | null {
    const row = this.db.prepare("SELECT * FROM commitments WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : mapCommitmentRow(row);
  }

  list(options: CommitmentListOptions = {}): CommitmentRecord[] {
    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.activeOnly === true) {
      filters.push("revoked_at IS NULL");
      filters.push("superseded_by IS NULL");
    }

    if (options.audience !== undefined && options.audience !== null) {
      filters.push("(restricted_audience IS NULL OR restricted_audience = ?)");
      values.push(options.audience);
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

  revoke(id: CommitmentId, timestamp = this.clock.now()): CommitmentRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    this.db.prepare("UPDATE commitments SET revoked_at = ? WHERE id = ?").run(timestamp, id);
    return {
      ...current,
      revoked_at: timestamp,
    };
  }

  supersede(id: CommitmentId, nextId: CommitmentId): CommitmentRecord | null {
    const current = this.get(id);

    if (current === null) {
      return null;
    }

    this.db.prepare("UPDATE commitments SET superseded_by = ? WHERE id = ?").run(nextId, id);
    return {
      ...current,
      superseded_by: nextId,
    };
  }

  getApplicable(options: CommitmentApplicableOptions = {}): CommitmentRecord[] {
    const nowMs = options.nowMs ?? this.clock.now();

    return this.list({
      activeOnly: true,
      audience: options.audience ?? null,
      aboutEntity: options.aboutEntity ?? null,
    }).filter((record) => record.expires_at === null || record.expires_at > nowMs);
  }
}
