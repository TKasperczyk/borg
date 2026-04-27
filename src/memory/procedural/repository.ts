import {
  LanceDbTable,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import {
  parseJsonArray,
  quoteSqlString,
  type JsonArrayCodecOptions,
} from "../../storage/codecs.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import type { EmbeddingClient } from "../../embeddings/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createProceduralEvidenceId,
  createSkillId,
  parseEpisodeId,
  parseSkillId,
  type EntityId,
  type EpisodeId,
  type ProceduralEvidenceId,
  type SkillId,
} from "../../util/ids.js";

import { computeBetaStats, type BetaStats } from "./bayes.js";
import {
  proceduralEvidenceSchema,
  proceduralOutcomeClassificationSchema,
  skillInsertSchema,
  skillSchema,
  type PendingProceduralAttemptValue,
  type ProceduralEvidenceRecord,
  type ProceduralOutcomeClassification,
  type SkillRecord,
  type SkillSearchCandidate,
} from "./types.js";

type SkillSqlRow = {
  id: string;
  applies_when: string;
  approach: string;
  alpha: number;
  beta: number;
  attempts: number;
  successes: number;
  failures: number;
  alternatives: string;
  source_episode_ids: string;
  last_used: number | null;
  last_successful: number | null;
  created_at: number;
  updated_at: number;
};

const SKILL_JSON_ARRAY_CODEC = {
  errorCode: "SKILL_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse skill ${label}`,
} satisfies JsonArrayCodecOptions;

const PROCEDURAL_EVIDENCE_JSON_ARRAY_CODEC = {
  errorCode: "PROCEDURAL_EVIDENCE_ROW_INVALID",
  errorMessage: (label: string) => `Failed to parse procedural evidence ${label}`,
} satisfies JsonArrayCodecOptions;

function rowFromSkill(skill: SkillRecord): SkillSqlRow {
  return {
    id: skill.id,
    applies_when: skill.applies_when,
    approach: skill.approach,
    alpha: skill.alpha,
    beta: skill.beta,
    attempts: skill.attempts,
    successes: skill.successes,
    failures: skill.failures,
    alternatives: serializeJsonValue(skill.alternatives),
    source_episode_ids: serializeJsonValue(skill.source_episode_ids),
    last_used: skill.last_used,
    last_successful: skill.last_successful,
    created_at: skill.created_at,
    updated_at: skill.updated_at,
  };
}

function skillFromRow(row: Record<string, unknown>): SkillRecord {
  const parsed = skillSchema.safeParse({
    id: row.id,
    applies_when: row.applies_when,
    approach: row.approach,
    alpha: Number(row.alpha),
    beta: Number(row.beta),
    attempts: Number(row.attempts),
    successes: Number(row.successes),
    failures: Number(row.failures),
    alternatives: parseJsonArray<string>(
      String(row.alternatives ?? "[]"),
      "alternatives",
      SKILL_JSON_ARRAY_CODEC,
    ).map((value) => parseSkillId(value)),
    source_episode_ids: parseJsonArray<string>(
      String(row.source_episode_ids ?? "[]"),
      "source_episode_ids",
      SKILL_JSON_ARRAY_CODEC,
    ).map((value) => parseEpisodeId(value)),
    last_used: row.last_used === null || row.last_used === undefined ? null : Number(row.last_used),
    last_successful:
      row.last_successful === null || row.last_successful === undefined
        ? null
        : Number(row.last_successful),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Skill row failed validation", {
      cause: parsed.error,
      code: "SKILL_ROW_INVALID",
    });
  }

  return parsed.data;
}

function parsePendingAttemptSnapshot(value: string): PendingProceduralAttemptValue {
  try {
    const parsed = JSON.parse(value) as unknown;
    return proceduralEvidenceSchema.shape.pending_attempt_snapshot.parse(parsed);
  } catch (error) {
    throw new StorageError("Failed to parse procedural evidence pending attempt snapshot", {
      cause: error,
      code: "PROCEDURAL_EVIDENCE_ROW_INVALID",
    });
  }
}

function proceduralEvidenceFromRow(row: Record<string, unknown>): ProceduralEvidenceRecord {
  const parsed = proceduralEvidenceSchema.safeParse({
    id: row.id,
    pending_attempt_snapshot: parsePendingAttemptSnapshot(
      String(row.pending_attempt_snapshot ?? "{}"),
    ),
    classification: row.classification,
    evidence_text: row.evidence_text,
    grounded:
      row.grounded === null || row.grounded === undefined ? true : Number(row.grounded) !== 0,
    resolved_episode_ids: parseJsonArray<string>(
      String(row.resolved_episode_ids ?? "[]"),
      "resolved_episode_ids",
      PROCEDURAL_EVIDENCE_JSON_ARRAY_CODEC,
    ).map((value) => parseEpisodeId(value)),
    audience_entity_id:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : (String(row.audience_entity_id) as EntityId),
    consumed_at:
      row.consumed_at === null || row.consumed_at === undefined ? null : Number(row.consumed_at),
    created_at: Number(row.created_at),
  });

  if (!parsed.success) {
    throw new StorageError("Procedural evidence row failed validation", {
      cause: parsed.error,
      code: "PROCEDURAL_EVIDENCE_ROW_INVALID",
    });
  }

  return parsed.data;
}

export function createSkillsTableSchema(dimensions: number) {
  return schema([utf8Field("id"), utf8Field("applies_when"), vectorField("embedding", dimensions)]);
}

export type SkillRepositoryOptions = {
  table: LanceDbTable;
  db: SqliteDatabase;
  embeddingClient: EmbeddingClient;
  clock?: Clock;
};

export class SkillRepository {
  private readonly clock: Clock;

  constructor(private readonly options: SkillRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get table(): LanceDbTable {
    return this.options.table;
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private upsertSqlRow(skill: SkillRecord): void {
    const row = rowFromSkill(skill);

    this.db
      .prepare(
        `
          INSERT INTO skills (
            id, applies_when, approach, alpha, beta, attempts, successes, failures,
            alternatives, source_episode_ids, last_used, last_successful, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (id) DO UPDATE SET
            applies_when = excluded.applies_when,
            approach = excluded.approach,
            alpha = excluded.alpha,
            beta = excluded.beta,
            attempts = excluded.attempts,
            successes = excluded.successes,
            failures = excluded.failures,
            alternatives = excluded.alternatives,
            source_episode_ids = excluded.source_episode_ids,
            last_used = excluded.last_used,
            last_successful = excluded.last_successful,
            updated_at = excluded.updated_at
        `,
      )
      .run(
        row.id,
        row.applies_when,
        row.approach,
        row.alpha,
        row.beta,
        row.attempts,
        row.successes,
        row.failures,
        row.alternatives,
        row.source_episode_ids,
        row.last_used,
        row.last_successful,
        row.created_at,
        row.updated_at,
      );
  }

  async add(input: {
    id?: SkillId;
    applies_when: string;
    approach: string;
    alternatives?: readonly SkillId[];
    sourceEpisodes: readonly EpisodeId[];
    priorAlpha?: number;
    priorBeta?: number;
    createdAt?: number;
  }): Promise<SkillRecord> {
    const nowMs = input.createdAt ?? this.clock.now();
    const skill = skillInsertSchema.parse({
      id: input.id ?? createSkillId(),
      applies_when: input.applies_when,
      approach: input.approach,
      alpha: input.priorAlpha ?? 1,
      beta: input.priorBeta ?? 1,
      attempts: 0,
      successes: 0,
      failures: 0,
      alternatives: input.alternatives ?? [],
      source_episode_ids: input.sourceEpisodes,
      last_used: null,
      last_successful: null,
      created_at: nowMs,
      updated_at: nowMs,
    });
    const embedding = await this.options.embeddingClient.embed(skill.applies_when);

    try {
      await this.table.upsert(
        [
          {
            id: skill.id,
            applies_when: skill.applies_when,
            embedding: Array.from(embedding),
          },
        ],
        { on: "id" },
      );

      try {
        this.db.transaction(() => {
          this.upsertSqlRow(skill);
        })();
      } catch (error) {
        await this.table.remove(`id = ${quoteSqlString(skill.id)}`);
        throw error;
      }

      return skill;
    } catch (error) {
      throw new StorageError(`Failed to insert skill ${skill.id}`, {
        cause: error,
        code: "SKILL_INSERT_FAILED",
      });
    }
  }

  async replace(skill: SkillRecord): Promise<SkillRecord> {
    const parsed = skillSchema.parse(skill);
    const embedding = await this.options.embeddingClient.embed(parsed.applies_when);

    try {
      await this.table.upsert(
        [
          {
            id: parsed.id,
            applies_when: parsed.applies_when,
            embedding: Array.from(embedding),
          },
        ],
        { on: "id" },
      );

      this.db.transaction(() => {
        this.upsertSqlRow(parsed);
      })();

      return parsed;
    } catch (error) {
      throw new StorageError(`Failed to replace skill ${parsed.id}`, {
        cause: error,
        code: "SKILL_REPLACE_FAILED",
      });
    }
  }

  get(id: SkillId): SkillRecord | null {
    const row = this.db.prepare("SELECT * FROM skills WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : skillFromRow(row);
  }

  getMany(ids: readonly SkillId[]): Array<SkillRecord | null> {
    if (ids.length === 0) {
      return [];
    }

    const rows = this.db
      .prepare(`SELECT * FROM skills WHERE id IN (${ids.map(() => "?").join(", ")})`)
      .all(...ids) as Record<string, unknown>[];
    const byId = new Map(rows.map((row) => [String(row.id), skillFromRow(row)]));
    return ids.map((id) => byId.get(id) ?? null);
  }

  list(limit = 50): SkillRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM skills
          ORDER BY updated_at DESC, id ASC
          LIMIT ?
        `,
      )
      .all(limit) as Record<string, unknown>[];

    return rows.map((row) => skillFromRow(row));
  }

  async delete(id: SkillId): Promise<boolean> {
    const existing = this.get(id);

    if (existing === null) {
      return false;
    }

    try {
      this.db.transaction(() => {
        this.db.prepare("DELETE FROM skills WHERE id = ?").run(id);
      })();
      await this.table.remove(`id = ${quoteSqlString(id)}`);
      return true;
    } catch (error) {
      throw new StorageError(`Failed to delete skill ${id}`, {
        cause: error,
        code: "SKILL_DELETE_FAILED",
      });
    }
  }

  async searchByContext(text: string, limit = 10): Promise<SkillSearchCandidate[]> {
    if (text.trim().length === 0) {
      return [];
    }

    const embedding = await this.options.embeddingClient.embed(text);
    const rows = await this.table.search(embedding, {
      limit,
      distanceType: "cosine",
    });
    const ids = rows
      .map((row) => row.id)
      .filter((value): value is string => typeof value === "string")
      .map((value) => parseSkillId(value));
    const records = this.getMany(ids);
    const byId = new Map(
      records
        .filter((record): record is SkillRecord => record !== null)
        .map((record) => [record.id, record]),
    );

    return rows
      .map((row) => {
        const id = typeof row.id === "string" ? parseSkillId(row.id) : null;
        const skill = id === null ? null : (byId.get(id) ?? null);
        const distance = typeof row._distance === "number" ? row._distance : undefined;

        if (skill === null) {
          return null;
        }

        return {
          skill,
          similarity: Math.max(0, Math.min(1, 1 - (distance ?? 1))),
        } satisfies SkillSearchCandidate;
      })
      .filter((item): item is SkillSearchCandidate => item !== null);
  }

  recordOutcome(
    skillId: SkillId,
    success: boolean,
    episodeIds?: EpisodeId | readonly EpisodeId[],
  ): SkillRecord {
    const nowMs = this.clock.now();
    const incrementAlpha = success ? 1 : 0;
    const incrementBeta = success ? 0 : 1;
    const incrementSuccesses = success ? 1 : 0;
    const incrementFailures = success ? 0 : 1;
    const updateOutcome = this.db.prepare(
      `
        UPDATE skills
        SET
          alpha = alpha + ?,
          beta = beta + ?,
          attempts = attempts + 1,
          successes = successes + ?,
          failures = failures + ?,
          last_used = ?,
          last_successful = CASE WHEN ? THEN ? ELSE last_successful END,
          updated_at = ?
        WHERE id = ?
      `,
    );
    const appendSourceEpisode = this.db.prepare(
      `
        UPDATE skills
        SET source_episode_ids = CASE
          WHEN EXISTS (
            SELECT 1
            FROM json_each(source_episode_ids)
            WHERE value = ?
          ) THEN source_episode_ids
          ELSE json_insert(
            source_episode_ids,
            '$[' || json_array_length(source_episode_ids) || ']',
            ?
          )
        END
        WHERE id = ?
      `,
    );

    const sourceEpisodeIds =
      episodeIds === undefined ? [] : Array.isArray(episodeIds) ? episodeIds : [episodeIds];

    this.db.transaction(() => {
      const result = updateOutcome.run(
        incrementAlpha,
        incrementBeta,
        incrementSuccesses,
        incrementFailures,
        nowMs,
        success ? 1 : 0,
        nowMs,
        nowMs,
        skillId,
      );

      if (Number(result.changes) === 0) {
        throw new StorageError(`Unknown skill id: ${skillId}`, {
          code: "SKILL_NOT_FOUND",
        });
      }

      for (const episodeId of sourceEpisodeIds) {
        appendSourceEpisode.run(episodeId, episodeId, skillId);
      }
    })();

    const next = this.get(skillId);

    if (next === null) {
      throw new StorageError(`Unknown skill id: ${skillId}`, {
        code: "SKILL_NOT_FOUND",
      });
    }

    return next;
  }

  getStats(id: SkillId): BetaStats {
    const current = this.get(id);

    if (current === null) {
      throw new StorageError(`Unknown skill id: ${id}`, {
        code: "SKILL_NOT_FOUND",
      });
    }

    return computeBetaStats(current.alpha, current.beta);
  }
}

export type ProceduralEvidenceRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

function uniqueEpisodeIds(ids: readonly EpisodeId[]): EpisodeId[] {
  return [...new Set(ids)];
}

export class ProceduralEvidenceRepository {
  private readonly clock: Clock;

  constructor(private readonly options: ProceduralEvidenceRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  insert(input: {
    id?: ProceduralEvidenceId;
    pendingAttemptSnapshot: PendingProceduralAttemptValue;
    classification: ProceduralOutcomeClassification;
    evidenceText: string;
    grounded?: boolean;
    resolvedEpisodeIds?: readonly EpisodeId[];
    audienceEntityId?: EntityId | null;
    createdAt?: number;
  }): ProceduralEvidenceRecord {
    const snapshot = proceduralEvidenceSchema.shape.pending_attempt_snapshot.parse(
      input.pendingAttemptSnapshot,
    );
    const snapshotJson = serializeJsonValue(snapshot);
    const existing = this.db
      .prepare(
        `
          SELECT *
          FROM procedural_evidence
          WHERE pending_attempt_snapshot = ?
          ORDER BY created_at ASC, id ASC
          LIMIT 1
        `,
      )
      .get(snapshotJson) as Record<string, unknown> | undefined;

    if (existing !== undefined) {
      // Sprint 55: a pending attempt may be graded as `unclear` first
      // (Sprint 53 keeps it pending) and later get a grounded
      // success/failure signal. Upgrade the existing row in place so the
      // synthesizer sees the actionable classification. Anything else
      // (same classification, downgrade, ungrounded retry) dedups.
      const existingRecord = proceduralEvidenceFromRow(existing);
      const incomingClassification = proceduralOutcomeClassificationSchema.parse(
        input.classification,
      );
      const incomingGrounded = input.grounded ?? true;
      const isUpgrade =
        incomingGrounded &&
        existingRecord.classification === "unclear" &&
        (incomingClassification === "success" || incomingClassification === "failure");

      if (!isUpgrade) {
        return existingRecord;
      }

      const resolvedEpisodeIds = uniqueEpisodeIds([
        ...(input.resolvedEpisodeIds ?? []),
      ]);
      this.db
        .prepare(
          `
            UPDATE procedural_evidence
            SET classification = ?,
                evidence_text = ?,
                grounded = 1,
                resolved_episode_ids = ?,
                audience_entity_id = ?
            WHERE id = ?
          `,
        )
        .run(
          incomingClassification,
          input.evidenceText.trim(),
          serializeJsonValue(resolvedEpisodeIds),
          input.audienceEntityId ?? existingRecord.audience_entity_id,
          existingRecord.id,
        );

      return {
        ...existingRecord,
        classification: incomingClassification,
        evidence_text: input.evidenceText.trim(),
        grounded: true,
        resolved_episode_ids: resolvedEpisodeIds,
        audience_entity_id: input.audienceEntityId ?? existingRecord.audience_entity_id,
      };
    }

    const record = proceduralEvidenceSchema.parse({
      id: input.id ?? createProceduralEvidenceId(),
      pending_attempt_snapshot: snapshot,
      classification: proceduralOutcomeClassificationSchema.parse(input.classification),
      evidence_text: input.evidenceText.trim(),
      grounded: input.grounded ?? true,
      resolved_episode_ids: uniqueEpisodeIds([...(input.resolvedEpisodeIds ?? [])]),
      audience_entity_id: input.audienceEntityId ?? snapshot.audience_entity_id,
      consumed_at: null,
      created_at: input.createdAt ?? this.clock.now(),
    });

    this.db
      .prepare(
        `
          INSERT INTO procedural_evidence (
            id, pending_attempt_snapshot, classification, evidence_text, grounded,
            resolved_episode_ids, audience_entity_id, consumed_at, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        record.id,
        serializeJsonValue(record.pending_attempt_snapshot),
        record.classification,
        record.evidence_text,
        record.grounded ? 1 : 0,
        serializeJsonValue(record.resolved_episode_ids),
        record.audience_entity_id,
        record.consumed_at,
        record.created_at,
      );

    return record;
  }

  get(id: ProceduralEvidenceId): ProceduralEvidenceRecord | null {
    const row = this.db.prepare("SELECT * FROM procedural_evidence WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : proceduralEvidenceFromRow(row);
  }

  list(limit = 100): ProceduralEvidenceRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM procedural_evidence
          ORDER BY created_at ASC, id ASC
          LIMIT ?
        `,
      )
      .all(limit) as Record<string, unknown>[];

    return rows.map((row) => proceduralEvidenceFromRow(row));
  }

  listUnconsumed(limit = 100): ProceduralEvidenceRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM procedural_evidence
          WHERE consumed_at IS NULL
          ORDER BY created_at ASC, id ASC
          LIMIT ?
        `,
      )
      .all(limit) as Record<string, unknown>[];

    return rows.map((row) => proceduralEvidenceFromRow(row));
  }

  markConsumed(ids: readonly ProceduralEvidenceId[], consumedAt = this.clock.now()): void {
    if (ids.length === 0) {
      return;
    }

    const statement = this.db.prepare(
      "UPDATE procedural_evidence SET consumed_at = ? WHERE id = ?",
    );

    this.db.transaction(() => {
      for (const id of ids) {
        statement.run(consumedAt, id);
      }
    })();
  }

  markUnconsumed(ids: readonly ProceduralEvidenceId[]): void {
    if (ids.length === 0) {
      return;
    }

    const statement = this.db.prepare(
      "UPDATE procedural_evidence SET consumed_at = NULL WHERE id = ?",
    );

    this.db.transaction(() => {
      for (const id of ids) {
        statement.run(id);
      }
    })();
  }

  updateResolvedEpisodeIds(
    id: ProceduralEvidenceId,
    episodeIds: readonly EpisodeId[],
  ): ProceduralEvidenceRecord {
    const current = this.get(id);

    if (current === null) {
      throw new StorageError(`Unknown procedural evidence id: ${id}`, {
        code: "PROCEDURAL_EVIDENCE_NOT_FOUND",
      });
    }

    const nextIds = uniqueEpisodeIds([...current.resolved_episode_ids, ...episodeIds]);

    this.db
      .prepare("UPDATE procedural_evidence SET resolved_episode_ids = ? WHERE id = ?")
      .run(serializeJsonValue(nextIds), id);

    const updated = this.get(id);

    if (updated === null) {
      throw new StorageError(`Unknown procedural evidence id: ${id}`, {
        code: "PROCEDURAL_EVIDENCE_NOT_FOUND",
      });
    }

    return updated;
  }
}
