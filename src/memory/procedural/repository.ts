import { LanceDbTable, schema, utf8Field, vectorField } from "../../storage/lancedb/index.js";
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
import { proceduralContextSchema, type ProceduralContext } from "./context.js";
import {
  proceduralEvidenceSchema,
  proceduralOutcomeClassificationSchema,
  skillContextStatsSchema,
  skillInsertSchema,
  skillSchema,
  type PendingProceduralAttemptValue,
  type ProceduralEvidenceRecord,
  type ProceduralOutcomeClassification,
  type SkillContextStatsRecord,
  type SkillRecord,
  type SkillSearchCandidate,
} from "./types.js";

type SkillSqlRow = {
  id: string;
  applies_when: string;
  approach: string;
  status: "active" | "superseded";
  alpha: number;
  beta: number;
  attempts: number;
  successes: number;
  failures: number;
  alternatives: string;
  superseded_by: string;
  superseded_at: number | null;
  splitting_at: number | null;
  last_split_attempt_at: number | null;
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
    status: skill.status,
    alpha: skill.alpha,
    beta: skill.beta,
    attempts: skill.attempts,
    successes: skill.successes,
    failures: skill.failures,
    alternatives: serializeJsonValue(skill.alternatives),
    superseded_by: serializeJsonValue(skill.superseded_by),
    superseded_at: skill.superseded_at,
    splitting_at: skill.splitting_at,
    last_split_attempt_at: skill.last_split_attempt_at ?? null,
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
    status: row.status ?? "active",
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
    superseded_by: parseJsonArray<string>(
      String(row.superseded_by ?? "[]"),
      "superseded_by",
      SKILL_JSON_ARRAY_CODEC,
    ).map((value) => parseSkillId(value)),
    superseded_at:
      row.superseded_at === null || row.superseded_at === undefined
        ? null
        : Number(row.superseded_at),
    splitting_at:
      row.splitting_at === null || row.splitting_at === undefined
        ? null
        : Number(row.splitting_at),
    last_split_attempt_at:
      row.last_split_attempt_at === null || row.last_split_attempt_at === undefined
        ? null
        : Number(row.last_split_attempt_at),
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

function parseProceduralContextColumn(value: unknown): ProceduralContext | undefined {
  if (value === null || value === undefined) {
    return undefined;
  }

  try {
    const parsed = typeof value === "string" ? (JSON.parse(value) as unknown) : value;
    return proceduralContextSchema.parse(parsed);
  } catch (error) {
    throw new StorageError("Failed to parse procedural context", {
      cause: error,
      code: "PROCEDURAL_CONTEXT_INVALID",
    });
  }
}

function serializeProceduralContext(context: ProceduralContext | null | undefined): string | null {
  return context === null || context === undefined ? null : serializeJsonValue(context);
}

function proceduralEvidenceFromRow(row: Record<string, unknown>): ProceduralEvidenceRecord {
  const proceduralContext = parseProceduralContextColumn(row.procedural_context);
  const parsed = proceduralEvidenceSchema.safeParse({
    id: row.id,
    pending_attempt_snapshot: parsePendingAttemptSnapshot(
      String(row.pending_attempt_snapshot ?? "{}"),
    ),
    classification: row.classification,
    evidence_text: row.evidence_text,
    grounded:
      row.grounded === null || row.grounded === undefined ? true : Number(row.grounded) !== 0,
    skill_actually_applied:
      row.skill_actually_applied === null || row.skill_actually_applied === undefined
        ? true
        : Number(row.skill_actually_applied) !== 0,
    ...(proceduralContext === undefined ? {} : { procedural_context: proceduralContext }),
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

function skillContextStatsFromRow(row: Record<string, unknown>): SkillContextStatsRecord {
  const parsed = skillContextStatsSchema.safeParse({
    skill_id: row.skill_id,
    context_key: row.context_key,
    alpha: Number(row.alpha),
    beta: Number(row.beta),
    attempts: Number(row.attempts),
    successes: Number(row.successes),
    failures: Number(row.failures),
    last_used: row.last_used === null || row.last_used === undefined ? null : Number(row.last_used),
    last_successful:
      row.last_successful === null || row.last_successful === undefined
        ? null
        : Number(row.last_successful),
    updated_at: Number(row.updated_at),
  });

  if (!parsed.success) {
    throw new StorageError("Skill context stats row failed validation", {
      cause: parsed.error,
      code: "SKILL_CONTEXT_STATS_ROW_INVALID",
    });
  }

  return parsed.data;
}

function assertContextKey(contextKey: string): string {
  const trimmed = contextKey.trim();

  if (trimmed.length === 0) {
    throw new StorageError("Context key must not be empty", {
      code: "SKILL_CONTEXT_KEY_INVALID",
    });
  }

  return trimmed;
}

function recordContextOutcomeInTransaction(
  db: SqliteDatabase,
  input: {
    skillId: SkillId;
    contextKey: string;
    success: boolean;
    ts: number;
  },
): void {
  const contextKey = assertContextKey(input.contextKey);
  const incrementAlpha = input.success ? 1 : 0;
  const incrementBeta = input.success ? 0 : 1;
  const incrementSuccesses = input.success ? 1 : 0;
  const incrementFailures = input.success ? 0 : 1;

  db.prepare(
    `
      INSERT INTO skill_context_stats (
        skill_id, context_key, alpha, beta, attempts, successes, failures,
        last_used, last_successful, updated_at
      ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?)
      ON CONFLICT (skill_id, context_key) DO UPDATE SET
        alpha = alpha + ?,
        beta = beta + ?,
        attempts = attempts + 1,
        successes = successes + ?,
        failures = failures + ?,
        last_used = ?,
        last_successful = CASE WHEN ? THEN ? ELSE last_successful END,
        updated_at = ?
    `,
  ).run(
    input.skillId,
    contextKey,
    1 + incrementAlpha,
    1 + incrementBeta,
    incrementSuccesses,
    incrementFailures,
    input.ts,
    input.success ? input.ts : null,
    input.ts,
    incrementAlpha,
    incrementBeta,
    incrementSuccesses,
    incrementFailures,
    input.ts,
    input.success ? 1 : 0,
    input.ts,
    input.ts,
  );
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

export type SkillSplitPartInput = {
  applies_when: string;
  approach: string;
  target_contexts: readonly string[];
};

export type SkillSplitApplyResult = {
  previous: SkillRecord;
  superseded: SkillRecord;
  created: SkillRecord[];
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
            id, applies_when, approach, status, alpha, beta, attempts, successes, failures,
            alternatives, superseded_by, superseded_at, splitting_at, last_split_attempt_at,
            source_episode_ids, last_used, last_successful, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (id) DO UPDATE SET
            applies_when = excluded.applies_when,
            approach = excluded.approach,
            status = excluded.status,
            alpha = excluded.alpha,
            beta = excluded.beta,
            attempts = excluded.attempts,
            successes = excluded.successes,
            failures = excluded.failures,
            alternatives = excluded.alternatives,
            superseded_by = excluded.superseded_by,
            superseded_at = excluded.superseded_at,
            splitting_at = excluded.splitting_at,
            last_split_attempt_at = excluded.last_split_attempt_at,
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
        row.status,
        row.alpha,
        row.beta,
        row.attempts,
        row.successes,
        row.failures,
        row.alternatives,
        row.superseded_by,
        row.superseded_at,
        row.splitting_at,
        row.last_split_attempt_at,
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
      status: "active",
      alpha: input.priorAlpha ?? 1,
      beta: input.priorBeta ?? 1,
      attempts: 0,
      successes: 0,
      failures: 0,
      alternatives: input.alternatives ?? [],
      superseded_by: [],
      superseded_at: null,
      splitting_at: null,
      last_split_attempt_at: null,
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

  listContextStatsForSkill(skillId: SkillId): SkillContextStatsRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM skill_context_stats
          WHERE skill_id = ?
          ORDER BY updated_at DESC, context_key ASC
        `,
      )
      .all(skillId) as Record<string, unknown>[];

    return rows.map((row) => skillContextStatsFromRow(row));
  }

  batchListContextStatsForSkills(
    skillIds: readonly SkillId[],
  ): Map<SkillId, SkillContextStatsRecord[]> {
    const uniqueSkillIds = [...new Set(skillIds)];

    if (uniqueSkillIds.length === 0) {
      return new Map();
    }

    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM skill_context_stats
          WHERE skill_id IN (${uniqueSkillIds.map(() => "?").join(", ")})
          ORDER BY skill_id ASC, updated_at DESC, context_key ASC
        `,
      )
      .all(...uniqueSkillIds) as Record<string, unknown>[];
    const result = new Map<SkillId, SkillContextStatsRecord[]>(
      uniqueSkillIds.map((skillId) => [skillId, []]),
    );

    for (const row of rows) {
      const stats = skillContextStatsFromRow(row);
      const bucket = result.get(stats.skill_id) ?? [];
      bucket.push(stats);
      result.set(stats.skill_id, bucket);
    }

    return result;
  }

  claimSplit(input: { skillId: SkillId; claimedAt: number; staleBefore: number }): boolean {
    const result = this.db
      .prepare(
        `
          UPDATE skills
          SET splitting_at = ?, updated_at = ?
          WHERE id = ?
            AND status = 'active'
            AND (splitting_at IS NULL OR splitting_at < ?)
        `,
      )
      .run(input.claimedAt, input.claimedAt, input.skillId, input.staleBefore);

    return Number(result.changes) > 0;
  }

  recordSplitAttemptAndClearClaim(input: {
    skillId: SkillId;
    attemptedAt: number;
    claimedAt?: number | null;
  }): void {
    if (input.claimedAt === undefined || input.claimedAt === null) {
      this.db
        .prepare(
          `
            UPDATE skills
            SET
              last_split_attempt_at = max(COALESCE(last_split_attempt_at, 0), ?),
              updated_at = ?
            WHERE id = ?
          `,
        )
        .run(input.attemptedAt, input.attemptedAt, input.skillId);
      return;
    }

    this.db
      .prepare(
        `
          UPDATE skills
          SET
            last_split_attempt_at = max(COALESCE(last_split_attempt_at, 0), ?),
            splitting_at = CASE WHEN splitting_at = ? THEN NULL ELSE splitting_at END,
            updated_at = ?
          WHERE id = ?
        `,
      )
      .run(input.attemptedAt, input.claimedAt, input.attemptedAt, input.skillId);
  }

  clearSplitClaim(input: { skillId: SkillId; claimedAt: number; clearedAt: number }): void {
    this.db
      .prepare(
        `
          UPDATE skills
          SET splitting_at = NULL, updated_at = ?
          WHERE id = ? AND splitting_at = ?
        `,
      )
      .run(input.clearedAt, input.skillId, input.claimedAt);
  }

  restoreContextStats(statsRows: readonly SkillContextStatsRecord[]): void {
    const insert = this.db.prepare(
      `
        INSERT INTO skill_context_stats (
          skill_id, context_key, alpha, beta, attempts, successes, failures,
          last_used, last_successful, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (skill_id, context_key) DO UPDATE SET
          alpha = excluded.alpha,
          beta = excluded.beta,
          attempts = excluded.attempts,
          successes = excluded.successes,
          failures = excluded.failures,
          last_used = excluded.last_used,
          last_successful = excluded.last_successful,
          updated_at = excluded.updated_at
      `,
    );

    this.db.transaction(() => {
      for (const stats of statsRows) {
        const parsed = skillContextStatsSchema.parse(stats);
        insert.run(
          parsed.skill_id,
          parsed.context_key,
          parsed.alpha,
          parsed.beta,
          parsed.attempts,
          parsed.successes,
          parsed.failures,
          parsed.last_used,
          parsed.last_successful,
          parsed.updated_at,
        );
      }
    })();
  }

  async supersedeWithSplits(input: {
    skillId: SkillId;
    parts: readonly SkillSplitPartInput[];
    supersededAt?: number;
    claimedAt?: number | null;
  }): Promise<SkillSplitApplyResult | null> {
    const original = this.get(input.skillId);

    if (original === null) {
      throw new StorageError(`Unknown skill id: ${input.skillId}`, {
        code: "SKILL_NOT_FOUND",
      });
    }

    if (original.status !== "active") {
      return null;
    }

    if (
      input.claimedAt !== undefined &&
      input.claimedAt !== null &&
      original.splitting_at !== input.claimedAt
    ) {
      return null;
    }

    if (input.parts.length === 0) {
      throw new StorageError("Skill split must include at least one part", {
        code: "SKILL_SPLIT_EMPTY",
      });
    }

    const contextRowsByKey = new Map(
      this.listContextStatsForSkill(input.skillId).map((stats) => [stats.context_key, stats]),
    );
    const assignedContexts = new Set<string>();
    const nowMs = input.supersededAt ?? this.clock.now();
    const newSkills = input.parts.map((part) => {
      const targetContexts = [...new Set(part.target_contexts.map((contextKey) => contextKey.trim()))]
        .filter((contextKey) => contextKey.length > 0)
        .sort();

      if (targetContexts.length === 0) {
        throw new StorageError("Skill split part must target at least one context", {
          code: "SKILL_SPLIT_TARGETS_EMPTY",
        });
      }

      for (const contextKey of targetContexts) {
        if (!contextRowsByKey.has(contextKey)) {
          throw new StorageError(`Unknown split context bucket: ${contextKey}`, {
            code: "SKILL_SPLIT_CONTEXT_MISSING",
          });
        }

        if (assignedContexts.has(contextKey)) {
          throw new StorageError(`Split context assigned to more than one part: ${contextKey}`, {
            code: "SKILL_SPLIT_CONTEXT_DUPLICATE",
          });
        }

        assignedContexts.add(contextKey);
      }

      const targetStats = targetContexts.map((contextKey) => contextRowsByKey.get(contextKey)!);
      const lastUsed = Math.max(...targetStats.map((stats) => stats.last_used ?? 0));
      const lastSuccessful = Math.max(...targetStats.map((stats) => stats.last_successful ?? 0));
      const id = createSkillId();

      return skillInsertSchema.parse({
        id,
        applies_when: part.applies_when.trim(),
        approach: part.approach.trim(),
        status: "active",
        alpha: targetStats.reduce((sum, stats) => sum + stats.alpha, 0),
        beta: targetStats.reduce((sum, stats) => sum + stats.beta, 0),
        attempts: targetStats.reduce((sum, stats) => sum + stats.attempts, 0),
        successes: targetStats.reduce((sum, stats) => sum + stats.successes, 0),
        failures: targetStats.reduce((sum, stats) => sum + stats.failures, 0),
        alternatives: [original.id, ...original.alternatives],
        superseded_by: [],
        superseded_at: null,
        splitting_at: null,
        last_split_attempt_at: null,
        source_episode_ids: original.source_episode_ids,
        last_used: lastUsed === 0 ? null : lastUsed,
        last_successful: lastSuccessful === 0 ? null : lastSuccessful,
        created_at: nowMs,
        updated_at: nowMs,
      });
    });
    const embeddings = await Promise.all(
      newSkills.map((skill) => this.options.embeddingClient.embed(skill.applies_when)),
    );
    const insertedSkillIds = newSkills.map((skill) => skill.id);

    try {
      await this.table.upsert(
        newSkills.map((skill, index) => ({
          id: skill.id,
          applies_when: skill.applies_when,
          embedding: Array.from(embeddings[index]!),
        })),
        { on: "id" },
      );

      try {
        this.db.transaction(() => {
          const current = this.get(input.skillId);

          if (current === null) {
            throw new StorageError(`Unknown skill id: ${input.skillId}`, {
              code: "SKILL_NOT_FOUND",
            });
          }

          if (current.status !== "active") {
            throw new StorageError(`Skill already superseded: ${input.skillId}`, {
              code: "SKILL_SPLIT_ALREADY_APPLIED",
            });
          }

          if (
            input.claimedAt !== undefined &&
            input.claimedAt !== null &&
            current.splitting_at !== input.claimedAt
          ) {
            throw new StorageError(`Skill split claim lost: ${input.skillId}`, {
              code: "SKILL_SPLIT_CLAIM_LOST",
            });
          }

          for (const skill of newSkills) {
            this.upsertSqlRow(skill);
          }

          const insertContextStats = this.db.prepare(
            `
              INSERT INTO skill_context_stats (
                skill_id, context_key, alpha, beta, attempts, successes, failures,
                last_used, last_successful, updated_at
              ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            `,
          );

          for (const [partIndex, part] of input.parts.entries()) {
            const newSkill = newSkills[partIndex]!;
            const targetContexts = [...new Set(part.target_contexts.map((contextKey) => contextKey.trim()))]
              .filter((contextKey) => contextKey.length > 0)
              .sort();

            for (const contextKey of targetContexts) {
              const stats = contextRowsByKey.get(contextKey)!;
              insertContextStats.run(
                newSkill.id,
                stats.context_key,
                stats.alpha,
                stats.beta,
                stats.attempts,
                stats.successes,
                stats.failures,
                stats.last_used,
                stats.last_successful,
                stats.updated_at,
              );
            }
          }

          this.db
            .prepare(
              `
                DELETE FROM skill_context_stats
                WHERE skill_id = ? AND context_key IN (${[...assignedContexts].map(() => "?").join(", ")})
              `,
            )
            .run(input.skillId, ...assignedContexts);

          this.upsertSqlRow(
            skillSchema.parse({
              ...current,
              status: "superseded",
              superseded_by: insertedSkillIds,
              superseded_at: nowMs,
              splitting_at: null,
              last_split_attempt_at: nowMs,
              updated_at: nowMs,
            }),
          );
        })();
      } catch (error) {
        await Promise.all(insertedSkillIds.map((id) => this.table.remove(`id = ${quoteSqlString(id)}`)));

        if (
          error instanceof StorageError &&
          (error.code === "SKILL_SPLIT_ALREADY_APPLIED" ||
            error.code === "SKILL_SPLIT_CLAIM_LOST")
        ) {
          return null;
        }

        throw error;
      }

      const supersededOriginal = this.get(input.skillId);

      if (supersededOriginal === null) {
        throw new StorageError(`Unknown skill id: ${input.skillId}`, {
          code: "SKILL_NOT_FOUND",
        });
      }

      return {
        previous: original,
        superseded: supersededOriginal,
        created: newSkills,
      };
    } catch (error) {
      throw new StorageError(`Failed to split skill ${input.skillId}`, {
        cause: error,
        code: "SKILL_SPLIT_FAILED",
      });
    }
  }

  async delete(id: SkillId): Promise<boolean> {
    const existing = this.get(id);

    if (existing === null) {
      return false;
    }

    try {
      this.db.transaction(() => {
        this.db.prepare("DELETE FROM skill_context_stats WHERE skill_id = ?").run(id);
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
        .filter((record): record is SkillRecord => record !== null && record.status === "active")
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
    proceduralContext?: ProceduralContext | null,
  ): SkillRecord {
    const nowMs = this.clock.now();
    const parsedContext =
      proceduralContext === null || proceduralContext === undefined
        ? null
        : proceduralContextSchema.parse(proceduralContext);
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
          AND status = 'active'
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
        const existing = this.get(skillId);

        if (existing !== null && existing.status === "superseded") {
          return;
        }

        throw new StorageError(`Unknown skill id: ${skillId}`, {
          code: "SKILL_NOT_FOUND",
        });
      }

      for (const episodeId of sourceEpisodeIds) {
        appendSourceEpisode.run(episodeId, episodeId, skillId);
      }

      if (parsedContext !== null) {
        recordContextOutcomeInTransaction(this.db, {
          skillId,
          contextKey: parsedContext.context_key,
          success,
          ts: nowMs,
        });
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

export type ProceduralContextStatsRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
};

export class ProceduralContextStatsRepository {
  private readonly clock: Clock;

  constructor(private readonly options: ProceduralContextStatsRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  recordContextOutcome(input: {
    skillId: SkillId;
    contextKey: string;
    success: boolean;
    ts?: number;
  }): SkillContextStatsRecord {
    const ts = input.ts ?? this.clock.now();
    const contextKey = assertContextKey(input.contextKey);

    this.db.transaction(() => {
      recordContextOutcomeInTransaction(this.db, {
        skillId: input.skillId,
        contextKey,
        success: input.success,
        ts,
      });
    })();

    const next = this.getContextStats(input.skillId, contextKey);

    if (next === null) {
      throw new StorageError("Failed to record skill context outcome", {
        code: "SKILL_CONTEXT_STATS_RECORD_FAILED",
      });
    }

    return next;
  }

  getContextStats(skillId: SkillId, contextKey: string): SkillContextStatsRecord | null {
    const row = this.db
      .prepare(
        `
          SELECT *
          FROM skill_context_stats
          WHERE skill_id = ? AND context_key = ?
        `,
      )
      .get(skillId, assertContextKey(contextKey)) as Record<string, unknown> | undefined;

    return row === undefined ? null : skillContextStatsFromRow(row);
  }

  batchGetContextStats(
    contextKey: string,
    skillIds: readonly SkillId[],
  ): Map<SkillId, SkillContextStatsRecord> {
    const contextKeyValue = assertContextKey(contextKey);

    if (skillIds.length === 0) {
      return new Map();
    }

    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM skill_context_stats
          WHERE context_key = ? AND skill_id IN (${skillIds.map(() => "?").join(", ")})
        `,
      )
      .all(contextKeyValue, ...skillIds) as Record<string, unknown>[];

    return new Map(
      rows.map((row) => {
        const stats = skillContextStatsFromRow(row);
        return [stats.skill_id, stats] as const;
      }),
    );
  }

  listForSkill(skillId: SkillId): SkillContextStatsRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM skill_context_stats
          WHERE skill_id = ?
          ORDER BY updated_at DESC, context_key ASC
        `,
      )
      .all(skillId) as Record<string, unknown>[];

    return rows.map((row) => skillContextStatsFromRow(row));
  }

  listGlobalUsage(contextKey: string): SkillContextStatsRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM skill_context_stats
          WHERE context_key = ?
          ORDER BY updated_at DESC, skill_id ASC
        `,
      )
      .all(assertContextKey(contextKey)) as Record<string, unknown>[];

    return rows.map((row) => skillContextStatsFromRow(row));
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
    skillActuallyApplied?: boolean;
    proceduralContext?: ProceduralContext | null;
    resolvedEpisodeIds?: readonly EpisodeId[];
    audienceEntityId?: EntityId | null;
    createdAt?: number;
  }): ProceduralEvidenceRecord {
    const contextSource =
      input.proceduralContext ?? input.pendingAttemptSnapshot.procedural_context ?? null;
    const proceduralContext =
      contextSource === null || contextSource === undefined
        ? null
        : proceduralContextSchema.parse(contextSource);
    const snapshot = proceduralEvidenceSchema.shape.pending_attempt_snapshot.parse(
      proceduralContext === null
        ? input.pendingAttemptSnapshot
        : {
            ...input.pendingAttemptSnapshot,
            procedural_context: proceduralContext,
          },
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
        ...existingRecord.resolved_episode_ids,
        ...(input.resolvedEpisodeIds ?? []),
      ]);
      const skillActuallyApplied = input.skillActuallyApplied ?? true;
      const nextProceduralContext = proceduralContext ?? existingRecord.procedural_context ?? null;
      this.db
        .prepare(
          `
            UPDATE procedural_evidence
            SET classification = ?,
                evidence_text = ?,
                grounded = 1,
                skill_actually_applied = ?,
                procedural_context = ?,
                resolved_episode_ids = ?,
                audience_entity_id = ?
            WHERE id = ?
          `,
        )
        .run(
          incomingClassification,
          input.evidenceText.trim(),
          skillActuallyApplied ? 1 : 0,
          serializeProceduralContext(nextProceduralContext),
          serializeJsonValue(resolvedEpisodeIds),
          input.audienceEntityId ?? existingRecord.audience_entity_id,
          existingRecord.id,
        );

      return {
        ...existingRecord,
        classification: incomingClassification,
        evidence_text: input.evidenceText.trim(),
        grounded: true,
        skill_actually_applied: skillActuallyApplied,
        ...(nextProceduralContext === null ? {} : { procedural_context: nextProceduralContext }),
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
      skill_actually_applied: input.skillActuallyApplied ?? true,
      ...(proceduralContext === null ? {} : { procedural_context: proceduralContext }),
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
            skill_actually_applied, procedural_context, resolved_episode_ids,
            audience_entity_id, consumed_at, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        record.id,
        serializeJsonValue(record.pending_attempt_snapshot),
        record.classification,
        record.evidence_text,
        record.grounded ? 1 : 0,
        record.skill_actually_applied ? 1 : 0,
        serializeProceduralContext(record.procedural_context),
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
