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
  createSkillId,
  parseEpisodeId,
  parseSkillId,
  type EpisodeId,
  type SkillId,
} from "../../util/ids.js";

import { computeBetaStats, type BetaStats } from "./bayes.js";
import {
  skillInsertSchema,
  skillSchema,
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

  recordOutcome(skillId: SkillId, success: boolean, episodeId?: EpisodeId): SkillRecord {
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

      if (episodeId !== undefined) {
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
