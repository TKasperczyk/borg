import { Buffer } from "node:buffer";

import {
  LanceDbTable,
  booleanField,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import {
  parseJsonArray,
  quoteSqlString,
  toFloat32Array,
  type Float32ArrayCodecOptions,
  type JsonArrayCodecOptions,
} from "../../storage/codecs.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { parseEntityId, parseEpisodeId, type EntityId, type EpisodeId } from "../../util/ids.js";
import { emotionalArcSchema } from "../affective/types.js";
import { computeEpisodeHeat, computeEpisodeHeatForTimestamp } from "./heat.js";
import {
  isEpisodeInGlobalIdentityScope,
  isEpisodeVisibleToAudience,
  normalizeEpisodeAccess,
} from "./access.js";

import {
  type Episode,
  type EpisodeListOptions,
  type EpisodeListResult,
  type EpisodePatch,
  type EpisodeSearchCandidate,
  type EpisodeSearchOptions,
  type EpisodeStats,
  type EpisodeStatsPatch,
  type EpisodeVisibilityOptions,
  episodeInsertSchema,
  episodePatchSchema,
  episodeSchema,
  episodeStatsPatchSchema,
  episodeStatsSchema,
} from "./types.js";

type EpisodeRow = {
  id: string;
  title: string;
  narrative: string;
  participants: string;
  location: string | null;
  start_time: number;
  end_time: number;
  source_stream_ids: string;
  significance: number;
  tags: string;
  confidence: number;
  lineage_derived_from: string;
  lineage_supersedes: string;
  source_fingerprint: string | null;
  audience_entity_id: string | null;
  shared: boolean | number | null;
  emotional_arc: string | null;
  embedding: number[];
  created_at: number;
  updated_at: number;
  _distance?: number;
};

type CursorPayload = {
  updatedAt: number;
  id: string;
};

type IndexedEpisodeOrder = "recent" | "heat";

type IndexedVisibilityBranch = {
  where: string;
  params: unknown[];
  indexName: string;
};

type IndexedEpisodeIdRow = {
  episode_id: string;
};

type IndexedEpisodeStatsProjectionRow = {
  updated_at: number;
};

const DEFAULT_LIST_LIMIT = 20;
const DEFAULT_SEARCH_LIMIT = 10;
const EPISODE_INDEX_BACKFILLED_KEY = "lance_backfilled_at";
const EPISODE_JSON_ARRAY_CODEC = {
  errorCode: "EPISODE_ROW_INVALID",
  errorMessage: (label: string) => `Failed to decode episode ${label}`,
} satisfies JsonArrayCodecOptions;
const EPISODE_VECTOR_CODEC = {
  arrayLikeErrorMessage: "Episode row embedding must be array-like",
  nonFiniteErrorMessage: "Episode row embedding contains a non-finite value",
  errorCode: "EPISODE_ROW_INVALID",
} satisfies Float32ArrayCodecOptions;

function assertPositiveLimit(limit: number | undefined, label: string): number {
  const resolved = limit ?? DEFAULT_LIST_LIMIT;

  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new StorageError(`${label} must be a positive integer`);
  }

  return resolved;
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function buildSourceFingerprint(sourceStreamIds: readonly string[]): string {
  return [...new Set(sourceStreamIds)].sort().join("\n");
}

function encodeCursor(payload: CursorPayload): string {
  return Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
}

function decodeCursor(cursor: string): CursorPayload {
  try {
    const raw = Buffer.from(cursor, "base64url").toString("utf8");
    const parsed = JSON.parse(raw) as unknown;

    if (
      parsed === null ||
      typeof parsed !== "object" ||
      Array.isArray(parsed) ||
      typeof (parsed as { updatedAt?: unknown }).updatedAt !== "number" ||
      typeof (parsed as { id?: unknown }).id !== "string"
    ) {
      throw new TypeError("Invalid cursor payload");
    }

    const cursorPayload = parsed as {
      updatedAt: number;
      id: string;
    };

    return {
      updatedAt: cursorPayload.updatedAt,
      id: cursorPayload.id,
    };
  } catch (error) {
    throw new StorageError("Invalid episode cursor", {
      cause: error,
      code: "EPISODE_CURSOR_INVALID",
    });
  }
}

function compareEpisodes(left: Episode, right: Episode): number {
  if (left.updated_at !== right.updated_at) {
    return right.updated_at - left.updated_at;
  }

  return right.id.localeCompare(left.id);
}

function compareAfterCursor(episode: Episode, cursor: CursorPayload): boolean {
  if (episode.updated_at < cursor.updatedAt) {
    return true;
  }

  if (episode.updated_at > cursor.updatedAt) {
    return false;
  }

  return episode.id.localeCompare(cursor.id) < 0;
}

function combineWhereClauses(...clauses: Array<string | undefined>): string | undefined {
  const definedClauses = clauses.filter((clause): clause is string => clause !== undefined);

  if (definedClauses.length === 0) {
    return undefined;
  }

  return definedClauses.map((clause) => `(${clause})`).join(" AND ");
}

function normalizeTerm(value: string): string {
  return value.trim().toLowerCase();
}

function sqlPlaceholders(count: number): string {
  return Array.from({ length: count }, () => "?").join(", ");
}

function getDistance(row: Record<string, unknown>): number | undefined {
  const value = row._distance;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function toSimilarity(distance: number | undefined): number {
  if (distance === undefined) {
    return 0;
  }

  return Math.max(0, Math.min(1, 1 - distance));
}

function toEpisodeRow(episode: Episode): EpisodeRow {
  const normalized = normalizeEpisodeAccess(episode);

  return {
    id: normalized.id,
    title: normalized.title,
    narrative: normalized.narrative,
    participants: serializeJsonValue(normalized.participants),
    location: normalized.location,
    start_time: normalized.start_time,
    end_time: normalized.end_time,
    source_stream_ids: serializeJsonValue(normalized.source_stream_ids),
    significance: normalized.significance,
    tags: serializeJsonValue(normalized.tags),
    confidence: normalized.confidence,
    lineage_derived_from: serializeJsonValue(normalized.lineage.derived_from),
    lineage_supersedes: serializeJsonValue(normalized.lineage.supersedes),
    source_fingerprint: buildSourceFingerprint(normalized.source_stream_ids),
    audience_entity_id: normalized.audience_entity_id,
    shared: normalized.shared,
    emotional_arc:
      normalized.emotional_arc === null ? null : serializeJsonValue(normalized.emotional_arc),
    embedding: Array.from(normalized.embedding),
    created_at: normalized.created_at,
    updated_at: normalized.updated_at,
  };
}

function fromEpisodeRow(row: Record<string, unknown>): Episode {
  const emotionalArc = (() => {
    if (row.emotional_arc === null || row.emotional_arc === undefined || row.emotional_arc === "") {
      return null;
    }

    try {
      return emotionalArcSchema.parse(JSON.parse(String(row.emotional_arc)) as unknown);
    } catch (error) {
      throw new StorageError("Failed to decode episode emotional arc", {
        cause: error,
        code: "EPISODE_ROW_INVALID",
      });
    }
  })();
  const candidate = {
    id: row.id,
    title: row.title,
    narrative: row.narrative,
    participants: parseJsonArray<string>(
      String(row.participants ?? "[]"),
      "participants",
      EPISODE_JSON_ARRAY_CODEC,
    ),
    location: row.location === null || row.location === undefined ? null : String(row.location),
    start_time: Number(row.start_time),
    end_time: Number(row.end_time),
    source_stream_ids: parseJsonArray<string>(
      String(row.source_stream_ids ?? "[]"),
      "source_stream_ids",
      EPISODE_JSON_ARRAY_CODEC,
    ),
    significance: Number(row.significance),
    tags: parseJsonArray<string>(String(row.tags ?? "[]"), "tags", EPISODE_JSON_ARRAY_CODEC),
    confidence: Number(row.confidence),
    lineage: {
      derived_from: parseJsonArray<string>(
        String(row.lineage_derived_from ?? "[]"),
        "lineage.derived_from",
        EPISODE_JSON_ARRAY_CODEC,
      ),
      supersedes: parseJsonArray<string>(
        String(row.lineage_supersedes ?? "[]"),
        "lineage.supersedes",
        EPISODE_JSON_ARRAY_CODEC,
      ),
    },
    emotional_arc: emotionalArc,
    audience_entity_id:
      row.audience_entity_id === null || row.audience_entity_id === undefined
        ? null
        : parseEntityId(String(row.audience_entity_id)),
    shared:
      row.shared === null || row.shared === undefined
        ? row.audience_entity_id === null || row.audience_entity_id === undefined
        : row.shared === true || Number(row.shared) === 1,
    embedding: toFloat32Array(row.embedding, EPISODE_VECTOR_CODEC),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  };
  const parsed = episodeSchema.safeParse(normalizeEpisodeAccess(candidate));

  if (!parsed.success) {
    throw new StorageError("Episode row failed validation", {
      cause: parsed.error,
      code: "EPISODE_ROW_INVALID",
    });
  }

  return parsed.data;
}

function defaultEpisodeStats(episode: Episode): EpisodeStats {
  const valenceMean =
    episode.emotional_arc === null
      ? 0
      : (episode.emotional_arc.start.valence +
          episode.emotional_arc.peak.valence +
          episode.emotional_arc.end.valence) /
        3;

  return {
    episode_id: episode.id,
    retrieval_count: 0,
    use_count: 0,
    last_retrieved: null,
    win_rate: 0,
    tier: "T1",
    promoted_at: episode.created_at,
    promoted_from: null,
    gist: null,
    gist_generated_at: null,
    last_decayed_at: null,
    heat_multiplier: 1,
    valence_mean: valenceMean,
    archived: false,
  };
}

function fromEpisodeStatsRow(row: Record<string, unknown>): EpisodeStats {
  const parsed = episodeStatsSchema.safeParse({
    episode_id: row.episode_id,
    retrieval_count: Number(row.retrieval_count),
    use_count: Number(row.use_count),
    last_retrieved:
      row.last_retrieved === null || row.last_retrieved === undefined
        ? null
        : Number(row.last_retrieved),
    win_rate: Number(row.win_rate),
    tier: row.tier,
    promoted_at: Number(row.promoted_at),
    promoted_from:
      row.promoted_from === null || row.promoted_from === undefined
        ? null
        : String(row.promoted_from),
    gist: row.gist === null || row.gist === undefined ? null : String(row.gist),
    gist_generated_at:
      row.gist_generated_at === null || row.gist_generated_at === undefined
        ? null
        : Number(row.gist_generated_at),
    last_decayed_at:
      row.last_decayed_at === null || row.last_decayed_at === undefined
        ? null
        : Number(row.last_decayed_at),
    heat_multiplier:
      row.heat_multiplier === null || row.heat_multiplier === undefined
        ? 1
        : Number(row.heat_multiplier),
    valence_mean:
      row.valence_mean === null || row.valence_mean === undefined ? 0 : Number(row.valence_mean),
    archived: row.archived === true || Number(row.archived) === 1,
  });

  if (!parsed.success) {
    throw new StorageError("Episode stats row failed validation", {
      cause: parsed.error,
      code: "EPISODE_STATS_INVALID",
    });
  }

  return parsed.data;
}

export function createEpisodesTableSchema(dimensions: number) {
  return schema([
    utf8Field("id"),
    utf8Field("title"),
    utf8Field("narrative"),
    utf8Field("participants"),
    utf8Field("location", true),
    float64Field("start_time"),
    float64Field("end_time"),
    utf8Field("source_stream_ids"),
    float64Field("significance"),
    utf8Field("tags"),
    float64Field("confidence"),
    utf8Field("lineage_derived_from"),
    utf8Field("lineage_supersedes"),
    utf8Field("source_fingerprint", true),
    utf8Field("audience_entity_id", true),
    booleanField("shared", true),
    utf8Field("emotional_arc", true),
    vectorField("embedding", dimensions),
    float64Field("created_at"),
    float64Field("updated_at"),
  ]);
}

export type ReconciliationReport = {
  createdMissingStats: number;
  deletedOrphanStats: number;
  deletedOrphanRetrievalLogs: number;
  deletedOrphanValueSources: number;
};

export type EpisodicRepositoryOptions = {
  table: LanceDbTable;
  db: SqliteDatabase;
  clock?: Clock;
};

export type EpisodeGetOptions = {
  includeArchived?: boolean;
};

export class EpisodicRepository {
  private readonly clock: Clock;

  constructor(private readonly options: EpisodicRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get table(): LanceDbTable {
    return this.options.table;
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private deleteSqlRowsForEpisode(episodeId: EpisodeId): {
    deletedStats: number;
    deletedRetrievalLogs: number;
    deletedValueSources: number;
  } {
    const deleteIndex = this.db.prepare("DELETE FROM episode_index WHERE episode_id = ?");
    const deleteStats = this.db.prepare("DELETE FROM episode_stats WHERE episode_id = ?");
    const deleteRetrievalLog = this.db.prepare("DELETE FROM retrieval_log WHERE episode_id = ?");
    const deleteValueSources = this.db.prepare("DELETE FROM value_sources WHERE episode_id = ?");
    const apply = this.db.transaction((targetEpisodeId: EpisodeId) => {
      deleteIndex.run(targetEpisodeId);
      const deletedStats = deleteStats.run(targetEpisodeId).changes;
      const deletedRetrievalLogs = deleteRetrievalLog.run(targetEpisodeId).changes;
      const deletedValueSources = deleteValueSources.run(targetEpisodeId).changes;

      return {
        deletedStats,
        deletedRetrievalLogs,
        deletedValueSources,
      };
    });

    return apply(episodeId) as {
      deletedStats: number;
      deletedRetrievalLogs: number;
      deletedValueSources: number;
    };
  }

  private buildVisibilityWhereClause(
    audienceEntityId: EntityId | null | undefined,
    crossAudience = false,
    globalIdentitySelfAudienceEntityId?: EntityId | null,
  ): string | undefined {
    if (globalIdentitySelfAudienceEntityId !== undefined) {
      return globalIdentitySelfAudienceEntityId === null
        ? "audience_entity_id IS NULL"
        : `(audience_entity_id IS NULL OR audience_entity_id = ${quoteSqlString(
            globalIdentitySelfAudienceEntityId,
          )})`;
    }

    if (crossAudience) {
      return undefined;
    }

    if (audienceEntityId === null || audienceEntityId === undefined) {
      return "(audience_entity_id IS NULL OR shared = true)";
    }

    return `(audience_entity_id IS NULL OR audience_entity_id = ${quoteSqlString(
      audienceEntityId,
    )} OR shared = true)`;
  }

  private buildOptionsVisibilityWhereClause(options: EpisodeVisibilityOptions): string | undefined {
    return this.buildVisibilityWhereClause(
      options.audienceEntityId,
      options.crossAudience,
      options.globalIdentitySelfAudienceEntityId,
    );
  }

  private buildIndexedVisibilityWhereClause(
    options: EpisodeVisibilityOptions,
    alias: string,
  ): {
    sql: string;
    params: unknown[];
  } {
    if (options.globalIdentitySelfAudienceEntityId !== undefined) {
      return options.globalIdentitySelfAudienceEntityId === null
        ? {
            sql: `${alias}.audience_entity_id IS NULL`,
            params: [],
          }
        : {
            sql: `(${alias}.audience_entity_id IS NULL OR ${alias}.audience_entity_id = ?)`,
            params: [options.globalIdentitySelfAudienceEntityId],
          };
    }

    if (options.crossAudience === true) {
      return {
        sql: "1 = 1",
        params: [],
      };
    }

    if (options.audienceEntityId === null || options.audienceEntityId === undefined) {
      return {
        sql: `(${alias}.audience_entity_id IS NULL OR ${alias}.shared = 1)`,
        params: [],
      };
    }

    return {
      sql: `(${alias}.audience_entity_id IS NULL OR ${alias}.audience_entity_id = ? OR ${alias}.shared = 1)`,
      params: [options.audienceEntityId],
    };
  }

  private buildIndexedVisibilityBranches(
    options: EpisodeVisibilityOptions,
    order: IndexedEpisodeOrder,
  ): IndexedVisibilityBranch[] {
    const globalIndexName =
      order === "heat" ? "idx_episode_index_heat" : "idx_episode_index_recent";
    const audienceIndexName =
      order === "heat" ? "idx_episode_index_audience_heat" : "idx_episode_index_audience_recent";
    const sharedIndexName =
      order === "heat" ? "idx_episode_index_shared_heat" : "idx_episode_index_shared_recent";

    if (
      options.crossAudience === true &&
      options.globalIdentitySelfAudienceEntityId === undefined
    ) {
      return [
        {
          where: "archived = 0",
          params: [],
          indexName: globalIndexName,
        },
      ];
    }

    if (options.globalIdentitySelfAudienceEntityId !== undefined) {
      const publicBranch = {
        where: "archived = 0 AND audience_entity_id IS NULL",
        params: [],
        indexName: audienceIndexName,
      };

      if (options.globalIdentitySelfAudienceEntityId === null) {
        return [publicBranch];
      }

      return [
        publicBranch,
        {
          where: "archived = 0 AND audience_entity_id = ?",
          params: [options.globalIdentitySelfAudienceEntityId],
          indexName: audienceIndexName,
        },
      ];
    }

    const branches: IndexedVisibilityBranch[] = [
      {
        where: "archived = 0 AND audience_entity_id IS NULL",
        params: [],
        indexName: audienceIndexName,
      },
      {
        where: "archived = 0 AND shared = 1",
        params: [],
        indexName: sharedIndexName,
      },
    ];

    if (options.audienceEntityId !== null && options.audienceEntityId !== undefined) {
      branches.push({
        where: "archived = 0 AND audience_entity_id = ?",
        params: [options.audienceEntityId],
        indexName: audienceIndexName,
      });
    }

    return branches;
  }

  private async listEpisodesWhere(where: string | undefined): Promise<Episode[]> {
    const rows = await this.table.list(where === undefined ? {} : { where });
    return rows.map((row) => fromEpisodeRow(row));
  }

  async listVisibleEpisodes(
    options: EpisodeVisibilityOptions = {},
    extraWhere?: string,
  ): Promise<Episode[]> {
    return this.listEpisodesWhere(
      combineWhereClauses(this.buildOptionsVisibilityWhereClause(options), extraWhere),
    );
  }

  private computeEpisodeIndexHeatScore(updatedAt: number, stats: EpisodeStats): number {
    return computeEpisodeHeatForTimestamp(updatedAt, stats, this.clock.now());
  }

  private syncEpisodeIndexStats(stats: EpisodeStats): void {
    const row = this.db
      .prepare("SELECT updated_at FROM episode_index WHERE episode_id = ?")
      .get(stats.episode_id) as IndexedEpisodeStatsProjectionRow | undefined;

    if (row === undefined) {
      return;
    }

    this.db
      .prepare(
        `
          UPDATE episode_index
          SET retrieval_count = ?,
              win_rate = ?,
              last_retrieved = ?,
              tier = ?,
              archived = ?,
              heat_multiplier = ?,
              heat_score = ?
          WHERE episode_id = ?
        `,
      )
      .run(
        stats.retrieval_count,
        stats.win_rate,
        stats.last_retrieved,
        stats.tier,
        stats.archived ? 1 : 0,
        stats.heat_multiplier,
        this.computeEpisodeIndexHeatScore(Number(row.updated_at), stats),
        stats.episode_id,
      );
  }

  private upsertEpisodeIndex(episode: Episode, statsOverride?: EpisodeStats): void {
    const normalized = normalizeEpisodeAccess(episode);
    const stats = statsOverride ?? this.getStats(normalized.id) ?? defaultEpisodeStats(normalized);
    const heatScore = computeEpisodeHeat(normalized, stats, this.clock.now());

    this.db
      .prepare(
        `
          INSERT INTO episode_index (
            episode_id, audience_entity_id, shared, start_time, end_time, created_at, updated_at,
            retrieval_count, win_rate, last_retrieved, tier, archived, heat_multiplier, heat_score
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (episode_id) DO UPDATE SET
            audience_entity_id = excluded.audience_entity_id,
            shared = excluded.shared,
            start_time = excluded.start_time,
            end_time = excluded.end_time,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            retrieval_count = excluded.retrieval_count,
            win_rate = excluded.win_rate,
            last_retrieved = excluded.last_retrieved,
            tier = excluded.tier,
            archived = excluded.archived,
            heat_multiplier = excluded.heat_multiplier,
            heat_score = excluded.heat_score
        `,
      )
      .run(
        normalized.id,
        normalized.audience_entity_id,
        normalized.shared ? 1 : 0,
        normalized.start_time,
        normalized.end_time,
        normalized.created_at,
        normalized.updated_at,
        stats.retrieval_count,
        stats.win_rate,
        stats.last_retrieved,
        stats.tier,
        stats.archived ? 1 : 0,
        stats.heat_multiplier,
        heatScore,
      );

    this.db.prepare("DELETE FROM episode_participants WHERE episode_id = ?").run(normalized.id);
    this.db.prepare("DELETE FROM episode_tags WHERE episode_id = ?").run(normalized.id);

    const insertParticipant = this.db.prepare(
      "INSERT OR IGNORE INTO episode_participants (episode_id, term, value) VALUES (?, ?, ?)",
    );
    const insertTag = this.db.prepare(
      "INSERT OR IGNORE INTO episode_tags (episode_id, term, value) VALUES (?, ?, ?)",
    );

    for (const participant of uniqueStrings(normalized.participants)) {
      const term = normalizeTerm(participant);

      if (term.length > 0) {
        insertParticipant.run(normalized.id, term, participant);
      }
    }

    for (const tag of uniqueStrings(normalized.tags)) {
      const term = normalizeTerm(tag);

      if (term.length > 0) {
        insertTag.run(normalized.id, term, tag);
      }
    }
  }

  private isEpisodeIndexBackfilled(): boolean {
    return (
      this.db
        .prepare("SELECT 1 FROM episode_index_metadata WHERE key = ? LIMIT 1")
        .get(EPISODE_INDEX_BACKFILLED_KEY) !== undefined
    );
  }

  private markEpisodeIndexBackfilled(): void {
    this.db
      .prepare(
        `
          INSERT INTO episode_index_metadata (key, value)
          VALUES (?, ?)
          ON CONFLICT (key) DO UPDATE SET value = excluded.value
        `,
      )
      .run(EPISODE_INDEX_BACKFILLED_KEY, String(this.clock.now()));
  }

  private async ensureEpisodeIndexBackfilled(): Promise<void> {
    if (this.isEpisodeIndexBackfilled()) {
      return;
    }

    const episodes = await this.listEpisodesWhere(undefined);
    const statsById = this.getStatsMany(episodes.map((episode) => episode.id));
    const apply = this.db.transaction((backfillEpisodes: readonly Episode[]) => {
      for (const episode of backfillEpisodes) {
        const stats = statsById.get(episode.id) ?? defaultEpisodeStats(episode);

        if (!statsById.has(episode.id)) {
          this.upsertStats(stats);
        }

        this.upsertEpisodeIndex(episode, stats);
      }

      this.markEpisodeIndexBackfilled();
    });

    apply(episodes);
  }

  private queryVisibleIndexedEpisodeIds(
    options: EpisodeVisibilityOptions,
    order: IndexedEpisodeOrder,
    limit: number,
  ): EpisodeId[] {
    const branches = this.buildIndexedVisibilityBranches(options, order);
    const orderBy =
      order === "heat"
        ? "heat_score DESC, updated_at DESC, episode_id DESC"
        : "updated_at DESC, episode_id DESC";
    const branchSql = branches.map(
      (branch) => `
        SELECT episode_id, updated_at, heat_score
        FROM (
          SELECT episode_id, updated_at, heat_score
          FROM episode_index INDEXED BY ${branch.indexName}
          WHERE ${branch.where}
          ORDER BY ${orderBy}
          LIMIT ?
        )
      `,
    );
    const params = branches.flatMap((branch) => [...branch.params, limit]);
    const rows = this.db
      .prepare(
        `
          SELECT episode_id
          FROM (
            ${branchSql.join("\nUNION\n")}
          )
          ORDER BY ${orderBy}
          LIMIT ?
        `,
      )
      .all(...params, limit) as IndexedEpisodeIdRow[];

    return rows.map((row) => parseEpisodeId(row.episode_id));
  }

  private async hydrateCandidatesByIds(
    ids: readonly EpisodeId[],
  ): Promise<EpisodeSearchCandidate[]> {
    if (ids.length === 0) {
      return [];
    }

    const episodes = await this.getMany(ids);
    const episodeById = new Map(episodes.map((episode) => [episode.id, episode]));
    const orderedEpisodes = ids
      .map((id) => episodeById.get(id))
      .filter((episode): episode is Episode => episode !== undefined);
    const statsById = this.getStatsMany(orderedEpisodes.map((episode) => episode.id));

    return this.hydrateSearchCandidates(orderedEpisodes, statsById);
  }

  private hydrateSearchCandidates(
    episodes: readonly Episode[],
    statsById: ReadonlyMap<EpisodeId, EpisodeStats>,
    similarityById?: ReadonlyMap<EpisodeId, number>,
  ): EpisodeSearchCandidate[] {
    const results: EpisodeSearchCandidate[] = [];

    for (const episode of episodes) {
      const stats = statsById.get(episode.id) ?? defaultEpisodeStats(episode);

      if (stats.archived) {
        continue;
      }

      results.push({
        episode,
        stats,
        similarity: similarityById?.get(episode.id) ?? 0,
      });
    }

    return results;
  }

  private rankEpisodesByHeat(
    episodes: readonly Episode[],
    statsById: ReadonlyMap<EpisodeId, EpisodeStats>,
  ): Episode[] {
    const nowMs = this.clock.now();

    return [...episodes].sort((left, right) => {
      const leftStats = statsById.get(left.id) ?? defaultEpisodeStats(left);
      const rightStats = statsById.get(right.id) ?? defaultEpisodeStats(right);
      const leftHeat = computeEpisodeHeat(left, leftStats, nowMs);
      const rightHeat = computeEpisodeHeat(right, rightStats, nowMs);

      return rightHeat - leftHeat || compareEpisodes(left, right);
    });
  }

  private upsertStats(stats: EpisodeStats): void {
    const parsed = episodeStatsSchema.parse(stats);

    this.db
      .prepare(
        `
          INSERT INTO episode_stats (
            episode_id, retrieval_count, use_count, last_retrieved, win_rate, tier,
            promoted_at, promoted_from, gist, gist_generated_at, last_decayed_at,
            heat_multiplier, valence_mean, archived
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (episode_id) DO UPDATE SET
            retrieval_count = excluded.retrieval_count,
            use_count = excluded.use_count,
            last_retrieved = excluded.last_retrieved,
            win_rate = excluded.win_rate,
            tier = excluded.tier,
            promoted_at = excluded.promoted_at,
            promoted_from = excluded.promoted_from,
            gist = excluded.gist,
            gist_generated_at = excluded.gist_generated_at,
            last_decayed_at = excluded.last_decayed_at,
            heat_multiplier = excluded.heat_multiplier,
            valence_mean = excluded.valence_mean,
            archived = excluded.archived
        `,
      )
      .run(
        parsed.episode_id,
        parsed.retrieval_count,
        parsed.use_count,
        parsed.last_retrieved,
        parsed.win_rate,
        parsed.tier,
        parsed.promoted_at,
        parsed.promoted_from,
        parsed.gist,
        parsed.gist_generated_at,
        parsed.last_decayed_at,
        parsed.heat_multiplier,
        parsed.valence_mean,
        parsed.archived ? 1 : 0,
      );

    this.syncEpisodeIndexStats(parsed);
  }

  async insert(episode: Episode): Promise<Episode> {
    const parsed = episodeInsertSchema.safeParse(normalizeEpisodeAccess(episode));

    if (!parsed.success) {
      throw new StorageError("Invalid episode payload", {
        cause: parsed.error,
        code: "EPISODE_INVALID",
      });
    }

    if (parsed.data.source_stream_ids.length === 0) {
      throw new StorageError("Episodes must include at least one source stream id", {
        code: "EPISODE_SOURCE_ANCHOR_REQUIRED",
      });
    }

    try {
      await this.table.upsert([toEpisodeRow(parsed.data)], { on: "id" });

      try {
        const apply = this.db.transaction(() => {
          const stats = defaultEpisodeStats(parsed.data);

          this.upsertStats(stats);
          this.upsertEpisodeIndex(parsed.data, stats);
        });
        apply();
      } catch (error) {
        await this.table.remove(`id = ${quoteSqlString(parsed.data.id)}`);
        throw error;
      }

      return parsed.data;
    } catch (error) {
      throw new StorageError(`Failed to insert episode ${parsed.data.id}`, {
        cause: error,
        code: "EPISODE_INSERT_FAILED",
      });
    }
  }

  async get(id: EpisodeId, options: EpisodeGetOptions = {}): Promise<Episode | null> {
    const rows = await this.table.list({
      where: `id = ${quoteSqlString(id)}`,
      limit: 1,
    });
    const row = rows[0];

    if (row === undefined) {
      return null;
    }

    const episode = fromEpisodeRow(row);
    const stats = this.getStats(id);

    if (options.includeArchived !== true && (stats?.archived ?? false)) {
      return null;
    }

    return episode;
  }

  async getMany(ids: readonly EpisodeId[]): Promise<Episode[]> {
    if (ids.length === 0) {
      return [];
    }

    const where = `id IN (${ids.map((id) => quoteSqlString(id)).join(", ")})`;
    const rows = await this.table.list({ where, limit: ids.length });
    const episodeById = new Map(rows.map((row) => [String(row.id), fromEpisodeRow(row)]));
    return ids
      .map((id) => episodeById.get(id))
      .filter((value): value is Episode => value !== undefined);
  }

  async update(id: EpisodeId, patch: EpisodePatch): Promise<Episode | null> {
    const current = await this.get(id, { includeArchived: true });

    if (current === null) {
      return null;
    }

    const parsedPatch = episodePatchSchema.safeParse(patch);

    if (!parsedPatch.success) {
      throw new StorageError("Invalid episode patch payload", {
        cause: parsedPatch.error,
        code: "EPISODE_PATCH_INVALID",
      });
    }

    const patchIncludesEmotionalArc = Object.prototype.hasOwnProperty.call(patch, "emotional_arc");
    const merged = normalizeEpisodeAccess({
      ...current,
      ...parsedPatch.data,
      emotional_arc: patchIncludesEmotionalArc
        ? (parsedPatch.data.emotional_arc ?? null)
        : current.emotional_arc,
      lineage: {
        ...current.lineage,
        ...parsedPatch.data.lineage,
      },
      updated_at: this.clock.now(),
    });
    const parsedEpisode = episodeSchema.safeParse(merged);

    if (!parsedEpisode.success) {
      throw new StorageError("Failed to update episode", {
        cause: parsedEpisode.error,
        code: "EPISODE_PATCH_INVALID",
      });
    }

    const previousRow = toEpisodeRow(current);

    try {
      await this.table.upsert([toEpisodeRow(parsedEpisode.data)], { on: "id" });

      try {
        const apply = this.db.transaction(() => {
          const stats = this.updateStats(id, {
            valence_mean:
              parsedEpisode.data.emotional_arc === null
                ? 0
                : (parsedEpisode.data.emotional_arc.start.valence +
                    parsedEpisode.data.emotional_arc.peak.valence +
                    parsedEpisode.data.emotional_arc.end.valence) /
                  3,
          });
          this.upsertEpisodeIndex(parsedEpisode.data, stats);
        });
        apply();
      } catch (error) {
        const currentAfterFailure = await this.get(id, { includeArchived: true });

        if (currentAfterFailure?.updated_at === parsedEpisode.data.updated_at) {
          await this.table.upsert([previousRow], { on: "id" });
        } else {
          console.warn("Skipped episode rollback because newer Lance state exists.", {
            episodeId: id,
            attemptedUpdatedAt: parsedEpisode.data.updated_at,
            currentUpdatedAt: currentAfterFailure?.updated_at ?? null,
          });
        }
        throw error;
      }

      return parsedEpisode.data;
    } catch (error) {
      throw new StorageError(`Failed to update episode ${id}`, {
        cause: error,
        code: "EPISODE_UPDATE_FAILED",
      });
    }
  }

  async updateSignificance(id: EpisodeId, significance: number): Promise<Episode | null> {
    const current = await this.get(id, { includeArchived: true });

    if (current === null) {
      return null;
    }

    const parsedEpisode = episodeSchema.safeParse({
      ...current,
      significance,
    });

    if (!parsedEpisode.success) {
      throw new StorageError("Invalid episode significance patch", {
        cause: parsedEpisode.error,
        code: "EPISODE_PATCH_INVALID",
      });
    }

    try {
      await this.table.upsert([toEpisodeRow(parsedEpisode.data)], { on: "id" });
      return parsedEpisode.data;
    } catch (error) {
      throw new StorageError(`Failed to update episode significance ${id}`, {
        cause: error,
        code: "EPISODE_UPDATE_FAILED",
      });
    }
  }

  async delete(id: EpisodeId): Promise<boolean> {
    const existing = await this.get(id, { includeArchived: true });

    if (existing === null) {
      return false;
    }

    try {
      await this.table.remove(`id = ${quoteSqlString(id)}`);

      try {
        this.deleteSqlRowsForEpisode(id);
      } catch (error) {
        console.warn("Episode delete left orphaned SQLite rows for reconciliation.", {
          episodeId: id,
          error,
        });
        throw error;
      }

      return true;
    } catch (error) {
      throw new StorageError(`Failed to delete episode ${id}`, {
        cause: error,
        code: "EPISODE_DELETE_FAILED",
      });
    }
  }

  async reconcileCrossStoreState(): Promise<ReconciliationReport> {
    const episodes = await this.listEpisodesWhere(undefined);
    const episodeIds = new Set(episodes.map((episode) => episode.id));
    const statsRows = this.db
      .prepare("SELECT episode_id FROM episode_stats ORDER BY episode_id ASC")
      .all() as Array<{ episode_id: string }>;
    const retrievalLogRows = this.db
      .prepare("SELECT DISTINCT episode_id FROM retrieval_log ORDER BY episode_id ASC")
      .all() as Array<{ episode_id: string }>;
    const valueSourceRows = this.db
      .prepare("SELECT DISTINCT episode_id FROM value_sources ORDER BY episode_id ASC")
      .all() as Array<{ episode_id: string }>;
    const indexRows = this.db
      .prepare("SELECT episode_id FROM episode_index ORDER BY episode_id ASC")
      .all() as Array<{ episode_id: string }>;
    const statsIdSet = new Set(statsRows.map((row) => parseEpisodeId(row.episode_id)));
    const referencedSqlEpisodeIds = new Set<EpisodeId>([
      ...statsRows.map((row) => parseEpisodeId(row.episode_id)),
      ...retrievalLogRows.map((row) => parseEpisodeId(row.episode_id)),
      ...valueSourceRows.map((row) => parseEpisodeId(row.episode_id)),
      ...indexRows.map((row) => parseEpisodeId(row.episode_id)),
    ]);
    const missingStats = episodes.filter((episode) => !statsIdSet.has(episode.id));
    const orphanEpisodeIds = [...referencedSqlEpisodeIds].filter(
      (episodeId) => !episodeIds.has(episodeId),
    );
    let createdMissingStats = 0;
    let deletedOrphanStats = 0;
    let deletedOrphanRetrievalLogs = 0;
    let deletedOrphanValueSources = 0;

    if (missingStats.length > 0) {
      const apply = this.db.transaction((episodesWithoutStats: readonly Episode[]) => {
        for (const episode of episodesWithoutStats) {
          this.upsertStats(defaultEpisodeStats(episode));
        }
      });
      apply(missingStats);
      createdMissingStats = missingStats.length;
    }

    const statsById = this.getStatsMany(episodes.map((episode) => episode.id));
    const syncIndex = this.db.transaction((indexedEpisodes: readonly Episode[]) => {
      for (const episode of indexedEpisodes) {
        this.upsertEpisodeIndex(episode, statsById.get(episode.id) ?? defaultEpisodeStats(episode));
      }

      this.markEpisodeIndexBackfilled();
    });
    syncIndex(episodes);

    for (const episodeId of orphanEpisodeIds) {
      const deleted = this.deleteSqlRowsForEpisode(episodeId);
      deletedOrphanStats += deleted.deletedStats;
      deletedOrphanRetrievalLogs += deleted.deletedRetrievalLogs;
      deletedOrphanValueSources += deleted.deletedValueSources;
    }

    return {
      createdMissingStats,
      deletedOrphanStats,
      deletedOrphanRetrievalLogs,
      deletedOrphanValueSources,
    };
  }

  getStats(id: EpisodeId): EpisodeStats | null {
    const row = this.db
      .prepare(
        `
          SELECT
            episode_id, retrieval_count, use_count, last_retrieved, win_rate, tier,
            promoted_at, promoted_from, gist, gist_generated_at, last_decayed_at,
            heat_multiplier, valence_mean, archived
          FROM episode_stats
          WHERE episode_id = ?
        `,
      )
      .get(id) as Record<string, unknown> | undefined;

    return row === undefined ? null : fromEpisodeStatsRow(row);
  }

  getStatsMany(ids: readonly EpisodeId[]): Map<EpisodeId, EpisodeStats> {
    const uniqueIds = [...new Set(ids)];

    if (uniqueIds.length === 0) {
      return new Map();
    }

    const rows = this.db
      .prepare(
        `
          SELECT
            episode_id, retrieval_count, use_count, last_retrieved, win_rate, tier,
            promoted_at, promoted_from, gist, gist_generated_at, last_decayed_at,
            heat_multiplier, valence_mean, archived
          FROM episode_stats
          WHERE episode_id IN (${uniqueIds.map(() => "?").join(", ")})
        `,
      )
      .all(...uniqueIds) as Record<string, unknown>[];

    return new Map(
      rows.map((row) => {
        const stats = fromEpisodeStatsRow(row);
        return [stats.episode_id, stats] as const;
      }),
    );
  }

  updateStats(episodeId: EpisodeId, patch: EpisodeStatsPatch): EpisodeStats {
    const current = this.getStats(episodeId);

    if (current === null) {
      throw new StorageError(`Missing episode_stats row for ${episodeId}`, {
        code: "EPISODE_STATS_MISSING",
      });
    }

    const parsedPatch = episodeStatsPatchSchema.safeParse(patch);

    if (!parsedPatch.success) {
      throw new StorageError("Invalid episode stats patch", {
        cause: parsedPatch.error,
        code: "EPISODE_STATS_PATCH_INVALID",
      });
    }

    const next = {
      ...current,
      ...parsedPatch.data,
    };
    const parsed = episodeStatsSchema.parse(next);
    this.upsertStats(parsed);
    return parsed;
  }

  listStats(): EpisodeStats[] {
    const rows = this.db
      .prepare(
        `
          SELECT
            episode_id, retrieval_count, use_count, last_retrieved, win_rate, tier,
            promoted_at, promoted_from, gist, gist_generated_at, last_decayed_at,
            heat_multiplier, valence_mean, archived
          FROM episode_stats
          ORDER BY promoted_at DESC, episode_id ASC
        `,
      )
      .all() as Record<string, unknown>[];

    return rows.map((row) => {
      const stats = fromEpisodeStatsRow(row);
      return {
        ...stats,
        episode_id: parseEpisodeId(String(stats.episode_id)),
      };
    });
  }

  async searchByVector(
    vector: Float32Array,
    options: EpisodeSearchOptions = {},
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "Search limit");
    const searchLimit = Math.max(limit * 5, limit, 20);
    const rows = await this.table.search(Array.from(vector), {
      limit: searchLimit,
      vectorColumn: "embedding",
      distanceType: "cosine",
      where: this.buildOptionsVisibilityWhereClause(options),
    });
    const ranked = rows.map((row) => {
      const episode = fromEpisodeRow(row);
      return {
        episode,
        similarity: toSimilarity(getDistance(row)),
      };
    });
    const statsById = this.getStatsMany(ranked.map((item) => item.episode.id));
    const results: EpisodeSearchCandidate[] = [];

    for (const item of ranked) {
      const episode = item.episode;
      const stats = statsById.get(episode.id) ?? defaultEpisodeStats(episode);
      const similarity = item.similarity;

      if (
        options.globalIdentitySelfAudienceEntityId !== undefined &&
        !isEpisodeInGlobalIdentityScope(episode, options.globalIdentitySelfAudienceEntityId)
      ) {
        continue;
      }

      if (options.minSimilarity !== undefined && similarity < options.minSimilarity) {
        continue;
      }

      if (
        options.globalIdentitySelfAudienceEntityId === undefined &&
        !isEpisodeVisibleToAudience(episode, options.audienceEntityId, {
          crossAudience: options.crossAudience,
        })
      ) {
        continue;
      }

      if (
        options.tagFilter !== undefined &&
        options.tagFilter.length > 0 &&
        !options.tagFilter.every((tag) => episode.tags.includes(tag))
      ) {
        continue;
      }

      if (
        options.tierFilter !== undefined &&
        options.tierFilter.length > 0 &&
        !options.tierFilter.includes(stats.tier)
      ) {
        continue;
      }

      if (stats.archived) {
        continue;
      }

      if (options.timeRange !== undefined) {
        const overlaps =
          episode.start_time <= options.timeRange.end &&
          episode.end_time >= options.timeRange.start;

        if (!overlaps) {
          continue;
        }
      }

      results.push({
        episode,
        stats,
        similarity,
      });

      if (results.length >= limit) {
        break;
      }
    }

    return results;
  }

  async searchByTimeRange(
    range: { start: number; end: number },
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "Search limit");
    await this.ensureEpisodeIndexBackfilled();

    const visibility = this.buildIndexedVisibilityWhereClause(options, "ei");
    const clauses = ["ei.archived = 0", visibility.sql];
    const params: unknown[] = [...visibility.params];

    if (Number.isFinite(range.end)) {
      clauses.push("ei.start_time <= ?");
      params.push(range.end);
    }

    if (Number.isFinite(range.start)) {
      clauses.push("ei.end_time >= ?");
      params.push(range.start);
    }

    const rows = this.db
      .prepare(
        `
          SELECT ei.episode_id
          FROM episode_index AS ei INDEXED BY idx_episode_index_time_start
          WHERE ${clauses.join(" AND ")}
          ORDER BY ei.updated_at DESC, ei.episode_id DESC
          LIMIT ?
        `,
      )
      .all(...params, limit) as IndexedEpisodeIdRow[];

    return this.hydrateCandidatesByIds(rows.map((row) => parseEpisodeId(row.episode_id)));
  }

  async listByAudience(
    audienceEntityId: EntityId,
    options: {
      limit?: number;
      orderBy: "recent" | "heat";
    },
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "List limit");
    await this.ensureEpisodeIndexBackfilled();

    const orderBy =
      options.orderBy === "heat"
        ? "heat_score DESC, updated_at DESC, episode_id DESC"
        : "updated_at DESC, episode_id DESC";
    const indexName =
      options.orderBy === "heat"
        ? "idx_episode_index_audience_heat"
        : "idx_episode_index_audience_recent";
    const rows = this.db
      .prepare(
        `
          SELECT episode_id
          FROM episode_index INDEXED BY ${indexName}
          WHERE archived = 0 AND audience_entity_id = ?
          ORDER BY ${orderBy}
          LIMIT ?
        `,
      )
      .all(audienceEntityId, limit) as IndexedEpisodeIdRow[];

    return this.hydrateCandidatesByIds(rows.map((row) => parseEpisodeId(row.episode_id)));
  }

  async searchByParticipantsOrTags(
    terms: readonly string[],
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "Search limit");
    const normalizedTerms = new Set(
      terms.map((term) => normalizeTerm(term)).filter((term) => term.length > 0),
    );

    if (normalizedTerms.size === 0) {
      return [];
    }

    await this.ensureEpisodeIndexBackfilled();

    const normalizedTermList = [...normalizedTerms];
    const termPlaceholders = sqlPlaceholders(normalizedTermList.length);
    const visibility = this.buildIndexedVisibilityWhereClause(options, "ei");
    const visibilityParams = visibility.params;
    const rows = this.db
      .prepare(
        `
          SELECT episode_id
          FROM (
            SELECT ei.episode_id, ei.updated_at
            FROM episode_participants AS ep INDEXED BY idx_episode_participants_term
            JOIN episode_index AS ei ON ei.episode_id = ep.episode_id
            WHERE ep.term IN (${termPlaceholders})
              AND ei.archived = 0
              AND ${visibility.sql}
            UNION
            SELECT ei.episode_id, ei.updated_at
            FROM episode_tags AS et INDEXED BY idx_episode_tags_term
            JOIN episode_index AS ei ON ei.episode_id = et.episode_id
            WHERE et.term IN (${termPlaceholders})
              AND ei.archived = 0
              AND ${visibility.sql}
          )
          ORDER BY updated_at DESC, episode_id DESC
          LIMIT ?
        `,
      )
      .all(
        ...normalizedTermList,
        ...visibilityParams,
        ...normalizedTermList,
        ...visibilityParams,
        limit,
      ) as IndexedEpisodeIdRow[];

    return this.hydrateCandidatesByIds(rows.map((row) => parseEpisodeId(row.episode_id)));
  }

  async listRecent(
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "List limit");
    await this.ensureEpisodeIndexBackfilled();
    return this.hydrateCandidatesByIds(
      this.queryVisibleIndexedEpisodeIds(options, "recent", limit),
    );
  }

  async listHottest(
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "List limit");
    await this.ensureEpisodeIndexBackfilled();
    return this.hydrateCandidatesByIds(this.queryVisibleIndexedEpisodeIds(options, "heat", limit));
  }

  async list(options: EpisodeListOptions = {}): Promise<EpisodeListResult> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_LIST_LIMIT, "List limit");
    const cursor = options.cursor === undefined ? undefined : decodeCursor(options.cursor);
    const rows = await this.table.list();
    const episodes = rows.map((row) => fromEpisodeRow(row)).sort(compareEpisodes);
    const filtered =
      cursor === undefined
        ? episodes
        : episodes.filter((episode) => compareAfterCursor(episode, cursor));
    const items = filtered.slice(0, limit);
    const lastItem = items.at(-1);

    return {
      items,
      nextCursor:
        filtered.length > limit && lastItem !== undefined
          ? encodeCursor({ updatedAt: lastItem.updated_at, id: lastItem.id })
          : undefined,
    };
  }

  async listAll(): Promise<Episode[]> {
    const rows = await this.table.list();
    return rows.map((row) => fromEpisodeRow(row)).sort(compareEpisodes);
  }

  recordRetrieval(episodeId: EpisodeId, timestamp: number, score: number): void {
    const apply = this.db.transaction(() => {
      this.db
        .prepare("INSERT INTO retrieval_log (episode_id, timestamp, score) VALUES (?, ?, ?)")
        .run(episodeId, timestamp, score);

      this.db
        .prepare(
          `
            UPDATE episode_stats
            SET retrieval_count = retrieval_count + 1,
                last_retrieved = ?
            WHERE episode_id = ?
          `,
        )
        .run(timestamp, episodeId);

      const stats = this.getStats(episodeId);

      if (stats !== null) {
        this.syncEpisodeIndexStats(stats);
      }
    });

    apply();
  }

  mergeEpisodeFields(current: Episode, patch: Partial<Episode>): Episode {
    const merged = normalizeEpisodeAccess({
      ...current,
      ...patch,
      participants:
        patch.participants === undefined
          ? current.participants
          : uniqueStrings([...current.participants, ...patch.participants]),
      source_stream_ids:
        patch.source_stream_ids === undefined
          ? current.source_stream_ids
          : uniqueStrings([...current.source_stream_ids, ...patch.source_stream_ids]),
      tags:
        patch.tags === undefined ? current.tags : uniqueStrings([...current.tags, ...patch.tags]),
      lineage: {
        derived_from: uniqueStrings([
          ...current.lineage.derived_from,
          ...(patch.lineage?.derived_from ?? []),
        ]) as Episode["lineage"]["derived_from"],
        supersedes: uniqueStrings([
          ...current.lineage.supersedes,
          ...(patch.lineage?.supersedes ?? []),
        ]) as Episode["lineage"]["supersedes"],
      },
      updated_at: this.clock.now(),
    });

    return episodeSchema.parse(merged);
  }

  async findBySourceStreamIds(
    sourceStreamIds: ReadonlyArray<Episode["source_stream_ids"][number]>,
  ): Promise<Episode | null> {
    const fingerprint = buildSourceFingerprint(sourceStreamIds);
    const byFingerprint = await this.table.list({
      where: `source_fingerprint = ${quoteSqlString(fingerprint)}`,
      limit: 1,
    });
    const fingerprintMatch = byFingerprint[0];

    if (fingerprintMatch !== undefined) {
      return fromEpisodeRow(fingerprintMatch);
    }

    const legacyRows = await this.table.list();

    for (const row of legacyRows) {
      if (row.source_fingerprint !== null && row.source_fingerprint !== undefined) {
        continue;
      }

      const episode = fromEpisodeRow(row);

      if (buildSourceFingerprint(episode.source_stream_ids) === fingerprint) {
        return episode;
      }
    }

    return null;
  }
}
