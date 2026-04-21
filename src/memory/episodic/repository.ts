import { Buffer } from "node:buffer";

import {
  LanceDbTable,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import { parseEpisodeId, type EpisodeId } from "../../util/ids.js";
import { createNeutralEmotionalArc, emotionalArcSchema } from "../affective/types.js";

import {
  type Episode,
  type EpisodeListOptions,
  type EpisodeListResult,
  type EpisodePatch,
  type EpisodeSearchCandidate,
  type EpisodeSearchOptions,
  type EpisodeStats,
  type EpisodeStatsPatch,
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

const DEFAULT_LIST_LIMIT = 20;
const DEFAULT_SEARCH_LIMIT = 10;

function assertPositiveLimit(limit: number | undefined, label: string): number {
  const resolved = limit ?? DEFAULT_LIST_LIMIT;

  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new StorageError(`${label} must be a positive integer`);
  }

  return resolved;
}

function quoteSqlString(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

function uniqueStrings(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function parseJsonArray<T>(value: string, label: string): T[] {
  try {
    const parsed = JSON.parse(value) as unknown;

    if (!Array.isArray(parsed)) {
      throw new TypeError(`${label} must be an array`);
    }

    return parsed as T[];
  } catch (error) {
    throw new StorageError(`Failed to decode episode ${label}`, {
      cause: error,
      code: "EPISODE_ROW_INVALID",
    });
  }
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

function toFloat32Array(vector: unknown): Float32Array {
  if (vector instanceof Float32Array) {
    return vector;
  }

  const candidate = Array.isArray(vector)
    ? vector
    : ArrayBuffer.isView(vector)
      ? Array.from(vector as unknown as ArrayLike<number>)
      : vector !== null &&
          typeof vector === "object" &&
          "length" in vector &&
          typeof vector.length === "number"
        ? Array.from(vector as ArrayLike<number>)
        : null;

  if (candidate === null) {
    throw new StorageError("Episode row embedding must be array-like", {
      code: "EPISODE_ROW_INVALID",
    });
  }

  const values = candidate.map((value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new StorageError("Episode row embedding contains a non-finite value", {
        code: "EPISODE_ROW_INVALID",
      });
    }

    return value;
  });

  return Float32Array.from(values);
}

function toEpisodeRow(episode: Episode): EpisodeRow {
  return {
    id: episode.id,
    title: episode.title,
    narrative: episode.narrative,
    participants: serializeJsonValue(episode.participants),
    location: episode.location,
    start_time: episode.start_time,
    end_time: episode.end_time,
    source_stream_ids: serializeJsonValue(episode.source_stream_ids),
    significance: episode.significance,
    tags: serializeJsonValue(episode.tags),
    confidence: episode.confidence,
    lineage_derived_from: serializeJsonValue(episode.lineage.derived_from),
    lineage_supersedes: serializeJsonValue(episode.lineage.supersedes),
    emotional_arc:
      episode.emotional_arc === null ? null : serializeJsonValue(episode.emotional_arc),
    embedding: Array.from(episode.embedding),
    created_at: episode.created_at,
    updated_at: episode.updated_at,
  };
}

function fromEpisodeRow(row: Record<string, unknown>): Episode {
  const emotionalArc = (() => {
    if (row.emotional_arc === null || row.emotional_arc === undefined || row.emotional_arc === "") {
      return createNeutralEmotionalArc();
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
    participants: parseJsonArray<string>(String(row.participants ?? "[]"), "participants"),
    location: row.location === null || row.location === undefined ? null : String(row.location),
    start_time: Number(row.start_time),
    end_time: Number(row.end_time),
    source_stream_ids: parseJsonArray<string>(
      String(row.source_stream_ids ?? "[]"),
      "source_stream_ids",
    ),
    significance: Number(row.significance),
    tags: parseJsonArray<string>(String(row.tags ?? "[]"), "tags"),
    confidence: Number(row.confidence),
    lineage: {
      derived_from: parseJsonArray<string>(
        String(row.lineage_derived_from ?? "[]"),
        "lineage.derived_from",
      ),
      supersedes: parseJsonArray<string>(
        String(row.lineage_supersedes ?? "[]"),
        "lineage.supersedes",
      ),
    },
    emotional_arc: emotionalArc,
    embedding: toFloat32Array(row.embedding),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  };
  const parsed = episodeSchema.safeParse(candidate);

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
    valence_mean: valenceMean,
    archived: false,
  };
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
    utf8Field("emotional_arc", true),
    vectorField("embedding", dimensions),
    float64Field("created_at"),
    float64Field("updated_at"),
  ]);
}

export type EpisodicRepositoryOptions = {
  table: LanceDbTable;
  db: SqliteDatabase;
  clock?: Clock;
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

  private upsertStats(stats: EpisodeStats): void {
    const parsed = episodeStatsSchema.parse(stats);

    this.db
      .prepare(
        `
          INSERT INTO episode_stats (
            episode_id, retrieval_count, use_count, last_retrieved, win_rate, tier,
            promoted_at, promoted_from, gist, gist_generated_at, last_decayed_at, valence_mean, archived
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        parsed.valence_mean,
        parsed.archived ? 1 : 0,
      );
  }

  async insert(episode: Episode): Promise<Episode> {
    const parsed = episodeInsertSchema.safeParse(episode);

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

    await this.table.upsert([toEpisodeRow(parsed.data)], { on: "id" });
    this.upsertStats(defaultEpisodeStats(parsed.data));
    return parsed.data;
  }

  async get(id: EpisodeId): Promise<Episode | null> {
    const rows = await this.table.list({
      where: `id = ${quoteSqlString(id)}`,
      limit: 1,
    });
    const row = rows[0];
    return row === undefined ? null : fromEpisodeRow(row);
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
    const current = await this.get(id);

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
    const merged = {
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
    };
    const parsedEpisode = episodeSchema.safeParse(merged);

    if (!parsedEpisode.success) {
      throw new StorageError("Failed to update episode", {
        cause: parsedEpisode.error,
        code: "EPISODE_PATCH_INVALID",
      });
    }

    await this.table.upsert([toEpisodeRow(parsedEpisode.data)], { on: "id" });
    this.updateStats(id, {
      valence_mean:
        parsedEpisode.data.emotional_arc === null
          ? 0
          : (parsedEpisode.data.emotional_arc.start.valence +
              parsedEpisode.data.emotional_arc.peak.valence +
              parsedEpisode.data.emotional_arc.end.valence) /
            3,
    });
    return parsedEpisode.data;
  }

  async delete(id: EpisodeId): Promise<boolean> {
    const existing = await this.get(id);

    if (existing === null) {
      return false;
    }

    await this.table.remove(`id = ${quoteSqlString(id)}`);
    this.db.prepare("DELETE FROM episode_stats WHERE episode_id = ?").run(id);
    this.db.prepare("DELETE FROM retrieval_log WHERE episode_id = ?").run(id);
    this.db.prepare("DELETE FROM value_sources WHERE episode_id = ?").run(id);
    return true;
  }

  getStats(id: EpisodeId): EpisodeStats | null {
    const row = this.db
      .prepare(
        `
          SELECT
            episode_id, retrieval_count, use_count, last_retrieved, win_rate, tier,
            promoted_at, promoted_from, gist, gist_generated_at, last_decayed_at, valence_mean, archived
          FROM episode_stats
          WHERE episode_id = ?
        `,
      )
      .get(id) as Record<string, unknown> | undefined;

    if (row === undefined) {
      return null;
    }

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
            promoted_at, promoted_from, gist, gist_generated_at, last_decayed_at, valence_mean, archived
          FROM episode_stats
          ORDER BY promoted_at DESC, episode_id ASC
        `,
      )
      .all() as Record<string, unknown>[];

    return rows.map((row) => {
      const episodeId = parseEpisodeId(String(row.episode_id));
      const stats = this.getStats(episodeId);

      if (stats === null) {
        throw new StorageError(`Missing episode_stats row for ${episodeId}`, {
          code: "EPISODE_STATS_MISSING",
        });
      }

      return stats;
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
    });
    const results: EpisodeSearchCandidate[] = [];

    for (const row of rows) {
      const episode = fromEpisodeRow(row);
      const stats = this.getStats(episode.id) ?? defaultEpisodeStats(episode);
      const similarity = toSimilarity(getDistance(row));

      if (options.minSimilarity !== undefined && similarity < options.minSimilarity) {
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
    });

    apply();
  }

  mergeEpisodeFields(current: Episode, patch: Partial<Episode>): Episode {
    const merged = {
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
    };

    return episodeSchema.parse(merged);
  }
}
