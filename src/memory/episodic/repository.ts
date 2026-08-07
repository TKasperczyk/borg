import { Buffer } from "node:buffer";

import {
  LanceDbTable,
  booleanField,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import { getDistance, toSimilarity } from "../../storage/lancedb/vector-results.js";
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
import {
  createMaintenanceRunId,
  parseConsolidationFamilyId,
  parseEntityId,
  parseEpisodeId,
  parseStreamEntryId,
  type ConsolidationFamilyId,
  type EntityId,
  type EpisodeId,
  type MaintenanceRunId,
} from "../../util/ids.js";
import { emotionalArcSchema } from "../affective/types.js";
import { computeEpisodeHeat, computeEpisodeHeatForTimestamp } from "./heat.js";
import {
  isEpisodeVisibleToCapability,
  normalizeEpisodeAccess,
  resolveViewerCapability,
  type ViewerCapability,
} from "./access.js";

import {
  type Episode,
  type EpisodeCognitionRecallOptions,
  type EpisodeListOptions,
  type EpisodeListResult,
  type EpisodePatch,
  type EpisodeSearchCandidate,
  type EpisodeSearchOptions,
  type EpisodeStats,
  type EpisodeStatsPatch,
  type EpisodeVisibilityOptions,
  type EpisodeKind,
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
  origin_audience_entity_ids: string | null;
  shared: boolean | number | null;
  emotional_arc: string | null;
  episode_kind: string | null;
  consolidation_family_id: string | null;
  consolidation_coverage_hash: string | null;
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
type EpisodeSearchVisibilityMode = "cognition" | "disclosure";

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

type EpisodeEffectiveVisibilityRow = {
  episode_id: string;
};

type ConsolidationFamilyRow = {
  family_id: string;
  current_version_episode_id: string;
  coverage_hash: string;
  policy_version: number;
  created_at: number;
  updated_at: number;
};

type ConsolidationMemberRow = {
  family_id: string;
  raw_episode_id: string;
  source_stream_ids_json: string;
  added_by_version_episode_id: string;
};

export type EpisodeLifecycleAuditInput = {
  caller: string;
  reason: string;
  process: string;
  runId?: MaintenanceRunId;
};

export type ConsolidationFamilyRecord = {
  family_id: ConsolidationFamilyId;
  current_version_episode_id: EpisodeId;
  coverage_hash: string;
  policy_version: number;
  created_at: number;
  updated_at: number;
};

export type ConsolidationMemberRecord = {
  family_id: ConsolidationFamilyId;
  raw_episode_id: EpisodeId;
  source_stream_ids: Episode["source_stream_ids"];
  added_by_version_episode_id: EpisodeId;
};

export type ConsolidationMemberInput = {
  raw_episode_id: EpisodeId;
  source_stream_ids: Episode["source_stream_ids"];
  added_by_version_episode_id: EpisodeId;
};

type EpisodeStatsPatchKey = keyof EpisodeStatsPatch;

export const HOT_LANE_RETRIEVAL_COOLDOWN_MS = 5 * 60 * 1000;
export const HOT_LANE_RETRIEVAL_CANDIDATE_BUFFER = 20;
export const HOT_LANE_RETRIEVAL_CANDIDATE_MULTIPLIER = 5;

const DEFAULT_LIST_LIMIT = 20;
const DEFAULT_SEARCH_LIMIT = 10;
const EPISODE_INDEX_BACKFILLED_KEY = "lance_backfilled_at";
const EPISODE_STATS_PATCH_COLUMNS = {
  retrieval_count: "retrieval_count",
  use_count: "use_count",
  last_retrieved: "last_retrieved",
  win_rate: "win_rate",
  tier: "tier",
  promoted_at: "promoted_at",
  promoted_from: "promoted_from",
  gist: "gist",
  gist_generated_at: "gist_generated_at",
  last_decayed_at: "last_decayed_at",
  heat_multiplier: "heat_multiplier",
  valence_mean: "valence_mean",
  archived: "archived",
} satisfies Record<EpisodeStatsPatchKey, string>;
const EPISODE_LIFECYCLE_AUDIT_PROCESSES = [
  "consolidator",
  "reflector",
  "semantic-extractor",
  "curator",
  "overseer",
  "associator",
  "review-resolver",
  "ruminator",
  "self-narrator",
  "procedural-synthesizer",
  "belief-reviser",
  "creator-directive-reconciler",
  "commitment-reconciler",
  "episodic-repository",
  "correction",
] as const;
const EPISODE_LIFECYCLE_AUDIT_PROCESS_SET: ReadonlySet<string> = new Set(
  EPISODE_LIFECYCLE_AUDIT_PROCESSES,
);
type EpisodeLifecycleAuditProcess = (typeof EPISODE_LIFECYCLE_AUDIT_PROCESSES)[number];
const EPISODE_JSON_ARRAY_CODEC = {
  errorCode: "EPISODE_ROW_INVALID",
  errorMessage: (label: string) => `Failed to decode episode ${label}`,
} satisfies JsonArrayCodecOptions;
const EPISODE_VECTOR_CODEC = {
  arrayLikeErrorMessage: "Episode row embedding must be array-like",
  nonFiniteErrorMessage: "Episode row embedding contains a non-finite value",
  errorCode: "EPISODE_ROW_INVALID",
} satisfies Float32ArrayCodecOptions;

function encodeEpisodeStatsPatchValue(key: EpisodeStatsPatchKey, value: unknown): unknown {
  return key === "archived" ? ((value as boolean) ? 1 : 0) : value;
}

function resolveEpisodeLifecycleAuditProcess(process: string): EpisodeLifecycleAuditProcess {
  return EPISODE_LIFECYCLE_AUDIT_PROCESS_SET.has(process)
    ? (process as EpisodeLifecycleAuditProcess)
    : "episodic-repository";
}

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

export function buildConsolidationCoverageHash(sourceStreamIds: readonly string[]): string {
  return buildSourceFingerprint(sourceStreamIds);
}

function fromConsolidationFamilyRow(row: ConsolidationFamilyRow): ConsolidationFamilyRecord {
  return {
    family_id: parseConsolidationFamilyId(row.family_id),
    current_version_episode_id: parseEpisodeId(row.current_version_episode_id),
    coverage_hash: row.coverage_hash,
    policy_version: Number(row.policy_version),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
  };
}

function fromConsolidationMemberRow(row: ConsolidationMemberRow): ConsolidationMemberRecord {
  return {
    family_id: parseConsolidationFamilyId(row.family_id),
    raw_episode_id: parseEpisodeId(row.raw_episode_id),
    source_stream_ids: parseJsonArray<string>(
      row.source_stream_ids_json,
      "consolidation member source stream ids",
      EPISODE_JSON_ARRAY_CODEC,
    ).map((sourceStreamId) => parseStreamEntryId(sourceStreamId)) as Episode["source_stream_ids"],
    added_by_version_episode_id: parseEpisodeId(row.added_by_version_episode_id),
  };
}

function lancePublicOriginSql(): string {
  return "((origin_audience_entity_ids IS NULL OR origin_audience_entity_ids = '[]') AND audience_entity_id IS NULL AND (shared IS NULL OR shared = true))";
}

function lanceOriginContainsSql(audienceEntityId: EntityId): string {
  return `(origin_audience_entity_ids LIKE ${quoteSqlString(
    `%"${audienceEntityId}"%`,
  )} OR audience_entity_id = ${quoteSqlString(audienceEntityId)})`;
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

const LEXICAL_SCAN_PAGE_SIZE = 64;
const LEXICAL_TOKEN_MIN_LENGTH = 3;
const LEXICAL_SHORT_TOKEN_LENGTH = 5;
const LEXICAL_TEXT_COLUMNS = ["title", "narrative", "participants"] as const;

// These are source handles already identified by the recall-expansion LLM. The
// repository only performs a bounded exact lexical lookup; it does not infer
// entities or topics from user text.
export function episodeLexicalSearchTokens(term: string): string[] {
  return [
    ...new Set(
      term
        .toLowerCase()
        // The Unicode token split also strips LIKE metacharacters, so they
        // cannot reach the query pattern as wildcards.
        .split(/[^\p{L}\p{M}\p{N}]+/u)
        .filter((token) => [...token].length >= LEXICAL_TOKEN_MIN_LENGTH),
    ),
  ];
}

function episodeLexicalTokenWhereClause(token: string): string {
  const tokenLength = [...token].length;

  return LEXICAL_TEXT_COLUMNS.map((column) => {
    if (tokenLength < LEXICAL_SHORT_TOKEN_LENGTH) {
      // Short names must be complete Unicode tokens. Inflected forms belong in
      // the LLM expansion output, not in this exact-handle lookup.
      const exactTokenPattern = `(^|[^\\p{L}\\p{M}\\p{N}])${token}($|[^\\p{L}\\p{M}\\p{N}])`;
      return `regexp_like(lower(${column}), ${quoteSqlString(exactTokenPattern)})`;
    }

    return `lower(${column}) LIKE ${quoteSqlString(`%${token}%`)}`;
  }).join(" OR ");
}

function sqlPlaceholders(count: number): string {
  return Array.from({ length: count }, () => "?").join(", ");
}

function indexedOriginColumn(alias: string): string {
  return `${alias}.origin_audience_entity_ids`;
}

function indexedPublicOriginSql(alias: string): string {
  const originColumn = indexedOriginColumn(alias);
  return [
    `(${alias}.shared = 1`,
    `AND (${originColumn} IS NULL OR json_array_length(${originColumn}) = 0)`,
    `AND ${alias}.audience_entity_id IS NULL)`,
  ].join(" ");
}

function indexedOriginContainsSql(alias: string): string {
  return `(EXISTS (SELECT 1 FROM json_each(${indexedOriginColumn(alias)}) WHERE value = ?) OR ${alias}.audience_entity_id = ?)`;
}

function normalizedEpisodeKind(episode: Episode): EpisodeKind {
  return episode.episode_kind ?? "raw";
}

function normalizedConsolidationFamilyId(episode: Episode): ConsolidationFamilyId | null {
  return episode.consolidation_family_id ?? null;
}

function normalizedConsolidationCoverageHash(episode: Episode): string | null {
  return episode.consolidation_coverage_hash ?? null;
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
    origin_audience_entity_ids: serializeJsonValue(normalized.origin_audience_entity_ids),
    shared: normalized.shared,
    emotional_arc:
      normalized.emotional_arc === null ? null : serializeJsonValue(normalized.emotional_arc),
    episode_kind: normalizedEpisodeKind(normalized),
    consolidation_family_id: normalizedConsolidationFamilyId(normalized),
    consolidation_coverage_hash: normalizedConsolidationCoverageHash(normalized),
    embedding: Array.from(normalized.embedding),
    created_at: normalized.created_at,
    updated_at: normalized.updated_at,
  };
}

function episodeKindFromRow(row: Record<string, unknown>): EpisodeKind {
  if (
    row.episode_kind === null ||
    row.episode_kind === undefined ||
    String(row.episode_kind) === ""
  ) {
    return "raw";
  }

  return String(row.episode_kind) as EpisodeKind;
}

function consolidationFamilyIdFromRow(row: Record<string, unknown>): ConsolidationFamilyId | null {
  if (
    row.consolidation_family_id === null ||
    row.consolidation_family_id === undefined ||
    String(row.consolidation_family_id) === ""
  ) {
    return null;
  }

  return parseConsolidationFamilyId(String(row.consolidation_family_id));
}

function consolidationCoverageHashFromRow(row: Record<string, unknown>): string | null {
  if (
    row.consolidation_coverage_hash === null ||
    row.consolidation_coverage_hash === undefined ||
    String(row.consolidation_coverage_hash) === ""
  ) {
    return null;
  }

  return String(row.consolidation_coverage_hash);
}

function originAudienceEntityIdsFromRow(row: Record<string, unknown>): EntityId[] {
  const legacyAudienceEntityId =
    row.audience_entity_id === null || row.audience_entity_id === undefined
      ? null
      : parseEntityId(String(row.audience_entity_id));

  if (
    row.origin_audience_entity_ids === null ||
    row.origin_audience_entity_ids === undefined ||
    row.origin_audience_entity_ids === ""
  ) {
    return legacyAudienceEntityId === null ? [] : [legacyAudienceEntityId];
  }

  const parsed = parseJsonArray<string>(
    String(row.origin_audience_entity_ids),
    "origin_audience_entity_ids",
    EPISODE_JSON_ARRAY_CODEC,
  ).map((value) => parseEntityId(String(value)));

  if (parsed.length === 0 && legacyAudienceEntityId !== null) {
    return [legacyAudienceEntityId];
  }

  return [...new Set(parsed)];
}

function fromEpisodeRow(row: Record<string, unknown>): Episode {
  const originAudienceEntityIds = originAudienceEntityIdsFromRow(row);
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
    origin_audience_entity_ids: originAudienceEntityIds,
    shared:
      row.shared === null || row.shared === undefined
        ? originAudienceEntityIds.length === 0
        : row.shared === true || Number(row.shared) === 1,
    episode_kind: episodeKindFromRow(row),
    consolidation_family_id: consolidationFamilyIdFromRow(row),
    consolidation_coverage_hash: consolidationCoverageHashFromRow(row),
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

function hotLaneCandidateLimit(limit: number): number {
  return Math.max(
    limit + HOT_LANE_RETRIEVAL_CANDIDATE_BUFFER,
    limit * HOT_LANE_RETRIEVAL_CANDIDATE_MULTIPLIER,
  );
}

function isHotLaneCooled(stats: EpisodeStats, cooldownCutoff: number): boolean {
  return stats.last_retrieved !== null && stats.last_retrieved >= cooldownCutoff;
}

function applyHotLaneCooldownPenalty(
  candidates: readonly EpisodeSearchCandidate[],
  nowMs: number,
  limit: number,
): EpisodeSearchCandidate[] {
  const cooldownCutoff = nowMs - HOT_LANE_RETRIEVAL_COOLDOWN_MS;

  return candidates
    .map((candidate, index) => ({
      candidate,
      index,
      cooled: isHotLaneCooled(candidate.stats, cooldownCutoff),
    }))
    .sort((left, right) => {
      if (left.cooled !== right.cooled) {
        return left.cooled ? 1 : -1;
      }

      return left.index - right.index;
    })
    .slice(0, limit)
    .map((entry) => entry.candidate);
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
    utf8Field("origin_audience_entity_ids", true),
    booleanField("shared", true),
    utf8Field("emotional_arc", true),
    utf8Field("episode_kind", true),
    utf8Field("consolidation_family_id", true),
    utf8Field("consolidation_coverage_hash", true),
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

  private buildVisibilityWhereClause(capability: ViewerCapability): string | undefined {
    switch (capability.kind) {
      case "unrestricted":
        return undefined;
      case "audience":
        return capability.audienceEntityId === null
          ? lancePublicOriginSql()
          : `(${lancePublicOriginSql()} OR ${lanceOriginContainsSql(capability.audienceEntityId)})`;
      default: {
        const exhaustive: never = capability;
        throw new Error(`unhandled ViewerCapability kind: ${JSON.stringify(exhaustive)}`);
      }
    }
  }

  private buildOptionsVisibilityWhereClause(options: EpisodeVisibilityOptions): string | undefined {
    return this.buildVisibilityWhereClause(resolveViewerCapability(options));
  }

  private buildIndexedVisibilityWhereClause(
    capability: ViewerCapability,
    alias: string,
  ): {
    sql: string;
    params: unknown[];
  } {
    switch (capability.kind) {
      case "unrestricted":
        return { sql: "1 = 1", params: [] };
      case "audience":
        return capability.audienceEntityId === null
          ? {
              sql: indexedPublicOriginSql(alias),
              params: [],
            }
          : {
              sql: `(${indexedPublicOriginSql(alias)} OR ${indexedOriginContainsSql(alias)})`,
              params: [capability.audienceEntityId, capability.audienceEntityId],
            };
      default: {
        const exhaustive: never = capability;
        throw new Error(`unhandled ViewerCapability kind: ${JSON.stringify(exhaustive)}`);
      }
    }
  }

  private buildEffectiveVisibilityWhereClause(alias: string): string {
    return [
      `${alias}.archived = 0`,
      `AND EXISTS (`,
      `  SELECT 1`,
      `  FROM episode_stats AS effective_stats`,
      `  WHERE effective_stats.episode_id = ${alias}.episode_id`,
      `    AND effective_stats.archived = 0`,
      `)`,
      `AND ((`,
      `  ${alias}.episode_kind = 'raw'`,
      `  AND NOT EXISTS (`,
      `    SELECT 1`,
      `    FROM consolidation_members AS cm`,
      `    JOIN consolidation_families AS cf ON cf.family_id = cm.family_id`,
      `    JOIN episode_index AS current_version`,
      `      ON current_version.episode_id = cf.current_version_episode_id`,
      `    JOIN episode_stats AS current_version_stats`,
      `      ON current_version_stats.episode_id = current_version.episode_id`,
      `    WHERE cm.raw_episode_id = ${alias}.episode_id`,
      `      AND current_version.archived = 0`,
      `      AND current_version_stats.archived = 0`,
      `  )`,
      `) OR (`,
      `  ${alias}.episode_kind = 'consolidation_version'`,
      `  AND EXISTS (`,
      `    SELECT 1`,
      `    FROM consolidation_families AS cf`,
      `    WHERE cf.family_id = ${alias}.consolidation_family_id`,
      `      AND cf.current_version_episode_id = ${alias}.episode_id`,
      `  )`,
      `))`,
    ].join("\n");
  }

  private queryEffectivelyVisibleEpisodeIdSet(ids: readonly EpisodeId[]): Set<EpisodeId> {
    const uniqueIds = [...new Set(ids)];

    if (uniqueIds.length === 0) {
      return new Set();
    }

    const rows = this.db
      .prepare(
        `
          SELECT ei.episode_id
          FROM episode_index AS ei
          WHERE ei.episode_id IN (${sqlPlaceholders(uniqueIds.length)})
            AND ${this.buildEffectiveVisibilityWhereClause("ei")}
        `,
      )
      .all(...uniqueIds) as EpisodeEffectiveVisibilityRow[];

    return new Set(rows.map((row) => parseEpisodeId(row.episode_id)));
  }

  isEpisodeEffectivelyVisible(episodeId: EpisodeId): boolean {
    return this.queryEffectivelyVisibleEpisodeIdSet([episodeId]).has(episodeId);
  }

  private buildIndexedVisibilityBranches(
    capability: ViewerCapability,
    order: IndexedEpisodeOrder,
  ): IndexedVisibilityBranch[] {
    const globalIndexName =
      order === "heat" ? "idx_episode_index_heat" : "idx_episode_index_recent";
    const visibility = this.buildIndexedVisibilityWhereClause(capability, "episode_index");
    const effectiveVisibility = this.buildEffectiveVisibilityWhereClause("episode_index");

    return [
      {
        where: `${effectiveVisibility} AND ${visibility.sql}`,
        params: visibility.params,
        indexName: globalIndexName,
      },
    ];
  }

  private async listEpisodesWhere(where: string | undefined): Promise<Episode[]> {
    const rows = await this.table.list(where === undefined ? {} : { where });
    return rows.map((row) => fromEpisodeRow(row));
  }

  async listVisibleEpisodes(
    options: EpisodeVisibilityOptions = {},
    extraWhere?: string,
  ): Promise<Episode[]> {
    const viewer = resolveViewerCapability(options);
    const episodes = await this.listEpisodesWhere(
      combineWhereClauses(this.buildOptionsVisibilityWhereClause(options), extraWhere),
    );
    await this.ensureEpisodeIndexBackfilled();
    const effectivelyVisibleEpisodeIds = this.queryEffectivelyVisibleEpisodeIdSet(
      episodes.map((episode) => episode.id),
    );

    return episodes.filter(
      (episode) =>
        effectivelyVisibleEpisodeIds.has(episode.id) &&
        isEpisodeVisibleToCapability(episode, viewer),
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
            episode_id, audience_entity_id, origin_audience_entity_ids, shared, episode_kind,
            consolidation_family_id, consolidation_coverage_hash, start_time, end_time, created_at,
            updated_at, retrieval_count, win_rate, last_retrieved, tier, archived, heat_multiplier,
            heat_score
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (episode_id) DO UPDATE SET
            audience_entity_id = excluded.audience_entity_id,
            origin_audience_entity_ids = excluded.origin_audience_entity_ids,
            shared = excluded.shared,
            episode_kind = excluded.episode_kind,
            consolidation_family_id = excluded.consolidation_family_id,
            consolidation_coverage_hash = excluded.consolidation_coverage_hash,
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
        serializeJsonValue(normalized.origin_audience_entity_ids),
        normalized.shared ? 1 : 0,
        normalizedEpisodeKind(normalized),
        normalizedConsolidationFamilyId(normalized),
        normalizedConsolidationCoverageHash(normalized),
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
    const branches = this.buildIndexedVisibilityBranches(resolveViewerCapability(options), order);
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

  private queryAllIndexedEpisodeIds(order: IndexedEpisodeOrder, limit: number): EpisodeId[] {
    const orderBy =
      order === "heat"
        ? "heat_score DESC, updated_at DESC, episode_id DESC"
        : "updated_at DESC, episode_id DESC";
    const indexName = order === "heat" ? "idx_episode_index_heat" : "idx_episode_index_recent";
    const rows = this.db
      .prepare(
        `
          SELECT episode_id
          FROM episode_index INDEXED BY ${indexName}
          WHERE ${this.buildEffectiveVisibilityWhereClause("episode_index")}
          ORDER BY ${orderBy}
          LIMIT ?
        `,
      )
      .all(limit) as IndexedEpisodeIdRow[];

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
    const effectivelyVisibleEpisodeIds = this.queryEffectivelyVisibleEpisodeIdSet(
      orderedEpisodes.map((episode) => episode.id),
    );

    return this.hydrateSearchCandidates(
      orderedEpisodes,
      statsById,
      undefined,
      effectivelyVisibleEpisodeIds,
    );
  }

  private hydrateSearchCandidates(
    episodes: readonly Episode[],
    statsById: ReadonlyMap<EpisodeId, EpisodeStats>,
    similarityById?: ReadonlyMap<EpisodeId, number>,
    effectivelyVisibleEpisodeIds?: ReadonlySet<EpisodeId>,
  ): EpisodeSearchCandidate[] {
    const results: EpisodeSearchCandidate[] = [];

    for (const episode of episodes) {
      const stats = statsById.get(episode.id) ?? defaultEpisodeStats(episode);

      if (effectivelyVisibleEpisodeIds?.has(episode.id) === false || stats.archived) {
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

  private validateEpisodeForWrite(episode: Episode, errorCode: string): Episode {
    const parsed = episodeInsertSchema.safeParse(normalizeEpisodeAccess(episode));

    if (!parsed.success) {
      throw new StorageError("Invalid episode payload", {
        cause: parsed.error,
        code: errorCode,
      });
    }

    return parsed.data;
  }

  async createEpisode(episode: Episode): Promise<Episode> {
    const parsed = this.validateEpisodeForWrite(episode, "EPISODE_INVALID");
    const existing = await this.get(parsed.id, { includeArchived: true });

    if (existing !== null) {
      throw new StorageError(`Episode ${parsed.id} already exists`, {
        code: "EPISODE_ALREADY_EXISTS",
      });
    }

    try {
      await this.table.upsert([toEpisodeRow(parsed)], { on: "id" });

      try {
        const apply = this.db.transaction(() => {
          const stats = defaultEpisodeStats(parsed);

          this.upsertStats(stats);
          this.upsertEpisodeIndex(parsed, stats);
        });
        apply();
      } catch (error) {
        const currentAfterFailure = await this.get(parsed.id, { includeArchived: true });

        if (currentAfterFailure?.updated_at === parsed.updated_at) {
          await this.table.remove(`id = ${quoteSqlString(parsed.id)}`);
        } else {
          console.warn("Skipped episode create rollback because newer Lance state exists.", {
            episodeId: parsed.id,
            attemptedUpdatedAt: parsed.updated_at,
            currentUpdatedAt: currentAfterFailure?.updated_at ?? null,
          });
        }
        throw error;
      }

      return parsed;
    } catch (error) {
      throw new StorageError(`Failed to create episode ${parsed.id}`, {
        cause: error,
        code: "EPISODE_INSERT_FAILED",
      });
    }
  }

  async upsertEpisodeBodyPreservingStats(episode: Episode): Promise<Episode> {
    const parsed = this.validateEpisodeForWrite(episode, "EPISODE_INVALID");
    const current = await this.get(parsed.id, { includeArchived: true });
    const stats = this.getStats(parsed.id);

    if (current === null || stats === null) {
      throw new StorageError(`Missing episode row for ${parsed.id}`, {
        code: "EPISODE_MISSING",
      });
    }

    const previousRow = toEpisodeRow(current);

    try {
      await this.table.upsert([toEpisodeRow(parsed)], { on: "id" });

      try {
        const apply = this.db.transaction(() => {
          this.upsertEpisodeIndex(parsed, stats);
        });
        apply();
      } catch (error) {
        const currentAfterFailure = await this.get(parsed.id, { includeArchived: true });

        if (currentAfterFailure?.updated_at === parsed.updated_at) {
          await this.table.upsert([previousRow], { on: "id" });
        } else {
          console.warn("Skipped episode body rollback because newer Lance state exists.", {
            episodeId: parsed.id,
            attemptedUpdatedAt: parsed.updated_at,
            currentUpdatedAt: currentAfterFailure?.updated_at ?? null,
          });
        }
        throw error;
      }

      return parsed;
    } catch (error) {
      throw new StorageError(`Failed to upsert episode body ${parsed.id}`, {
        cause: error,
        code: "EPISODE_UPDATE_FAILED",
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

    if (
      options.includeArchived !== true &&
      ((stats?.archived ?? false) || !this.isEpisodeEffectivelyVisible(id))
    ) {
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

  private validateLifecycleAuditInput(
    input: EpisodeLifecycleAuditInput,
  ): EpisodeLifecycleAuditInput {
    const caller = input.caller.trim();
    const reason = input.reason.trim();
    const process = input.process.trim();

    if (caller.length === 0) {
      throw new StorageError("Episode lifecycle audit caller is required", {
        code: "EPISODE_LIFECYCLE_AUDIT_INVALID",
      });
    }

    if (reason.length === 0) {
      throw new StorageError("Episode lifecycle audit reason is required", {
        code: "EPISODE_LIFECYCLE_AUDIT_INVALID",
      });
    }

    if (process.length === 0) {
      throw new StorageError("Episode lifecycle audit process is required", {
        code: "EPISODE_LIFECYCLE_AUDIT_INVALID",
      });
    }

    return {
      ...input,
      caller,
      reason,
      process,
    };
  }

  private recordEpisodeLifecycleAudit(input: {
    action: "archive_episode" | "reactivate_episode" | "unarchive_episode";
    episodeId: EpisodeId;
    previousArchived: boolean;
    nextArchived: boolean;
    audit: EpisodeLifecycleAuditInput;
  }): void {
    const audit = this.validateLifecycleAuditInput(input.audit);
    const auditProcess = resolveEpisodeLifecycleAuditProcess(audit.process);

    this.db
      .prepare(
        `
          INSERT INTO maintenance_audit (
            run_id, process, action, targets, reversal, applied_at, reverted_at, reverted_by
          ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)
        `,
      )
      .run(
        audit.runId ?? createMaintenanceRunId(),
        auditProcess,
        input.action,
        serializeJsonValue({
          episode_id: input.episodeId,
          caller: audit.caller,
          reason: audit.reason,
          initiating_process: audit.process,
          lifecycle_owner: "episodic-repository",
          previous_archived: input.previousArchived,
          next_archived: input.nextArchived,
        }),
        serializeJsonValue({
          episode_id: input.episodeId,
          archived: input.previousArchived,
        }),
        this.clock.now(),
      );
  }

  archiveEpisode(episodeId: EpisodeId, audit: EpisodeLifecycleAuditInput): EpisodeStats {
    const apply = this.db.transaction(() => {
      const current = this.getStats(episodeId);

      if (current === null) {
        throw new StorageError(`Missing episode_stats row for ${episodeId}`, {
          code: "EPISODE_STATS_MISSING",
        });
      }

      if (current.archived) {
        return current;
      }

      const owningFamily = this.db
        .prepare(
          `
            SELECT family_id
            FROM consolidation_families
            WHERE current_version_episode_id = ?
          `,
        )
        .get(episodeId) as { family_id: string } | undefined;

      if (owningFamily !== undefined) {
        throw new StorageError(
          `Episode ${episodeId} is the current version of consolidation family ${owningFamily.family_id}; tear down via revertConsolidationVersion, do not archive it directly`,
          {
            code: "EPISODE_ARCHIVE_CURRENT_CONSOLIDATION_VERSION",
          },
        );
      }

      const result = this.db
        .prepare(
          `
            UPDATE episode_stats
            SET archived = 1
            WHERE episode_id = ?
              AND archived = 0
          `,
        )
        .run(episodeId);

      if (result.changes !== 1) {
        throw new StorageError(`Stale episode archive transition for ${episodeId}`, {
          code: "EPISODE_ARCHIVE_STALE",
        });
      }

      const next = this.getStats(episodeId);

      if (next === null) {
        throw new StorageError(`Missing episode_stats row for ${episodeId}`, {
          code: "EPISODE_STATS_MISSING",
        });
      }

      this.syncEpisodeIndexStats(next);
      this.recordEpisodeLifecycleAudit({
        action: "archive_episode",
        episodeId,
        previousArchived: false,
        nextArchived: true,
        audit,
      });

      return next;
    });

    return apply() as EpisodeStats;
  }

  private transitionArchivedEpisodeToActive(
    episodeId: EpisodeId,
    audit: EpisodeLifecycleAuditInput,
    action: "reactivate_episode" | "unarchive_episode",
  ): EpisodeStats {
    const apply = this.db.transaction(() => {
      const current = this.getStats(episodeId);

      if (current === null) {
        throw new StorageError(`Missing episode_stats row for ${episodeId}`, {
          code: "EPISODE_STATS_MISSING",
        });
      }

      if (!current.archived) {
        return current;
      }

      const result = this.db
        .prepare(
          `
            UPDATE episode_stats
            SET archived = 0
            WHERE episode_id = ?
              AND archived = 1
          `,
        )
        .run(episodeId);

      if (result.changes !== 1) {
        throw new StorageError(`Stale episode activation transition for ${episodeId}`, {
          code:
            action === "unarchive_episode" ? "EPISODE_UNARCHIVE_STALE" : "EPISODE_REACTIVATE_STALE",
        });
      }

      const next = this.getStats(episodeId);

      if (next === null) {
        throw new StorageError(`Missing episode_stats row for ${episodeId}`, {
          code: "EPISODE_STATS_MISSING",
        });
      }

      this.syncEpisodeIndexStats(next);
      this.recordEpisodeLifecycleAudit({
        action,
        episodeId,
        previousArchived: true,
        nextArchived: false,
        audit,
      });

      return next;
    });

    return apply() as EpisodeStats;
  }

  reactivateEpisode(episodeId: EpisodeId, audit: EpisodeLifecycleAuditInput): EpisodeStats {
    return this.transitionArchivedEpisodeToActive(episodeId, audit, "reactivate_episode");
  }

  // Explicit operator-invoked reversal paths only, never happy-path writers. The June lockdown
  // intentionally keeps every generic restore patch archived-neutral.
  unarchiveEpisode(episodeId: EpisodeId, audit: EpisodeLifecycleAuditInput): EpisodeStats {
    return this.transitionArchivedEpisodeToActive(episodeId, audit, "unarchive_episode");
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

    if (parsedPatch.data.archived !== undefined) {
      throw new StorageError(
        `Episode ${episodeId} archive state must change via archiveEpisode/reactivateEpisode/unarchiveEpisode, not updateStats`,
        {
          code: "EPISODE_ARCHIVED_REQUIRES_LIFECYCLE_API",
        },
      );
    }

    const patchEntries: Array<[EpisodeStatsPatchKey, unknown]> = [];

    for (const [rawKey, value] of Object.entries(parsedPatch.data)) {
      if (value !== undefined) {
        patchEntries.push([rawKey as EpisodeStatsPatchKey, value]);
      }
    }

    if (patchEntries.length === 0) {
      return current;
    }

    episodeStatsSchema.parse({
      ...current,
      ...parsedPatch.data,
    });
    const assignments = patchEntries.map(([key]) => `${EPISODE_STATS_PATCH_COLUMNS[key]} = ?`);
    const values = patchEntries.map(([key, value]) => encodeEpisodeStatsPatchValue(key, value));

    this.db
      .prepare(
        `
          UPDATE episode_stats
          SET ${assignments.join(", ")}
          WHERE episode_id = ?
        `,
      )
      .run(...values, episodeId);

    const updated = this.getStats(episodeId);

    if (updated === null) {
      throw new StorageError(`Missing episode_stats row for ${episodeId}`, {
        code: "EPISODE_STATS_MISSING",
      });
    }

    this.syncEpisodeIndexStats(updated);
    return updated;
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

  async recallByVectorForCognition(
    vector: Float32Array,
    options: EpisodeCognitionRecallOptions = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByVectorInternal(vector, options, "cognition");
  }

  async searchByVectorForDisclosure(
    vector: Float32Array,
    options: EpisodeSearchOptions = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByVectorInternal(vector, options, "disclosure");
  }

  async searchByVector(
    vector: Float32Array,
    options: EpisodeSearchOptions = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByVectorForDisclosure(vector, options);
  }

  private async searchByVectorInternal(
    vector: Float32Array,
    options: EpisodeCognitionRecallOptions | EpisodeSearchOptions,
    visibilityMode: EpisodeSearchVisibilityMode,
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "Search limit");
    const searchLimit = Math.max(limit * 5, limit, 20);
    const rows = await this.table.search(Array.from(vector), {
      limit: searchLimit,
      vectorColumn: "embedding",
      distanceType: "cosine",
      where:
        visibilityMode === "disclosure"
          ? this.buildOptionsVisibilityWhereClause(options as EpisodeSearchOptions)
          : undefined,
    });
    const ranked = rows.map((row) => {
      const episode = fromEpisodeRow(row);
      return {
        episode,
        similarity: toSimilarity(getDistance(row)),
      };
    });
    const statsById = this.getStatsMany(ranked.map((item) => item.episode.id));
    await this.ensureEpisodeIndexBackfilled();
    const effectivelyVisibleEpisodeIds = this.queryEffectivelyVisibleEpisodeIdSet(
      ranked.map((item) => item.episode.id),
    );
    const results: EpisodeSearchCandidate[] = [];
    const viewer =
      visibilityMode === "disclosure"
        ? resolveViewerCapability(options as EpisodeSearchOptions)
        : null;

    for (const item of ranked) {
      const episode = item.episode;
      const stats = statsById.get(episode.id) ?? defaultEpisodeStats(episode);
      const similarity = item.similarity;

      if (viewer !== null && !isEpisodeVisibleToCapability(episode, viewer)) {
        continue;
      }

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

      if (!effectivelyVisibleEpisodeIds.has(episode.id) || stats.archived) {
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

  async recallByTimeRangeForCognition(
    range: { start: number; end: number },
    options: {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByTimeRangeInternal(range, options, "cognition");
  }

  async searchByTimeRangeForDisclosure(
    range: { start: number; end: number },
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByTimeRangeInternal(range, options, "disclosure");
  }

  async searchByTimeRange(
    range: { start: number; end: number },
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByTimeRangeForDisclosure(range, options);
  }

  private async searchByTimeRangeInternal(
    range: { start: number; end: number },
    options: EpisodeVisibilityOptions & {
      limit?: number;
    },
    visibilityMode: EpisodeSearchVisibilityMode,
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "Search limit");
    await this.ensureEpisodeIndexBackfilled();

    const clauses = [this.buildEffectiveVisibilityWhereClause("ei")];
    const params: unknown[] = [];

    if (visibilityMode === "disclosure") {
      const visibility = this.buildIndexedVisibilityWhereClause(
        resolveViewerCapability(options),
        "ei",
      );
      clauses.push(visibility.sql);
      params.push(...visibility.params);
    }

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
      options.orderBy === "heat" ? "idx_episode_index_heat" : "idx_episode_index_recent";
    const rows = this.db
      .prepare(
        `
          SELECT episode_id
          FROM episode_index INDEXED BY ${indexName}
          WHERE ${this.buildEffectiveVisibilityWhereClause("episode_index")}
            AND ${indexedOriginContainsSql("episode_index")}
          ORDER BY ${orderBy}
          LIMIT ?
        `,
      )
      .all(audienceEntityId, audienceEntityId, limit) as IndexedEpisodeIdRow[];

    return this.hydrateCandidatesByIds(rows.map((row) => parseEpisodeId(row.episode_id)));
  }

  async recallByParticipantsOrTagsForCognition(
    terms: readonly string[],
    options: {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByParticipantsOrTagsInternal(terms, options, "cognition");
  }

  async searchByParticipantsOrTagsForDisclosure(
    terms: readonly string[],
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByParticipantsOrTagsInternal(terms, options, "disclosure");
  }

  async searchByParticipantsOrTags(
    terms: readonly string[],
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByParticipantsOrTagsForDisclosure(terms, options);
  }

  private async searchByParticipantsOrTagsInternal(
    terms: readonly string[],
    options: EpisodeVisibilityOptions & {
      limit?: number;
    },
    visibilityMode: EpisodeSearchVisibilityMode,
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
    const visibility =
      visibilityMode === "disclosure"
        ? this.buildIndexedVisibilityWhereClause(resolveViewerCapability(options), "ei")
        : { sql: "1 = 1", params: [] };
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
              AND ${this.buildEffectiveVisibilityWhereClause("ei")}
              AND ${visibility.sql}
            UNION
            SELECT ei.episode_id, ei.updated_at
            FROM episode_tags AS et INDEXED BY idx_episode_tags_term
            JOIN episode_index AS ei ON ei.episode_id = et.episode_id
            WHERE et.term IN (${termPlaceholders})
              AND ${this.buildEffectiveVisibilityWhereClause("ei")}
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

  async recallByLexicalTermsForCognition(
    terms: readonly string[],
    options: {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByLexicalTermsInternal(terms, options, "cognition");
  }

  async searchByLexicalTermsForDisclosure(
    terms: readonly string[],
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.searchByLexicalTermsInternal(terms, options, "disclosure");
  }

  private async searchByLexicalTermsInternal(
    terms: readonly string[],
    options: EpisodeVisibilityOptions & {
      limit?: number;
    },
    visibilityMode: EpisodeSearchVisibilityMode,
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "Search limit");
    const tokenGroupsByKey = new Map<string, string[]>();

    for (const term of terms) {
      const tokens = episodeLexicalSearchTokens(term);

      if (tokens.length > 0) {
        tokenGroupsByKey.set(tokens.join("\u0000"), tokens);
      }
    }

    if (tokenGroupsByKey.size === 0) {
      return [];
    }

    await this.ensureEpisodeIndexBackfilled();

    const visibility =
      visibilityMode === "disclosure"
        ? this.buildIndexedVisibilityWhereClause(resolveViewerCapability(options), "ei")
        : { sql: "1 = 1", params: [] };
    const results: EpisodeSearchCandidate[] = [];
    const seenIds = new Set<EpisodeId>();
    const lexicalWhere = [...tokenGroupsByKey.values()]
      .map((tokens) =>
        tokens.map((token) => `(${episodeLexicalTokenWhereClause(token)})`).join(" AND "),
      )
      .map((group) => `(${group})`)
      .join(" OR ");
    const pageSize = Math.max(LEXICAL_SCAN_PAGE_SIZE, limit * 2);
    let offset = 0;

    while (results.length < limit) {
      const indexedRows = this.db
        .prepare(
          `
            SELECT ei.episode_id
            FROM episode_index AS ei
            WHERE ${this.buildEffectiveVisibilityWhereClause("ei")}
              AND ${visibility.sql}
            ORDER BY ei.updated_at DESC, ei.episode_id DESC
            LIMIT ? OFFSET ?
          `,
        )
        .all(...visibility.params, pageSize, offset) as IndexedEpisodeIdRow[];

      if (indexedRows.length === 0) {
        break;
      }

      offset += indexedRows.length;
      const pageIds = indexedRows.map((row) => parseEpisodeId(row.episode_id));
      const idWhere = `id IN (${pageIds.map((id) => quoteSqlString(id)).join(", ")})`;
      const matchingRows = await this.table.list({
        columns: ["id"],
        where: combineWhereClauses(idWhere, `(${lexicalWhere})`),
        limit: pageIds.length,
      });
      const matchingIds = new Set(matchingRows.map((row) => parseEpisodeId(String(row.id))));
      const candidates = await this.hydrateCandidatesByIds(
        pageIds.filter((episodeId) => matchingIds.has(episodeId)),
      );

      for (const candidate of candidates) {
        if (!seenIds.has(candidate.episode.id)) {
          seenIds.add(candidate.episode.id);
          results.push(candidate);

          if (results.length >= limit) {
            break;
          }
        }
      }

      if (indexedRows.length < pageSize) {
        break;
      }
    }

    return results;
  }

  async listRecentForCognition(
    options: {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "List limit");
    await this.ensureEpisodeIndexBackfilled();
    return this.hydrateCandidatesByIds(this.queryAllIndexedEpisodeIds("recent", limit));
  }

  async listRecentForDisclosure(
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

  async listRecent(
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.listRecentForDisclosure(options);
  }

  async listHottestForCognition(
    options: {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "List limit");
    await this.ensureEpisodeIndexBackfilled();
    const candidates = await this.hydrateCandidatesByIds(
      this.queryAllIndexedEpisodeIds("heat", hotLaneCandidateLimit(limit)),
    );

    return applyHotLaneCooldownPenalty(candidates, this.clock.now(), limit);
  }

  async listHottestForDisclosure(
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit ?? DEFAULT_SEARCH_LIMIT, "List limit");
    await this.ensureEpisodeIndexBackfilled();
    return this.hydrateCandidatesByIds(this.queryVisibleIndexedEpisodeIds(options, "heat", limit));
  }

  async listHottest(
    options: EpisodeVisibilityOptions & {
      limit?: number;
    } = {},
  ): Promise<EpisodeSearchCandidate[]> {
    return this.listHottestForDisclosure(options);
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

  async listUnarchivedEpisodeIds(): Promise<EpisodeId[]> {
    await this.ensureEpisodeIndexBackfilled();

    const rows = this.db
      .prepare(
        `
          SELECT episode_id
          FROM episode_index
          WHERE archived = 0
          ORDER BY episode_id ASC
        `,
      )
      .all() as IndexedEpisodeIdRow[];

    return rows.map((row) => parseEpisodeId(row.episode_id));
  }

  async listEffectivelyVisible(): Promise<Episode[]> {
    const episodes = await this.listAll();
    await this.ensureEpisodeIndexBackfilled();
    const effectivelyVisibleEpisodeIds = this.queryEffectivelyVisibleEpisodeIdSet(
      episodes.map((episode) => episode.id),
    );

    return episodes.filter((episode) => effectivelyVisibleEpisodeIds.has(episode.id));
  }

  listConsolidationFamilies(): ConsolidationFamilyRecord[] {
    const rows = this.db
      .prepare(
        `
          SELECT
            family_id, current_version_episode_id, coverage_hash, policy_version,
            created_at, updated_at
          FROM consolidation_families
          ORDER BY updated_at DESC, family_id ASC
        `,
      )
      .all() as ConsolidationFamilyRow[];

    return rows.map(fromConsolidationFamilyRow);
  }

  getConsolidationFamily(familyId: ConsolidationFamilyId): ConsolidationFamilyRecord | null {
    const row = this.db
      .prepare(
        `
          SELECT
            family_id, current_version_episode_id, coverage_hash, policy_version,
            created_at, updated_at
          FROM consolidation_families
          WHERE family_id = ?
        `,
      )
      .get(familyId) as ConsolidationFamilyRow | undefined;

    return row === undefined ? null : fromConsolidationFamilyRow(row);
  }

  listConsolidationMembers(familyId?: ConsolidationFamilyId): ConsolidationMemberRecord[] {
    const rows =
      familyId === undefined
        ? (this.db
            .prepare(
              `
                SELECT
                  family_id, raw_episode_id, source_stream_ids_json, added_by_version_episode_id
                FROM consolidation_members
                ORDER BY family_id ASC, raw_episode_id ASC
              `,
            )
            .all() as ConsolidationMemberRow[])
        : (this.db
            .prepare(
              `
                SELECT
                  family_id, raw_episode_id, source_stream_ids_json, added_by_version_episode_id
                FROM consolidation_members
                WHERE family_id = ?
                ORDER BY raw_episode_id ASC
              `,
            )
            .all(familyId) as ConsolidationMemberRow[]);

    return rows.map(fromConsolidationMemberRow);
  }

  createConsolidationFamily(input: {
    familyId: ConsolidationFamilyId;
    currentVersionEpisodeId: EpisodeId;
    coverageHash: string;
    policyVersion: number;
    members: readonly ConsolidationMemberInput[];
  }): void {
    const nowMs = this.clock.now();
    const insertFamily = this.db.prepare(
      `
        INSERT INTO consolidation_families (
          family_id, current_version_episode_id, coverage_hash, policy_version, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
      `,
    );
    const insertMember = this.db.prepare(
      `
        INSERT OR IGNORE INTO consolidation_members (
          family_id, raw_episode_id, source_stream_ids_json, added_by_version_episode_id
        ) VALUES (?, ?, ?, ?)
      `,
    );
    const apply = this.db.transaction(() => {
      insertFamily.run(
        input.familyId,
        input.currentVersionEpisodeId,
        input.coverageHash,
        input.policyVersion,
        nowMs,
        nowMs,
      );

      for (const member of input.members) {
        insertMember.run(
          input.familyId,
          member.raw_episode_id,
          serializeJsonValue(member.source_stream_ids),
          member.added_by_version_episode_id,
        );
      }
    });

    apply();
  }

  extendConsolidationFamily(input: {
    familyId: ConsolidationFamilyId;
    expectedCurrentVersionEpisodeId: EpisodeId;
    nextVersionEpisodeId: EpisodeId;
    coverageHash: string;
    policyVersion: number;
    members: readonly ConsolidationMemberInput[];
  }): void {
    const nowMs = this.clock.now();
    const insertMember = this.db.prepare(
      `
        INSERT OR IGNORE INTO consolidation_members (
          family_id, raw_episode_id, source_stream_ids_json, added_by_version_episode_id
        ) VALUES (?, ?, ?, ?)
      `,
    );
    const updateFamily = this.db.prepare(
      `
        UPDATE consolidation_families
        SET current_version_episode_id = ?,
            coverage_hash = ?,
            policy_version = ?,
            updated_at = ?
        WHERE family_id = ?
          AND current_version_episode_id = ?
      `,
    );
    const apply = this.db.transaction(() => {
      for (const member of input.members) {
        insertMember.run(
          input.familyId,
          member.raw_episode_id,
          serializeJsonValue(member.source_stream_ids),
          member.added_by_version_episode_id,
        );
      }

      const result = updateFamily.run(
        input.nextVersionEpisodeId,
        input.coverageHash,
        input.policyVersion,
        nowMs,
        input.familyId,
        input.expectedCurrentVersionEpisodeId,
      );

      if (result.changes !== 1) {
        throw new StorageError(`Stale consolidation family update for ${input.familyId}`, {
          code: "CONSOLIDATION_FAMILY_STALE",
        });
      }
    });

    apply();
  }

  async revertConsolidationVersion(input: {
    familyId: ConsolidationFamilyId;
    versionEpisodeId: EpisodeId;
    previousCurrentVersionEpisodeId: EpisodeId | null;
    previousCoverageHash: string | null;
    previousPolicyVersion: number | null;
  }): Promise<void> {
    const nowMs = this.clock.now();
    const deleteVersionMembers = this.db.prepare(
      `
        DELETE FROM consolidation_members
        WHERE family_id = ?
          AND added_by_version_episode_id = ?
      `,
    );
    const restoreFamily = this.db.prepare(
      `
        UPDATE consolidation_families
        SET current_version_episode_id = ?,
            coverage_hash = ?,
            policy_version = ?,
            updated_at = ?
        WHERE family_id = ?
          AND current_version_episode_id = ?
      `,
    );
    const deleteFamily = this.db.prepare(
      `
        DELETE FROM consolidation_families
        WHERE family_id = ?
          AND current_version_episode_id = ?
      `,
    );
    const apply = this.db.transaction(() => {
      deleteVersionMembers.run(input.familyId, input.versionEpisodeId);

      if (input.previousCurrentVersionEpisodeId === null) {
        const result = deleteFamily.run(input.familyId, input.versionEpisodeId);

        if (result.changes !== 1) {
          throw new StorageError(`Stale consolidation family deletion for ${input.familyId}`, {
            code: "CONSOLIDATION_FAMILY_STALE",
          });
        }
        return;
      }

      if (input.previousCoverageHash === null || input.previousPolicyVersion === null) {
        throw new StorageError(
          `Missing previous consolidation family metadata for ${input.familyId}`,
          {
            code: "CONSOLIDATION_FAMILY_STALE",
          },
        );
      }

      const result = restoreFamily.run(
        input.previousCurrentVersionEpisodeId,
        input.previousCoverageHash,
        input.previousPolicyVersion,
        nowMs,
        input.familyId,
        input.versionEpisodeId,
      );

      if (result.changes !== 1) {
        throw new StorageError(`Stale consolidation family restore for ${input.familyId}`, {
          code: "CONSOLIDATION_FAMILY_STALE",
        });
      }
    });

    apply();
    await this.delete(input.versionEpisodeId);
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

  countRetrievalLogBefore(timestamp: number): number {
    const row = this.db
      .prepare("SELECT COUNT(*) AS count FROM retrieval_log WHERE timestamp < ?")
      .get(timestamp) as { count: number };

    return row.count;
  }

  pruneRetrievalLogBefore(timestamp: number): number {
    const result = this.db.prepare("DELETE FROM retrieval_log WHERE timestamp < ?").run(timestamp);

    return result.changes;
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
    const rows = await this.table.list({
      where: `source_fingerprint = ${quoteSqlString(fingerprint)}`,
      limit: 1,
    });

    return rows[0] === undefined ? null : fromEpisodeRow(rows[0]);
  }

  async findBySourceStreamIdsContaining(
    sourceStreamIds: ReadonlyArray<Episode["source_stream_ids"][number]>,
  ): Promise<Episode | null> {
    const requiredSourceIds = [...new Set(sourceStreamIds)];

    if (requiredSourceIds.length === 0) {
      return null;
    }

    const exact = await this.findBySourceStreamIds(requiredSourceIds);

    if (exact !== null) {
      return exact;
    }

    const rows = await this.table.list();

    for (const row of rows) {
      const episode = fromEpisodeRow(row);

      if (requiredSourceIds.every((streamId) => episode.source_stream_ids.includes(streamId))) {
        return episode;
      }
    }

    return null;
  }
}
