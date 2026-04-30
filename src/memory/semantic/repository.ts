import { z } from "zod";

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
import { SemanticError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createSemanticEdgeId,
  parseEpisodeId,
  parseSemanticEdgeId,
  parseSemanticNodeId,
  type SemanticEdgeId,
  type SemanticNodeId,
} from "../../util/ids.js";
import type { ReviewQueueInsertInput } from "./review-queue.js";
import {
  invalidationProcessSchema,
  semanticEdgeIdSchema,
  semanticEdgePatchSchema,
  semanticEdgeSchema,
  semanticNodeIdSchema,
  semanticNodePatchSchema,
  semanticNodeSchema,
  semanticRelationSchema,
  type SemanticEdge,
  type SemanticEdgeListOptions,
  type SemanticNode,
  type SemanticNodeListOptions,
  type SemanticNodePatch,
  type SemanticNodeSearchCandidate,
  type SemanticNodeSearchOptions,
} from "./types.js";
import { canonicalizeDomain } from "./domain.js";

const semanticEdgeInvalidationInputSchema = z.object({
  at: z.number().finite(),
  by_edge_id: semanticEdgeIdSchema.optional(),
  by_review_id: z.number().int().optional(),
  by_process: invalidationProcessSchema,
  reason: z.string().min(1).optional(),
});
type SemanticNodeRow = {
  id: string;
  kind: string;
  label: string;
  description: string;
  domain: string | null;
  aliases: string;
  confidence: number;
  source_episode_ids: string;
  created_at: number;
  updated_at: number;
  last_verified_at: number;
  embedding: number[];
  archived: number | boolean;
  superseded_by: string | null;
  _distance?: number;
};

const SEMANTIC_JSON_ARRAY_CODEC = {
  errorCode: "SEMANTIC_ROW_INVALID",
  errorMessage: (label: string) => `Failed to decode semantic ${label}`,
  createError: (message, options) => new SemanticError(message, options),
} satisfies JsonArrayCodecOptions;
const SEMANTIC_VECTOR_CODEC = {
  arrayLikeErrorMessage: "Semantic embedding must be array-like",
  nonFiniteErrorMessage: "Semantic embedding contains a non-finite value",
  errorCode: "SEMANTIC_ROW_INVALID",
  createError: (message, options) => new SemanticError(message, options),
} satisfies Float32ArrayCodecOptions;
const DEFAULT_VECTOR_SYNC_OUTBOX_LIMIT = 64;
const DEFAULT_CONFIDENCE_ADJUSTMENT_REASON = "semantic_node_confidence_adjustment";

function assertPositiveLimit(limit: number | undefined, label: string, fallback: number): number {
  const resolved = limit ?? fallback;

  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new SemanticError(`${label} must be a positive integer`, {
      code: "SEMANTIC_LIMIT_INVALID",
    });
  }

  return resolved;
}

function toSimilarity(distance: number | undefined): number {
  if (distance === undefined) {
    return 0;
  }

  return Math.max(0, Math.min(1, 1 - distance));
}

function getDistance(row: Record<string, unknown>): number | undefined {
  return typeof row._distance === "number" && Number.isFinite(row._distance)
    ? row._distance
    : undefined;
}

function normalizeAliases(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter((value) => value.length > 0))];
}

function normalizeVectorSyncOutboxLimit(limit: number | undefined): number {
  return assertPositiveLimit(
    limit,
    "Semantic node vector sync outbox limit",
    DEFAULT_VECTOR_SYNC_OUTBOX_LIMIT,
  );
}

function normalizeOutboxReason(reason: string | undefined): string {
  const normalized = reason?.trim() ?? "";

  return normalized.length === 0 ? DEFAULT_CONFIDENCE_ADJUSTMENT_REASON : normalized;
}

function parseConfidence(value: unknown, label: string): number {
  const confidence = Number(value);

  if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new SemanticError(`${label} must be a finite number between 0 and 1`, {
      code: "SEMANTIC_NODE_CONFIDENCE_INVALID",
    });
  }

  return confidence;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function errorCode(error: unknown): string | undefined {
  if (error instanceof Error && "code" in error) {
    const code = (error as { code?: unknown }).code;

    return typeof code === "string" ? code : undefined;
  }

  return undefined;
}

function nodeToRow(node: SemanticNode): SemanticNodeRow {
  return {
    id: node.id,
    kind: node.kind,
    label: node.label,
    description: node.description,
    domain: node.domain,
    aliases: serializeJsonValue(node.aliases),
    confidence: node.confidence,
    source_episode_ids: serializeJsonValue(node.source_episode_ids),
    created_at: node.created_at,
    updated_at: node.updated_at,
    last_verified_at: node.last_verified_at,
    embedding: Array.from(node.embedding),
    archived: node.archived ? 1 : 0,
    superseded_by: node.superseded_by,
  };
}

function nodeFromRow(row: Record<string, unknown>): SemanticNode {
  const parsed = semanticNodeSchema.safeParse({
    id: row.id,
    kind: row.kind,
    label: row.label,
    description: row.description,
    domain: row.domain === undefined ? null : row.domain,
    aliases: normalizeAliases(
      parseJsonArray<string>(String(row.aliases ?? "[]"), "aliases", SEMANTIC_JSON_ARRAY_CODEC),
    ),
    confidence: Number(row.confidence),
    source_episode_ids: parseJsonArray<string>(
      String(row.source_episode_ids ?? "[]"),
      "source_episode_ids",
      SEMANTIC_JSON_ARRAY_CODEC,
    ).map((value) => parseEpisodeId(value)),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
    last_verified_at: Number(row.last_verified_at),
    embedding: toFloat32Array(row.embedding, SEMANTIC_VECTOR_CODEC),
    archived: row.archived === true || Number(row.archived) === 1,
    superseded_by:
      row.superseded_by === null || row.superseded_by === undefined
        ? null
        : parseSemanticNodeId(String(row.superseded_by)),
  });

  if (!parsed.success) {
    throw new SemanticError("Semantic node row failed validation", {
      cause: parsed.error,
      code: "SEMANTIC_ROW_INVALID",
    });
  }

  return parsed.data;
}

function edgeFromRow(row: Record<string, unknown>): SemanticEdge {
  const parsed = semanticEdgeSchema.safeParse({
    id: row.id,
    from_node_id: row.from_node_id,
    to_node_id: row.to_node_id,
    relation: row.relation,
    confidence: Number(row.confidence),
    evidence_episode_ids: parseJsonArray<string>(
      String(row.evidence_episode_ids ?? "[]"),
      "evidence_episode_ids",
      SEMANTIC_JSON_ARRAY_CODEC,
    ).map((value) => parseEpisodeId(value)),
    created_at: Number(row.created_at),
    last_verified_at: Number(row.last_verified_at),
    valid_from: Number(row.valid_from),
    valid_to: row.valid_to === null || row.valid_to === undefined ? null : Number(row.valid_to),
    invalidated_at:
      row.invalidated_at === null || row.invalidated_at === undefined
        ? null
        : Number(row.invalidated_at),
    invalidated_by_edge_id:
      row.invalidated_by_edge_id === null || row.invalidated_by_edge_id === undefined
        ? null
        : parseSemanticEdgeId(String(row.invalidated_by_edge_id)),
    invalidated_by_review_id:
      row.invalidated_by_review_id === null || row.invalidated_by_review_id === undefined
        ? null
        : Number(row.invalidated_by_review_id),
    invalidated_by_process:
      row.invalidated_by_process === null || row.invalidated_by_process === undefined
        ? null
        : row.invalidated_by_process,
    invalidated_reason:
      row.invalidated_reason === null || row.invalidated_reason === undefined
        ? null
        : String(row.invalidated_reason),
  });

  if (!parsed.success) {
    throw new SemanticError("Semantic edge row failed validation", {
      cause: parsed.error,
      code: "SEMANTIC_EDGE_INVALID",
    });
  }

  return parsed.data;
}

export function createSemanticNodesTableSchema(dimensions: number) {
  return schema([
    utf8Field("id"),
    utf8Field("kind"),
    utf8Field("label"),
    utf8Field("description"),
    utf8Field("domain", true),
    utf8Field("aliases"),
    float64Field("confidence"),
    utf8Field("source_episode_ids"),
    float64Field("created_at"),
    float64Field("updated_at"),
    float64Field("last_verified_at"),
    booleanField("archived"),
    utf8Field("superseded_by", true),
    vectorField("embedding", dimensions),
  ]);
}

export type SemanticNodeRepositoryOptions = {
  table: LanceDbTable;
  db: SqliteDatabase;
  clock?: Clock;
};

export type SemanticNodeConfidenceAdjustmentInput = {
  id: SemanticNodeId;
  adjust: (currentConfidence: number) => number | null;
  updatedAt?: number;
  reason?: string;
};

export type SemanticNodeConfidenceAdjustment = {
  id: SemanticNodeId;
  previousConfidence: number;
  nextConfidence: number;
  updatedAt: number;
};

export type SemanticNodeVectorSyncFailure = {
  outboxId: number;
  nodeId: SemanticNodeId;
  message: string;
  code?: string;
};

export type SemanticNodeVectorSyncResult = {
  synced: number;
  failed: SemanticNodeVectorSyncFailure[];
  pending: number;
};

export type SemanticNodeVectorSyncOptions = {
  limit?: number;
  nodeIds?: readonly SemanticNodeId[];
};

type SemanticNodeVectorSyncOutboxRow = {
  id: number;
  node_id: string;
  reason: string;
  attempts: number;
  generation: number;
};

type SemanticNodeVectorSyncOutcome =
  | {
      kind: "synced";
    }
  | {
      kind: "source_missing";
      message: string;
      code: string;
    };

export class SemanticNodeRepository {
  private readonly clock: Clock;

  constructor(private readonly options: SemanticNodeRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get table(): LanceDbTable {
    return this.options.table;
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private getSqlNodeRow(id: SemanticNodeId): Record<string, unknown> | null {
    const row = this.db
      .prepare(
        `
          SELECT id, kind, label, description, domain, aliases, confidence, source_episode_ids,
                 created_at, updated_at, last_verified_at, archived, superseded_by
          FROM semantic_nodes
          WHERE id = ?
        `,
      )
      .get(id) as Record<string, unknown> | undefined;

    return row ?? null;
  }

  private enqueueVectorSyncRow(input: {
    nodeId: SemanticNodeId;
    reason?: string;
    createdAt: number;
  }): void {
    this.db
      .prepare(
        `
          INSERT INTO semantic_node_vector_sync_outbox (
            node_id, reason, created_at, generation, attempts, last_attempt_at, last_error
          ) VALUES (?, ?, ?, 1, 0, NULL, NULL)
          ON CONFLICT(node_id) DO UPDATE SET
            reason = excluded.reason,
            created_at = excluded.created_at,
            generation = semantic_node_vector_sync_outbox.generation + 1,
            attempts = 0,
            last_attempt_at = NULL,
            last_error = NULL
        `,
      )
      .run(input.nodeId, normalizeOutboxReason(input.reason), input.createdAt);
  }

  private enqueueVectorSyncTransactional(input: {
    nodeId: SemanticNodeId;
    reason?: string;
    createdAt: number;
  }): void {
    this.enqueueVectorSyncRow(input);
  }

  private enqueueVectorSyncRowForResync(nodeId: SemanticNodeId, reason: string): void {
    this.enqueueVectorSyncRow({
      nodeId,
      reason,
      createdAt: this.clock.now(),
    });
  }

  private countPendingVectorSyncs(nodeIds: readonly SemanticNodeId[] | undefined): number {
    if (nodeIds !== undefined) {
      if (nodeIds.length === 0) {
        return 0;
      }

      const placeholders = nodeIds.map(() => "?").join(", ");

      return (
        this.db
          .prepare(
            `SELECT COUNT(*) AS count FROM semantic_node_vector_sync_outbox WHERE node_id IN (${placeholders})`,
          )
          .get(...nodeIds) as { count: number }
      ).count;
    }

    return (
      this.db.prepare("SELECT COUNT(*) AS count FROM semantic_node_vector_sync_outbox").get() as {
        count: number;
      }
    ).count;
  }

  private listPendingVectorSyncs(options: {
    limit: number;
    nodeIds?: readonly SemanticNodeId[];
  }): SemanticNodeVectorSyncOutboxRow[] {
    if (options.nodeIds !== undefined) {
      if (options.nodeIds.length === 0) {
        return [];
      }

      const placeholders = options.nodeIds.map(() => "?").join(", ");

      return this.db
        .prepare(
          `
            SELECT id, node_id, reason, attempts, generation
            FROM semantic_node_vector_sync_outbox
            WHERE node_id IN (${placeholders})
            ORDER BY CASE WHEN attempts = 0 THEN 0 ELSE 1 END, created_at ASC, id ASC
            LIMIT ?
          `,
        )
        .all(...options.nodeIds, options.limit) as SemanticNodeVectorSyncOutboxRow[];
    }

    return this.db
      .prepare(
        `
          SELECT id, node_id, reason, attempts, generation
          FROM semantic_node_vector_sync_outbox
          ORDER BY CASE WHEN attempts = 0 THEN 0 ELSE 1 END, created_at ASC, id ASC
          LIMIT ?
        `,
      )
      .all(options.limit) as SemanticNodeVectorSyncOutboxRow[];
  }

  private recordVectorSyncFailure(row: SemanticNodeVectorSyncOutboxRow, error: unknown): boolean {
    const result = this.db
      .prepare(
        `
          UPDATE semantic_node_vector_sync_outbox
          SET attempts = attempts + 1,
              last_attempt_at = ?,
              last_error = ?
          WHERE id = ? AND generation = ?
        `,
      )
      .run(this.clock.now(), errorMessage(error), row.id, row.generation);

    return result.changes === 1;
  }

  private deleteVectorSyncOutboxRow(row: SemanticNodeVectorSyncOutboxRow): boolean {
    const result = this.db
      .prepare("DELETE FROM semantic_node_vector_sync_outbox WHERE id = ? AND generation = ?")
      .run(row.id, row.generation);

    return result.changes === 1;
  }

  private async syncVectorOutboxRow(
    row: SemanticNodeVectorSyncOutboxRow,
  ): Promise<SemanticNodeVectorSyncOutcome> {
    const nodeId = parseSemanticNodeId(row.node_id);
    const sqlRow = this.getSqlNodeRow(nodeId);

    if (sqlRow === null) {
      try {
        await this.table.remove(`id = ${quoteSqlString(nodeId)}`);
      } catch (error) {
        throw new SemanticError(
          `Semantic node ${nodeId} is missing in SQLite; LanceDB cleanup failed: ${errorMessage(error)}`,
          {
            cause: error,
            code: "SEMANTIC_NODE_VECTOR_SYNC_SOURCE_MISSING_CLEANUP_FAILED",
          },
        );
      }

      return {
        kind: "source_missing",
        message: `Semantic node ${nodeId} is missing in SQLite; cleared pending vector sync`,
        code: "SEMANTIC_NODE_VECTOR_SYNC_SOURCE_MISSING",
      };
    }

    const vectorRows = await this.table.list({
      where: `id = ${quoteSqlString(nodeId)}`,
      limit: 1,
    });
    const vectorRow = vectorRows[0];

    if (vectorRow === undefined) {
      throw new SemanticError(`Cannot sync semantic node ${nodeId}; LanceDB row is missing`, {
        code: "SEMANTIC_NODE_VECTOR_SYNC_TARGET_MISSING",
      });
    }

    const currentVectorNode = nodeFromRow(vectorRow);
    const next = nodeFromRow({
      ...sqlRow,
      embedding: currentVectorNode.embedding,
    });

    await this.table.upsert([nodeToRow(next)], {
      on: "id",
    });

    return {
      kind: "synced",
    };
  }

  private upsertSqlRow(node: SemanticNode): void {
    this.db
      .prepare(
        `
          INSERT INTO semantic_nodes (
            id, kind, label, description, domain, aliases, confidence, source_episode_ids,
            created_at, updated_at, last_verified_at, archived, superseded_by
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (id) DO UPDATE SET
            kind = excluded.kind,
            label = excluded.label,
            description = excluded.description,
            domain = excluded.domain,
            aliases = excluded.aliases,
            confidence = excluded.confidence,
            source_episode_ids = excluded.source_episode_ids,
            updated_at = excluded.updated_at,
            last_verified_at = excluded.last_verified_at,
            archived = excluded.archived,
            superseded_by = excluded.superseded_by
        `,
      )
      .run(
        node.id,
        node.kind,
        node.label,
        node.description,
        node.domain,
        serializeJsonValue(node.aliases),
        node.confidence,
        serializeJsonValue(node.source_episode_ids),
        node.created_at,
        node.updated_at,
        node.last_verified_at,
        node.archived ? 1 : 0,
        node.superseded_by,
      );
  }

  async insert(input: z.input<typeof semanticNodeSchema>): Promise<SemanticNode> {
    const parsed = semanticNodeSchema.parse(input);
    const normalizedNode = semanticNodeSchema.parse({
      ...parsed,
      domain: canonicalizeDomain(parsed.domain),
    });
    const row = nodeToRow(normalizedNode);

    try {
      await this.table.upsert([row], {
        on: "id",
      });

      try {
        const apply = this.db.transaction(() => {
          this.upsertSqlRow(normalizedNode);
        });

        apply();
      } catch (error) {
        await this.table.remove(`id = ${quoteSqlString(normalizedNode.id)}`);
        throw error;
      }
    } catch (error) {
      throw new SemanticError(`Failed to insert semantic node ${normalizedNode.id}`, {
        cause: error,
        code: "SEMANTIC_NODE_INSERT_FAILED",
      });
    }

    return normalizedNode;
  }

  async restore(input: z.input<typeof semanticNodeSchema>): Promise<SemanticNode> {
    const parsed = semanticNodeSchema.parse(input);
    const normalizedNode = semanticNodeSchema.parse({
      ...parsed,
      domain: canonicalizeDomain(parsed.domain),
    });
    const current = await this.get(normalizedNode.id);
    const previousRow = current === null ? null : nodeToRow(current);

    try {
      await this.table.upsert([nodeToRow(normalizedNode)], {
        on: "id",
      });

      try {
        const apply = this.db.transaction(() => {
          this.upsertSqlRow(normalizedNode);
        });
        apply();
      } catch (error) {
        if (previousRow === null) {
          await this.table.remove(`id = ${quoteSqlString(normalizedNode.id)}`);
        } else {
          await this.table.upsert([previousRow], {
            on: "id",
          });
        }
        throw error;
      }
    } catch (error) {
      throw new SemanticError(`Failed to restore semantic node ${normalizedNode.id}`, {
        cause: error,
        code: "SEMANTIC_NODE_RESTORE_FAILED",
      });
    }

    return normalizedNode;
  }

  async get(id: SemanticNodeId): Promise<SemanticNode | null> {
    const rows = await this.table.list({
      where: `id = ${quoteSqlString(id)}`,
      limit: 1,
    });
    const row = rows[0];

    return row === undefined ? null : nodeFromRow(row);
  }

  getStoredConfidence(id: SemanticNodeId): number | null {
    const row = this.getSqlNodeRow(semanticNodeIdSchema.parse(id));

    return row === null ? null : parseConfidence(row.confidence, "Semantic node confidence");
  }

  adjustConfidenceTransactional(
    input: SemanticNodeConfidenceAdjustmentInput,
  ): SemanticNodeConfidenceAdjustment | null {
    if (!this.db.raw.inTransaction) {
      throw new SemanticError(
        "Semantic node confidence adjustments must run inside a SQLite transaction",
        {
          code: "SEMANTIC_NODE_CONFIDENCE_ADJUSTMENT_REQUIRES_TRANSACTION",
        },
      );
    }

    const id = semanticNodeIdSchema.parse(input.id);
    const updatedAt = input.updatedAt ?? this.clock.now();

    if (!Number.isFinite(updatedAt)) {
      throw new SemanticError("Semantic node confidence update time must be finite", {
        code: "SEMANTIC_NODE_CONFIDENCE_UPDATED_AT_INVALID",
      });
    }

    const current = this.getSqlNodeRow(id);

    if (current === null) {
      return null;
    }

    const previousConfidence = parseConfidence(
      current.confidence,
      "Current semantic node confidence",
    );
    const adjusted = input.adjust(previousConfidence);

    if (adjusted === null) {
      return null;
    }

    const nextConfidence = parseConfidence(adjusted, "Next semantic node confidence");

    if (nextConfidence === previousConfidence) {
      return null;
    }

    const result = this.db
      .prepare("UPDATE semantic_nodes SET confidence = ?, updated_at = ? WHERE id = ?")
      .run(nextConfidence, updatedAt, id);

    if (result.changes !== 1) {
      return null;
    }

    this.enqueueVectorSyncTransactional({
      nodeId: id,
      reason: input.reason,
      createdAt: updatedAt,
    });

    return {
      id,
      previousConfidence,
      nextConfidence,
      updatedAt,
    };
  }

  async syncPendingVectorUpdates(
    options: SemanticNodeVectorSyncOptions = {},
  ): Promise<SemanticNodeVectorSyncResult> {
    const limit = normalizeVectorSyncOutboxLimit(options.limit);
    const nodeIds =
      options.nodeIds === undefined
        ? undefined
        : [...new Set(options.nodeIds.map((id) => semanticNodeIdSchema.parse(id)))];
    const rows = this.listPendingVectorSyncs({
      limit,
      nodeIds,
    });
    const failed: SemanticNodeVectorSyncFailure[] = [];
    let synced = 0;

    for (const row of rows) {
      const nodeId = parseSemanticNodeId(row.node_id);

      try {
        const outcome = await this.syncVectorOutboxRow(row);
        const deleted = this.deleteVectorSyncOutboxRow(row);

        if (!deleted) {
          this.enqueueVectorSyncRowForResync(nodeId, "stale_upsert_repair");
          continue;
        }

        if (outcome.kind === "source_missing") {
          failed.push({
            outboxId: row.id,
            nodeId,
            message: outcome.message,
            code: outcome.code,
          });
          continue;
        }

        synced += 1;
      } catch (error) {
        if (!this.recordVectorSyncFailure(row, error)) {
          continue;
        }

        failed.push({
          outboxId: row.id,
          nodeId,
          message: errorMessage(error),
          code: errorCode(error),
        });
      }
    }

    return {
      synced,
      failed,
      pending: this.countPendingVectorSyncs(nodeIds),
    };
  }

  async getMany(
    ids: readonly SemanticNodeId[],
    options: { includeArchived?: boolean } = {},
  ): Promise<Array<SemanticNode | null>> {
    if (ids.length === 0) {
      return [];
    }

    const where = `id IN (${ids.map((id) => quoteSqlString(id)).join(", ")})`;
    const rows = await this.table.list({
      where,
    });
    const byId = new Map(rows.map((row) => [String(row.id), nodeFromRow(row)]));

    return ids.map((id) => {
      const node = byId.get(id) ?? null;

      if (node === null) {
        return null;
      }

      if (options.includeArchived !== true && node.archived) {
        return null;
      }

      return node;
    });
  }

  async findByExactLabelOrAlias(
    query: string,
    limit = 10,
    options: { includeArchived?: boolean } = {},
  ): Promise<SemanticNode[]> {
    const normalized = query.trim().toLowerCase();

    if (normalized.length === 0) {
      return [];
    }

    const rows = this.db
      .prepare(
        `
          SELECT id, kind, label, description, aliases, confidence, source_episode_ids,
                 created_at, updated_at, last_verified_at, archived, superseded_by, domain
          FROM semantic_nodes
          ORDER BY updated_at DESC, id ASC
        `,
      )
      .all() as Record<string, unknown>[];
    const matchedIds: SemanticNodeId[] = [];

    for (const row of rows) {
      const archived = row.archived === true || Number(row.archived) === 1;

      if (options.includeArchived !== true && archived) {
        continue;
      }

      const label = String(row.label ?? "").toLowerCase();
      const aliases = parseJsonArray<string>(
        String(row.aliases ?? "[]"),
        "aliases",
        SEMANTIC_JSON_ARRAY_CODEC,
      ).map((value) => value.toLowerCase());

      if (label === normalized || aliases.includes(normalized)) {
        matchedIds.push(parseSemanticNodeId(String(row.id)));
      }

      if (matchedIds.length >= limit) {
        break;
      }
    }

    return (await this.getMany(matchedIds, options)).filter(
      (value): value is SemanticNode => value !== null,
    );
  }

  async searchByVector(
    vector: Float32Array,
    options: SemanticNodeSearchOptions = {},
  ): Promise<SemanticNodeSearchCandidate[]> {
    const limit = assertPositiveLimit(options.limit, "Semantic search limit", 10);
    const searchLimit = Math.max(limit * 5, 20);
    const rows = await this.table.search(Array.from(vector), {
      limit: searchLimit,
      vectorColumn: "embedding",
      distanceType: "cosine",
    });
    const results: SemanticNodeSearchCandidate[] = [];

    for (const row of rows) {
      const node = nodeFromRow(row);
      const similarity = toSimilarity(getDistance(row));

      if (options.minSimilarity !== undefined && similarity < options.minSimilarity) {
        continue;
      }

      if (options.includeArchived !== true && node.archived) {
        continue;
      }

      if (
        options.kindFilter !== undefined &&
        options.kindFilter.length > 0 &&
        !options.kindFilter.includes(node.kind)
      ) {
        continue;
      }

      results.push({
        node,
        similarity,
      });

      if (results.length >= limit) {
        break;
      }
    }

    return results;
  }

  async list(options: SemanticNodeListOptions = {}): Promise<SemanticNode[]> {
    const filters: string[] = [];
    const values: unknown[] = [];
    const limit = assertPositiveLimit(options.limit, "Semantic list limit", 50);

    if (options.kind !== undefined) {
      filters.push("kind = ?");
      values.push(options.kind);
    }

    if (options.includeArchived !== true) {
      filters.push("archived = 0");
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT id
          FROM semantic_nodes
          ${whereClause}
          ORDER BY updated_at DESC, id ASC
          LIMIT ?
        `,
      )
      .all(...values, limit) as Array<{ id: string }>;

    return (await this.getMany(rows.map((row) => parseSemanticNodeId(row.id)))).filter(
      (value): value is SemanticNode => value !== null,
    );
  }

  async update(id: SemanticNodeId, patch: SemanticNodePatch): Promise<SemanticNode | null> {
    const current = await this.get(id);

    if (current === null) {
      return null;
    }

    const parsedPatch = semanticNodePatchSchema.parse(patch);
    const next = semanticNodeSchema.parse({
      ...current,
      ...parsedPatch,
      domain: canonicalizeDomain(parsedPatch.domain ?? current.domain),
      aliases:
        parsedPatch.aliases === undefined
          ? current.aliases
          : parsedPatch.replace_aliases === true
            ? normalizeAliases(parsedPatch.aliases)
            : normalizeAliases([...current.aliases, ...parsedPatch.aliases]),
      source_episode_ids:
        parsedPatch.source_episode_ids === undefined
          ? current.source_episode_ids
          : parsedPatch.replace_source_episode_ids === true
            ? [...new Set(parsedPatch.source_episode_ids)]
            : [...new Set([...current.source_episode_ids, ...parsedPatch.source_episode_ids])],
      updated_at: this.clock.now(),
    });
    const previousRow = nodeToRow(current);

    try {
      await this.table.upsert([nodeToRow(next)], {
        on: "id",
      });

      try {
        const apply = this.db.transaction(() => {
          this.upsertSqlRow(next);
        });
        apply();
      } catch (error) {
        await this.table.upsert([previousRow], {
          on: "id",
        });
        throw error;
      }
    } catch (error) {
      throw new SemanticError(`Failed to update semantic node ${id}`, {
        cause: error,
        code: "SEMANTIC_NODE_UPDATE_FAILED",
      });
    }

    return next;
  }

  async delete(id: SemanticNodeId): Promise<boolean> {
    const current = await this.get(id);

    if (current === null) {
      return false;
    }

    try {
      const apply = this.db.transaction(() => {
        this.db.prepare("DELETE FROM semantic_nodes WHERE id = ?").run(id);
        this.db.prepare("DELETE FROM semantic_node_vector_sync_outbox WHERE node_id = ?").run(id);
        this.db
          .prepare("DELETE FROM semantic_edges WHERE from_node_id = ? OR to_node_id = ?")
          .run(id, id);
      });
      apply();
      await this.table.remove(`id = ${quoteSqlString(id)}`);
      return true;
    } catch (error) {
      throw new SemanticError(`Failed to delete semantic node ${id}`, {
        cause: error,
        code: "SEMANTIC_NODE_DELETE_FAILED",
      });
    }
  }
}

export type SemanticEdgeRepositoryOptions = {
  db: SqliteDatabase;
  clock?: Clock;
  enqueueReview?: (input: ReviewQueueInsertInput) => unknown;
};

type SemanticEdgeValidityKey =
  | "valid_from"
  | "valid_to"
  | "invalidated_at"
  | "invalidated_by_edge_id"
  | "invalidated_by_review_id"
  | "invalidated_by_process"
  | "invalidated_reason";

export type SemanticEdgeInsertInput = Omit<SemanticEdge, "id" | SemanticEdgeValidityKey> &
  Partial<Pick<SemanticEdge, SemanticEdgeValidityKey>> & {
    id?: SemanticEdgeId;
  };

export type SemanticEdgeInvalidationInput = z.input<typeof semanticEdgeInvalidationInputSchema>;

export class SemanticEdgeRepository {
  private readonly clock: Clock;

  constructor(private readonly options: SemanticEdgeRepositoryOptions) {
    this.clock = options.clock ?? new SystemClock();
  }

  private get db(): SqliteDatabase {
    return this.options.db;
  }

  private assertNodeExists(id: SemanticNodeId, field: "from_node_id" | "to_node_id"): void {
    const row = this.db.prepare("SELECT id FROM semantic_nodes WHERE id = ?").get(id) as
      | { id: string }
      | undefined;

    if (row !== undefined) {
      return;
    }

    throw new SemanticError(`Semantic edge ${field} does not exist: ${id}`, {
      code: "SEMANTIC_EDGE_DANGLING",
    });
  }

  private hasSupportPath(fromId: SemanticNodeId, toId: SemanticNodeId, maxDepth = 3): boolean {
    const queue: Array<{ id: SemanticNodeId; depth: number }> = [{ id: fromId, depth: 0 }];
    const visited = new Set<string>([fromId]);

    while (queue.length > 0) {
      const next = queue.shift();

      if (next === undefined || next.depth >= maxDepth) {
        continue;
      }

      const edges = this.listEdges({
        fromId: next.id,
        relation: "supports",
      });

      for (const edge of edges) {
        if (edge.to_node_id === toId) {
          return true;
        }

        if (visited.has(edge.to_node_id)) {
          continue;
        }

        visited.add(edge.to_node_id);
        queue.push({
          id: edge.to_node_id,
          depth: next.depth + 1,
        });
      }
    }

    return false;
  }

  private getNodeLabel(id: SemanticNodeId): string | null {
    const row = this.db.prepare("SELECT label FROM semantic_nodes WHERE id = ?").get(id) as
      | { label: string }
      | undefined;

    return row?.label ?? null;
  }

  private insertEdgeRow(edge: SemanticEdge): void {
    this.db
      .prepare(
        `
          INSERT INTO semantic_edges (
            id,
            from_node_id,
            to_node_id,
            relation,
            confidence,
            evidence_episode_ids,
            created_at,
            last_verified_at,
            valid_from,
            valid_to,
            invalidated_at,
            invalidated_by_edge_id,
            invalidated_by_review_id,
            invalidated_by_process,
            invalidated_reason
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        edge.id,
        edge.from_node_id,
        edge.to_node_id,
        edge.relation,
        edge.confidence,
        serializeJsonValue(edge.evidence_episode_ids),
        edge.created_at,
        edge.last_verified_at,
        edge.valid_from,
        edge.valid_to,
        edge.invalidated_at,
        edge.invalidated_by_edge_id,
        edge.invalidated_by_review_id,
        edge.invalidated_by_process,
        edge.invalidated_reason,
      );
  }

  private insertSupportDependency(edge: SemanticEdge): void {
    if (edge.relation !== "supports") {
      return;
    }

    this.db
      .prepare(
        `
          INSERT OR IGNORE INTO semantic_belief_dependencies (
            target_type,
            target_id,
            source_edge_id,
            dependency_kind,
            created_at
          ) VALUES ('semantic_node', ?, ?, 'supports', ?)
        `,
      )
      .run(edge.to_node_id, edge.id, edge.created_at);
  }

  private insertEdgeWithDependencies(edge: SemanticEdge): void {
    const write = () => {
      this.insertEdgeRow(edge);
      this.insertSupportDependency(edge);
    };

    if (this.db.raw.inTransaction) {
      write();
      return;
    }

    this.db.transaction(write)();
  }

  addEdge(input: SemanticEdgeInsertInput): SemanticEdge {
    const now = this.clock.now();
    const edge = semanticEdgeSchema.parse({
      ...input,
      id: input.id ?? createSemanticEdgeId(),
      valid_from: input.valid_from ?? now,
      valid_to: input.valid_to ?? null,
      invalidated_at: input.invalidated_at ?? null,
      invalidated_by_edge_id: input.invalidated_by_edge_id ?? null,
      invalidated_by_review_id: input.invalidated_by_review_id ?? null,
      invalidated_by_process: input.invalidated_by_process ?? null,
      invalidated_reason: input.invalidated_reason ?? null,
    });

    this.assertNodeExists(edge.from_node_id, "from_node_id");
    this.assertNodeExists(edge.to_node_id, "to_node_id");

    if (edge.valid_to === null) {
      const duplicate = this.db
        .prepare(
          `
            SELECT id
            FROM semantic_edges
            WHERE from_node_id = ? AND to_node_id = ? AND relation = ? AND valid_to IS NULL
          `,
        )
        .get(edge.from_node_id, edge.to_node_id, edge.relation);

      if (duplicate !== undefined) {
        throw new SemanticError("Duplicate semantic edge", {
          code: "SEMANTIC_EDGE_DUPLICATE",
        });
      }
    }

    this.insertEdgeWithDependencies(edge);

    if (edge.relation === "contradicts" && this.options.enqueueReview !== undefined) {
      const conflictsWithSupportChain =
        this.hasSupportPath(edge.from_node_id, edge.to_node_id) ||
        this.hasSupportPath(edge.to_node_id, edge.from_node_id);

      this.options.enqueueReview({
        kind: "contradiction",
        refs: {
          node_ids: [edge.from_node_id, edge.to_node_id],
          node_labels: [
            this.getNodeLabel(edge.from_node_id) ?? edge.from_node_id,
            this.getNodeLabel(edge.to_node_id) ?? edge.to_node_id,
          ],
          edge_id: edge.id,
        },
        reason: conflictsWithSupportChain
          ? "Direct contradiction edge recorded for review; conflicts_with_support_chain"
          : "Direct contradiction edge recorded for review",
      });
    }

    return edge;
  }

  getEdge(id: SemanticEdgeId): SemanticEdge | null {
    const row = this.db.prepare("SELECT * FROM semantic_edges WHERE id = ?").get(id) as
      | Record<string, unknown>
      | undefined;

    return row === undefined ? null : edgeFromRow(row);
  }

  listEdges(options: SemanticEdgeListOptions = {}): SemanticEdge[] {
    if (options.relation !== undefined) {
      semanticRelationSchema.parse(options.relation);
    }

    if (options.asOf !== undefined && !Number.isFinite(options.asOf)) {
      throw new SemanticError("Semantic edge asOf must be finite", {
        code: "SEMANTIC_EDGE_AS_OF_INVALID",
      });
    }

    const filters: string[] = [];
    const values: unknown[] = [];

    if (options.fromId !== undefined) {
      filters.push("from_node_id = ?");
      values.push(options.fromId);
    }

    if (options.toId !== undefined) {
      filters.push("to_node_id = ?");
      values.push(options.toId);
    }

    if (options.relation !== undefined) {
      filters.push("relation = ?");
      values.push(options.relation);
    }

    if (options.includeInvalid !== true) {
      const asOf = options.asOf ?? this.clock.now();

      filters.push("valid_from <= ?");
      values.push(asOf);
      filters.push("(valid_to IS NULL OR valid_to > ?)");
      values.push(asOf);
    }

    const whereClause = filters.length === 0 ? "" : `WHERE ${filters.join(" AND ")}`;
    const rows = this.db
      .prepare(
        `
          SELECT *
          FROM semantic_edges
          ${whereClause}
          ORDER BY created_at ASC, id ASC
        `,
      )
      .all(...values) as Record<string, unknown>[];

    return rows.map((row) => edgeFromRow(row));
  }

  invalidateEdge(id: SemanticEdgeId, input: SemanticEdgeInvalidationInput): SemanticEdge | null {
    const current = this.getEdge(id);

    if (current === null) {
      return null;
    }

    if (current.valid_to !== null) {
      return current;
    }

    const parsed = semanticEdgeInvalidationInputSchema.parse(input);

    if (parsed.at < current.valid_from) {
      throw new SemanticError("Semantic edge invalidation 'at' precedes valid_from", {
        code: "SEMANTIC_EDGE_INVALIDATE_BEFORE_VALID_FROM",
      });
    }

    const invalidatedAt = this.clock.now();

    this.db
      .prepare(
        `
          UPDATE semantic_edges
          SET valid_to = ?,
              invalidated_at = ?,
              invalidated_by_edge_id = ?,
              invalidated_by_review_id = ?,
              invalidated_by_process = ?,
              invalidated_reason = ?
          WHERE id = ? AND valid_to IS NULL
        `,
      )
      .run(
        parsed.at,
        invalidatedAt,
        parsed.by_edge_id ?? null,
        parsed.by_review_id ?? null,
        parsed.by_process,
        parsed.reason ?? null,
        id,
      );

    return semanticEdgeSchema.parse({
      ...current,
      valid_to: parsed.at,
      invalidated_at: invalidatedAt,
      invalidated_by_edge_id: parsed.by_edge_id ?? null,
      invalidated_by_review_id: parsed.by_review_id ?? null,
      invalidated_by_process: parsed.by_process,
      invalidated_reason: parsed.reason ?? null,
    });
  }

  delete(id: SemanticEdgeId): boolean {
    const result = this.db.prepare("DELETE FROM semantic_edges WHERE id = ?").run(id);
    return result.changes > 0;
  }

  updateConfidence(
    id: SemanticEdgeId,
    confidence: number,
    lastVerifiedAt = this.clock.now(),
  ): SemanticEdge | null {
    const current = this.getEdge(id);

    if (current === null) {
      return null;
    }

    const next = semanticEdgePatchSchema.parse({
      confidence,
      last_verified_at: lastVerifiedAt,
    });
    const merged = semanticEdgeSchema.parse({
      ...current,
      ...next,
    });

    this.db
      .prepare(
        `
          UPDATE semantic_edges
          SET confidence = ?, last_verified_at = ?
          WHERE id = ?
        `,
      )
      .run(merged.confidence, merged.last_verified_at, id);

    return merged;
  }
}
