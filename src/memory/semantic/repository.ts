import {
  LanceDbTable,
  booleanField,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import { SqliteDatabase } from "../../storage/sqlite/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { SemanticError, StorageError } from "../../util/errors.js";
import { serializeJsonValue } from "../../util/json-value.js";
import {
  createSemanticEdgeId,
  createSemanticNodeId,
  parseEpisodeId,
  parseSemanticEdgeId,
  parseSemanticNodeId,
  type EpisodeId,
  type SemanticEdgeId,
  type SemanticNodeId,
} from "../../util/ids.js";
import { tokenizeText } from "../../util/text/tokenize.js";
import type { ReviewQueueInsertInput } from "./review-queue.js";
import {
  semanticEdgePatchSchema,
  semanticEdgeSchema,
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
  type SemanticRelation,
} from "./types.js";

type SemanticNodeRow = {
  id: string;
  kind: string;
  label: string;
  description: string;
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

function assertPositiveLimit(limit: number | undefined, label: string, fallback: number): number {
  const resolved = limit ?? fallback;

  if (!Number.isInteger(resolved) || resolved <= 0) {
    throw new SemanticError(`${label} must be a positive integer`, {
      code: "SEMANTIC_LIMIT_INVALID",
    });
  }

  return resolved;
}

function quoteSqlString(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

function parseJsonArray<T>(value: string, label: string): T[] {
  try {
    const parsed = JSON.parse(value) as unknown;

    if (!Array.isArray(parsed)) {
      throw new TypeError(`${label} must be an array`);
    }

    return parsed as T[];
  } catch (error) {
    throw new SemanticError(`Failed to decode semantic ${label}`, {
      cause: error,
      code: "SEMANTIC_ROW_INVALID",
    });
  }
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
    throw new SemanticError("Semantic embedding must be array-like", {
      code: "SEMANTIC_ROW_INVALID",
    });
  }

  return Float32Array.from(
    candidate.map((value) => {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new SemanticError("Semantic embedding contains a non-finite value", {
          code: "SEMANTIC_ROW_INVALID",
        });
      }

      return value;
    }),
  );
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

function nodeToRow(node: SemanticNode): SemanticNodeRow {
  return {
    id: node.id,
    kind: node.kind,
    label: node.label,
    description: node.description,
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
    aliases: normalizeAliases(parseJsonArray<string>(String(row.aliases ?? "[]"), "aliases")),
    confidence: Number(row.confidence),
    source_episode_ids: parseJsonArray<string>(
      String(row.source_episode_ids ?? "[]"),
      "source_episode_ids",
    ).map((value) => parseEpisodeId(value)),
    created_at: Number(row.created_at),
    updated_at: Number(row.updated_at),
    last_verified_at: Number(row.last_verified_at),
    embedding: toFloat32Array(row.embedding),
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
    ).map((value) => parseEpisodeId(value)),
    created_at: Number(row.created_at),
    last_verified_at: Number(row.last_verified_at),
  });

  if (!parsed.success) {
    throw new SemanticError("Semantic edge row failed validation", {
      cause: parsed.error,
      code: "SEMANTIC_EDGE_INVALID",
    });
  }

  return parsed.data;
}

function hasNegationConflict(
  left: Pick<SemanticNode, "label" | "description">,
  right: Pick<SemanticNode, "label" | "description">,
): boolean {
  const leftText = `${left.label} ${left.description}`.toLowerCase();
  const rightText = `${right.label} ${right.description}`.toLowerCase();
  const leftNegated = /\b(no|not|never|without|cannot|can't|won't)\b/.test(leftText);
  const rightNegated = /\b(no|not|never|without|cannot|can't|won't)\b/.test(rightText);

  if (leftNegated === rightNegated) {
    return false;
  }

  const leftTokens = tokenizeText(leftText);
  const rightTokens = tokenizeText(rightText);
  let overlap = 0;

  for (const token of leftTokens) {
    if (rightTokens.has(token)) {
      overlap += 1;
    }
  }

  return overlap >= 2;
}

export function createSemanticNodesTableSchema(dimensions: number) {
  return schema([
    utf8Field("id"),
    utf8Field("kind"),
    utf8Field("label"),
    utf8Field("description"),
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
  enqueueReview?: (input: ReviewQueueInsertInput) => ReviewQueueInsertInput | unknown;
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

  private maybeQueueDuplicateReview(node: SemanticNode): void {
    void this.detectDuplicateReview(node);
  }

  private async detectDuplicateReview(node: SemanticNode): Promise<void> {
    if (this.options.enqueueReview === undefined || node.kind !== "proposition") {
      return;
    }

    const matches = await this.searchByVector(node.embedding, {
      limit: 3,
      minSimilarity: 0.9,
      kindFilter: ["proposition"],
      includeArchived: false,
    });

    for (const match of matches) {
      if (match.node.id === node.id) {
        continue;
      }

      const substantiveDifference =
        match.node.label.toLowerCase() !== node.label.toLowerCase() ||
        match.node.description.toLowerCase() !== node.description.toLowerCase();

      if (!substantiveDifference || !hasNegationConflict(node, match.node)) {
        continue;
      }

      this.options.enqueueReview({
        kind: "duplicate",
        refs: {
          node_ids: [node.id, match.node.id],
        },
        reason: `Nearby proposition appears to conflict with ${match.node.label}`,
      });
      break;
    }
  }

  private upsertSqlRow(node: SemanticNode): void {
    this.db
      .prepare(
        `
          INSERT INTO semantic_nodes (
            id, kind, label, description, aliases, confidence, source_episode_ids,
            created_at, updated_at, last_verified_at, archived, superseded_by
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
          ON CONFLICT (id) DO UPDATE SET
            kind = excluded.kind,
            label = excluded.label,
            description = excluded.description,
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

  async insert(input: SemanticNode): Promise<SemanticNode> {
    const parsed = semanticNodeSchema.parse(input);
    const row = nodeToRow(parsed);

    try {
      await this.table.upsert([row], {
        on: "id",
      });

      try {
        const apply = this.db.transaction(() => {
          this.upsertSqlRow(parsed);
        });

        apply();
      } catch (error) {
        await this.table.remove(`id = ${quoteSqlString(parsed.id)}`);
        throw error;
      }
    } catch (error) {
      throw new SemanticError(`Failed to insert semantic node ${parsed.id}`, {
        cause: error,
        code: "SEMANTIC_NODE_INSERT_FAILED",
      });
    }

    this.maybeQueueDuplicateReview(parsed);
    return parsed;
  }

  async get(id: SemanticNodeId): Promise<SemanticNode | null> {
    const rows = await this.table.list({
      where: `id = ${quoteSqlString(id)}`,
      limit: 1,
    });
    const row = rows[0];

    return row === undefined ? null : nodeFromRow(row);
  }

  async getMany(ids: readonly SemanticNodeId[]): Promise<Array<SemanticNode | null>> {
    if (ids.length === 0) {
      return [];
    }

    const where = `id IN (${ids.map((id) => quoteSqlString(id)).join(", ")})`;
    const rows = await this.table.list({
      where,
    });
    const byId = new Map(rows.map((row) => [String(row.id), nodeFromRow(row)]));
    return ids.map((id) => byId.get(id) ?? null);
  }

  async findByLabelOrAlias(query: string, limit = 10): Promise<SemanticNode[]> {
    const normalized = query.trim().toLowerCase();

    if (normalized.length === 0) {
      return [];
    }

    const rows = this.db
      .prepare(
        `
          SELECT id, kind, label, description, aliases, confidence, source_episode_ids,
                 created_at, updated_at, last_verified_at, archived, superseded_by
          FROM semantic_nodes
          ORDER BY updated_at DESC, id ASC
        `,
      )
      .all() as Record<string, unknown>[];
    const matchedIds: SemanticNodeId[] = [];

    for (const row of rows) {
      const label = String(row.label ?? "").toLowerCase();
      const aliases = parseJsonArray<string>(String(row.aliases ?? "[]"), "aliases").map((value) =>
        value.toLowerCase(),
      );

      if (
        label === normalized ||
        label.includes(normalized) ||
        aliases.includes(normalized) ||
        aliases.some((alias) => alias.includes(normalized))
      ) {
        matchedIds.push(parseSemanticNodeId(String(row.id)));
      }

      if (matchedIds.length >= limit) {
        break;
      }
    }

    return (await this.getMany(matchedIds)).filter(
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
      aliases:
        parsedPatch.aliases === undefined
          ? current.aliases
          : normalizeAliases([...current.aliases, ...parsedPatch.aliases]),
      source_episode_ids:
        parsedPatch.source_episode_ids === undefined
          ? current.source_episode_ids
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

  addEdge(input: Omit<SemanticEdge, "id"> & { id?: SemanticEdgeId }): SemanticEdge {
    const edge = semanticEdgeSchema.parse({
      ...input,
      id: input.id ?? createSemanticEdgeId(),
    });

    this.assertNodeExists(edge.from_node_id, "from_node_id");
    this.assertNodeExists(edge.to_node_id, "to_node_id");
    const duplicate = this.db
      .prepare(
        `
          SELECT id
          FROM semantic_edges
          WHERE from_node_id = ? AND to_node_id = ? AND relation = ?
        `,
      )
      .get(edge.from_node_id, edge.to_node_id, edge.relation);

    if (duplicate !== undefined) {
      throw new SemanticError("Duplicate semantic edge", {
        code: "SEMANTIC_EDGE_DUPLICATE",
      });
    }

    this.db
      .prepare(
        `
          INSERT INTO semantic_edges (
            id, from_node_id, to_node_id, relation, confidence, evidence_episode_ids, created_at, last_verified_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
      );

    if (edge.relation === "contradicts" && this.options.enqueueReview !== undefined) {
      const conflictsWithSupportChain =
        this.hasSupportPath(edge.from_node_id, edge.to_node_id) ||
        this.hasSupportPath(edge.to_node_id, edge.from_node_id);

      this.options.enqueueReview({
        kind: "contradiction",
        refs: {
          node_ids: [edge.from_node_id, edge.to_node_id],
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
