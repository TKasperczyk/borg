/** Read-only evidence assembly. Label equality selects work; it never establishes identity. */
import { z } from "zod";
import { EntityRepository } from "../src/memory/commitments/repository.js";
import type { EpisodicRepository } from "../src/memory/episodic/repository.js";
import { episodeIdSchema } from "../src/memory/episodic/types.js";
import { semanticNodeIdSchema } from "../src/memory/semantic/types.js";
import type { SqliteDatabase } from "../src/storage/sqlite/index.js";
import { utf16SafePrefixEnd } from "../src/util/utf16-boundary.js";
import { fingerprint, refuse } from "./audited-data-repair.js";
import type { Row } from "./repair-entity-graph-storage.js";

export const DOSSIER_EPISODE_LIMIT = 6;
export const DOSSIER_EXCERPT_CHARS = 800;
const hashSchema = z.string().regex(/^[a-f0-9]{64}$/);
const excerptSchema = z.object({ text: z.string(), truncated: z.boolean() }).strict();
export const verdictSchema = z
  .object({
    verdict: z.enum(["self", "other", "unsure"]),
    reason: z.string().min(1),
    confidence: z.number().min(0).max(1),
  })
  .strict();
export const dossierSchema = z
  .object({
    version: z.literal(1),
    node: z
      .object({
        id: semanticNodeIdSchema,
        kind: z.string(),
        label: z.string(),
        description: excerptSchema,
        domain: z.string().nullable(),
        aliases: z.array(z.string()),
        confidence: z.number(),
        created_at: z.number(),
        source_episode_ids: z.array(episodeIdSchema),
        snapshot_hash: hashSchema,
      })
      .strict(),
    self: z
      .object({
        entity_id: z.string(),
        canonical_name: z.string(),
        aliases: z.array(z.string()),
        node_id: semanticNodeIdSchema,
        description: excerptSchema,
        snapshot_hash: hashSchema,
      })
      .strict(),
    edges: z.array(
      z
        .object({
          id: z.string(),
          direction: z.enum(["incoming", "outgoing"]),
          relation: z.string(),
          other: z.object({ id: z.string(), label: z.string(), kind: z.string() }).strict(),
          confidence: z.number(),
          valid_from: z.number(),
          valid_to: z.number().nullable(),
          snapshot_hash: hashSchema,
        })
        .strict(),
    ),
    episodes: z.array(
      z
        .object({
          id: episodeIdSchema,
          missing: z.boolean(),
          title: excerptSchema.nullable(),
          excerpt: excerptSchema.nullable(),
          excerpt_source: z.enum(["gist", "narrative"]).nullable(),
          archived: z.boolean().nullable(),
          origin_audience_entity_ids: z.array(z.string()),
          shared: z.boolean().nullable(),
          snapshot_hash: hashSchema.nullable(),
        })
        .strict(),
    ),
    episodes_omitted: z.number().int().nonnegative(),
  })
  .strict();
export type SelfLabelDossier = z.infer<typeof dossierSchema>;
export const verdictRecordSchema = verdictSchema
  .extend({
    node_id: semanticNodeIdSchema,
    dossier: dossierSchema,
    dossier_hash: hashSchema,
    model: z.string().min(1),
    judged_at: z.number().finite(),
  })
  .strict();
export const verdictFileSchema = z
  .object({
    version: z.literal(1),
    kind: z.literal("judge-self-label-nodes"),
    data_dir: z.string().min(1),
    started_at: z.number().finite(),
    completed_at: z.number().finite(),
    model: z.string().min(1),
    model_role: z.literal("extraction"),
    verdicts: z.array(verdictRecordSchema),
    matching_entities_report_only: z.array(z.record(z.string(), z.unknown())),
  })
  .strict();
export type VerdictFile = z.infer<typeof verdictFileSchema>;
export type VerdictRecord = z.infer<typeof verdictRecordSchema>;

// Operator-specified inventory criterion, NOT an identity decision.
function normalizedLabel(value: string): string {
  return value.trim().toLowerCase();
}
function strings(row: Row, field: string): string[] {
  return z.array(z.string()).parse(JSON.parse(String(row[field])));
}
function excerpt(value: string, limit: number) {
  return {
    text: value.slice(0, utf16SafePrefixEnd(value, limit)),
    truncated: value.length > limit,
  };
}

export function selfLabelInventory(db: SqliteDatabase) {
  const self = new EntityRepository({ db }).getSelf() ?? refuse("Missing self entity");
  const anchors = db
    .prepare(
      `SELECT * FROM semantic_nodes WHERE kind = 'self'
    AND archived = 0 AND status = 'active' AND superseded_by IS NULL ORDER BY id`,
    )
    .all();
  if (anchors.length !== 1) refuse("Expected exactly one active canonical kind=self semantic node");
  const canonical = anchors[0]!;
  const names = new Set([self.canonical_name, ...self.aliases].map(normalizedLabel));
  const candidates = db
    .prepare(
      `SELECT * FROM semantic_nodes
    WHERE kind IN ('entity', 'person') AND archived = 0
    AND status <> 'superseded' AND superseded_by IS NULL ORDER BY id`,
    )
    .all()
    .filter((row) => names.has(normalizedLabel(String(row.label))));
  const matchingEntities = db
    .prepare("SELECT * FROM entities WHERE id <> ? ORDER BY id")
    .all(self.id)
    .filter(
      (row) => normalizedLabel(String(row.canonical_name)) === normalizedLabel(self.canonical_name),
    );
  return { self, canonical, candidates, matchingEntities };
}

export async function buildSelfLabelDossier(
  db: SqliteDatabase,
  episodes: Pick<EpisodicRepository, "get" | "getStats"> | null,
  nodeId: string,
): Promise<SelfLabelDossier> {
  const { self, canonical } = selfLabelInventory(db);
  const node =
    db.prepare("SELECT * FROM semantic_nodes WHERE id = ?").get(nodeId) ??
    refuse(`Missing judged node ${nodeId}`);
  const sourceIds = strings(node, "source_episode_ids").map((id) => episodeIdSchema.parse(id));
  const edges = db
    .prepare(
      `SELECT * FROM semantic_edges
    WHERE from_node_id = ? OR to_node_id = ? ORDER BY id`,
    )
    .all(nodeId, nodeId)
    .map((edge) => {
      const outgoing = edge.from_node_id === nodeId;
      const otherId = outgoing ? edge.to_node_id : edge.from_node_id;
      const other =
        db.prepare("SELECT * FROM semantic_nodes WHERE id = ?").get(otherId) ??
        refuse(`Missing edge endpoint ${otherId}`);
      return {
        id: String(edge.id),
        direction: outgoing ? ("outgoing" as const) : ("incoming" as const),
        relation: String(edge.relation),
        other: { id: String(other.id), label: String(other.label), kind: String(other.kind) },
        confidence: Number(edge.confidence),
        valid_from: Number(edge.valid_from),
        valid_to: edge.valid_to === null ? null : Number(edge.valid_to),
        snapshot_hash: fingerprint({ edge, other }),
      };
    });
  const evidence: SelfLabelDossier["episodes"] = [];
  for (const id of sourceIds.slice(0, DOSSIER_EPISODE_LIMIT)) {
    // Evidence includes archived sources; this is an offline administrative read.
    const episode = await episodes?.get(id, { includeArchived: true });
    const stats = episodes?.getStats(id);
    const gist = stats?.gist ?? null;
    const { embedding: _vector, ...record } = episode ?? {};
    evidence.push({
      id,
      missing: episode == null,
      title: episode ? excerpt(episode.title, 240) : null,
      excerpt: episode ? excerpt(gist ?? episode.narrative, DOSSIER_EXCERPT_CHARS) : null,
      excerpt_source: episode ? (gist === null ? "narrative" : "gist") : null,
      archived: stats?.archived ?? null,
      origin_audience_entity_ids: episode?.origin_audience_entity_ids ?? [],
      shared: episode?.shared ?? null,
      snapshot_hash: episode
        ? fingerprint({ record, gist, archived: stats?.archived ?? null })
        : null,
    });
  }
  return dossierSchema.parse({
    version: 1,
    node: {
      id: node.id,
      kind: node.kind,
      label: node.label,
      description: excerpt(String(node.description), 4000),
      domain: node.domain,
      aliases: strings(node, "aliases"),
      confidence: node.confidence,
      created_at: node.created_at,
      source_episode_ids: sourceIds,
      snapshot_hash: fingerprint(node),
    },
    self: {
      entity_id: self.id,
      canonical_name: self.canonical_name,
      aliases: self.aliases,
      node_id: canonical.id,
      description: excerpt(String(canonical.description), 4000),
      snapshot_hash: fingerprint({ self, canonical }),
    },
    edges,
    episodes: evidence,
    episodes_omitted: Math.max(0, sourceIds.length - evidence.length),
  });
}

export function parseVerdictFile(value: unknown): VerdictFile {
  const file = verdictFileSchema.parse(value);
  const ids = new Set<string>();
  for (const record of file.verdicts) {
    if (
      ids.has(record.node_id) ||
      record.node_id !== record.dossier.node.id ||
      record.model !== file.model ||
      fingerprint(record.dossier) !== record.dossier_hash
    )
      refuse(`Invalid or duplicate judgment/dossier: ${record.node_id}`);
    ids.add(record.node_id);
  }
  return file;
}
