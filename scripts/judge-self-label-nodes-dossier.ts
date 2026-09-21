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
const selfNodeAnchorSchema = z.object({ id: semanticNodeIdSchema, label: z.string() }).strict();
export type SelfNodeAnchor = z.infer<typeof selfNodeAnchorSchema>;
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
export const failedVerdictSchema = z
  .object({
    status: z.literal("failed"),
    node_id: semanticNodeIdSchema,
    dossier: dossierSchema,
    dossier_hash: hashSchema,
    model: z.string().min(1),
    failed_at: z.number().finite(),
    error_class: z.string().min(1),
    message: z.string(),
    cause_class: z.string().nullable(),
    cause_message: z.string().nullable(),
  })
  .strict();
export type FailedVerdict = z.infer<typeof failedVerdictSchema>;
export const verdictFileSchema = z
  .object({
    version: z.union([z.literal(1), z.literal(2), z.literal(3)]),
    kind: z.literal("judge-self-label-nodes"),
    data_dir: z.string().min(1),
    started_at: z.number().finite(),
    completed_at: z.number().finite().nullable(),
    model: z.string().min(1),
    model_role: z.literal("extraction"),
    verdicts: z.array(verdictRecordSchema),
    failures: z.array(failedVerdictSchema).default([]),
    candidate_node_ids: z.array(semanticNodeIdSchema).optional(),
    self_node: selfNodeAnchorSchema.optional(),
    matching_entities_report_only: z.array(z.record(z.string(), z.unknown())),
  })
  .strict()
  .superRefine((file, ctx) => {
    if (file.version >= 2 && file.candidate_node_ids === undefined)
      ctx.addIssue({
        code: "custom",
        path: ["candidate_node_ids"],
        message: "Version 2 and later require the candidate manifest",
      });
    if (file.version === 3 && file.self_node === undefined)
      ctx.addIssue({
        code: "custom",
        path: ["self_node"],
        message: "Version 3 requires the selected self node",
      });
  });
export type VerdictFile = z.infer<typeof verdictFileSchema>;
export type VerdictRecord = z.infer<typeof verdictRecordSchema>;

// Operator-specified inventory criterion, NOT an identity decision.
// Duplicates private normalizeName in memory/commitments/repository.ts to avoid widening its exports.
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

export function selfLabelInventory(db: SqliteDatabase, selfNode?: string) {
  const self = new EntityRepository({ db }).getSelf() ?? refuse("Missing self entity");
  const anchors = db
    .prepare(
      `SELECT * FROM semantic_nodes WHERE kind = 'self'
    AND archived = 0 AND status = 'active' AND superseded_by IS NULL ORDER BY id`,
    )
    .all();
  const edgeCount = db.prepare(
    "SELECT COUNT(*) AS count FROM semantic_edges WHERE from_node_id = ? OR to_node_id = ?",
  );
  const summaries = anchors.map((node) => ({
    id: String(node.id),
    label: String(node.label),
    created_at: Number(node.created_at),
    edge_count: Number(edgeCount.get(node.id, node.id)!.count),
    source_count: strings(node, "source_episode_ids").length,
  }));
  if (selfNode === undefined && anchors.length !== 1)
    refuse(
      `${anchors.length ? "Multiple active kind=self semantic nodes; --self-node is required" : "No active kind=self semantic nodes"}. Candidates: ${JSON.stringify(summaries)}`,
    );
  const canonical =
    (selfNode === undefined ? anchors[0] : anchors.find((node) => node.id === selfNode)) ??
    refuse(
      `--self-node must name an active, unarchived, unsuperseded kind=self semantic node: ${selfNode}. Candidates: ${JSON.stringify(summaries)}`,
    );
  const anchor = selfNodeAnchorSchema.parse({ id: canonical.id, label: canonical.label });
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
  return {
    self,
    canonical,
    anchor,
    candidates,
    matchingEntities,
    otherSelfNodes: summaries.filter((node) => node.id !== anchor.id),
  };
}

export async function buildSelfLabelDossier(
  db: SqliteDatabase,
  episodes: Pick<EpisodicRepository, "get" | "getStats"> | null,
  nodeId: string,
  selfNode?: string,
): Promise<SelfLabelDossier> {
  const { self, canonical } = selfLabelInventory(db, selfNode);
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
  const candidates =
    file.candidate_node_ids === undefined ? undefined : new Set(file.candidate_node_ids);
  if (candidates && candidates.size !== file.candidate_node_ids!.length)
    refuse("Duplicate candidate IDs");
  for (const record of [...file.verdicts, ...file.failures]) {
    if (
      ids.has(record.node_id) ||
      (candidates !== undefined && !candidates.has(record.node_id)) ||
      record.node_id !== record.dossier.node.id ||
      (file.self_node !== undefined && record.dossier.self.node_id !== file.self_node.id) ||
      record.model !== file.model ||
      fingerprint(record.dossier) !== record.dossier_hash
    )
      refuse(`Invalid or duplicate judgment/dossier: ${record.node_id}`);
    ids.add(record.node_id);
  }
  return file;
}

export function assertCompleteVerdicts(file: VerdictFile, currentCandidateIds: string[]): void {
  const successful = new Set(file.verdicts.map((row) => String(row.node_id)));
  const failed = new Set(file.failures.map((row) => String(row.node_id)));
  const incomplete = new Set(
    [...currentCandidateIds, ...(file.candidate_node_ids ?? []), ...failed].filter(
      (id) => !successful.has(id),
    ),
  );
  if (incomplete.size)
    refuse(
      `Failed or missing judgments: ${[...incomplete]
        .sort()
        .map((id) => `${id} (${failed.has(id) ? "failed" : "missing"})`)
        .join(", ")}; rerun judge on the existing --out file`,
    );
}
