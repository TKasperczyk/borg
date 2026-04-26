import { z } from "zod";

import { episodeIdSchema } from "../episodic/types.js";
import {
  semanticEdgeIdHelpers,
  semanticNodeIdHelpers,
  type EpisodeId,
  type SemanticEdgeId,
  type SemanticNodeId,
} from "../../util/ids.js";

export const SEMANTIC_NODE_KINDS = ["concept", "entity", "proposition"] as const;
export const SEMANTIC_RELATIONS = [
  "is_a",
  "part_of",
  "causes",
  "prevents",
  "supports",
  "contradicts",
  "related_to",
  "instance_of",
] as const;
export const INVALIDATION_PROCESSES = [
  "extractor",
  "overseer",
  "manual",
  "review",
  "maintenance",
] as const;

export const semanticNodeIdSchema = z
  .string()
  .refine((value) => semanticNodeIdHelpers.is(value), {
    message: "Invalid semantic node id",
  })
  .transform((value) => value as SemanticNodeId);

export const semanticEdgeIdSchema = z
  .string()
  .refine((value) => semanticEdgeIdHelpers.is(value), {
    message: "Invalid semantic edge id",
  })
  .transform((value) => value as SemanticEdgeId);

export const semanticNodeKindSchema = z.enum(SEMANTIC_NODE_KINDS);
export const semanticRelationSchema = z.enum(SEMANTIC_RELATIONS);
export const invalidationProcessSchema = z.enum(INVALIDATION_PROCESSES);

const float32ArraySchema = z.custom<Float32Array>((value) => value instanceof Float32Array, {
  message: "Expected Float32Array embedding",
});

export const semanticNodeSchema = z.object({
  id: semanticNodeIdSchema,
  kind: semanticNodeKindSchema,
  label: z.string().min(1),
  description: z.string().min(1),
  domain: z.string().min(1).nullable().default(null),
  aliases: z.array(z.string().min(1)),
  confidence: z.number().min(0).max(1),
  source_episode_ids: z.array(episodeIdSchema).min(1),
  created_at: z.number().finite(),
  updated_at: z.number().finite(),
  last_verified_at: z.number().finite(),
  embedding: float32ArraySchema,
  archived: z.boolean().default(false),
  superseded_by: semanticNodeIdSchema.nullable().default(null),
});

export const semanticNodeInsertSchema = semanticNodeSchema;
export const semanticNodePatchSchema = semanticNodeSchema
  .omit({
    id: true,
    created_at: true,
  })
  .partial()
  .extend({
    replace_aliases: z.boolean().optional(),
    replace_source_episode_ids: z.boolean().optional(),
  });

const semanticEdgeBaseSchema = z.object({
  id: semanticEdgeIdSchema,
  from_node_id: semanticNodeIdSchema,
  to_node_id: semanticNodeIdSchema,
  relation: semanticRelationSchema,
  confidence: z.number().min(0).max(1),
  evidence_episode_ids: z.array(episodeIdSchema).min(1),
  created_at: z.number().finite(),
  last_verified_at: z.number().finite(),
  // Knowledge-valid interval; created_at remains row insertion time.
  valid_from: z.number().finite(),
  valid_to: z.number().finite().nullable(),
  invalidated_at: z.number().finite().nullable(),
  invalidated_by_edge_id: semanticEdgeIdSchema.nullable(),
  invalidated_by_review_id: z.number().int().nullable(),
  invalidated_by_process: invalidationProcessSchema.nullable(),
  invalidated_reason: z.string().min(1).nullable(),
});

export const semanticEdgeSchema = semanticEdgeBaseSchema.refine(
  (value) => value.from_node_id !== value.to_node_id,
  {
    message: "Semantic edges cannot be self-edges",
    path: ["to_node_id"],
  },
);

export const semanticEdgePatchSchema = semanticEdgeBaseSchema
  .omit({
    id: true,
    from_node_id: true,
    to_node_id: true,
    relation: true,
    created_at: true,
  })
  .partial();

export type SemanticNode = z.infer<typeof semanticNodeSchema>;
export type SemanticNodePatch = z.infer<typeof semanticNodePatchSchema>;
export type SemanticNodeKind = z.infer<typeof semanticNodeKindSchema>;
export type SemanticRelation = z.infer<typeof semanticRelationSchema>;
export type InvalidationProcess = z.infer<typeof invalidationProcessSchema>;
export type SemanticEdge = z.infer<typeof semanticEdgeSchema>;
export type SemanticEdgePatch = z.infer<typeof semanticEdgePatchSchema>;

export type SemanticNodeSearchOptions = {
  limit?: number;
  minSimilarity?: number;
  kindFilter?: readonly SemanticNodeKind[];
  includeArchived?: boolean;
};

export type SemanticNodeSearchCandidate = {
  node: SemanticNode;
  similarity: number;
};

export type SemanticNodeListOptions = {
  kind?: SemanticNodeKind;
  includeArchived?: boolean;
  limit?: number;
};

export type SemanticEdgeListOptions = {
  fromId?: SemanticNodeId;
  toId?: SemanticNodeId;
  relation?: SemanticRelation;
  asOf?: number;
  includeInvalid?: boolean;
};

export type SemanticWalkOptions = {
  relations?: readonly SemanticRelation[];
  direction?: "out" | "in" | "both";
  depth?: number;
  maxNodes?: number;
  asOf?: number;
  includeInvalid?: boolean;
};

export type SemanticWalkStep = {
  node: SemanticNode;
  edgePath: SemanticEdge[];
};

export type SemanticContext = {
  supports: SemanticNode[];
  contradicts: SemanticNode[];
  categories: SemanticNode[];
};

export type SemanticNodeIdValue = SemanticNodeId;
export type SemanticEdgeIdValue = SemanticEdgeId;
export type EpisodeIdValue = EpisodeId;
