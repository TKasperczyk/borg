export {
  SemanticExtractor,
  type ExtractSemanticResult,
  type SemanticExtractorOptions,
} from "./extractor.js";
export { canonicalizeDomain, DOMAIN_SYNONYMS } from "./domain.js";
export { SemanticGraph, type SemanticGraphOptions } from "./graph.js";
export { semanticMigrations } from "./migrations.js";
export {
  ReviewQueueRepository,
  REVIEW_KINDS,
  REVIEW_RESOLUTIONS,
  reviewKindSchema,
  reviewQueueItemSchema,
  reviewResolutionSchema,
  reviewResolutionInputSchema,
  type ReviewKind,
  type ReviewQueueInsertInput,
  type ReviewQueueItem,
  type ReviewResolution,
  type ReviewResolutionInput,
} from "./review-queue.js";
export {
  SemanticReviewService,
  type SemanticReviewServiceOptions,
} from "./review-service.js";
export {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
  type SemanticEdgeRepositoryOptions,
  type SemanticNodeRepositoryOptions,
} from "./repository.js";
export {
  SEMANTIC_NODE_KINDS,
  SEMANTIC_RELATIONS,
  semanticEdgeIdSchema,
  semanticEdgePatchSchema,
  semanticEdgeSchema,
  semanticNodeIdSchema,
  semanticNodeInsertSchema,
  semanticNodeKindSchema,
  semanticNodePatchSchema,
  semanticNodeSchema,
  semanticRelationSchema,
  type SemanticContext,
  type SemanticEdge,
  type SemanticEdgeListOptions,
  type SemanticNode,
  type SemanticNodeKind,
  type SemanticNodeListOptions,
  type SemanticNodePatch,
  type SemanticNodeSearchCandidate,
  type SemanticNodeSearchOptions,
  type SemanticRelation,
  type SemanticWalkOptions,
  type SemanticWalkStep,
} from "./types.js";
