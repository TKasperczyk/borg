export {
  SemanticExtractor,
  type ExtractSemanticResult,
  type SemanticExtractorOptions,
  type SemanticRelationshipEvidenceStreamEntryTrustResult,
  type SemanticRelationshipEvidenceStreamEntryTrustValidator,
} from "./extractor.js";
export { createUserStreamEntryRelationshipEvidenceTrustValidator } from "./source-trust.js";
export { canonicalizeDomain } from "./domain.js";
export { SemanticGraph, type SemanticGraphOptions } from "./graph.js";
export { semanticMigrations } from "./migrations.js";
export {
  ReviewQueueRepository,
  REVIEW_KINDS,
  REVIEW_RESOLUTIONS,
  ReviewQueueHandlerRegistry,
  reviewKindSchema,
  reviewQueueItemSchema,
  reviewResolutionSchema,
  reviewResolutionInputSchema,
  type ReviewKind,
  type BeliefRevisionTarget,
  type BeliefRevisionReasonCode,
  type BeliefRevisionVisibilityOptions,
  type OpenBeliefRevisionStatus,
  type ReviewApplyingStateSpec,
  type ReviewApplyDecision,
  type ReviewApplyOutcome,
  type ReviewHandlerContext,
  type ReviewQueueHandler,
  type ReviewQueueInsertInput,
  type ReviewQueueItem,
  type ReviewResolveOptions,
  type ReviewResolution,
  type ReviewResolutionSource,
  type ReviewResolutionInput,
  type ReviewTransactionScope,
} from "./review-queue.js";
export {
  beliefRevisionReviewRefsSchema,
  createBeliefRevisionReviewQueueHandler,
  type BeliefRevisionReviewRefs,
} from "./review-handlers/belief-revision.js";
export {
  correctionReviewRefsSchema,
  createCorrectionReviewHandler,
  type CorrectionReviewHandlerOptions,
  type CorrectionReviewRefs,
} from "./review-handlers/correction.js";
export { registerBuiltinReviewQueueHandlers } from "./review-handlers/defaults.js";
export {
  createIdentityInconsistencyReviewQueueHandler,
  identityInconsistencyReviewRefsSchema,
  type IdentityInconsistencyReviewRefs,
} from "./review-handlers/identity-inconsistency.js";
export {
  createNewInsightReviewQueueHandler,
  newInsightReviewRefsSchema,
  type NewInsightReviewRefs,
} from "./review-handlers/new-insight.js";
export {
  createMisattributionReviewQueueHandler,
  misattributionReviewRefsSchema,
  type MisattributionReviewRefs,
} from "./review-handlers/misattribution.js";
export {
  createSkillSplitReviewQueueHandler,
  skillSplitReviewPayloadSchema,
  type SkillSplitReviewApplyResult,
  type SkillSplitReviewHandler,
  type SkillSplitReviewPayload,
} from "./review-handlers/skill-split.js";
export {
  createSemanticPairReviewQueueHandler,
  semanticPairReviewRefsSchema,
  type SemanticPairReviewKind,
  type SemanticPairReviewRefs,
} from "./review-handlers/semantic-pair.js";
export {
  createTemporalDriftReviewQueueHandler,
  temporalDriftReviewRefsSchema,
  type TemporalDriftReviewRefs,
} from "./review-handlers/temporal-drift.js";
export {
  SemanticReviewService,
  type SemanticReviewQueueOptions,
  type SemanticReviewServiceOptions,
} from "./review-service.js";
export {
  SemanticBeliefDependencyRepository,
  semanticBeliefDependencyInputSchema,
  semanticBeliefDependencyKindSchema,
  semanticBeliefDependencySchema,
  semanticBeliefDependencyTargetTypeSchema,
  type SemanticBeliefDependency,
  type SemanticBeliefDependencyInput,
  type SemanticBeliefDependencyKind,
  type SemanticBeliefDependencyRepositoryOptions,
  type SemanticBeliefDependencyTargetType,
} from "./revision-dependencies.js";
export {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
  type SemanticEdgeRepositoryOptions,
  type SemanticNodeConfidenceAdjustment,
  type SemanticNodeConfidenceAdjustmentInput,
  type SemanticNodeRepositoryOptions,
  type SemanticNodeStatusCounts,
  type SemanticNodeStatusTransition,
  type SemanticNodeVectorSyncFailure,
  type SemanticNodeVectorSyncOptions,
  type SemanticNodeVectorSyncResult,
} from "./repository.js";
export {
  SEMANTIC_NODE_KINDS,
  SEMANTIC_NODE_STATUSES,
  SEMANTIC_RELATIONS,
  semanticEdgeIdSchema,
  semanticEdgePatchSchema,
  semanticEdgeSchema,
  semanticNodeIdSchema,
  semanticNodeCorrectionRefSchema,
  semanticNodeInsertSchema,
  semanticNodeKindSchema,
  semanticObservationMetadataSchema,
  semanticNodePatchSchema,
  semanticNodeSchema,
  semanticNodeStatusSchema,
  semanticRelationSchema,
  type SemanticContext,
  type SemanticObservationMetadata,
  type SemanticNodeCorrectionRef,
  type SemanticEdge,
  type SemanticEdgeListOptions,
  type SemanticEdgePatch,
  type SemanticNode,
  type SemanticNodeKind,
  type SemanticNodeListOptions,
  type SemanticNodePatch,
  type SemanticNodeSearchCandidate,
  type SemanticNodeSearchOptions,
  type SemanticNodeStatus,
  type SemanticRelation,
  type SemanticWalkOptions,
  type SemanticWalkStep,
} from "./types.js";
