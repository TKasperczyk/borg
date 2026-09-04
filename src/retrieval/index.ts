export { retrievalMigrations } from "./migrations.js";
export type {
  EvidenceItem,
  EvidencePool,
  RecallEvidenceHandle,
  RecallIntent,
} from "./recall-types.js";
export {
  DEFAULT_RECALL_STATE_MAX_ACTIVE_HANDLES,
  DEFAULT_RECALL_STATE_MAX_NEW_HANDLES_PER_TURN,
  DEFAULT_RECALL_STATE_MAX_WARM_EVIDENCE_RENDERED,
  DEFAULT_RECALL_STATE_TTL_TURNS,
  DEFAULT_RECALL_STATE_WARM_SUPPRESSION_TURNS,
  RecallStateRepository,
  createEmptyRecallState,
  deriveRecallEvidenceHandle,
  recallEvidenceHandleKey,
  recallEvidenceHandleSchema,
  recallStateHandleSchema,
  recallStateSchema,
  type RecallState,
  type RecallStateHandle,
} from "./recall-state.js";
export {
  computeRetrievalConfidence,
  EPISODE_EVIDENCE_STRENGTH_BOUND,
  SEMANTIC_EVIDENCE_STRENGTH_SCALE,
  type ComputeRetrievalConfidenceInput,
  type RetrievalConfidence,
} from "./confidence.js";
export {
  computeGoalRelevance,
  computeGoalRelevanceFromEmbeddings,
  computeValueAlignment,
  computeValueAlignmentFromEmbeddings,
} from "./attention-relevance.js";
export { applyMmr, type MmrCandidate } from "./mmr.js";
export type { EpisodeRecencyPrior } from "./evidence-projections.js";
export {
  MEMORY_DISCLOSURE_CLASSES,
  MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL,
  MEMORY_DISCLOSURE_INTERNAL_USE_NOTE,
  SEMANTIC_SOURCE_DISCLOSURE_INTERNAL_USE_NOTE,
  SELF_RECALL_SCOPE,
  combineDisclosureLabelForEpisodeIds,
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelFromMetadata,
  memoryDisclosureInternalUseNote,
  memoryDisclosureLabelMetadata,
  memoryDisclosureLabelMetadataSchema,
  memoryDisclosureLabelSchema,
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
  renderMemoryDisclosureLabelFieldsForModel,
  renderMemoryDisclosureLabelForModel,
  renderSemanticSourceDisclosureLabelForModel,
  resolveDisclosureLabelsByEpisodeId,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type CognitionRecallContext,
  type DisclosureContext,
  type MemoryDisclosureClass,
  type MemoryDisclosureLabel,
  type MemoryDisclosureLabelMetadata,
  type MemoryDisclosureLabelRenderContext,
} from "./recall-context.js";
export { countRetrievedContradictionRelations } from "./context-assembly.js";
export {
  RetrievalPipeline,
  type CognitionRecallSearchOptions,
  type CognitionRetrievalOptions,
  type DisclosureRetrievalOptions,
  type RetrievedContext,
  type RetrievedContradictionRouting,
  type RetrievedContradictionRoutingItem,
  type RetrievedContradictionSessionScope,
  type RetrievedEpisode,
  type RetrievedSemanticHit,
  type RetrievedSemanticNode,
  type RetrievedSemanticUnderReview,
  type RetrievedSemantic,
  type RetrievalGetEpisodeOptions,
  type RetrievalPipelineOptions,
  type RetrievalSharedOptions,
  type RetrievalSearchOptions,
} from "./pipeline.js";
export type { RecallQueryReformulationContext } from "./recall-expansion.js";
export {
  mapWithDisclosureConcurrency,
  memoryDisclosureLabelForEpisodeIds,
  resolveMemoryDisclosureLabelForEpisodeIds,
  resolveMemoryDisclosureLabelsByEpisodeId,
  resolveSemanticContextForCognition,
  resolveSemanticContextForDisclosure,
  resolveSemanticDisclosureSourceAdapter,
  type SemanticSourceAdapter,
} from "./semantic-retrieval.js";
