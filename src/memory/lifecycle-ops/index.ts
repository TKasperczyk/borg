export {
  canonicalizeActionWithSharedStateEntry,
  canonicalizeCommitmentWithSharedStateEntry,
  canonicalizeGoalWithSharedStateEntry,
  canonicalizeOpenQuestionWithSharedStateEntry,
  isSharedStateArtifactCanonicalizableCommitmentType,
  isTerminalCommitment,
  isTerminalGoalStatus,
  isTerminalOpenQuestionStatus,
  SHARED_STATE_RECONCILIATION_PROVENANCE,
  type CanonicalizeCommitmentRepository,
  type CanonicalizeGoalRepository,
  type CanonicalizeOpenQuestionRepository,
} from "./canonicalize.js";
export { completeAction, isTerminalActionState, type CompleteActionRepository } from "./complete.js";
export {
  SHARED_STATE_COMMITMENT_CANONICALIZATION_TYPES,
  type SharedStateCommitmentCanonicalizationType,
} from "./commitment-types.js";
export {
  resolveOpenQuestionThroughIdentityService,
  resolveOpenQuestionWithEvidence,
  type ResolveOpenQuestionIdentityService,
  type ResolveOpenQuestionRepository,
} from "./resolve.js";
export {
  markSemanticContradicted,
  markSemanticSuperseded,
  type SemanticLifecycleTraceSource,
  type SemanticStatusRepository,
} from "./semantic-revise.js";
export { supersedeCommitment, type SupersedeCommitmentRepository } from "./supersede.js";
export type {
  LifecycleOperationResult,
  LifecycleTraceData,
  LifecycleTraceEventName,
  LifecycleTracer,
} from "./types.js";
