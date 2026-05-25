import { type LLMClient, type LLMToolDefinition, toToolInputSchema } from "../../llm/index.js";
import {
  type SharedStateArtifact,
  type SharedStateRepository,
  type SharedStateSourceTrustValidator,
} from "../../memory/decision-artifacts/index.js";
import type { SyncRelationshipEvidenceStreamEntryTrustValidator } from "../../memory/source-trust.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import type { SharedStateRenderOptions } from "./render.js";
import type { SharedStatePromptSummaryOptions } from "./summary.js";
import type {
  SharedStateReconciliationRepositories,
  SharedStateSemanticBeliefRevisionDependencies,
} from "./reconciliation.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { ParticipantRoster } from "../perception/index.js";
import {
  DECISION_ARTIFACT_TOOL_NAME,
  SHARED_STATE_ACCEPTED_TOOL_NAMES,
  SHARED_STATE_TOOL_NAME,
} from "./constants.js";
import {
  sharedStatePatchSchema,
  type SharedStateArtifactParticipantContext,
  type SharedStateCanonicalizationCandidates,
  type SharedStateCompileDegradedReason,
  type SharedStateLedgerMode,
  type SharedStateLifecycleOptions,
  type SharedStateRelationalSlotContext,
} from "./types.js";

export {
  DECISION_ARTIFACT_TOOL_NAME,
  MAX_PATCH_OUTPUT_TOKENS,
  SHARED_STATE_ACCEPTED_TOOL_NAMES,
  SHARED_STATE_TOOL_ENTRY_KINDS,
  SHARED_STATE_TOOL_NAME,
  SHARED_STATE_TOOL_NAME_ALIASES,
  SHARED_STATE_PROMPT_WARNING_TOKEN_THRESHOLD,
} from "./constants.js";
export {
  canonicalizesSchema,
  sharedStatePatchSchema,
  type AllowedCanonicalizationIds,
  type CanonicalizationDuplicateDrop,
  type CanonicalizeIdChannel,
  type DroppedCanonicalizeId,
  type EmitDecisionArtifactPatch,
  type EmitSharedStatePatch,
  type EmptyUpdateDrop,
  type NonLockedCanonicalizesDrop,
  type ParsedCanonicalizes,
  type ParsedPatchOperation,
  type PatchRejection,
  type SharedStateActionCanonicalizationCandidate,
  type SharedStateArtifactParticipantContext,
  type SharedStateCanonicalizationCandidate,
  type SharedStateCanonicalizationCandidates,
  type SharedStateCommitmentCanonicalizationCandidate,
  type SharedStateCompileDegradedReason,
  type SharedStateLedgerMode,
  type SharedStateLifecycleOptions,
  type SharedStateRelationalSlotContext,
} from "./types.js";

function sharedStateToolDefinition(name: (typeof SHARED_STATE_ACCEPTED_TOOL_NAMES)[number]) {
  return {
    name,
    description:
      "Compile durable shared state for this audience into additive, updating, superseding, or pruning operations.",
    inputSchema: toToolInputSchema(sharedStatePatchSchema),
  } satisfies LLMToolDefinition;
}

export const SHARED_STATE_TOOL = sharedStateToolDefinition(SHARED_STATE_TOOL_NAME);
export const DECISION_ARTIFACT_TOOL_ALIAS = sharedStateToolDefinition(DECISION_ARTIFACT_TOOL_NAME);
export const SHARED_STATE_TOOLS = [
  SHARED_STATE_TOOL,
  DECISION_ARTIFACT_TOOL_ALIAS,
] as const satisfies readonly LLMToolDefinition[];
export const DECISION_ARTIFACT_TOOL = DECISION_ARTIFACT_TOOL_ALIAS;

export type CompileSharedStateArtifactInput = {
  llmClient?: LLMClient;
  model?: string;
  repository?: Pick<SharedStateRepository, "get" | "upsert">;
  audienceEntityId: EntityId;
  selfEntityId: EntityId;
  speakerEntityId?: EntityId | null;
  participants: readonly SharedStateArtifactParticipantContext[];
  participantRoster?: ParticipantRoster | null;
  currentUserMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  promptVisibleLedger: string;
  previousArtifact?: SharedStateArtifact | null;
  relationalSlotsContext?: readonly SharedStateRelationalSlotContext[];
  allowedSourceStreamEntryIds?: readonly StreamEntryId[];
  offLimitsSourceStreamEntryIds?: readonly StreamEntryId[];
  sourceTrustValidator?: SharedStateSourceTrustValidator;
  relationshipEvidenceStreamEntryTrust?: SyncRelationshipEvidenceStreamEntryTrustValidator;
  canonicalizationCandidates?: SharedStateCanonicalizationCandidates;
  reconciliation?: SharedStateReconciliationRepositories;
  semanticBeliefRevision?: Omit<SharedStateSemanticBeliefRevisionDependencies, "llmClient">;
  clock?: Clock;
  tracer?: TurnTracer;
  turnId?: string;
  turnCounter?: number;
  lifecycle?: SharedStateLifecycleOptions;
  renderOptions?: SharedStateRenderOptions;
  previousArtifactSummaryOptions?: SharedStatePromptSummaryOptions;
  ledgerMode?: SharedStateLedgerMode;
  onDegraded?: (reason: SharedStateCompileDegradedReason, error?: unknown) => Promise<void> | void;
};

export class MissingSharedStateArtifactToolCallError extends Error {}
