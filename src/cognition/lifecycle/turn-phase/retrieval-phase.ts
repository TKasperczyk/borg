import {
  appendCommitmentIfMissing,
  type CorrectivePreferenceTurnService,
} from "../../commitments/corrective-preference-service.js";
import type {
  CreatorDirectiveBriefing,
  CurrentTimePromptContext,
  DeliberationRoutingOverride,
} from "../../deliberation/types.js";
import {
  EvidenceLedgerBuilder,
  compactEvidenceLedger,
  estimateEvidenceLedgerPromptTokens,
  renderEvidenceLedger,
  summarizeEvidenceLedgerTrace,
  summarizeSharedStateArtifactRender,
  type CompactedEvidenceLedger,
  type EvidenceLedger,
  type EvidenceLedgerBuildInput,
} from "../../evidence-ledger/index.js";
import type { ActualFrameAnomalyClassification } from "../../frame-anomaly/index.js";
import type { ActiveParticipant, ParticipantProfileContext } from "../../participants.js";
import type { ParticipantRoster } from "../../perception/index.js";
import type { RecencyMessage } from "../../recency/index.js";
import {
  commitmentMemoryDisclosureLabel,
  goalMemoryDisclosureLabel,
  memoryDisclosurePayloadFields,
  openQuestionMemoryDisclosureLabel,
} from "../../../memory/common/disclosure-serializers.js";
import {
  compileSharedStateArtifact,
  findUnsettledSharedStateReconciliation,
  type SharedStateCanonicalizationCandidates,
  type SharedStateReconciliationRepositories,
} from "../../shared-state/index.js";
import {
  buildSessionReentryContinuityPrompt,
  type SessionReentryContinuityPrompt,
} from "../../session-reentry-continuity.js";
import { AutobiographicalRecallService } from "../../autobiographical-recall.js";
import { toTraceJsonValue } from "../../../tracing/tracer.js";
import type { PerceptionResult } from "../../types.js";
import {
  hydrateTurnMechanismEvidence,
  type TurnMechanismEvidence,
} from "../../mechanism-evidence.js";
import type { LLMClient } from "../../../llm/index.js";
import {
  effectiveCommitmentEnforcementClass,
  type BorgRole,
  type EntityRepository,
  type CommitmentRecord,
} from "../../../memory/commitments/index.js";
import type {
  CreatorDirective,
  CreatorDirectiveApplicable,
  CreatorDirectiveKind,
} from "../../../memory/creator-directives/index.js";
import { creatorDirectiveDisclosureBlocksPrivateOperation } from "../../../memory/creator-directives/index.js";
import {
  RECENT_LIVED_EXPERIENCE_DAILY_SPINE_WINDOW_MS,
  selectRecentLivedExperienceRows,
  selectCrossSessionSelfActivity,
} from "../../../memory/activity/index.js";
import { selectSelfDecisionIntrospection } from "../../../memory/self-decisions/index.js";
import {
  DEFAULT_OBSERVED_EVENT_INTROSPECTION_CAP,
  DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS,
  recallObservedEventsForCognition,
} from "../../../memory/observed-events/index.js";
import type { SharedStateArtifact } from "../../../memory/shared-state/index.js";
import { createLoadedUserStreamEntryRelationshipEvidenceTrustValidator } from "../../../memory/source-trust.js";
import {
  SELF_RECALL_SCOPE,
  type CognitionRecallContext,
  type DisclosureContext,
} from "../../../retrieval/index.js";
import type { IndexedEntryFacts, StreamEntry } from "../../../stream/index.js";
import { filterActiveStreamEntries, loadSessionStreamEntries } from "../../../stream/index.js";
import type {
  ActionId,
  AttachmentId,
  CommitmentId,
  EntityId,
  GoalId,
  OpenQuestionId,
  SessionId,
  SharedStateEntryId,
  StreamEntryId,
} from "../../../util/ids.js";
import { dedupePreservingOrder } from "../../../util/collections.js";
import { utcDayStartMs } from "../../../util/utc-day.js";
import type { WorkingMemory } from "../../../memory/working/index.js";
import type { SessionAudienceRole } from "../../../sessions/index.js";
import type { ClosureLoopAssessment } from "../../generation/closure-loop.js";
import type { SharedStateCompilePass } from "../../prompts/shared-state.js";
import type { TurnPhaseCoordinatorOptions, TurnPhaseInput } from "./types.js";
import {
  buildContradictionRoutingOverride,
  listConstrainedRelationalSlotsForParticipants,
  listSharedStateRelationalSlotsForParticipants,
} from "./context-build.js";
import { evidenceLedgerCompactionChanged } from "./trace-metrics.js";
import { traceTurnPhase } from "./phase-trace.js";
import {
  advanceSharedStateCompileSkipAnchor,
  buildSharedStateLedgerPromptContext,
  buildSharedStateSourceTrustValidator,
  collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIds,
  compactSharedStateArtifactCandidateText,
  isSharedStateCommitmentCanonicalizationRecord,
  selectCurrentAudienceSharedStateActionCandidatesForCanonicalization,
  shouldSkipSharedStateCompile,
} from "./shared-state-phase.js";
import { runSharedStateArtifactRetryOnlyReconciliation } from "./reconciliation-phase.js";
import { shouldRenderRecentLivedExperience } from "./recent-lived-experience-gap.js";
import { sharedStateRenderOptions } from "./utils.js";
import type { TurnExtractionPhaseResult } from "./extraction-phase.js";

export type EvidenceLedgerFinalizerContext = {
  ledger: EvidenceLedger | null;
  promptSection: string | null;
  sessionReentryContinuityPromptSection: string | null;
  sharedStateAppliedOperationCount: number;
  openQuestionsRenderedToFinalizerCount: number;
};

export type SharedStateArtifactForEvidenceLedgerResult = {
  artifact: SharedStateArtifact | null;
  appliedOperationCount: number;
  renderOptions?: ReturnType<typeof sharedStateRenderOptions>;
};

export type SharedStateCompilerAssistantResponse = {
  streamEntryId: StreamEntryId;
  text: string;
};

export type EvidenceLedgerFinalizerBuildInput = EvidenceLedgerBuildInput & {
  globalTurnCounter?: number;
  isUserTurn: boolean;
  perception: PerceptionResult;
  closureLoopAssessment: ClosureLoopAssessment | null;
  participantRoster?: ParticipantRoster | null;
};

export type CompactedEvidenceLedgerWithoutSharedStateResult = {
  compacted: CompactedEvidenceLedger;
  ledger: EvidenceLedger;
  rendered: string | null;
};

export type TurnRetrievalPhaseResult = {
  selfContext: Awaited<ReturnType<TurnPhaseCoordinatorOptions["selfContextBuilder"]["build"]>>;
  selfSnapshot: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["selfContextBuilder"]["build"]>
  >["selfSnapshot"];
  executiveFocusWithStep: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["selfContextBuilder"]["build"]>
  >["executiveFocus"];
  retrievalContext: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >;
  applicableCommitments: readonly CommitmentRecord[];
  actionApplicableCommitments: readonly CommitmentRecord[];
  pendingCorrections: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["pendingCorrections"];
  pendingCommitmentReviews: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["pendingCommitmentReviews"];
  affectiveTrajectory: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["affectiveTrajectory"];
  retrieval: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["retrieval"];
  turnMechanismEvidence: TurnMechanismEvidence;
  retrievedEpisodes: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["retrievedEpisodes"];
  retrievedSemantic: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["retrievedSemantic"];
  proceduralContext: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["proceduralContext"];
  selectedSkill: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["selectedSkill"];
  currentTimeContext: CurrentTimePromptContext;
  relationalSlots: ReturnType<typeof listConstrainedRelationalSlotsForParticipants>;
  participantRoster: ParticipantRoster | null;
  creatorDirectiveBriefing: CreatorDirectiveBriefing | null;
  evidenceLedgerContext: EvidenceLedgerFinalizerContext;
  routingOverride: DeliberationRoutingOverride | null;
};

function evidenceLedgerEntryCount(ledger: EvidenceLedger | null): number {
  return ledger?.sections.reduce((sum, section) => sum + section.entries.length, 0) ?? 0;
}

function summarizeEvidenceLedgerContext(context: EvidenceLedgerFinalizerContext): string {
  if (context.ledger === null) {
    return "disabled";
  }

  return `entries=${evidenceLedgerEntryCount(context.ledger)} shared_ops=${context.sharedStateAppliedOperationCount} images=${context.ledger.imageAttachments?.length ?? 0}`;
}

function uniqueStreamEntryIds(ids: readonly StreamEntryId[]): StreamEntryId[] {
  return dedupePreservingOrder(ids);
}

function uniqueEntityIds(ids: readonly EntityId[]): EntityId[] {
  return dedupePreservingOrder(ids);
}

function singleSenderEntityId(
  entries: readonly Pick<StreamEntry, "sender_entity_id">[],
): EntityId | null {
  const senderIds = uniqueEntityIds(
    entries.flatMap((entry) => {
      const senderEntityId = entry.sender_entity_id ?? null;
      return senderEntityId === null ? [] : [senderEntityId];
    }),
  );

  return senderIds.length === 1 ? (senderIds[0] ?? null) : null;
}

function participantEntityIds(input: {
  audienceEntityId: EntityId | null;
  audienceEntityKind?: "person" | "group" | "self" | "abstract" | null;
  activeParticipants: readonly ActiveParticipant[];
}): EntityId[] {
  const concreteParticipants = input.activeParticipants.filter(
    (participant) =>
      input.audienceEntityKind !== "group" ||
      input.audienceEntityId === null ||
      participant.entityId !== input.audienceEntityId ||
      input.activeParticipants.length === 1,
  );

  if (concreteParticipants.length > 0) {
    return uniqueEntityIds(concreteParticipants.map((participant) => participant.entityId));
  }

  return input.audienceEntityId === null ? [] : [input.audienceEntityId];
}

function subjectLabelForCreatorDirective(
  directive: CreatorDirective,
  entityRepository: Pick<EntityRepository, "get">,
): string {
  switch (directive.subject_kind) {
    case "borg_self":
      return "Borg";
    case "system":
      return "system";
    case "unknown":
      return "unknown";
    case "entity":
      return directive.subject_entity_id === null
        ? "unknown"
        : (entityRepository.get(directive.subject_entity_id)?.canonical_name ?? "unknown");
  }
}

function contentPayloadForCreatorDirective(directive: CreatorDirective): {
  semanticValue: string | null;
  canonicalFact: string | null;
  operationalDirective: string | null;
} | null {
  if (isPrivateOperationCreatorDirectiveKind(directive.kind)) {
    return {
      // Preserve the pre-compact auxiliary semantic value for the byte-pinned
      // legacy renderer, which dispatches semantic slots first. Compact
      // renderers dispatch the structural kind first and use the exact
      // operational directive below.
      semanticValue: directive.semantic_slot === null ? null : directive.canonical_fact,
      canonicalFact: null,
      operationalDirective: directive.operational_directive,
    };
  }

  if (directive.semantic_slot !== null) {
    return directive.canonical_fact === null
      ? null
      : {
          semanticValue: directive.canonical_fact,
          canonicalFact: null,
          operationalDirective: null,
        };
  }

  switch (directive.kind) {
    case "self_identity":
    case "subject_fact":
    case "disclosure_boundary":
      return directive.canonical_fact === null
        ? null
        : {
            semanticValue: null,
            canonicalFact: directive.canonical_fact,
            operationalDirective: null,
          };
  }
}

type CreatorDirectivePrivateOperationKind = Extract<
  CreatorDirectiveKind,
  "response_policy" | "routing_instruction"
>;

function isPrivateOperationCreatorDirectiveKind(
  kind: CreatorDirectiveKind,
): kind is CreatorDirectivePrivateOperationKind {
  return kind === "response_policy" || kind === "routing_instruction";
}

type CreatorDirectiveContentBearingKind = Extract<
  CreatorDirectiveKind,
  "self_identity" | "subject_fact" | "disclosure_boundary"
>;

function isContentBearingCreatorDirectiveKind(
  kind: CreatorDirectiveKind,
): kind is CreatorDirectiveContentBearingKind {
  return kind === "self_identity" || kind === "subject_fact" || kind === "disclosure_boundary";
}

function canRenderCreatorDirectivePrivateOperation(
  item: CreatorDirectiveApplicable,
): item is CreatorDirectiveApplicable & {
  directive: CreatorDirective & {
    kind: CreatorDirectivePrivateOperationKind;
    operational_directive: string;
  };
} {
  if (!item.activation.active) {
    return false;
  }

  if (item.disclosure.render_mode === "content") {
    return false;
  }

  if (!isPrivateOperationCreatorDirectiveKind(item.directive.kind)) {
    return false;
  }

  if (item.directive.operational_directive === null) {
    return false;
  }

  return !creatorDirectiveDisclosureBlocksPrivateOperation({
    directive: item.directive,
    recipientEntityIds: item.recipient_entity_ids,
  });
}

// A fact-bearing directive that governs the current session but whose content the current
// audience may not be told. Borg holds it privately for orientation/action; it must not be
// disclosed. Mirror of canRenderCreatorDirectivePrivateOperation for content-bearing kinds.
// Gated to render_mode === "omit": "content" is the disclose-to-audience lane and "boundary"
// has its own visible-wall lane; "omit" is the active-but-silent case that otherwise has none.
function canRenderCreatorDirectivePrivateKnowledge(
  item: CreatorDirectiveApplicable,
): item is CreatorDirectiveApplicable & {
  directive: CreatorDirective & { kind: CreatorDirectiveContentBearingKind };
} {
  if (!item.activation.active) {
    return false;
  }

  if (item.render_mode !== "omit") {
    return false;
  }

  if (!isContentBearingCreatorDirectiveKind(item.directive.kind)) {
    return false;
  }

  return contentPayloadForCreatorDirective(item.directive) !== null;
}

function creatorDirectiveBriefingLane(
  item: CreatorDirectiveApplicable,
): "content" | "private_knowledge" | "boundary" | "private_operation" | "omitted" {
  if (!item.activation.active) {
    return "omitted";
  }

  if (item.render_mode === "content") {
    return contentPayloadForCreatorDirective(item.directive) === null ? "omitted" : "content";
  }

  if (
    item.render_mode === "boundary" &&
    item.directive.disclosure_policy.boundary_prompt !== null
  ) {
    return "boundary";
  }

  if (canRenderCreatorDirectivePrivateOperation(item)) {
    return "private_operation";
  }

  return canRenderCreatorDirectivePrivateKnowledge(item) ? "private_knowledge" : "omitted";
}

export function buildCreatorDirectiveBriefing(input: {
  applicable: readonly CreatorDirectiveApplicable[];
  entityRepository: Pick<EntityRepository, "get">;
}): CreatorDirectiveBriefing | null {
  const briefingScope = (item: CreatorDirectiveApplicable) => ({
    directiveId: item.directive.id,
    createdByEntityId: item.directive.created_by_entity_id,
    sourceSessionId: item.directive.source_session_id,
    contentScope: item.directive.disclosure_policy.content_scope,
    allowedEntityIds: [...item.directive.disclosure_policy.allowed_entity_ids],
    excludedEntityIds: [...item.directive.disclosure_policy.excluded_entity_ids],
    subjectMayKnow: item.directive.disclosure_policy.subject_may_know,
    mentionPolicy: item.directive.disclosure_policy.mention_policy,
    deniedAudienceBehavior: item.directive.disclosure_policy.denied_audience_behavior,
    activationScope: item.directive.activation_policy.scope,
    activationAllowedEntityIds: [...item.directive.activation_policy.allowed_entity_ids],
    activationExcludedEntityIds: [...item.directive.activation_policy.excluded_entity_ids],
  });
  const contentDirectives = input.applicable
    .flatMap((item) => {
      if (item.render_mode !== "content") {
        return [];
      }

      if (!item.activation.active) {
        return [];
      }

      const payload = contentPayloadForCreatorDirective(item.directive);

      if (payload === null) {
        return [];
      }

      return [
        {
          renderMode: "content" as const,
          kind: item.directive.kind,
          subjectKind: item.directive.subject_kind,
          subjectLabel: subjectLabelForCreatorDirective(item.directive, input.entityRepository),
          semanticSlot: item.directive.semantic_slot,
          semanticValue: payload.semanticValue,
          canonicalFact: payload.canonicalFact,
          operationalDirective: payload.operationalDirective,
          mentionPolicy: item.directive.disclosure_policy.mention_policy,
          scope: briefingScope(item),
          priority: item.directive.priority,
          createdAt: item.directive.created_at,
        },
      ];
    })
    .sort((left, right) => right.priority - left.priority || left.createdAt - right.createdAt);
  const privateDirectives = [
    ...input.applicable
      .filter(canRenderCreatorDirectivePrivateKnowledge)
      .flatMap((item) => {
        const payload = contentPayloadForCreatorDirective(item.directive);

        if (payload === null) {
          return [];
        }

        return [
          {
            renderMode: "private" as const,
            privateKind: "knowledge" as const,
            kind: item.directive.kind,
            subjectKind: item.directive.subject_kind,
            subjectLabel: subjectLabelForCreatorDirective(item.directive, input.entityRepository),
            semanticSlot: item.directive.semantic_slot,
            semanticValue: payload.semanticValue,
            canonicalFact: payload.canonicalFact,
            mentionPolicy: item.directive.disclosure_policy.mention_policy,
            scope: briefingScope(item),
            priority: item.directive.priority,
            createdAt: item.directive.created_at,
          },
        ];
      })
      .sort((left, right) => right.priority - left.priority || left.createdAt - right.createdAt),
    ...input.applicable
      .filter(canRenderCreatorDirectivePrivateOperation)
      .map((item) => ({
        renderMode: "private" as const,
        privateKind: "operation" as const,
        kind: item.directive.kind,
        operationalDirective: item.directive.operational_directive,
        scope: briefingScope(item),
        priority: item.directive.priority,
        createdAt: item.directive.created_at,
      }))
      .sort((left, right) => right.priority - left.priority || left.createdAt - right.createdAt),
  ];
  const boundaryDirectives = input.applicable
    .filter(
      (item) =>
        item.activation.active &&
        item.render_mode === "boundary" &&
        item.directive.disclosure_policy.boundary_prompt !== null,
    )
    .map((item) => ({
      renderMode: "boundary" as const,
      scope: briefingScope(item),
      priority: item.directive.priority,
      createdAt: item.directive.created_at,
    }))
    .sort((left, right) => right.priority - left.priority || left.createdAt - right.createdAt);
  const directives = [...contentDirectives, ...privateDirectives, ...boundaryDirectives];

  return directives.length === 0 ? null : { directives };
}

function currentTurnEligibleCreatorDirectives(input: {
  applicable: readonly CreatorDirectiveApplicable[];
  currentUserEntryId?: StreamEntryId;
  currentUserEntryIds?: readonly StreamEntryId[];
}): CreatorDirectiveApplicable[] {
  const currentUserEntryIds = new Set([
    ...(input.currentUserEntryId === undefined ? [] : [input.currentUserEntryId]),
    ...(input.currentUserEntryIds ?? []),
  ]);

  if (currentUserEntryIds.size === 0) {
    return [...input.applicable];
  }

  return input.applicable.filter(
    (item) =>
      !item.directive.authorization_stream_entry_ids.some((entryId) =>
        currentUserEntryIds.has(entryId),
      ),
  );
}

const CREATOR_DIRECTIVE_RENDER_TRACE_LIMIT = 50;
const SHARED_STATE_COGNITION_RECALL_LIMIT = 12;

function traceCreatorDirectiveRendered(input: {
  tracer: TurnPhaseCoordinatorOptions["tracer"];
  turnId: string;
  sessionId?: SessionId;
  applicable: readonly CreatorDirectiveApplicable[];
  currentUserEntryId?: StreamEntryId;
  currentUserEntryIds?: readonly StreamEntryId[];
  currentAudienceEntityId: EntityId | null;
  participantEntityIds: readonly EntityId[];
}): void {
  if (!input.tracer.enabled) {
    return;
  }

  const currentUserEntryIds = new Set([
    ...(input.currentUserEntryId === undefined ? [] : [input.currentUserEntryId]),
    ...(input.currentUserEntryIds ?? []),
  ]);

  for (const item of input.applicable.slice(0, CREATOR_DIRECTIVE_RENDER_TRACE_LIMIT)) {
    const sameTurnNPlusOne =
      currentUserEntryIds.size > 0 &&
      item.directive.authorization_stream_entry_ids.some((entryId) =>
        currentUserEntryIds.has(entryId),
      );
    const renderedMode = sameTurnNPlusOne ? "omitted" : creatorDirectiveBriefingLane(item);

    input.tracer.emit("creator_directive_rendered", {
      turnId: input.turnId,
      ...(input.sessionId !== undefined ? { session_id: input.sessionId } : {}),
      // The slice above caps the census at CREATOR_DIRECTIVE_RENDER_TRACE_LIMIT, and the cap is not
      // visible in a per-directive row. Counting render_mode across a turn's events then yields a
      // total that is silently the cap: measured 2026-08-23 over three days of traces, every turn
      // reported exactly 50 applicable directives while the prompt's own creator_directive_index
      // carried rows_total=109 on solitary turns. Without these two fields there is nothing in the
      // event stream that distinguishes "50 applicable" from "50 was as far as we looked", and the
      // lane split reads as a measurement when it is a head slice. Only the briefing builder sees
      // the full list; it is never sliced, so the prompt is complete and the trace is the lossy one.
      applicable_total: input.applicable.length,
      traced_total: Math.min(input.applicable.length, CREATOR_DIRECTIVE_RENDER_TRACE_LIMIT),
      directive_id: item.directive.id,
      current_audience_entity_id: input.currentAudienceEntityId,
      participant_entity_ids: [...input.participantEntityIds],
      render_mode: renderedMode,
      reason: sameTurnNPlusOne ? "same_turn_n_plus_one" : item.reason,
      activation_active: item.activation.active,
      activation_reason: item.activation.reason,
      disclosure_render_mode: item.disclosure.render_mode,
      disclosure_reason: item.disclosure.reason,
    });
  }
}

export function buildCreatorDirectiveBriefingForTurn(input: {
  applicable: readonly CreatorDirectiveApplicable[];
  currentUserEntryId?: StreamEntryId;
  currentUserEntryIds?: readonly StreamEntryId[];
  entityRepository: Pick<EntityRepository, "get">;
}): CreatorDirectiveBriefing | null {
  return buildCreatorDirectiveBriefing({
    applicable: currentTurnEligibleCreatorDirectives({
      applicable: input.applicable,
      currentUserEntryId: input.currentUserEntryId,
      currentUserEntryIds: input.currentUserEntryIds,
    }),
    entityRepository: input.entityRepository,
  });
}

function retrievedStreamEntryIds(
  input: Partial<Pick<EvidenceLedgerBuildInput, "retrievedEvidence" | "retrievedEpisodes">>,
): StreamEntryId[] {
  const retrievedEvidence = input.retrievedEvidence ?? [];
  const retrievedEpisodes = input.retrievedEpisodes ?? [];

  return uniqueStreamEntryIds([
    ...retrievedEvidence.flatMap((item) => item.provenance?.streamIds ?? []),
    ...retrievedEpisodes.flatMap((result) => result.episode.source_stream_ids),
    ...retrievedEpisodes.flatMap((result) => result.citationChain.map((entry) => entry.id)),
  ]);
}

function imageDerivedLastUpdatedTurns(input: {
  retrievedEvidence?: readonly EvidenceLedgerBuildInput["retrievedEvidence"][number][];
  attachmentRepository: Pick<TurnPhaseCoordinatorOptions["attachmentRepository"], "get">;
}): Record<string, number> {
  const result: Record<string, number> = {};

  for (const item of input.retrievedEvidence ?? []) {
    const attachmentId = item.imageAttachmentId ?? item.provenance?.attachmentId;
    if (attachmentId === undefined) {
      continue;
    }

    const attachment = input.attachmentRepository.get(attachmentId);
    const createdTurn = attachment?.created_turn_global;
    if (createdTurn === undefined || createdTurn === null || !Number.isFinite(createdTurn)) {
      continue;
    }

    for (const streamEntryId of item.provenance?.streamIds ?? []) {
      result[streamEntryId] = createdTurn;
    }
  }

  return result;
}

function recentlyRetrievedSharedStateEntryIds(input: {
  artifact: SharedStateArtifact | null;
  retrievedStreamEntryIds: readonly StreamEntryId[];
}): SharedStateEntryId[] {
  if (input.artifact === null || input.retrievedStreamEntryIds.length === 0) {
    return [];
  }

  const retrievedIds = new Set(input.retrievedStreamEntryIds);

  return input.artifact.entries
    .filter(
      (entry) =>
        entry.superseded_by_id === null &&
        entry.last_updated_stream_entry_ids.some((streamEntryId) =>
          retrievedIds.has(streamEntryId),
        ),
    )
    .map((entry) => entry.id);
}

export async function runRetrievalPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  turnId: string;
  turnInput: TurnPhaseInput;
  isSelfAudience: boolean;
  isUserTurn: boolean;
  cognitionInput: string;
  llmClient: LLMClient;
  recencyMessages: readonly RecencyMessage[];
  audienceEntityId: EntityId | null;
  audienceEntity: ReturnType<TurnPhaseCoordinatorOptions["entityRepository"]["get"]> | null;
  currentSenderBorgRole?: BorgRole | null;
  operatorOnlyDirectivesAllowed?: boolean;
  audienceProfile: ReturnType<TurnPhaseCoordinatorOptions["socialRepository"]["getProfile"]>;
  sessionAudienceRole?: SessionAudienceRole;
  perception: PerceptionResult;
  workingMemory: WorkingMemory;
  suppressionSet: Parameters<
    TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]
  >[0]["suppressionSet"];
  actionLinkSelfContext: TurnExtractionPhaseResult["actionLinkSelfContext"];
  persistedPromotions: TurnExtractionPhaseResult["persistedPromotions"];
  correctiveCommitment: Parameters<
    CorrectivePreferenceTurnService["persistCommitment"]
  >[0]["commitment"];
  activeParticipants: readonly ActiveParticipant[];
  participantRoster: ParticipantRoster | null;
  participantProfiles: readonly ParticipantProfileContext[];
  persistedUserEntry?: StreamEntry;
  currentUserEntries?: readonly StreamEntry[];
  currentTurnAttachmentIds?: readonly AttachmentId[];
  currentTurnFrameAnomaly: ActualFrameAnomalyClassification | null;
  closureLoopAssessment: ClosureLoopAssessment | null;
}): Promise<TurnRetrievalPhaseResult> {
  const creatorDirectiveParticipantEntityIds = participantEntityIds({
    audienceEntityId: input.audienceEntityId,
    audienceEntityKind: input.audienceEntity?.kind ?? null,
    activeParticipants: input.activeParticipants,
  });
  const isPrivateSelfCognition =
    input.isSelfAudience &&
    !input.isUserTurn &&
    input.audienceEntityId === null &&
    creatorDirectiveParticipantEntityIds.length === 0;
  const sessionAudienceRole = input.sessionAudienceRole ?? "participant";
  const nowMs = input.options.clock.now();
  const recallContext: CognitionRecallContext = {
    reader: SELF_RECALL_SCOPE,
    currentSessionId: input.sessionId,
    currentAudienceEntityId: input.audienceEntityId,
    currentParticipantEntityIds: creatorDirectiveParticipantEntityIds,
  };
  const disclosureContext: DisclosureContext = {
    currentSessionId: input.sessionId,
    currentAudienceEntityId: input.audienceEntityId,
    audienceRole: sessionAudienceRole,
    senderEntityId: input.turnInput.senderEntityId ?? null,
    senderRole: input.currentSenderBorgRole ?? null,
    participantEntityIds: creatorDirectiveParticipantEntityIds,
    isPrivateSelfCognition,
  };
  const selfContext =
    input.actionLinkSelfContext !== null &&
    input.persistedPromotions.goalIds.length === 0 &&
    input.persistedPromotions.executiveStepIds.length === 0
      ? input.actionLinkSelfContext
      : await input.options.selfContextBuilder.build({
          turnId: input.turnId,
          sessionId: input.sessionId,
          cognitionInput: input.cognitionInput,
          perception: input.perception,
          autonomyTrigger: input.turnInput.autonomyTrigger,
          audienceEntityId: input.audienceEntityId,
        });
  const selfSnapshot = selfContext.selfSnapshot;
  const activeScoringValues = selfContext.activeScoringValues;
  const retrievalScoringFeatures = selfContext.retrievalScoringFeatures;
  const executiveFocusWithStep = selfContext.executiveFocus;

  const retrievalContext = await input.options.turnRetrievalCoordinator.coordinate({
    turnId: input.turnId,
    userMessage: input.turnInput.userMessage,
    recentMessages: input.recencyMessages.map((message) => ({
      role: message.role,
      content: message.content,
    })),
    cognitionInput: input.cognitionInput,
    inputAudience: input.turnInput.audience,
    isSelfAudience: input.isSelfAudience,
    recallContext,
    disclosureContext,
    audienceEntity: input.audienceEntity,
    audienceProfile: input.audienceProfile,
    perception: input.perception,
    workingMemory: input.workingMemory,
    selfSnapshot,
    executiveFocus: executiveFocusWithStep,
    activeValues: activeScoringValues,
    scoringFeatures: retrievalScoringFeatures,
    suppressionSet: input.suppressionSet,
    ...(input.currentTurnAttachmentIds === undefined || input.currentTurnAttachmentIds.length === 0
      ? {}
      : { currentTurnAttachmentIds: input.currentTurnAttachmentIds }),
    llmClient: input.llmClient,
    proceduralContextModel: input.options.config.anthropic.models.background,
  });
  const applicableCommitments = appendCommitmentIfMissing(
    retrievalContext.applicableCommitments,
    input.correctiveCommitment,
  );
  const actionApplicableCommitments = appendCommitmentIfMissing(
    retrievalContext.actionApplicableCommitments,
    input.correctiveCommitment,
  );
  const pendingCorrections = retrievalContext.pendingCorrections;
  const pendingCommitmentReviews = retrievalContext.pendingCommitmentReviews;
  const affectiveTrajectory = retrievalContext.affectiveTrajectory;
  const retrieval = retrievalContext.retrieval;
  const retrievedEpisodes = retrievalContext.retrievedEpisodes;
  const retrievedSemantic = retrievalContext.retrievedSemantic;
  const proceduralContext = retrievalContext.proceduralContext;
  const selectedSkill = retrievalContext.selectedSkill;
  let autonomySchedulerState: TurnMechanismEvidence["autonomySchedulerState"];

  if (input.options.autonomySchedulerStateProvider !== undefined) {
    try {
      const description = await input.options.autonomySchedulerStateProvider();

      if (description !== null) {
        autonomySchedulerState = {
          // The scheduler's own read, not this phase's `nowMs`. `nowMs` is
          // stamped when the phase starts and the provider is awaited well
          // after that, so using it dated every count in the block by the
          // whole retrieval span rather than by the ledger/compile gap the
          // surface says it is naming.
          observedAt: description.observed_at,
          enabled: description.enabled,
          tickInFlight: description.tick_in_flight,
          nextTickAt: description.next_tick_at,
          scheduledTickAt: description.scheduled_tick_at,
          budget: description.budget,
          fleetBrake: description.fleet_brake,
        };
      }
    } catch (error) {
      if (input.options.tracer.enabled) {
        input.options.tracer.emit("retrieval.degraded", {
          turnId: input.turnId,
          turn_id: input.turnId,
          component: "autonomy_scheduler_mechanism_evidence",
          reason: "scheduler_budget_unavailable",
          ...(input.options.tracer.includePayloads
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      }
    }
  }
  const turnMechanismEvidence = await hydrateTurnMechanismEvidence({
    dataDir: input.options.config.dataDir,
    sessionId: input.sessionId,
    workingMemory: input.workingMemory,
    ...(autonomySchedulerState === undefined ? {} : { autonomySchedulerState }),
    entryIndex: input.options.entryIndex,
    createStreamReader: input.options.createStreamReader,
  });
  const relationalSlots = listConstrainedRelationalSlotsForParticipants(
    input.options.relationalSlotRepository,
    input.activeParticipants,
  );
  const creatorDirectiveApplicableRaw =
    input.options.creatorDirectiveRepository === undefined
      ? []
      : input.options.creatorDirectiveRepository.listApplicable({
          currentAudienceEntityId: input.audienceEntityId,
          currentSenderBorgRole: input.currentSenderBorgRole ?? null,
          isPrivateSelfCognition,
          participantEntityIds: creatorDirectiveParticipantEntityIds,
          sessionRole: input.sessionAudienceRole ?? "participant",
        });
  const creatorDirectiveApplicable =
    input.operatorOnlyDirectivesAllowed === false
      ? creatorDirectiveApplicableRaw.map((item) =>
          item.directive.disclosure_policy.content_scope === "operator_only" ||
          item.directive.activation_policy.scope === "operator_only"
            ? {
                ...item,
                activation: {
                  active: false,
                  reason: "operator_only_omitted" as const,
                },
                disclosure: {
                  render_mode: "omit" as const,
                  reason: "operator_only_omitted" as const,
                },
                render_mode: "omit" as const,
                reason: "operator_only_omitted" as const,
              }
            : item,
        )
      : creatorDirectiveApplicableRaw;
  traceCreatorDirectiveRendered({
    tracer: input.options.tracer,
    turnId: input.turnId,
    sessionId: input.sessionId,
    applicable: creatorDirectiveApplicable,
    currentUserEntryId: input.persistedUserEntry?.id,
    currentUserEntryIds: input.currentUserEntries?.map((entry) => entry.id),
    currentAudienceEntityId: input.audienceEntityId,
    participantEntityIds: creatorDirectiveParticipantEntityIds,
  });
  const creatorDirectiveBriefing = buildCreatorDirectiveBriefing({
    applicable: currentTurnEligibleCreatorDirectives({
      applicable: creatorDirectiveApplicable,
      currentUserEntryId: input.persistedUserEntry?.id,
      currentUserEntryIds: input.currentUserEntries?.map((entry) => entry.id),
    }),
    entityRepository: input.options.entityRepository,
  });
  const recentLivedExperienceConfig =
    input.options.config.generation.evidenceLedger.recentLivedExperience;
  const recentLivedExperienceSinceMs = nowMs - recentLivedExperienceConfig.recencyWindowMs;
  const crossSessionSelfActivity =
    input.options.activityRepository === undefined
      ? []
      : selectCrossSessionSelfActivity({
          repository: input.options.activityRepository,
          currentSessionId: input.sessionId,
          nowMs,
          recencyWindowMs: recentLivedExperienceConfig.recencyWindowMs,
          cap: recentLivedExperienceConfig.cap,
        });
  const selfDecisionIntrospection =
    input.options.selfDecisionRepository === undefined
      ? []
      : selectSelfDecisionIntrospection({
          repository: input.options.selfDecisionRepository,
          nowMs,
          recencyWindowMs: recentLivedExperienceConfig.recencyWindowMs,
          cap: recentLivedExperienceConfig.cap,
        });
  const activityDensity =
    input.options.activityRepository?.listDailyOtherActiveSessionDensity?.({
      currentSessionId: input.sessionId,
      sinceMs: recentLivedExperienceSinceMs,
      untilMs: nowMs,
      limit: recentLivedExperienceConfig.densityCap,
    }) ?? [];
  const selfDecisionDensity =
    input.options.selfDecisionRepository?.listDailyAutonomousSelfPrivateDensity?.({
      sinceMs: recentLivedExperienceSinceMs,
      untilMs: nowMs,
      limit: recentLivedExperienceConfig.densityCap,
    }) ?? [];
  const crossSessionConversationTurnCount =
    input.options.activityRepository?.countOtherActiveSessionConversationTurns?.({
      currentSessionId: input.sessionId,
      sinceMs: recentLivedExperienceSinceMs,
      untilMs: nowMs,
    }) ?? 0;
  const autonomousReflectionCount =
    input.options.selfDecisionRepository?.countAutonomousSelfPrivateDecisions?.({
      sinceMs: recentLivedExperienceSinceMs,
      untilMs: nowMs,
    }) ?? 0;
  const currentSessionPreviousTurnAt = await currentSessionPreviousTurnAdjacencyAt({
    options: input.options,
    sessionId: input.sessionId,
    currentUserEntries: input.currentUserEntries,
    currentUserEntryId: input.persistedUserEntry?.id,
  });
  const previousUserMessageAt = await currentSessionPreviousUserMessageAt({
    options: input.options,
    sessionId: input.sessionId,
    currentUserEntries: input.currentUserEntries,
    currentUserEntryId: input.persistedUserEntry?.id,
  });
  const currentTimeContext = buildCurrentTimePromptContext({
    previousUserMessageAt,
    recentLifeWindowMs: recentLivedExperienceConfig.recencyWindowMs,
    autonomousReflectionCount,
    crossSessionConversationTurnCount,
  });
  const autobiographicalPeriodCutoffMs = nowMs - RECENT_LIVED_EXPERIENCE_DAILY_SPINE_WINDOW_MS;
  const livedExperienceAutobiographicalPeriods =
    input.options.autobiographicalRepository !== undefined &&
    autobiographicalPeriodCutoffMs > recentLivedExperienceSinceMs
      ? input.options.autobiographicalRepository.listPeriods({
          fromTs: recentLivedExperienceSinceMs,
          toTs: autobiographicalPeriodCutoffMs,
          limit: recentLivedExperienceConfig.densityCap,
        })
      : [];
  const livedExperienceClosedWindowEndMs = utcDayStartMs(nowMs) - 1;
  let livedExperienceDaySummaries = [] as ReturnType<
    NonNullable<typeof input.options.livedExperienceDaySummaryRepository>["listForWindow"]
  >;

  if (
    input.options.livedExperienceDaySummaryRepository !== undefined &&
    livedExperienceClosedWindowEndMs >= recentLivedExperienceSinceMs
  ) {
    const selfEntityForLivedExperienceSummaries = input.options.entityRepository.getSelf();

    livedExperienceDaySummaries =
      selfEntityForLivedExperienceSummaries === null
        ? []
        : input.options.livedExperienceDaySummaryRepository.listForWindow({
            selfEntityId: selfEntityForLivedExperienceSummaries.id,
            fromMs: recentLivedExperienceSinceMs,
            toMs: livedExperienceClosedWindowEndMs,
            limit: recentLivedExperienceConfig.densityCap,
          });
  }
  const recentLivedExperience = selectRecentLivedExperienceRows({
    nowMs,
    crossSessionSelfActivity,
    selfDecisionIntrospection,
    activityDensity,
    selfDecisionDensity,
    daySummaries: livedExperienceDaySummaries,
    autobiographicalPeriods: livedExperienceAutobiographicalPeriods,
    returnSilence: {
      currentAudienceLabel:
        input.audienceEntity?.canonical_name ?? input.turnInput.audience ?? null,
      currentSessionPreviousTurnAt,
    },
    windowStartMs: recentLivedExperienceSinceMs,
  });
  const mostRecentOtherSessionActivityAt =
    input.options.activityRepository?.getMostRecentOtherActiveSessionEventOccurredAt?.({
      currentSessionId: input.sessionId,
      sinceMs: recentLivedExperienceSinceMs,
    }) ?? null;
  const renderRecentLivedExperience =
    input.turnInput.origin === "autonomous" ||
    shouldRenderRecentLivedExperience({
      nowMs,
      mostRecentOtherSessionActivityAt,
      currentSessionPreviousTurnAt,
      gapThresholdMs: recentLivedExperienceConfig.gapThresholdMs,
    });
  const observedEventQueryText = input.cognitionInput.trim();
  let observedEventQueryVector: Float32Array | null = null;

  if (input.options.observedEventRepository !== undefined && observedEventQueryText.length > 0) {
    try {
      observedEventQueryVector = await input.options.embeddingClient.embed(observedEventQueryText);
    } catch (error) {
      if (input.options.tracer.enabled) {
        input.options.tracer.emit("observed_event_recall.degraded", {
          turnId: input.turnId,
          turn_id: input.turnId,
          reason: "query_embedding_failed",
          ...(input.options.tracer.includePayloads
            ? { error: error instanceof Error ? error.message : String(error) }
            : {}),
        });
      }
    }
  }

  const observedEventIntrospection =
    input.options.observedEventRepository === undefined
      ? []
      : await recallObservedEventsForCognition({
          repository: input.options.observedEventRepository,
          speakerEntityIds: creatorDirectiveParticipantEntityIds,
          queryVector: observedEventQueryVector,
          nowMs,
          recencyWindowMs: DEFAULT_OBSERVED_EVENT_INTROSPECTION_RECENCY_WINDOW_MS,
          cap: DEFAULT_OBSERVED_EVENT_INTROSPECTION_CAP,
        });
  const autobiographicalRecall = input.options.config.generation.evidenceLedger.enabled
    ? await new AutobiographicalRecallService({
        clock: input.options.clock,
        activityRepository: input.options.activityRepository,
        selfDecisionRepository: input.options.selfDecisionRepository,
        observedEventRepository: input.options.observedEventRepository,
        episodicRepository: input.options.episodicRepository,
        actionRepository: input.options.actionRepository,
        goalsRepository: input.options.goalsRepository,
        openQuestionsRepository: input.options.openQuestionsRepository,
        autobiographicalRepository: input.options.autobiographicalRepository,
        sessionsRepository: input.options.sessionsRepository,
        createStreamReader: input.options.createStreamReader,
      }).recall({
        sessionId: input.sessionId,
        temporalCue: input.perception.temporalCue,
        isSelfAudience: input.isSelfAudience,
        sessionAudienceRole,
        perceptionMode: input.perception.mode,
      })
    : null;
  const evidenceLedgerContext = await buildEvidenceLedgerFinalizerContext({
    options: input.options,
    input: {
      sessionId: input.sessionId,
      turnId: input.turnId,
      nowMs,
      audienceEntityId: input.audienceEntityId,
      currentUserMessage: input.turnInput.userMessage,
      currentUserEntry: input.persistedUserEntry ?? input.currentUserEntries?.[0],
      currentUserEntries: input.currentUserEntries,
      globalTurnCounter: input.turnInput.globalTurnCounter,
      workingMemory: input.workingMemory,
      applicableCommitments,
      retrievedEvidence: retrieval.evidence,
      retrievedEpisodes,
      retrievedSemantic,
      openQuestions: retrieval.open_questions,
      pendingCorrections,
      pendingCommitmentReviews,
      frameAnomaly: input.currentTurnFrameAnomaly,
      activeParticipants: input.activeParticipants,
      recentLivedExperience,
      renderRecentLivedExperience,
      observedEventIntrospection,
      autobiographicalRecall,
      participantRoster: input.participantRoster,
      isUserTurn: input.isUserTurn,
      perception: input.perception,
      closureLoopAssessment: input.closureLoopAssessment,
    },
  });
  const routingOverride = buildContradictionRoutingOverride({
    isUserTurn: input.isUserTurn,
    perception: input.perception,
    audienceEntityId: input.audienceEntityId,
    openQuestionsRepository: input.options.openQuestionsRepository,
    evidenceLedger: evidenceLedgerContext.ledger,
    enabled: input.options.config.deliberation.contradictionRouting.enabled,
  });

  return {
    selfContext,
    selfSnapshot,
    executiveFocusWithStep,
    retrievalContext,
    applicableCommitments,
    actionApplicableCommitments,
    pendingCorrections,
    pendingCommitmentReviews,
    affectiveTrajectory,
    retrieval,
    retrievedEpisodes,
    retrievedSemantic,
    turnMechanismEvidence,
    proceduralContext,
    selectedSkill,
    currentTimeContext,
    relationalSlots,
    participantRoster: input.participantRoster,
    creatorDirectiveBriefing,
    evidenceLedgerContext,
    routingOverride,
  };
}

async function buildEvidenceLedgerFinalizerContext(input: {
  options: TurnPhaseCoordinatorOptions;
  input: EvidenceLedgerFinalizerBuildInput;
}): Promise<EvidenceLedgerFinalizerContext> {
  return traceTurnPhase({
    tracer: input.options.tracer,
    clock: input.options.clock,
    turnId: input.input.turnId ?? "unknown",
    sessionId: input.input.sessionId,
    phase: "ledger",
    run: () => buildEvidenceLedgerFinalizerContextInternal(input),
    completedSub: summarizeEvidenceLedgerContext,
  });
}

async function buildEvidenceLedgerFinalizerContextInternal(input: {
  options: TurnPhaseCoordinatorOptions;
  input: EvidenceLedgerFinalizerBuildInput;
}): Promise<EvidenceLedgerFinalizerContext> {
  const config = input.options.config.generation.evidenceLedger;
  const previousSharedState =
    input.input.audienceEntityId === null
      ? null
      : input.options.sharedStateRepository.get(input.input.audienceEntityId);
  const priorUserTurnCount = await countPriorUserTurnsForSession({
    options: input.options,
    sessionId: input.input.sessionId,
    currentUserEntryId: input.input.currentUserEntry?.id,
    currentUserEntryIds: input.input.currentUserEntries?.map((entry) => entry.id),
    currentUserEntries: input.input.currentUserEntries,
  });
  const sessionReentryContinuity = buildSessionReentryContinuityPrompt({
    isUserTurn: input.input.isUserTurn,
    priorUserTurnCount,
    audienceEntityId: input.input.audienceEntityId,
    artifact: previousSharedState,
    ...(input.input.nowMs === undefined ? {} : { nowMs: input.input.nowMs }),
  });

  emitSessionReentryContinuityTrace({
    options: input.options,
    turnId: input.input.turnId ?? "unknown",
    sessionId: input.input.sessionId,
    continuity: sessionReentryContinuity,
  });

  if (!config.enabled) {
    return {
      ledger: null,
      promptSection: null,
      sessionReentryContinuityPromptSection: sessionReentryContinuity.promptSection,
      sharedStateAppliedOperationCount: 0,
      openQuestionsRenderedToFinalizerCount: 0,
    };
  }

  const sharedStateRecall =
    input.options.sharedStateRepository.listRecentEntriesForCognition?.({
      excludeAudienceEntityId: input.input.audienceEntityId,
      limit: SHARED_STATE_COGNITION_RECALL_LIMIT,
    }) ?? [];
  const finalizerInput: EvidenceLedgerFinalizerBuildInput = {
    ...input.input,
    sharedStateRecall,
  };
  const compacted = await buildCompactedEvidenceLedgerWithoutSharedState({
    options: input.options,
    input: finalizerInput,
  });
  const ledgerWithoutSharedState = compacted.ledger;
  const renderedWithoutSharedState = compacted.rendered;
  const sharedStateResult = await compileSharedStateArtifactForEvidenceLedgerResult({
    options: input.options,
    input: finalizerInput,
    previousArtifact: previousSharedState,
    ledger: ledgerWithoutSharedState,
    promptVisibleLedger: renderedWithoutSharedState ?? "",
  });
  const renderOptions =
    sharedStateResult.renderOptions ?? sharedStateRenderOptions(input.options.config);
  const ledger = withSharedStateArtifact(
    ledgerWithoutSharedState,
    previousSharedState,
    renderOptions,
  );
  const rendered = renderEvidenceLedger(ledger, {
    sharedState: renderOptions,
  });
  const sharedStateSummary = summarizeSharedStateArtifactRender(ledger.sharedState, renderOptions);
  const traceSummary = summarizeEvidenceLedgerTrace({
    ...ledger,
    estimatedTokens: estimateEvidenceLedgerPromptTokens(ledger, {
      sharedState: renderOptions,
    }),
  });

  if (
    input.options.tracer.enabled &&
    input.input.turnId !== undefined &&
    evidenceLedgerCompactionChanged(compacted.compacted.traceSummary)
  ) {
    input.options.tracer.emit("evidence_ledger.compaction.completed", {
      turnId: input.input.turnId,
      session_id: input.input.sessionId,
      pre_dedupe_tokens: compacted.compacted.traceSummary.preDedupeTokens,
      post_dedupe_tokens: compacted.compacted.traceSummary.postDedupeTokens,
      pre_cap_tokens: compacted.compacted.traceSummary.preCapTokens,
      post_section_cap_tokens: compacted.compacted.traceSummary.postSectionCapTokens,
      post_cap_tokens: compacted.compacted.traceSummary.postCapTokens,
      deduped_entry_count: compacted.compacted.traceSummary.dedupedEntryCount,
      omitted_entry_counts: toTraceJsonValue(
        compacted.compacted.traceSummary.omittedEntryCountsBySection,
      ),
      dropped_sections: compacted.compacted.traceSummary.droppedSections,
      target_tokens: compacted.compacted.traceSummary.targetTokens,
      hard_cap_tokens: compacted.compacted.traceSummary.hardCapTokens,
    });
  }

  if (input.options.tracer.enabled && input.input.turnId !== undefined) {
    input.options.tracer.emit("evidence_ledger.completed", {
      turnId: input.input.turnId,
      session_id: input.input.sessionId,
      entry_counts: toTraceJsonValue(traceSummary.entryCountsBySection),
      transcript_included: traceSummary.transcriptIncluded,
      transcript_compacted: traceSummary.transcriptCompacted,
      transcript_omitted_reason: traceSummary.transcriptOmittedReason ?? null,
      original_transcript_token_estimate: traceSummary.originalTranscriptTokenEstimate,
      compacted_transcript_token_estimate: traceSummary.compactedTranscriptTokenEstimate,
      compacted_entry_count: traceSummary.compactedEntryCount,
      raw_preserved_user_entry_count: traceSummary.rawPreservedUserEntryCount,
      total_estimated_tokens: traceSummary.totalEstimatedTokens,
      estimated_tokens_by_section: toTraceJsonValue(traceSummary.estimatedTokensBySection),
      shared_state_entry_count: sharedStateSummary.renderedEntryCount,
      shared_state_rendered_token_estimate: sharedStateSummary.estimatedTokens,
      shared_state_rendered_by_kind: toTraceJsonValue(sharedStateSummary.renderedByKind),
      shared_state_newest_entries_reserved: sharedStateSummary.newestReservedEntryCount,
    });

    input.options.tracer.emit("evidence_ledger.built", {
      turnId: input.input.turnId,
      turn_id: input.input.turnId,
      session_id: input.input.sessionId,
      entry_counts: toTraceJsonValue(traceSummary.entryCountsBySection),
      image_attachment_count: ledger.imageAttachments?.length ?? 0,
      shared_state_entry_count: ledger.sharedState?.entries.length ?? 0,
      total_estimated_tokens: traceSummary.totalEstimatedTokens,
      ...(input.options.tracer.includePayloads ? { ledger: toTraceJsonValue(ledger) } : {}),
    });
  }

  return {
    ledger,
    promptSection: rendered,
    sessionReentryContinuityPromptSection: sessionReentryContinuity.promptSection,
    sharedStateAppliedOperationCount: sharedStateResult.appliedOperationCount,
    openQuestionsRenderedToFinalizerCount: traceSummary.entryCountsBySection.open_questions,
  };
}

export async function buildCompactedEvidenceLedgerWithoutSharedState(input: {
  options: TurnPhaseCoordinatorOptions;
  input: EvidenceLedgerFinalizerBuildInput;
}): Promise<CompactedEvidenceLedgerWithoutSharedStateResult> {
  const config = input.options.config.generation.evidenceLedger;
  const builder = new EvidenceLedgerBuilder({
    createStreamReader: input.options.createStreamReader,
    relationalSlotRepository: input.options.relationalSlotRepository,
    actionRepository: input.options.actionRepository,
    commitmentRepository: input.options.commitmentRepository,
    goalsRepository: input.options.goalsRepository,
    openQuestionsRepository: input.options.openQuestionsRepository,
    currentSessionTranscriptTokenBudget: config.currentSessionTranscriptTokenBudget,
    actionThreadRenderLimit: config.actionThreadRenderLimit,
    actionThreadSimilarityThreshold: config.actionThreadSimilarityThreshold,
    actionThreadSourceRecordLimit: config.actionThreadSourceRecordLimit,
    actionThreadSalienceClassReservedSlots: config.actionThreadSalienceClassReservedSlots,
    actionThreadAudienceReservedSlots: config.actionThreadAudienceReservedSlots,
    entityRepository: input.options.entityRepository,
    attachmentRepository: input.options.attachmentRepository,
    maxImagesPerLedger: input.options.config.attachments.maxImagesPerLedger,
    maxLedgerImageBytes: input.options.config.attachments.maxLedgerImageBytes,
    imageRenderMaxDimension: input.options.config.attachments.imageRenderMaxDimension,
    tracer: input.options.tracer,
  });
  const builtLedger = await builder.build(input.input);
  const compacted = compactEvidenceLedger(builtLedger, {
    targetTokens: config.finalizerTargetTokens,
    hardCapTokens: config.finalizerHardCapTokens,
    maxEntryTextTokens: config.finalizerMaxEntryTextTokens,
    sectionOptions: config.sectionOptions,
  });

  return {
    compacted,
    ledger: compacted.ledger,
    rendered: renderEvidenceLedger(compacted.ledger),
  };
}

const CURRENT_SESSION_ADJACENCY_KINDS = [
  "agent_msg",
  "agent_suppressed",
  "agent_observed",
] as const;

function maxTimestamp(values: readonly number[]): number | null {
  const finite = values.filter((value) => Number.isFinite(value));

  return finite.length === 0 ? null : Math.max(...finite);
}

function isCurrentSessionAdjacencyEntry(entry: StreamEntry): boolean {
  return (
    CURRENT_SESSION_ADJACENCY_KINDS.some((kind) => entry.kind === kind) &&
    entry.turn_status !== "aborted"
  );
}

function isCurrentSessionUserMessageEntry(entry: StreamEntry): boolean {
  return entry.kind === "user_msg" && entry.turn_status !== "aborted";
}

function buildCurrentTimePromptContext(input: {
  previousUserMessageAt: number | null;
  recentLifeWindowMs: number;
  autonomousReflectionCount: number;
  crossSessionConversationTurnCount: number;
}): CurrentTimePromptContext {
  return {
    previousUserMessageAt: input.previousUserMessageAt,
    recentLifeElsewhere: {
      windowMs: input.recentLifeWindowMs,
      autonomousReflectionCount: nonNegativeInteger(input.autonomousReflectionCount),
      crossSessionConversationTurnCount: nonNegativeInteger(
        input.crossSessionConversationTurnCount,
      ),
    },
  };
}

function nonNegativeInteger(value: number): number {
  return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : 0;
}

async function currentSessionPreviousTurnAdjacencyAt(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  currentUserEntryId?: StreamEntryId;
  currentUserEntries?: readonly StreamEntry[];
}): Promise<number | null> {
  const currentEntryIds = dedupePreservingOrder([
    ...(input.currentUserEntryId === undefined ? [] : [input.currentUserEntryId]),
    ...(input.currentUserEntries ?? []).map((entry) => entry.id),
  ]);
  const currentEntryTimestamps = (input.currentUserEntries ?? []).map((entry) => entry.timestamp);

  if (input.options.entryIndex !== undefined) {
    const currentRecords =
      currentEntryIds.length === 0
        ? new Map()
        : input.options.entryIndex.lookupMany(currentEntryIds);
    const currentEntryIndexes = currentEntryIds.flatMap((entryId) => {
      const indexed = currentRecords.get(entryId)?.entry_index ?? null;

      return indexed === null ? [] : [indexed];
    });
    const currentTimestamps = [
      ...currentEntryTimestamps,
      ...currentEntryIds.flatMap((entryId) => {
        const timestamp = currentRecords.get(entryId)?.timestamp;

        return timestamp === undefined ? [] : [timestamp];
      }),
    ];
    const terminalRecords = CURRENT_SESSION_ADJACENCY_KINDS.flatMap((kind) =>
      input.options.entryIndex!.lookupSessionEntriesByKind({
        sessionId: input.sessionId,
        kind,
      }),
    ).filter((record) => record.active);

    if (currentEntryIndexes.length > 0) {
      const oldestCurrentEntryIndex = Math.min(...currentEntryIndexes);

      return maxTimestamp(
        terminalRecords
          .filter((record) => record.entry_index !== null)
          .filter((record) => record.entry_index! < oldestCurrentEntryIndex)
          .map((record) => record.timestamp),
      );
    }

    if (currentTimestamps.length > 0) {
      const oldestCurrentTimestamp = Math.min(...currentTimestamps);

      return maxTimestamp(
        terminalRecords
          .filter((record) => record.timestamp < oldestCurrentTimestamp)
          .map((record) => record.timestamp),
      );
    }

    return maxTimestamp(terminalRecords.map((record) => record.timestamp));
  }

  const entries = filterActiveStreamEntries(
    await loadSessionStreamEntries(input.options.createStreamReader(input.sessionId)),
  );
  const currentIds = new Set(currentEntryIds);
  const firstCurrentIndex =
    currentIds.size === 0 ? -1 : entries.findIndex((entry) => currentIds.has(entry.id));
  const priorEntries =
    firstCurrentIndex >= 0
      ? entries.slice(0, firstCurrentIndex)
      : currentEntryTimestamps.length === 0
        ? entries
        : entries.filter((entry) => entry.timestamp < Math.min(...currentEntryTimestamps));

  return maxTimestamp(
    priorEntries.filter(isCurrentSessionAdjacencyEntry).map((entry) => entry.timestamp),
  );
}

async function currentSessionPreviousUserMessageAt(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  currentUserEntryId?: StreamEntryId;
  currentUserEntries?: readonly StreamEntry[];
}): Promise<number | null> {
  const currentEntryIds = dedupePreservingOrder([
    ...(input.currentUserEntryId === undefined ? [] : [input.currentUserEntryId]),
    ...(input.currentUserEntries ?? []).map((entry) => entry.id),
  ]);
  const currentEntryTimestamps = (input.currentUserEntries ?? []).map((entry) => entry.timestamp);

  if (input.options.entryIndex !== undefined) {
    const currentRecords =
      currentEntryIds.length === 0
        ? new Map()
        : input.options.entryIndex.lookupMany(currentEntryIds);
    const currentEntryIndexes = currentEntryIds.flatMap((entryId) => {
      const indexed = currentRecords.get(entryId)?.entry_index ?? null;

      return indexed === null ? [] : [indexed];
    });
    const currentTimestamps = [
      ...currentEntryTimestamps,
      ...currentEntryIds.flatMap((entryId) => {
        const timestamp = currentRecords.get(entryId)?.timestamp;

        return timestamp === undefined ? [] : [timestamp];
      }),
    ];
    const userRecords = input.options.entryIndex
      .lookupSessionEntriesByKind({
        sessionId: input.sessionId,
        kind: "user_msg",
      })
      .filter((record) => record.active);

    if (currentEntryIndexes.length > 0) {
      const oldestCurrentEntryIndex = Math.min(...currentEntryIndexes);

      return maxTimestamp(
        userRecords
          .filter((record) => record.entry_index !== null)
          .filter((record) => record.entry_index! < oldestCurrentEntryIndex)
          .map((record) => record.timestamp),
      );
    }

    if (currentTimestamps.length > 0) {
      const oldestCurrentTimestamp = Math.min(...currentTimestamps);

      return maxTimestamp(
        userRecords
          .filter((record) => record.timestamp < oldestCurrentTimestamp)
          .map((record) => record.timestamp),
      );
    }

    return maxTimestamp(userRecords.map((record) => record.timestamp));
  }

  const entries = filterActiveStreamEntries(
    await loadSessionStreamEntries(input.options.createStreamReader(input.sessionId)),
  );
  const currentIds = new Set(currentEntryIds);
  const firstCurrentIndex =
    currentIds.size === 0 ? -1 : entries.findIndex((entry) => currentIds.has(entry.id));
  const priorEntries =
    firstCurrentIndex >= 0
      ? entries.slice(0, firstCurrentIndex)
      : currentEntryTimestamps.length === 0
        ? entries
        : entries.filter((entry) => entry.timestamp < Math.min(...currentEntryTimestamps));

  return maxTimestamp(
    priorEntries.filter(isCurrentSessionUserMessageEntry).map((entry) => entry.timestamp),
  );
}

async function countPriorUserTurnsForSession(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  currentUserEntryId?: StreamEntryId;
  currentUserEntryIds?: readonly StreamEntryId[];
  currentUserEntries?: readonly StreamEntry[];
}): Promise<number> {
  const currentUserEntryIds = new Set([
    ...(input.currentUserEntryId === undefined ? [] : [input.currentUserEntryId]),
    ...(input.currentUserEntryIds ?? []),
  ]);

  if (input.options.entryIndex !== undefined) {
    if (input.currentUserEntries !== undefined && input.currentUserEntries.length > 0) {
      const currentRecords = input.options.entryIndex.lookupMany(
        input.currentUserEntries.map((entry) => entry.id),
      );
      const currentEntryIndexes = input.currentUserEntries.flatMap((entry) => {
        const indexed = currentRecords.get(entry.id)?.entry_index ?? entry.entry_index ?? null;

        return indexed === null ? [] : [indexed];
      });

      if (currentEntryIndexes.length > 0) {
        const oldestCurrentEntryIndex = Math.min(...currentEntryIndexes);

        return input.options.entryIndex
          .lookupSessionEntriesByKind({
            sessionId: input.sessionId,
            kind: "user_msg",
          })
          .filter((record) => record.entry_index !== null)
          .filter((record) => record.active)
          .filter((record) => record.entry_index! < oldestCurrentEntryIndex).length;
      }
    }

    return input.options.entryIndex.countSessionEntriesByKind({
      sessionId: input.sessionId,
      kind: "user_msg",
      excludeEntryId: input.currentUserEntryId,
    });
  }

  const entries = await loadSessionStreamEntries(input.options.createStreamReader(input.sessionId));

  if (input.currentUserEntries !== undefined && input.currentUserEntries.length > 0) {
    const currentIds = new Set(input.currentUserEntries.map((entry) => entry.id));
    const firstCurrentIndex = entries.findIndex((entry) => currentIds.has(entry.id));

    if (firstCurrentIndex >= 0) {
      return entries.slice(0, firstCurrentIndex).filter((entry) => entry.kind === "user_msg")
        .length;
    }
  }

  return entries.filter((entry) => entry.kind === "user_msg" && !currentUserEntryIds.has(entry.id))
    .length;
}

function sharedStateLastUpdatedTurnByStreamEntryId(input: {
  entries: readonly Pick<StreamEntry, "id" | "turn_id">[];
  currentUserEntry: Pick<StreamEntry, "id" | "turn_id">;
  currentUserEntries?: readonly StreamEntry[];
  currentTurnId?: string;
  currentTurnCounter?: number;
}): Record<string, number> {
  if (input.currentTurnCounter === undefined) {
    return {};
  }

  const turnIds: string[] = [];
  const observedTurnIds = new Set<string>();

  for (const entry of input.entries) {
    if (entry.turn_id !== undefined && !observedTurnIds.has(entry.turn_id)) {
      turnIds.push(entry.turn_id);
      observedTurnIds.add(entry.turn_id);
    }
  }

  const currentTurnId = input.currentUserEntry.turn_id ?? input.currentTurnId;

  if (currentTurnId !== undefined && !observedTurnIds.has(currentTurnId)) {
    turnIds.push(currentTurnId);
    observedTurnIds.add(currentTurnId);
  }

  const turnCounterByTurnId = new Map<string, number>();
  const currentTurnIndex = currentTurnId === undefined ? -1 : turnIds.lastIndexOf(currentTurnId);

  if (currentTurnIndex >= 0) {
    for (let index = 0; index < turnIds.length; index += 1) {
      turnCounterByTurnId.set(
        turnIds[index]!,
        input.currentTurnCounter - (currentTurnIndex - index),
      );
    }
  } else {
    for (let index = 0; index < turnIds.length; index += 1) {
      turnCounterByTurnId.set(turnIds[index]!, input.currentTurnCounter - (turnIds.length - index));
    }
  }

  const turnCounterByStreamEntryId: Record<string, number> = {
    [input.currentUserEntry.id]: input.currentTurnCounter,
  };

  for (const currentUserEntry of input.currentUserEntries ?? []) {
    turnCounterByStreamEntryId[currentUserEntry.id] = input.currentTurnCounter;
  }

  for (const entry of input.entries) {
    if (entry.turn_id === undefined) {
      continue;
    }

    const turnCounter = turnCounterByTurnId.get(entry.turn_id);

    if (turnCounter !== undefined) {
      turnCounterByStreamEntryId[entry.id] = turnCounter;
    }
  }

  return turnCounterByStreamEntryId;
}

function streamEntryFromIndexedFacts(
  facts: IndexedEntryFacts,
): Pick<StreamEntry, "id" | "kind" | "turn_status"> & { active: boolean } {
  return {
    id: facts.entry_id as StreamEntryId,
    kind: facts.kind ?? "internal_event",
    turn_status: facts.turn_status ?? "active",
    active: facts.active,
  };
}

function createIndexedSourceTrustLookup(input: {
  entryIndex?: Pick<NonNullable<TurnPhaseCoordinatorOptions["entryIndex"]>, "lookupEntriesById">;
  currentUserEntries: readonly StreamEntry[];
}): {
  lookup: (streamEntryIds: readonly StreamEntryId[]) => Map<StreamEntryId, IndexedEntryFacts>;
  entriesForKnownFacts: () => Pick<StreamEntry, "id" | "kind" | "turn_id">[];
} {
  const factsById = new Map<StreamEntryId, IndexedEntryFacts>();

  const rememberCurrentUserEntry = (): void => {
    for (const currentUserEntry of input.currentUserEntries) {
      if (!factsById.has(currentUserEntry.id)) {
        factsById.set(currentUserEntry.id, {
          entry_id: currentUserEntry.id,
          session_id: currentUserEntry.session_id,
          timestamp: currentUserEntry.timestamp,
          kind: currentUserEntry.kind,
          turn_id: currentUserEntry.turn_id ?? null,
          turn_status: currentUserEntry.turn_status ?? "active",
          active: currentUserEntry.turn_status !== "aborted",
        });
      }
    }
  };

  const lookup = (
    streamEntryIds: readonly StreamEntryId[],
  ): Map<StreamEntryId, IndexedEntryFacts> => {
    rememberCurrentUserEntry();

    if (input.entryIndex !== undefined) {
      const missingIds = uniqueStreamEntryIds(
        streamEntryIds.filter((streamEntryId) => !factsById.has(streamEntryId)),
      );

      if (missingIds.length > 0) {
        for (const [streamEntryId, facts] of input.entryIndex.lookupEntriesById(missingIds)) {
          factsById.set(streamEntryId as StreamEntryId, facts);
        }
      }
    }

    const result = new Map<StreamEntryId, IndexedEntryFacts>();

    for (const streamEntryId of streamEntryIds) {
      const facts = factsById.get(streamEntryId);

      if (facts !== undefined) {
        result.set(streamEntryId, facts);
      }
    }

    return result;
  };

  return {
    lookup,
    entriesForKnownFacts: () =>
      [...factsById.values()].map((facts) => ({
        ...streamEntryFromIndexedFacts(facts),
        turn_id: facts.turn_id ?? undefined,
      })),
  };
}

function buildIndexedSharedStateSourceTrustValidator(input: {
  lookupFacts: (streamEntryIds: readonly StreamEntryId[]) => Map<StreamEntryId, IndexedEntryFacts>;
  quarantinedStreamEntryIds: ReadonlySet<StreamEntryId>;
  isActiveAttachmentStreamEntry?: (streamEntryId: StreamEntryId) => boolean | null;
  onMissingIndexedStreamEntry?: (streamEntryId: StreamEntryId) => void;
}) {
  const warnedStreamEntryIds = new Set<StreamEntryId>();

  return (streamEntryId: StreamEntryId) => {
    if (input.quarantinedStreamEntryIds.has(streamEntryId)) {
      return {
        allowed: false,
        reason: "quarantined",
      } as const;
    }

    const facts = input.lookupFacts([streamEntryId]).get(streamEntryId);

    if (facts === undefined && !warnedStreamEntryIds.has(streamEntryId)) {
      warnedStreamEntryIds.add(streamEntryId);
      input.onMissingIndexedStreamEntry?.(streamEntryId);
    }

    if (facts?.active === false) {
      return {
        allowed: false,
        reason: "inactive",
      } as const;
    }

    if (facts?.kind === "user_image_attachment") {
      const active = input.isActiveAttachmentStreamEntry?.(streamEntryId);

      if (active === false) {
        return {
          allowed: false,
          reason: "inactive",
        } as const;
      }
    }

    return { allowed: true } as const;
  };
}

function emitSessionReentryContinuityTrace(input: {
  options: TurnPhaseCoordinatorOptions;
  turnId: string | undefined;
  sessionId: SessionId;
  continuity: SessionReentryContinuityPrompt;
}): void {
  if (!input.options.tracer.enabled || input.turnId === undefined) {
    return;
  }

  const summary = input.continuity.summary;

  const traceData = {
    turnId: input.turnId,
    session_id: input.sessionId,
    status: summary.status,
    audience_entity_id: summary.audienceEntityId,
    active_entry_count: summary.activeEntryCount,
    active_keyed_entry_count: summary.activeKeyedEntryCount,
    active_legacy_entry_count: summary.activeLegacyEntryCount,
    active_state_key_count: summary.activeStateKeyCount,
    active_counts_by_kind: toTraceJsonValue(summary.activeCountsByKind),
    active_entries_by_key: toTraceJsonValue(summary.activeEntriesByKey),
    most_recent_update:
      summary.mostRecentUpdate === null ? null : toTraceJsonValue(summary.mostRecentUpdate),
  };

  input.options.tracer.emit("session_reentry.continuity.evaluated", traceData);

  if (summary.status === "rendered") {
    input.options.tracer.emit("session_reentry.continuity.rendered", traceData);
  }
}

export async function compileSharedStateArtifactForEvidenceLedger(input: {
  options: TurnPhaseCoordinatorOptions;
  input: EvidenceLedgerFinalizerBuildInput;
  previousArtifact?: SharedStateArtifact | null;
  ledger: EvidenceLedger;
  promptVisibleLedger: string;
  compilePass?: SharedStateCompilePass;
  assistantResponse?: SharedStateCompilerAssistantResponse | null;
  compileAnchorStreamEntryId?: StreamEntryId;
}): Promise<SharedStateArtifact | null> {
  return (await compileSharedStateArtifactForEvidenceLedgerResult(input)).artifact;
}

export async function compileSharedStateArtifactForEvidenceLedgerResult(input: {
  options: TurnPhaseCoordinatorOptions;
  input: EvidenceLedgerFinalizerBuildInput;
  previousArtifact?: SharedStateArtifact | null;
  ledger: EvidenceLedger;
  promptVisibleLedger: string;
  compilePass?: SharedStateCompilePass;
  assistantResponse?: SharedStateCompilerAssistantResponse | null;
  compileAnchorStreamEntryId?: StreamEntryId;
}): Promise<SharedStateArtifactForEvidenceLedgerResult> {
  return traceTurnPhase({
    tracer: input.options.tracer,
    clock: input.options.clock,
    turnId: input.input.turnId ?? "unknown",
    sessionId: input.input.sessionId,
    phase: "shared",
    run: () => compileSharedStateArtifactForEvidenceLedgerResultInternal(input),
    completedSub: (result) =>
      `entries=${result.artifact?.entries.length ?? 0} ops=${result.appliedOperationCount}`,
  });
}

async function compileSharedStateArtifactForEvidenceLedgerResultInternal(input: {
  options: TurnPhaseCoordinatorOptions;
  input: EvidenceLedgerFinalizerBuildInput;
  previousArtifact?: SharedStateArtifact | null;
  ledger: EvidenceLedger;
  promptVisibleLedger: string;
  compilePass?: SharedStateCompilePass;
  assistantResponse?: SharedStateCompilerAssistantResponse | null;
  compileAnchorStreamEntryId?: StreamEntryId;
}): Promise<SharedStateArtifactForEvidenceLedgerResult> {
  const audienceEntityId = input.input.audienceEntityId;

  if (audienceEntityId === null) {
    return { artifact: null, appliedOperationCount: 0 };
  }

  const previousArtifact =
    input.previousArtifact ?? input.options.sharedStateRepository.get(audienceEntityId);
  const compilePass = input.compilePass ?? "pre_answer";
  const assistantResponse = input.assistantResponse ?? null;

  const currentUserEntries =
    input.input.currentUserEntries === undefined || input.input.currentUserEntries.length === 0
      ? input.input.currentUserEntry === undefined
        ? []
        : [input.input.currentUserEntry]
      : [...input.input.currentUserEntries];
  const currentUserEntry = input.input.currentUserEntry ?? currentUserEntries[0];
  const currentUserStreamEntryIds = currentUserEntries.map((entry) => entry.id);
  const compileAnchorStreamEntryId =
    input.compileAnchorStreamEntryId ??
    (compilePass === "post_response" ? assistantResponse?.streamEntryId : undefined) ??
    currentUserEntry?.id;

  if (compilePass === "pre_answer" && (!input.input.isUserTurn || currentUserEntry === undefined)) {
    return { artifact: previousArtifact, appliedOperationCount: 0 };
  }

  if (compileAnchorStreamEntryId === undefined) {
    return { artifact: previousArtifact, appliedOperationCount: 0 };
  }

  const compileAnchorEntryForTurnAge: Pick<StreamEntry, "id" | "turn_id"> = currentUserEntry ?? {
    id: compileAnchorStreamEntryId,
    turn_id: input.input.turnId,
  };
  const postResponseStreamEntryIds =
    compilePass === "post_response" && assistantResponse !== null
      ? [assistantResponse.streamEntryId]
      : [];

  const quarantinedStreamEntryIds =
    input.options.entryIndex === undefined
      ? await collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIds(
          input.options.config.dataDir,
        )
      : await collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIds(
          input.options.entryIndex,
        );
  const indexedSourceTrustLookup =
    input.options.entryIndex === undefined
      ? null
      : createIndexedSourceTrustLookup({
          entryIndex: input.options.entryIndex,
          currentUserEntries,
        });
  const currentSessionTrustEntries =
    indexedSourceTrustLookup === null
      ? typeof input.options.createStreamReader === "function"
        ? await loadSessionStreamEntries(input.options.createStreamReader(input.input.sessionId))
        : []
      : indexedSourceTrustLookup.entriesForKnownFacts();
  const sourceTrustValidator =
    indexedSourceTrustLookup === null
      ? buildSharedStateSourceTrustValidator({
          currentSessionEntries:
            currentUserEntry === undefined ||
            currentSessionTrustEntries.some((entry) => entry.id === currentUserEntry.id)
              ? (currentSessionTrustEntries as StreamEntry[])
              : [...(currentSessionTrustEntries as StreamEntry[]), ...currentUserEntries],
          quarantinedStreamEntryIds,
        })
      : buildIndexedSharedStateSourceTrustValidator({
          lookupFacts: indexedSourceTrustLookup.lookup,
          quarantinedStreamEntryIds,
          isActiveAttachmentStreamEntry: (streamEntryId) =>
            input.options.attachmentRepository.isActiveForStreamEntry(streamEntryId),
          onMissingIndexedStreamEntry: (streamEntryId) => {
            console.warn(
              `Stream entry ${streamEntryId} was not found in the stream entry index during shared-state source trust validation`,
            );
          },
        });
  const sharedStateConfig = input.options.config.generation.evidenceLedger.decisionArtifact;
  const turnCounter = input.input.globalTurnCounter ?? input.input.workingMemory?.turn_counter;
  const ledgerPromptContext = buildSharedStateLedgerPromptContext({
    ledger: input.ledger,
    previousArtifact,
    fullPromptVisibleLedger: input.promptVisibleLedger,
    enabled: sharedStateConfig.ledgerDelta.enabled,
    minTailPerSection: sharedStateConfig.ledgerDelta.minTailPerSection,
    sourceTrustValidator,
  });
  const currentAudienceActionCandidatesForCanonicalization =
    selectCurrentAudienceSharedStateActionCandidatesForCanonicalization({
      actionRepository: input.options.actionRepository,
      audienceEntityId,
      activeParticipants: input.input.activeParticipants,
    });
  const activeGoals = input.options.goalsRepository.list({
    status: "active",
    visibleToAudienceEntityId: audienceEntityId,
  });
  const activeCommitments = input.options.commitmentRepository.list({
    activeOnly: true,
    audience: audienceEntityId,
  });
  const activeCommitmentCanonicalizationRecords = activeCommitments.filter(
    isSharedStateCommitmentCanonicalizationRecord,
  );
  const activeOpenQuestions = input.options.openQuestionsRepository.list({
    status: "open",
    visibleToAudienceEntityId: audienceEntityId,
    limit: 80,
  });
  const relationalSlotsContext = listSharedStateRelationalSlotsForParticipants(
    input.options.relationalSlotRepository,
    input.input.activeParticipants ?? [],
  );
  const relationalSlotEvidenceStreamEntryIds = uniqueStreamEntryIds(
    relationalSlotsContext.flatMap((slot) => slot.evidence_stream_entry_ids),
  );
  const sourceTrustFactIds = uniqueStreamEntryIds([
    ...currentUserStreamEntryIds,
    ...postResponseStreamEntryIds,
    ...ledgerPromptContext.visibleStreamEntryIds,
    ...ledgerPromptContext.offLimitsSourceStreamEntryIds,
    ...relationalSlotEvidenceStreamEntryIds,
  ]);
  const sourceTrustFacts =
    indexedSourceTrustLookup === null ? null : indexedSourceTrustLookup.lookup(sourceTrustFactIds);
  const lastUpdatedSourceTrustEntries =
    sourceTrustFacts === null
      ? currentSessionTrustEntries
      : [
          ...currentUserEntries.map((entry) => ({
            id: entry.id,
            turn_id: entry.turn_id,
          })),
          ...postResponseStreamEntryIds.map((streamEntryId) => ({
            id: streamEntryId,
            turn_id: input.input.turnId,
          })),
        ];
  const lastUpdatedTurnByStreamEntryId = sharedStateLastUpdatedTurnByStreamEntryId({
    entries: lastUpdatedSourceTrustEntries,
    currentUserEntry: compileAnchorEntryForTurnAge,
    currentUserEntries,
    currentTurnId: input.input.turnId,
    currentTurnCounter: turnCounter,
  });
  const imageLastUpdatedTurnByStreamEntryId = imageDerivedLastUpdatedTurns({
    retrievedEvidence: input.input.retrievedEvidence,
    attachmentRepository: input.options.attachmentRepository,
  });
  const recentRetrievalStreamEntryIds = retrievedStreamEntryIds(input.input);
  const recentlyRetrievedEntryIds = recentlyRetrievedSharedStateEntryIds({
    artifact: previousArtifact,
    retrievedStreamEntryIds: recentRetrievalStreamEntryIds,
  });
  const renderOptions = {
    ...sharedStateRenderOptions(input.options.config),
    currentUserStreamEntryId: compileAnchorStreamEntryId,
    ledgerStreamEntryIds: ledgerPromptContext.visibleStreamEntryIds,
    recentlyRetrievedEntryIds,
    activeOpenQuestionIds: activeOpenQuestions.map((question) => question.id as OpenQuestionId),
    activeActionIds: (currentAudienceActionCandidatesForCanonicalization.candidates ?? []).map(
      (action) => action.id as ActionId,
    ),
    activeGoalIds: activeGoals.map((goal) => goal.id as GoalId),
    activeCriticalCommitmentIds: activeCommitmentCanonicalizationRecords
      .filter((commitment) => effectiveCommitmentEnforcementClass(commitment) === "critical")
      .map((commitment) => commitment.id as CommitmentId),
    activeOperationalCommitmentIds: activeCommitmentCanonicalizationRecords
      .filter((commitment) => effectiveCommitmentEnforcementClass(commitment) !== "critical")
      .map((commitment) => commitment.id as CommitmentId),
    currentTurnCounter: turnCounter,
    lastUpdatedTurnByStreamEntryId: {
      ...lastUpdatedTurnByStreamEntryId,
      ...imageLastUpdatedTurnByStreamEntryId,
    },
  };
  const currentTurnIsFrameAnomaly =
    input.input.frameAnomaly !== null && input.input.frameAnomaly !== undefined;
  const reconciliationRepositories: SharedStateReconciliationRepositories = {
    goalsRepository: input.options.goalsRepository,
    commitmentRepository: input.options.commitmentRepository,
    actionRepository: input.options.actionRepository,
    openQuestionsRepository: input.options.openQuestionsRepository,
  };
  const unsettledReconciliation =
    sharedStateConfig.compilerPrefilter.enabled === true || currentTurnIsFrameAnomaly
      ? findUnsettledSharedStateReconciliation({
          previousArtifact,
          repositories: reconciliationRepositories,
          nowMs: input.options.clock.now(),
        })
      : null;

  const skip = shouldSkipSharedStateCompile({
    enabled: sharedStateConfig.compilerPrefilter.enabled,
    previousArtifact,
    perceptionMode: input.input.perception.mode,
    frameAnomaly: input.input.frameAnomaly,
    closureLoopAssessment: input.input.closureLoopAssessment,
    unsettledReconciliation: unsettledReconciliation?.summary ?? null,
  });

  if (skip !== null) {
    let skippedArtifact = previousArtifact;
    let advancedAnchor = false;

    if (previousArtifact !== null || compilePass !== "post_response") {
      try {
        const anchorAdvance = advanceSharedStateCompileSkipAnchor({
          repository: input.options.sharedStateRepository,
          audienceEntityId,
          previousArtifact,
          currentUserStreamEntryId: compileAnchorStreamEntryId,
          nowMs: input.options.clock.now(),
        });

        skippedArtifact = anchorAdvance.artifact;
        advancedAnchor = anchorAdvance.advanced;
      } catch {
        skippedArtifact = previousArtifact;
      }
    }

    if (skip.reason === "quarantined_current_turn" && unsettledReconciliation !== null) {
      runSharedStateArtifactRetryOnlyReconciliation({
        unsettledReconciliation,
        repositories: reconciliationRepositories,
        sourceTrustValidator,
        nowMs: input.options.clock.now(),
        tracer: input.options.tracer,
        turnId: input.input.turnId,
        sessionId: input.input.sessionId,
      });
    }

    if (input.options.tracer.enabled && input.input.turnId !== undefined) {
      input.options.tracer.emit("shared_state.compile.skipped", {
        turnId: input.input.turnId,
        session_id: input.input.sessionId,
        reason: skip.reason,
        previous_active_entry_count: skip.previousActiveEntryCount,
        perception_mode: skip.perceptionMode,
        advanced_anchor: advancedAnchor,
        ...(skip.closureShaped === undefined
          ? {}
          : {
              closure_shaped: skip.closureShaped,
              has_state_delta: skip.hasStateDelta ?? null,
            }),
      });
    }

    return {
      artifact: skippedArtifact,
      appliedOperationCount: 0,
      renderOptions,
    };
  }

  if (
    unsettledReconciliation !== null &&
    input.options.tracer.enabled &&
    input.input.turnId !== undefined
  ) {
    input.options.tracer.emit("shared_state.compile.transitioned", {
      turnId: input.input.turnId,
      session_id: input.input.sessionId,
      transition: "unblocked",
      shared_state_compile_transition_reason: "unsettled_reconciliation",
      ...unsettledReconciliation.summary,
    });
  }

  const selfEntityId = input.options.entityRepository.resolve("self", {
    kind: "self",
    provenance: "assistant_seeded",
  });
  const canonicalizationCandidates: SharedStateCanonicalizationCandidates = {
    goals: activeGoals.map((goal) => ({
      id: goal.id,
      text: compactSharedStateArtifactCandidateText(goal.description),
      ...memoryDisclosurePayloadFields(goalMemoryDisclosureLabel(goal)),
    })),
    commitments: activeCommitmentCanonicalizationRecords.map((commitment) => ({
      id: commitment.id,
      text: compactSharedStateArtifactCandidateText(commitment.directive),
      ...memoryDisclosurePayloadFields(commitmentMemoryDisclosureLabel(commitment)),
      kind: commitment.kind,
      type: commitment.type,
      directive_family: commitment.directive_family,
      enforcement_class: effectiveCommitmentEnforcementClass(commitment),
    })),
    actions: currentAudienceActionCandidatesForCanonicalization.candidates ?? [],
    openQuestions: activeOpenQuestions.map((question) => ({
      id: question.id,
      text: compactSharedStateArtifactCandidateText(question.question),
      ...memoryDisclosurePayloadFields(openQuestionMemoryDisclosureLabel(question)),
    })),
  };
  if (input.options.tracer.enabled && input.input.turnId !== undefined) {
    input.options.tracer.emit("shared_state.canonicalization.completed", {
      turnId: input.input.turnId,
      session_id: input.input.sessionId,
      candidate_count_by_scope: currentAudienceActionCandidatesForCanonicalization.countByScope,
      candidate_count_total: (currentAudienceActionCandidatesForCanonicalization.candidates ?? [])
        .length,
    });
  }

  const trustedRelationalSlotEvidenceStreamEntryIds = relationalSlotEvidenceStreamEntryIds.filter(
    (streamEntryId) => sourceTrustValidator(streamEntryId).allowed !== false,
  );
  const trustedPostResponseStreamEntryIds = postResponseStreamEntryIds.filter(
    (streamEntryId) => sourceTrustValidator(streamEntryId).allowed !== false,
  );
  const offLimitsRelationalSlotEvidenceStreamEntryIds = relationalSlotEvidenceStreamEntryIds.filter(
    (streamEntryId) => sourceTrustValidator(streamEntryId).allowed === false,
  );
  const currentUserStreamEntryId = currentUserEntry?.id ?? compileAnchorStreamEntryId;
  const audienceEntity =
    typeof input.options.entityRepository.get === "function"
      ? input.options.entityRepository.get(audienceEntityId)
      : null;
  const sharedStateLlmClient = input.options.llmFactory();
  const semanticBeliefRevision =
    input.options.semanticNodeRepository === undefined ||
    input.options.episodicRepository === undefined
      ? undefined
      : {
          semanticNodeRepository: input.options.semanticNodeRepository,
          episodicRepository: input.options.episodicRepository,
          embeddingClient: input.options.embeddingClient,
          model: input.options.config.anthropic.models.background,
        };

  const compileResult = await compileSharedStateArtifact({
    llmClient: sharedStateLlmClient,
    model: input.options.config.anthropic.models.recallExpansion,
    repository: input.options.sharedStateRepository,
    audienceEntityId,
    currentAudience: {
      entityId: audienceEntityId,
      displayName: audienceEntity?.canonical_name ?? null,
      kind: audienceEntity?.kind ?? null,
    },
    selfEntityId,
    speakerEntityId: singleSenderEntityId(currentUserEntries),
    participants: (input.input.activeParticipants ?? []).map((participant) => ({
      entityId: participant.entityId,
      displayName: participant.displayName,
    })),
    participantRoster: input.input.participantRoster ?? null,
    currentUserMessage: input.input.currentUserMessage,
    currentUserStreamEntryId,
    currentUserTurn:
      currentUserEntry === undefined
        ? null
        : {
            streamEntryId: currentUserEntry.id,
            text: input.input.currentUserMessage,
          },
    currentUserSourceStreamEntryIds: currentUserStreamEntryIds,
    compilePass,
    assistantResponse,
    compileAnchorStreamEntryId,
    ...(compilePass === "post_response" ? { createEmptyArtifactOnNoOp: false } : {}),
    promptVisibleLedger: ledgerPromptContext.promptVisibleLedger,
    previousArtifact,
    relationalSlotsContext,
    // N+1 durability boundary (8599f733): the message being answered is context for this
    // turn, never source material for a durable entry written during it. The compiler runs
    // in retrieval, before deliberation, so without this a fresh artifact citing the current
    // message could be read back as prior shared state inside the same turn. The message
    // stays citable from the next compile on, for as long as the ledger still shows it.
    allowedSourceStreamEntryIds: uniqueStreamEntryIds([
      ...ledgerPromptContext.visibleStreamEntryIds,
      ...trustedPostResponseStreamEntryIds,
      ...trustedRelationalSlotEvidenceStreamEntryIds,
    ]).filter((id) => !currentUserStreamEntryIds.some((sourceId) => sourceId === id)),
    offLimitsSourceStreamEntryIds: uniqueStreamEntryIds([
      ...currentUserStreamEntryIds,
      ...ledgerPromptContext.offLimitsSourceStreamEntryIds,
      ...offLimitsRelationalSlotEvidenceStreamEntryIds,
    ]),
    sourceTrustValidator,
    relationshipEvidenceStreamEntryTrust:
      createLoadedUserStreamEntryRelationshipEvidenceTrustValidator({
        entries:
          sourceTrustFacts === null
            ? currentSessionTrustEntries
            : [...sourceTrustFacts.values()].map(streamEntryFromIndexedFacts),
        isTrusted: (streamEntryId) => sourceTrustValidator(streamEntryId).allowed !== false,
        isActiveAttachmentStreamEntry: (streamEntryId) =>
          input.options.attachmentRepository.isActiveForStreamEntry(streamEntryId),
      }),
    canonicalizationCandidates,
    reconciliation: reconciliationRepositories,
    semanticBeliefRevision,
    clock: input.options.clock,
    tracer: input.options.tracer,
    turnId: input.input.turnId,
    sessionId: input.input.sessionId,
    turnCounter,
    lifecycle: {
      maxActiveEntries: sharedStateConfig.maxActiveEntries,
      maxLiveEntriesPerKey: sharedStateConfig.maxLiveEntriesPerKey,
      recentTurnThreshold: sharedStateConfig.recentTurnThreshold,
      dormantTurnThreshold: sharedStateConfig.dormantTurnThreshold,
      kindSoftCaps: sharedStateConfig.kindSoftCaps,
      newestStateChangeReservedSlots: sharedStateConfig.newestStateChangeReservedSlots,
    },
    renderOptions,
    previousArtifactSummaryOptions: {
      maxEntries: sharedStateConfig.previousArtifactSummary.maxEntries,
      summaryTokenBudget: sharedStateConfig.previousArtifactSummary.summaryTokenBudget,
      maxEntryTextTokens: sharedStateConfig.previousArtifactSummary.maxEntryTextTokens,
    },
    ledgerMode: ledgerPromptContext.ledgerMode,
  });

  return {
    artifact: input.options.sharedStateRepository.get(audienceEntityId),
    appliedOperationCount: compileResult.operations.length,
    renderOptions,
  };
}

function withSharedStateArtifact(
  ledger: EvidenceLedger,
  sharedState: SharedStateArtifact | null,
  renderOptions: ReturnType<typeof sharedStateRenderOptions>,
): EvidenceLedger {
  const ledgerWithSharedState = {
    ...ledger,
    sharedState,
  };

  return {
    ...ledgerWithSharedState,
    estimatedTokens: estimateEvidenceLedgerPromptTokens(ledgerWithSharedState, {
      sharedState: renderOptions,
    }),
  };
}
