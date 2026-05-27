import {
  appendCommitmentIfMissing,
  type CorrectivePreferenceTurnService,
} from "../../commitments/corrective-preference-service.js";
import type {
  CreatorDirectiveBriefing,
  DeliberationRoutingOverride,
} from "../../deliberation/types.js";
import {
  EvidenceLedgerBuilder,
  compactEvidenceLedger,
  estimateEvidenceLedgerPromptTokens,
  renderEvidenceLedger,
  summarizeEvidenceLedgerTrace,
  summarizeSharedStateArtifactRender,
  type EvidenceLedger,
  type EvidenceLedgerBuildInput,
} from "../../evidence-ledger/index.js";
import {
  isFrameAnomaly,
  type ActualFrameAnomalyClassification,
} from "../../frame-anomaly/index.js";
import type { ActiveParticipant, ParticipantProfileContext } from "../../participants.js";
import type { ParticipantRoster } from "../../perception/index.js";
import type { RecencyMessage } from "../../recency/index.js";
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
import { toTraceJsonValue } from "../../tracing/tracer.js";
import type { PerceptionResult } from "../../types.js";
import type { LLMClient } from "../../../llm/index.js";
import {
  effectiveCommitmentEnforcementClass,
  type EntityRepository,
  type CommitmentRecord,
} from "../../../memory/commitments/index.js";
import type {
  CreatorDirective,
  CreatorDirectiveApplicable,
} from "../../../memory/creator-directives/index.js";
import type { SharedStateArtifact } from "../../../memory/decision-artifacts/index.js";
import { createLoadedUserStreamEntryRelationshipEvidenceTrustValidator } from "../../../memory/source-trust.js";
import type { IndexedEntryFacts, StreamEntry } from "../../../stream/index.js";
import { loadSessionStreamEntries } from "../../../stream/index.js";
import type {
  ActionId,
  CommitmentId,
  EntityId,
  GoalId,
  OpenQuestionId,
  SessionId,
  SharedStateEntryId,
  StreamEntryId,
} from "../../../util/ids.js";
import { dedupePreservingOrder } from "../../../util/collections.js";
import type { WorkingMemory } from "../../../memory/working/index.js";
import type { SessionAudienceRole } from "../../../sessions/index.js";
import type { ClosureLoopAssessment } from "../../generation/closure-loop.js";
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
  selectSharedStateArtifactActionCandidates,
  shouldSkipSharedStateCompile,
} from "./shared-state-phase.js";
import { runSharedStateArtifactRetryOnlyReconciliation } from "./reconciliation-phase.js";
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

export type EvidenceLedgerFinalizerBuildInput = EvidenceLedgerBuildInput & {
  globalTurnCounter?: number;
  isUserTurn: boolean;
  perception: PerceptionResult;
  closureLoopAssessment: ClosureLoopAssessment | null;
  participantRoster?: ParticipantRoster | null;
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
  pendingCorrections: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["pendingCorrections"];
  affectiveTrajectory: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["affectiveTrajectory"];
  retrieval: Awaited<
    ReturnType<TurnPhaseCoordinatorOptions["turnRetrievalCoordinator"]["coordinate"]>
  >["retrieval"];
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

export function buildCreatorDirectiveBriefing(input: {
  applicable: readonly CreatorDirectiveApplicable[];
  entityRepository: Pick<EntityRepository, "get">;
}): CreatorDirectiveBriefing | null {
  const contentDirectives = input.applicable
    .filter((item) => item.render_mode === "content" && item.directive.canonical_fact !== null)
    .map((item) => ({
      renderMode: "content" as const,
      kind: item.directive.kind,
      subjectKind: item.directive.subject_kind,
      subjectLabel: subjectLabelForCreatorDirective(item.directive, input.entityRepository),
      canonicalFact: item.directive.canonical_fact!,
      mentionPolicy: item.directive.disclosure_policy.mention_policy,
      priority: item.directive.priority,
      createdAt: item.directive.created_at,
    }))
    .sort((left, right) => right.priority - left.priority || left.createdAt - right.createdAt);
  const boundaryDirectives = input.applicable
    .filter(
      (item) =>
        item.render_mode === "boundary" &&
        item.directive.disclosure_policy.boundary_prompt !== null,
    )
    .map((item) => ({
      renderMode: "boundary" as const,
      boundaryPrompt: item.directive.disclosure_policy.boundary_prompt!,
      topicTags: item.directive.disclosure_policy.topic_tags,
      priority: item.directive.priority,
      createdAt: item.directive.created_at,
    }))
    .sort((left, right) => right.priority - left.priority || left.createdAt - right.createdAt);
  const directives = [...contentDirectives, ...boundaryDirectives];

  return directives.length === 0 ? null : { directives };
}

function currentTurnEligibleCreatorDirectives(input: {
  applicable: readonly CreatorDirectiveApplicable[];
  currentUserEntryId?: StreamEntryId;
}): CreatorDirectiveApplicable[] {
  const currentUserEntryId = input.currentUserEntryId;

  if (currentUserEntryId === undefined) {
    return [...input.applicable];
  }

  return input.applicable.filter(
    (item) => !item.directive.authorization_stream_entry_ids.includes(currentUserEntryId),
  );
}

export function buildCreatorDirectiveBriefingForTurn(input: {
  applicable: readonly CreatorDirectiveApplicable[];
  currentUserEntryId?: StreamEntryId;
  entityRepository: Pick<EntityRepository, "get">;
}): CreatorDirectiveBriefing | null {
  return buildCreatorDirectiveBriefing({
    applicable: currentTurnEligibleCreatorDirectives({
      applicable: input.applicable,
      currentUserEntryId: input.currentUserEntryId,
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
  currentTurnFrameAnomaly: ActualFrameAnomalyClassification | null;
  closureLoopAssessment: ClosureLoopAssessment | null;
}): Promise<TurnRetrievalPhaseResult> {
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
    sessionId: input.sessionId,
    turnId: input.turnId,
    userMessage: input.turnInput.userMessage,
    recentMessages: input.recencyMessages.map((message) => ({
      role: message.role,
      content: message.content,
    })),
    cognitionInput: input.cognitionInput,
    inputAudience: input.turnInput.audience,
    isSelfAudience: input.isSelfAudience,
    audienceEntityId: input.audienceEntityId,
    audienceEntity: input.audienceEntity,
    audienceProfile: input.audienceProfile,
    perception: input.perception,
    workingMemory: input.workingMemory,
    selfSnapshot,
    executiveFocus: executiveFocusWithStep,
    activeValues: activeScoringValues,
    scoringFeatures: retrievalScoringFeatures,
    suppressionSet: input.suppressionSet,
    findEntityByName: (name) => input.options.entityRepository.findByName(name),
    llmClient: input.llmClient,
    proceduralContextModel: input.options.config.anthropic.models.background,
  });
  const applicableCommitments = appendCommitmentIfMissing(
    retrievalContext.applicableCommitments,
    input.correctiveCommitment,
  );
  const pendingCorrections = retrievalContext.pendingCorrections;
  const affectiveTrajectory = retrievalContext.affectiveTrajectory;
  const retrieval = retrievalContext.retrieval;
  const retrievedEpisodes = retrievalContext.retrievedEpisodes;
  const retrievedSemantic = retrievalContext.retrievedSemantic;
  const proceduralContext = retrievalContext.proceduralContext;
  const selectedSkill = retrievalContext.selectedSkill;
  const relationalSlots = listConstrainedRelationalSlotsForParticipants(
    input.options.relationalSlotRepository,
    input.activeParticipants,
  );
  const creatorDirectiveApplicable =
    input.options.creatorDirectiveRepository === undefined
      ? []
      : input.options.creatorDirectiveRepository.listApplicable({
          currentAudienceEntityId: input.audienceEntityId,
          participantEntityIds: participantEntityIds({
            audienceEntityId: input.audienceEntityId,
            audienceEntityKind: input.audienceEntity?.kind ?? null,
            activeParticipants: input.activeParticipants,
          }),
          topicTags: input.perception.entities,
          sessionRole: input.sessionAudienceRole ?? "participant",
        });
  const creatorDirectiveBriefing = buildCreatorDirectiveBriefingForTurn({
    applicable: creatorDirectiveApplicable,
    currentUserEntryId: input.persistedUserEntry?.id,
    entityRepository: input.options.entityRepository,
  });
  const evidenceLedgerContext = await buildEvidenceLedgerFinalizerContext({
    options: input.options,
    input: {
      sessionId: input.sessionId,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      currentUserMessage: input.turnInput.userMessage,
      currentUserEntry: input.persistedUserEntry,
      globalTurnCounter: input.turnInput.globalTurnCounter,
      workingMemory: input.workingMemory,
      applicableCommitments,
      retrievedEvidence: retrieval.evidence,
      retrievedEpisodes,
      retrievedSemantic,
      openQuestions: retrieval.open_questions,
      pendingCorrections,
      frameAnomaly: input.currentTurnFrameAnomaly,
      activeParticipants: input.activeParticipants,
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
    pendingCorrections,
    affectiveTrajectory,
    retrieval,
    retrievedEpisodes,
    retrievedSemantic,
    proceduralContext,
    selectedSkill,
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
  });
  const sessionReentryContinuity = buildSessionReentryContinuityPrompt({
    isUserTurn: input.input.isUserTurn,
    priorUserTurnCount,
    audienceEntityId: input.input.audienceEntityId,
    artifact: previousSharedState,
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
  const ledgerWithoutSharedState = compacted.ledger;
  const renderedWithoutSharedState = renderEvidenceLedger(ledgerWithoutSharedState);
  const sharedStateResult = await compileSharedStateArtifactForEvidenceLedgerResult({
    options: input.options,
    input: input.input,
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
    evidenceLedgerCompactionChanged(compacted.traceSummary)
  ) {
    input.options.tracer.emit("evidence_ledger.compaction.completed", {
      turnId: input.input.turnId,
      session_id: input.input.sessionId,
      pre_dedupe_tokens: compacted.traceSummary.preDedupeTokens,
      post_dedupe_tokens: compacted.traceSummary.postDedupeTokens,
      pre_cap_tokens: compacted.traceSummary.preCapTokens,
      post_section_cap_tokens: compacted.traceSummary.postSectionCapTokens,
      post_cap_tokens: compacted.traceSummary.postCapTokens,
      deduped_entry_count: compacted.traceSummary.dedupedEntryCount,
      omitted_entry_counts: toTraceJsonValue(compacted.traceSummary.omittedEntryCountsBySection),
      dropped_sections: compacted.traceSummary.droppedSections,
      target_tokens: compacted.traceSummary.targetTokens,
      hard_cap_tokens: compacted.traceSummary.hardCapTokens,
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
      decision_artifact_entry_count: sharedStateSummary.renderedEntryCount,
      decision_artifact_rendered_token_estimate: sharedStateSummary.estimatedTokens,
      decision_artifact_rendered_by_kind: toTraceJsonValue(sharedStateSummary.renderedByKind),
      decision_artifact_newest_entries_reserved: sharedStateSummary.newestReservedEntryCount,
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

async function countPriorUserTurnsForSession(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  currentUserEntryId?: StreamEntryId;
}): Promise<number> {
  if (input.options.entryIndex !== undefined) {
    return input.options.entryIndex.countSessionEntriesByKind({
      sessionId: input.sessionId,
      kind: "user_msg",
      excludeEntryId: input.currentUserEntryId,
    });
  }

  const entries = await loadSessionStreamEntries(input.options.createStreamReader(input.sessionId));

  return entries.filter(
    (entry) => entry.kind === "user_msg" && entry.id !== input.currentUserEntryId,
  ).length;
}

function sharedStateLastUpdatedTurnByStreamEntryId(input: {
  entries: readonly Pick<StreamEntry, "id" | "turn_id">[];
  currentUserEntry: StreamEntry;
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
  currentUserEntry: StreamEntry;
}): {
  lookup: (streamEntryIds: readonly StreamEntryId[]) => Map<StreamEntryId, IndexedEntryFacts>;
  entriesForKnownFacts: () => Pick<StreamEntry, "id" | "kind" | "turn_id">[];
} {
  const factsById = new Map<StreamEntryId, IndexedEntryFacts>();

  const rememberCurrentUserEntry = (): void => {
    if (!factsById.has(input.currentUserEntry.id)) {
      factsById.set(input.currentUserEntry.id, {
        entry_id: input.currentUserEntry.id,
        session_id: input.currentUserEntry.session_id,
        timestamp: input.currentUserEntry.timestamp,
        kind: input.currentUserEntry.kind,
        turn_id: input.currentUserEntry.turn_id ?? null,
        turn_status: input.currentUserEntry.turn_status ?? "active",
        active: input.currentUserEntry.turn_status !== "aborted",
      });
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
}): Promise<SharedStateArtifact | null> {
  return (await compileSharedStateArtifactForEvidenceLedgerResult(input)).artifact;
}

export async function compileSharedStateArtifactForEvidenceLedgerResult(input: {
  options: TurnPhaseCoordinatorOptions;
  input: EvidenceLedgerFinalizerBuildInput;
  previousArtifact?: SharedStateArtifact | null;
  ledger: EvidenceLedger;
  promptVisibleLedger: string;
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
}): Promise<SharedStateArtifactForEvidenceLedgerResult> {
  const audienceEntityId = input.input.audienceEntityId;

  if (audienceEntityId === null) {
    return { artifact: null, appliedOperationCount: 0 };
  }

  const previousArtifact =
    input.previousArtifact ?? input.options.sharedStateRepository.get(audienceEntityId);

  if (!input.input.isUserTurn || input.input.currentUserEntry === undefined) {
    return { artifact: previousArtifact, appliedOperationCount: 0 };
  }

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
          currentUserEntry: input.input.currentUserEntry,
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
          currentSessionEntries: currentSessionTrustEntries.some(
            (entry) => entry.id === input.input.currentUserEntry?.id,
          )
            ? (currentSessionTrustEntries as StreamEntry[])
            : [...(currentSessionTrustEntries as StreamEntry[]), input.input.currentUserEntry],
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
  const actionCanonicalizationCandidates = selectSharedStateArtifactActionCandidates({
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
    input.input.currentUserEntry.id,
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
          {
            id: input.input.currentUserEntry.id,
            turn_id: input.input.currentUserEntry.turn_id,
          },
        ];
  const lastUpdatedTurnByStreamEntryId = sharedStateLastUpdatedTurnByStreamEntryId({
    entries: lastUpdatedSourceTrustEntries,
    currentUserEntry: input.input.currentUserEntry,
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
    currentUserStreamEntryId: input.input.currentUserEntry.id,
    ledgerStreamEntryIds: ledgerPromptContext.visibleStreamEntryIds,
    recentlyRetrievedEntryIds,
    activeOpenQuestionIds: activeOpenQuestions.map((question) => question.id as OpenQuestionId),
    activeActionIds: (actionCanonicalizationCandidates.candidates ?? []).map(
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
  const currentTurnIsFrameAnomaly = isFrameAnomaly(input.input.frameAnomaly);
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

    try {
      const anchorAdvance = advanceSharedStateCompileSkipAnchor({
        repository: input.options.sharedStateRepository,
        audienceEntityId,
        previousArtifact,
        currentUserStreamEntryId: input.input.currentUserEntry.id,
        nowMs: input.options.clock.now(),
      });

      skippedArtifact = anchorAdvance.artifact;
      advancedAnchor = anchorAdvance.advanced;
    } catch {
      skippedArtifact = previousArtifact;
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
    })),
    commitments: activeCommitmentCanonicalizationRecords.map((commitment) => ({
      id: commitment.id,
      text: compactSharedStateArtifactCandidateText(commitment.directive),
      kind: commitment.kind,
      type: commitment.type,
      directive_family: commitment.directive_family,
      enforcement_class: effectiveCommitmentEnforcementClass(commitment),
    })),
    actions: actionCanonicalizationCandidates.candidates ?? [],
    openQuestions: activeOpenQuestions.map((question) => ({
      id: question.id,
      text: compactSharedStateArtifactCandidateText(question.question),
    })),
  };
  if (input.options.tracer.enabled && input.input.turnId !== undefined) {
    input.options.tracer.emit("shared_state.canonicalization.completed", {
      turnId: input.input.turnId,
      session_id: input.input.sessionId,
      candidate_count_by_scope: actionCanonicalizationCandidates.countByScope,
      candidate_count_total: (actionCanonicalizationCandidates.candidates ?? []).length,
    });
  }

  const trustedRelationalSlotEvidenceStreamEntryIds = relationalSlotEvidenceStreamEntryIds.filter(
    (streamEntryId) => sourceTrustValidator(streamEntryId).allowed !== false,
  );
  const offLimitsRelationalSlotEvidenceStreamEntryIds = relationalSlotEvidenceStreamEntryIds.filter(
    (streamEntryId) => sourceTrustValidator(streamEntryId).allowed === false,
  );
  const currentUserStreamEntryId = input.input.currentUserEntry.id;
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
    selfEntityId,
    speakerEntityId: input.input.currentUserEntry.sender_entity_id,
    participants: (input.input.activeParticipants ?? []).map((participant) => ({
      entityId: participant.entityId,
      displayName: participant.displayName,
    })),
    participantRoster: input.input.participantRoster ?? null,
    currentUserMessage: input.input.currentUserMessage,
    currentUserStreamEntryId,
    promptVisibleLedger: ledgerPromptContext.promptVisibleLedger,
    previousArtifact,
    relationalSlotsContext,
    allowedSourceStreamEntryIds: uniqueStreamEntryIds([
      ...ledgerPromptContext.visibleStreamEntryIds,
      ...trustedRelationalSlotEvidenceStreamEntryIds,
    ]).filter((id) => id !== currentUserStreamEntryId),
    offLimitsSourceStreamEntryIds: uniqueStreamEntryIds([
      currentUserStreamEntryId,
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
