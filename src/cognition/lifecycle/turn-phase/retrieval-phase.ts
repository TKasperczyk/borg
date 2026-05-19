import {
  appendCommitmentIfMissing,
  type CorrectivePreferenceTurnService,
} from "../../commitments/corrective-preference-service.js";
import type { DeliberationRoutingOverride } from "../../deliberation/types.js";
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
import type { RecencyMessage } from "../../recency/index.js";
import {
  compileSharedStateArtifact,
  findUnsettledSharedStateReconciliation,
  type SharedStateCanonicalizationCandidates,
  type SharedStateReconciliationRepositories,
} from "../../shared-state/index.js";
import { toTraceJsonValue } from "../../tracing/tracer.js";
import type { PerceptionResult } from "../../types.js";
import type { LLMClient } from "../../../llm/index.js";
import type { CommitmentRecord } from "../../../memory/commitments/index.js";
import type { SharedStateArtifact } from "../../../memory/decision-artifacts/index.js";
import type { StreamEntry } from "../../../stream/index.js";
import { loadSessionStreamEntries } from "../../../stream/index.js";
import type { EntityId, SessionId } from "../../../util/ids.js";
import type { WorkingMemory } from "../../../memory/working/index.js";
import type { ClosureLoopAssessment } from "../../generation/closure-loop.js";
import type { TurnPhaseCoordinatorOptions, TurnPhaseInput } from "./types.js";
import {
  buildContradictionRoutingOverride,
  listConstrainedRelationalSlotsForParticipants,
} from "./context-build.js";
import { evidenceLedgerCompactionChanged } from "./trace-metrics.js";
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
};

export type EvidenceLedgerFinalizerBuildInput = EvidenceLedgerBuildInput & {
  isUserTurn: boolean;
  perception: PerceptionResult;
  closureLoopAssessment: ClosureLoopAssessment | null;
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
  evidenceLedgerContext: EvidenceLedgerFinalizerContext;
  routingOverride: DeliberationRoutingOverride | null;
};

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
  const evidenceLedgerContext = await buildEvidenceLedgerFinalizerContext({
    options: input.options,
    input: {
      sessionId: input.sessionId,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      currentUserMessage: input.turnInput.userMessage,
      currentUserEntry: input.persistedUserEntry,
      workingMemory: input.workingMemory,
      applicableCommitments,
      retrievedEvidence: retrieval.evidence,
      retrievedEpisodes,
      retrievedSemantic,
      openQuestions: retrieval.open_questions,
      pendingCorrections,
      frameAnomaly: input.currentTurnFrameAnomaly,
      activeParticipants: input.activeParticipants,
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
    evidenceLedgerContext,
    routingOverride,
  };
}

async function buildEvidenceLedgerFinalizerContext(input: {
  options: TurnPhaseCoordinatorOptions;
  input: EvidenceLedgerFinalizerBuildInput;
}): Promise<EvidenceLedgerFinalizerContext> {
  const config = input.options.config.generation.evidenceLedger;

  if (!config.enabled) {
    return {
      ledger: null,
      promptSection: null,
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
  const sharedState = await compileSharedStateArtifactForEvidenceLedger({
    options: input.options,
    input: input.input,
    ledger: ledgerWithoutSharedState,
    promptVisibleLedger: renderedWithoutSharedState ?? "",
  });
  const renderOptions = sharedStateRenderOptions(input.options.config);
  const ledger = withSharedStateArtifact(ledgerWithoutSharedState, sharedState, renderOptions);
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
    input.options.tracer.emit("evidence_ledger_compacted", {
      turnId: input.input.turnId,
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
    input.options.tracer.emit("evidence_ledger_built", {
      turnId: input.input.turnId,
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
    });
  }

  return {
    ledger,
    promptSection: rendered,
  };
}

export async function compileSharedStateArtifactForEvidenceLedger(input: {
  options: TurnPhaseCoordinatorOptions;
  input: EvidenceLedgerFinalizerBuildInput;
  ledger: EvidenceLedger;
  promptVisibleLedger: string;
}): Promise<SharedStateArtifact | null> {
  const audienceEntityId = input.input.audienceEntityId;

  if (audienceEntityId === null) {
    return null;
  }

  const previousArtifact = input.options.sharedStateRepository.get(audienceEntityId);

  if (!input.input.isUserTurn || input.input.currentUserEntry === undefined) {
    return previousArtifact;
  }

  const sourceTrustEntries =
    typeof input.options.createStreamReader === "function"
      ? await loadSessionStreamEntries(input.options.createStreamReader(input.input.sessionId))
      : [];
  const currentSessionTrustEntries = sourceTrustEntries.some(
    (entry) => entry.id === input.input.currentUserEntry?.id,
  )
    ? sourceTrustEntries
    : [...sourceTrustEntries, input.input.currentUserEntry];
  const quarantinedStreamEntryIds =
    await collectCrossSessionQuarantinedSharedStateArtifactStreamEntryIds(
      input.options.config.dataDir,
    );
  const sourceTrustValidator = buildSharedStateSourceTrustValidator({
    currentSessionEntries: currentSessionTrustEntries,
    quarantinedStreamEntryIds,
  });
  const sharedStateConfig = input.options.config.generation.evidenceLedger.decisionArtifact;
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
      });
    }

    if (input.options.tracer.enabled && input.input.turnId !== undefined) {
      input.options.tracer.emit("decision_artifact_compile_skipped", {
        turnId: input.input.turnId,
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

    return skippedArtifact;
  }

  if (
    unsettledReconciliation !== null &&
    input.options.tracer.enabled &&
    input.input.turnId !== undefined
  ) {
    input.options.tracer.emit("decision_artifact_compile_unblocked", {
      turnId: input.input.turnId,
      decision_artifact_compile_unblocked_reason: "unsettled_reconciliation",
      ...unsettledReconciliation.summary,
    });
  }

  const ledgerPromptContext = buildSharedStateLedgerPromptContext({
    ledger: input.ledger,
    previousArtifact,
    fullPromptVisibleLedger: input.promptVisibleLedger,
    enabled: sharedStateConfig.ledgerDelta.enabled,
    minTailPerSection: sharedStateConfig.ledgerDelta.minTailPerSection,
    sourceTrustValidator,
  });
  const selfEntityId = input.options.entityRepository.resolve("self", {
    kind: "self",
    provenance: "assistant_seeded",
  });
  const actionCanonicalizationCandidates = selectSharedStateArtifactActionCandidates({
    actionRepository: input.options.actionRepository,
    audienceEntityId,
    activeParticipants: input.input.activeParticipants,
  });
  const canonicalizationCandidates: SharedStateCanonicalizationCandidates = {
    goals: input.options.goalsRepository
      .list({
        status: "active",
        visibleToAudienceEntityId: audienceEntityId,
      })
      .map((goal) => ({
        id: goal.id,
        text: compactSharedStateArtifactCandidateText(goal.description),
      })),
    commitments: input.options.commitmentRepository
      .list({
        activeOnly: true,
        audience: audienceEntityId,
      })
      .filter(isSharedStateCommitmentCanonicalizationRecord)
      .map((commitment) => ({
        id: commitment.id,
        text: compactSharedStateArtifactCandidateText(commitment.directive),
        type: commitment.type,
        directive_family: commitment.directive_family,
      })),
    actions: actionCanonicalizationCandidates.candidates ?? [],
    openQuestions: input.options.openQuestionsRepository
      .list({
        status: "open",
        visibleToAudienceEntityId: audienceEntityId,
        limit: 80,
      })
      .map((question) => ({
        id: question.id,
        text: compactSharedStateArtifactCandidateText(question.question),
      })),
  };

  if (input.options.tracer.enabled && input.input.turnId !== undefined) {
    input.options.tracer.emit("decision_artifact_canonicalization_candidates", {
      turnId: input.input.turnId,
      candidate_count_by_scope: actionCanonicalizationCandidates.countByScope,
      candidate_count_total: (actionCanonicalizationCandidates.candidates ?? []).length,
    });
  }

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

  await compileSharedStateArtifact({
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
    currentUserMessage: input.input.currentUserMessage,
    currentUserStreamEntryId: input.input.currentUserEntry.id,
    promptVisibleLedger: ledgerPromptContext.promptVisibleLedger,
    previousArtifact,
    allowedSourceStreamEntryIds: ledgerPromptContext.visibleStreamEntryIds,
    offLimitsSourceStreamEntryIds: ledgerPromptContext.offLimitsSourceStreamEntryIds,
    sourceTrustValidator,
    canonicalizationCandidates,
    reconciliation: reconciliationRepositories,
    semanticBeliefRevision,
    clock: input.options.clock,
    tracer: input.options.tracer,
    turnId: input.input.turnId,
    turnCounter: input.input.workingMemory?.turn_counter,
    lifecycle: {
      maxActiveEntries: sharedStateConfig.maxActiveEntries,
      kindSoftCaps: sharedStateConfig.kindSoftCaps,
    },
    renderOptions: sharedStateRenderOptions(input.options.config),
    previousArtifactSummaryOptions: {
      maxEntries: sharedStateConfig.previousArtifactSummary.maxEntries,
      summaryTokenBudget: sharedStateConfig.previousArtifactSummary.summaryTokenBudget,
      maxEntryTextTokens: sharedStateConfig.previousArtifactSummary.maxEntryTextTokens,
    },
    ledgerMode: ledgerPromptContext.ledgerMode,
  });

  return input.options.sharedStateRepository.get(audienceEntityId);
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
