import { SuppressionSet } from "../attention/index.js";
import { formatAutonomyTriggerContext } from "../autonomy-trigger.js";
import { ContradictionRoutingCooldown } from "../deliberation/contradiction-routing-cooldown.js";
import { GenerationGate } from "../generation/generation-gate.js";
import { isFrameAnomaly } from "../frame-anomaly/index.js";
import {
  resolveActiveParticipants,
  resolveParticipantProfiles,
  scanRecentParticipantStreamEntries,
} from "../participants.js";
import type { StreamWriter } from "../../stream/index.js";
import { CognitionError } from "../../util/errors.js";
import {
  classifyClosureLoopPhase,
  classifyFrameAnomalyPhase,
} from "./turn-phase/perception-phase.js";
export {
  advanceSharedStateCompileSkipAnchor,
  buildSharedStateLedgerPromptContext,
  shouldSkipSharedStateCompile,
  type SharedStateCompileSkip,
} from "./turn-phase/shared-state-phase.js";
import {
  audienceProfileForParticipants,
  buildFrameAnomalyConversationContext,
} from "./turn-phase/context-build.js";
export {
  buildContradictionRoutingOverride,
  type BuildContradictionRoutingOverrideInput,
} from "./turn-phase/context-build.js";
import { runExtractionPhase } from "./turn-phase/extraction-phase.js";
import { runRetrievalPhase } from "./turn-phase/retrieval-phase.js";
import { runDeliberationPhase } from "./turn-phase/deliberation-phase.js";
import {
  runPostGenerationPhase,
  suppressFromClosureLoopPhase,
  suppressFromGenerationGatePhase,
} from "./turn-phase/post-generation-phase.js";
import { appendHookFailureEvent, catchUpStreamIngestion } from "./turn-phase/utils.js";
import type {
  RunTurnPhasesInput,
  TurnPhaseCoordinatorOptions,
  TurnPhaseResult,
} from "./turn-phase/types.js";
export type {
  RunTurnPhasesInput,
  TurnPhaseCoordinatorOptions,
  TurnPhaseInput,
  TurnPhaseResult,
} from "./turn-phase/types.js";

export class TurnPhaseCoordinator {
  private readonly contradictionRoutingCooldown = new ContradictionRoutingCooldown();

  constructor(private readonly options: TurnPhaseCoordinatorOptions) {}

  async run(input: RunTurnPhasesInput): Promise<TurnPhaseResult> {
    const turnInput = input.input;
    const sessionId = input.sessionId;
    const turnId = input.turnId;
    const streamWriter = input.streamWriter;
    const lifecycleTracker = input.lifecycleTracker;
    const appendHookFailure = (
      targetStreamWriter: StreamWriter,
      hook: string,
      error: unknown,
      details?: Record<string, unknown>,
    ) => appendHookFailureEvent(targetStreamWriter, hook, error, details);
    const isSelfAudience = turnInput.audience === "self";
    const isUserTurn = turnInput.origin !== "autonomous";
    const preflightAudienceEntityId =
      turnInput.audience === undefined || isSelfAudience
        ? null
        : this.options.entityRepository.findByName(turnInput.audience);
    const preflightAudienceEntity =
      preflightAudienceEntityId === null
        ? null
        : this.options.entityRepository.get(preflightAudienceEntityId);

    if (
      preflightAudienceEntity?.kind === "group" &&
      isUserTurn &&
      (turnInput.senderEntityId === null || turnInput.senderEntityId === undefined)
    ) {
      throw new CognitionError("Group-audience user turns require senderEntityId", {
        code: "GROUP_SENDER_REQUIRED",
      });
    }

    await catchUpStreamIngestion({
      coordinator: this.options.streamIngestionCoordinator,
      sessionId,
      streamWriter,
      maxEntries: this.options.config.streamIngestion.preTurnCatchup.maxEntries,
      appendHookFailureEvent: appendHookFailure,
    });
    let workingMemory = this.options.workingMemoryStore.load(sessionId);
    lifecycleTracker.captureInitialWorkingMemory(workingMemory);
    const turnPerception = this.options.perceptionGateway.beginTurn({
      turnId,
      onHookFailure: (hook, error, details) =>
        appendHookFailure(streamWriter, hook, error, details),
    });
    const llmClient = this.options.llmFactory();
    const cognitionInput =
      turnInput.autonomyTrigger === null || turnInput.autonomyTrigger === undefined
        ? turnInput.userMessage
        : formatAutonomyTriggerContext(turnInput.autonomyTrigger);
    const audienceEntityId =
      turnInput.audience === undefined || isSelfAudience
        ? null
        : this.options.entityRepository.resolve(turnInput.audience, {
            provenance: "transport_audience_label",
          });
    const audienceEntity =
      audienceEntityId === null ? null : this.options.entityRepository.get(audienceEntityId);
    let audienceProfile =
      audienceEntityId === null ? null : this.options.socialRepository.getProfile(audienceEntityId);

    // In a group channel the social exchange belongs to the current speaker,
    // not to the abstract channel entity. Updating the group too is deferred.
    const socialInteractionEntityId =
      audienceEntity?.kind === "group" ? (turnInput.senderEntityId ?? null) : audienceEntityId;
    const groupSpeakerEntityId =
      audienceEntity?.kind === "group" ? (turnInput.senderEntityId ?? null) : null;
    const groupSpeakerDisplayName =
      groupSpeakerEntityId === null
        ? null
        : (this.options.entityRepository.get(groupSpeakerEntityId)?.canonical_name ?? null);
    const perceptionResult = await turnPerception.perceive({
      sessionId,
      isSelfAudience,
      origin: turnInput.origin,
      cognitionInput,
      workingMemory,
    });
    const perception = perceptionResult.perception;
    for (const userIdentityName of perception.userIdentityNames ?? []) {
      this.options.entityRepository.resolve(userIdentityName, {
        kind: "person",
        provenance: "user_declared",
      });
    }
    const recencyWindow = perceptionResult.recencyWindow;
    const workingMood = perceptionResult.workingMood;
    workingMemory = perceptionResult.workingMemory;
    const suppressionSet = SuppressionSet.fromEntries(
      workingMemory.suppressed,
      workingMemory.turn_counter,
    );
    const attributionResult = await this.options.attributionLifecycleService.settle({
      isUserTurn,
      audienceEntityId,
      socialEntityId: socialInteractionEntityId,
      perception,
      pendingSocialAttribution: workingMemory.pending_social_attribution,
      pendingTraitAttribution: workingMemory.pending_trait_attribution,
      audienceProfile,
      streamWriter,
      onHookFailure: (hook, error) => appendHookFailure(streamWriter, hook, error),
    });
    const pendingSocialAttribution = attributionResult.pendingSocialAttribution;
    const pendingTraitAttribution = attributionResult.pendingTraitAttribution;
    audienceProfile = attributionResult.audienceProfile;

    const openingPersistence = await this.options.turnOpeningPersistence.persist({
      streamWriter,
      turnId,
      userMessage: turnInput.userMessage,
      persistUserMessage: isUserTurn,
      audience: turnInput.audience,
      senderEntityId: turnInput.senderEntityId,
      workingMemory,
      pendingSocialAttribution,
      pendingTraitAttribution,
      suppressionSet,
      perception,
      now: () => this.options.clock.now(),
    });
    const persistedUserEntry = openingPersistence.persistedUserEntry;
    const persistedUserEntryId = persistedUserEntry?.id;
    const persistedPerceptionEntry = openingPersistence.persistedPerceptionEntry;
    workingMemory = openingPersistence.workingMemory;
    const activeParticipantLimit = this.options.config.generation.activeParticipantLimit;
    const participantScan =
      audienceEntity?.kind === "group"
        ? scanRecentParticipantStreamEntries(
            this.options.createStreamReader(sessionId),
            activeParticipantLimit,
          )
        : null;

    if (
      participantScan !== null &&
      participantScan.capReached !== null &&
      participantScan.foundUniqueParticipants < activeParticipantLimit &&
      this.options.tracer.enabled
    ) {
      this.options.tracer.emit("participant_scan.skipped", {
        turnId,
        reason: "cap_reached",
        cap: participantScan.capReached,
        scanned_entries: participantScan.scannedEntries,
        scanned_bytes: participantScan.scannedBytes,
        found_unique_participants: participantScan.foundUniqueParticipants,
        requested_limit: activeParticipantLimit,
      });
    }

    const activeParticipants = resolveActiveParticipants({
      audienceEntityId,
      senderEntityId: turnInput.senderEntityId ?? null,
      streamEntries: participantScan?.entries ?? [],
      entityRepository: this.options.entityRepository,
      limit: activeParticipantLimit,
    });
    const participantProfiles = resolveParticipantProfiles(
      activeParticipants,
      this.options.socialRepository,
    );
    if (activeParticipants.length > 0) {
      audienceProfile = audienceProfileForParticipants(participantProfiles, audienceEntityId);
    }

    const frameAnomalyConversationContext = buildFrameAnomalyConversationContext({
      audienceEntityId,
      audienceEntity,
      currentUserEntry: persistedUserEntry,
      activeParticipants,
      participantStreamEntries: participantScan?.entries ?? [],
      entityRepository: this.options.entityRepository,
    });
    const frameAnomalyClassification = await classifyFrameAnomalyPhase({
      options: this.options,
      appendHookFailureEvent: appendHookFailure,
      llmClient,
      turnId,
      isUserTurn,
      userMessage: turnInput.userMessage,
      recentHistory: recencyWindow.messages,
      conversationContext: frameAnomalyConversationContext,
      persistedUserEntryId,
      streamWriter,
    });
    const currentTurnFrameAnomaly = isFrameAnomaly(frameAnomalyClassification)
      ? frameAnomalyClassification
      : null;

    const extraction = await runExtractionPhase({
      options: this.options,
      appendHookFailureEvent: appendHookFailure,
      llmClient,
      turnId,
      sessionId,
      turnInput,
      isUserTurn,
      cognitionInput,
      perception,
      workingMemory,
      recentHistory: recencyWindow.messages,
      audienceEntityId,
      groupSpeakerEntityId,
      groupSpeakerDisplayName,
      persistedUserEntryId,
      frameAnomalyClassification,
      streamWriter,
      trackAppliedSlotNegation: (slot) => lifecycleTracker.trackAppliedSlotNegation(slot),
    });
    const correctiveCommitment = extraction.correctiveCommitment;
    const correctiveCommitmentSupersession = extraction.correctiveCommitmentSupersession;
    workingMemory = extraction.workingMemory;
    lifecycleTracker.trackCreatedActionIds(extraction.createdActionIds);
    lifecycleTracker.trackCreatedGoalIds(extraction.persistedPromotions.goalIds);
    lifecycleTracker.trackCreatedExecutiveStepIds(extraction.persistedPromotions.executiveStepIds);

    const closureLoopAssessment = await classifyClosureLoopPhase({
      options: this.options,
      appendHookFailureEvent: appendHookFailure,
      llmClient,
      turnId,
      isUserTurn,
      userMessage: turnInput.userMessage,
      recentHistory: recencyWindow.messages,
      persistedUserEntryId,
      workingMemory,
      streamWriter,
    });

    if (closureLoopAssessment?.currentUserSubstantive === true) {
      workingMemory = this.options.discourseStateService.clearClosureLoop({
        workingMemory,
        reason: closureLoopAssessment.reason,
        turnId,
      });
    } else if (
      closureLoopAssessment?.currentUserClosureShaped === true &&
      workingMemory.discourse_state?.closure_loop?.status === "named"
    ) {
      return suppressFromClosureLoopPhase({
        turnId,
        turnInput,
        streamWriter,
        appendHookFailureEvent: appendHookFailure,
        options: this.options,
        workingMemory,
        persistedUserEntryId,
        correctiveCommitment,
        correctiveCommitmentSupersession,
        perceptionMode: perception.mode,
        reason: closureLoopAssessment.reason,
      });
    } else if (closureLoopAssessment?.closureLoopDetected === true) {
      workingMemory = this.options.discourseStateService.setClosureLoopDetected({
        workingMemory,
        sourceStreamEntryIds: closureLoopAssessment.sourceStreamEntryIds,
        reason: closureLoopAssessment.reason,
        turnId,
      });
    }

    const generationGate = new GenerationGate({
      llmClient,
      embeddingClient: this.options.embeddingClient,
      model: this.options.config.anthropic.models.background,
      hardCapTurns: this.options.config.generation.discourseStateHardCapTurns,
      onDegraded: (reason, error) =>
        appendHookFailure(streamWriter, "generation_gate", error ?? reason, {
          reason,
        }),
    });
    const gateResult = await generationGate.evaluate({
      userMessage: turnInput.userMessage,
      workingMemory,
      recencyMessages: recencyWindow.messages,
    });

    if (gateResult.signals.hardCapDue) {
      await this.options.discourseStateService.appendHardCapEvent({
        streamWriter,
        turnId,
        activeTurns: gateResult.signals.hardCapActiveTurns,
        hardCapTurns: this.options.config.generation.discourseStateHardCapTurns,
        stateReason:
          workingMemory.discourse_state?.stop_until_substantive_content?.reason ?? "unknown",
      });
    }

    if (gateResult.clearDiscourseStop) {
      workingMemory = this.options.discourseStateService.clearStopState({
        workingMemory,
        reason: gateResult.explanation,
        turnId,
      });
    }

    if (gateResult.action === "suppress") {
      return suppressFromGenerationGatePhase({
        turnId,
        turnInput,
        streamWriter,
        appendHookFailureEvent: appendHookFailure,
        options: this.options,
        workingMemory,
        persistedUserEntryId,
        gateResult,
        correctiveCommitment,
        correctiveCommitmentSupersession,
        perceptionMode: perception.mode,
      });
    }

    const retrievalPhase = await runRetrievalPhase({
      options: this.options,
      sessionId,
      turnId,
      turnInput,
      isSelfAudience,
      isUserTurn,
      cognitionInput,
      llmClient,
      recencyMessages: recencyWindow.messages,
      audienceEntityId,
      audienceEntity,
      audienceProfile,
      perception,
      workingMemory,
      suppressionSet,
      actionLinkSelfContext: extraction.actionLinkSelfContext,
      persistedPromotions: extraction.persistedPromotions,
      correctiveCommitment,
      activeParticipants,
      participantProfiles,
      persistedUserEntry: persistedUserEntry ?? undefined,
      currentTurnFrameAnomaly,
      closureLoopAssessment,
    });
    const deliberationPhase = await runDeliberationPhase({
      options: this.options,
      llmClient,
      sessionId,
      turnId,
      turnInput,
      streamWriter,
      audienceEntityId,
      persistedUserEntryId,
      perception,
      workingMemory,
      activeParticipants,
      participantProfiles,
      audienceProfile,
      recencyMessages: recencyWindow.messages,
      currentTurnFrameAnomaly,
      retrievalPhase,
      contradictionRoutingCooldown: this.contradictionRoutingCooldown,
    });
    const deliberation = deliberationPhase.deliberation;
    workingMemory = deliberationPhase.workingMemory;

    return runPostGenerationPhase({
      options: this.options,
      appendHookFailureEvent: appendHookFailure,
      llmClient,
      sessionId,
      turnId,
      turnInput,
      streamWriter,
      lifecycleTracker,
      cognitionInput,
      perception,
      workingMemory,
      workingMood,
      persistedUserEntry: persistedUserEntry ?? undefined,
      persistedPerceptionEntry,
      persistedUserEntryId,
      correctiveCommitment,
      correctiveCommitmentSupersession,
      deliberation,
      origin: turnInput.origin,
      autonomyTrigger: turnInput.autonomyTrigger,
      retrievalPhase,
      closureLoopCurrentUserAct: closureLoopAssessment?.currentUserAct ?? null,
      audienceEntityId,
      audienceIsGroup: audienceEntity?.kind === "group",
      senderEntityId: turnInput.senderEntityId ?? null,
      socialInteractionEntityId,
      pendingSocialAttribution,
      suppressionSet,
      isUserTurn,
      frameAnomalyClassification,
    });
  }
}
