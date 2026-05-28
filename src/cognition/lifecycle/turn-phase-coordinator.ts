import { SuppressionSet } from "../attention/index.js";
import { formatAutonomyTriggerContext } from "../autonomy-trigger.js";
import { ContradictionRoutingCooldown } from "../deliberation/contradiction-routing-cooldown.js";
import { GenerationGate } from "../generation/generation-gate.js";
import { isFrameAnomaly } from "../frame-anomaly/index.js";
import { buildParticipantRosterFromRepositories } from "../perception/index.js";
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
import {
  buildOperatorSessionSnapshot,
  OPERATOR_SESSION_SNAPSHOT_CAP,
} from "./turn-phase/session-snapshot.js";
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
import { traceTurnPhase } from "./turn-phase/phase-trace.js";
import { appendHookFailureEvent, catchUpStreamIngestion } from "./turn-phase/utils.js";
import type {
  RunTurnPhasesInput,
  TurnPhaseCoordinatorOptions,
  TurnPhaseResult,
} from "./turn-phase/types.js";
import type { TurnExtractionPhaseResult } from "./turn-phase/extraction-phase.js";
import type { TurnRetrievalPhaseResult } from "./turn-phase/retrieval-phase.js";
import type { TurnDeliberationPhaseResult } from "./turn-phase/deliberation-phase.js";
export type {
  RunTurnPhasesInput,
  TurnPhaseCoordinatorOptions,
  TurnPhaseInput,
  TurnPhaseResult,
} from "./turn-phase/types.js";

function previewItems(items: readonly string[], limit = 4): string {
  const head = items.slice(0, limit).join(",");
  return items.length > limit ? `${head},+${items.length - limit}` : head;
}

function summarizePerceptionResult(
  result: Awaited<
    ReturnType<
      ReturnType<TurnPhaseCoordinatorOptions["perceptionGateway"]["beginTurn"]>["perceive"]
    >
  >,
): string {
  return `mode=${result.perception.mode} entities=[${previewItems(result.perception.entities)}]`;
}

function summarizeFrameClassification(
  classification: Awaited<ReturnType<typeof classifyFrameAnomalyPhase>>,
): string {
  if (classification === null) {
    return "skipped";
  }

  if (classification.status === "degraded") {
    return `degraded reason=${classification.reason}`;
  }

  return `kind=${classification.kind} conf=${classification.confidence}`;
}

function summarizeExtraction(result: TurnExtractionPhaseResult): string {
  return `actions=${result.createdActionIds.length} goals=${result.persistedPromotions.goalIds.length} steps=${result.persistedPromotions.executiveStepIds.length} creator_directives=${result.creatorDirectives.length} commitment=${result.correctiveCommitment === null ? "none" : "candidate"}`;
}

function summarizeRetrieval(result: TurnRetrievalPhaseResult): string {
  const ledgerEntries =
    result.evidenceLedgerContext.ledger?.sections.reduce(
      (sum, section) => sum + section.entries.length,
      0,
    ) ?? 0;

  return `episodes=${result.retrievedEpisodes.length} evidence=${result.retrieval.evidence.length} ledger=${ledgerEntries}`;
}

function summarizeDeliberation(result: TurnDeliberationPhaseResult): string {
  return `path=${result.deliberation.path} recommendation=${result.deliberation.emissionRecommendation ?? "emit"} stop=${result.deliberation.usage.stop_reason ?? "none"}`;
}

export class TurnPhaseCoordinator {
  private readonly contradictionRoutingCooldown = new ContradictionRoutingCooldown();

  constructor(private readonly options: TurnPhaseCoordinatorOptions) {}

  async run(input: RunTurnPhasesInput): Promise<TurnPhaseResult> {
    const turnInput = {
      ...input.input,
      globalTurnCounter: input.globalTurnCounter,
    };
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

    await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "ingest",
      sub: "pre_turn_catchup",
      run: () =>
        catchUpStreamIngestion({
          coordinator: this.options.streamIngestionCoordinator,
          sessionId,
          streamWriter,
          maxEntries: this.options.config.streamIngestion.preTurnCatchup.maxEntries,
          appendHookFailureEvent: appendHookFailure,
        }),
      completedSub: () => "pre_turn_catchup",
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
    const audienceResolution = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "audience",
      sub: "resolve_profile",
      run: async () => {
        const resolvedAudienceEntityId =
          turnInput.audience === undefined || isSelfAudience
            ? null
            : this.options.entityRepository.resolve(turnInput.audience, {
                provenance: "transport_audience_label",
              });
        const resolvedAudienceEntity =
          resolvedAudienceEntityId === null
            ? null
            : this.options.entityRepository.get(resolvedAudienceEntityId);
        const resolvedAudienceProfile =
          resolvedAudienceEntityId === null
            ? null
            : this.options.socialRepository.getProfile(resolvedAudienceEntityId);

        // In a group channel the social exchange belongs to the current speaker,
        // not to the abstract channel entity. Updating the group too is deferred.
        const resolvedSocialInteractionEntityId =
          resolvedAudienceEntity?.kind === "group"
            ? (turnInput.senderEntityId ?? null)
            : resolvedAudienceEntityId;
        const resolvedGroupSpeakerEntityId =
          resolvedAudienceEntity?.kind === "group" ? (turnInput.senderEntityId ?? null) : null;
        const resolvedGroupSpeakerDisplayName =
          resolvedGroupSpeakerEntityId === null
            ? null
            : (this.options.entityRepository.get(resolvedGroupSpeakerEntityId)?.canonical_name ??
              null);

        return {
          audienceEntityId: resolvedAudienceEntityId,
          audienceEntity: resolvedAudienceEntity,
          audienceProfile: resolvedAudienceProfile,
          socialInteractionEntityId: resolvedSocialInteractionEntityId,
          groupSpeakerEntityId: resolvedGroupSpeakerEntityId,
          groupSpeakerDisplayName: resolvedGroupSpeakerDisplayName,
        };
      },
      completedSub: (result) =>
        `entity=${result.audienceEntityId ?? "self"} kind=${result.audienceEntity?.kind ?? "self"}`,
    });
    const audienceEntityId = audienceResolution.audienceEntityId;
    const audienceEntity = audienceResolution.audienceEntity;
    let audienceProfile = audienceResolution.audienceProfile;
    const socialInteractionEntityId = audienceResolution.socialInteractionEntityId;
    const groupSpeakerEntityId = audienceResolution.groupSpeakerEntityId;
    const groupSpeakerDisplayName = audienceResolution.groupSpeakerDisplayName;
    const sessionRecord = this.options.sessionsRepository?.get(sessionId) ?? null;
    const sessionAudienceRole = sessionRecord?.audience_role ?? "participant";
    const participationPolicy = sessionRecord?.participation_policy ?? "active";
    const operatorSessionSnapshot =
      sessionAudienceRole === "operator" && this.options.sessionsRepository !== undefined
        ? buildOperatorSessionSnapshot({
            sessions: this.options.sessionsRepository.list({
              status: "active",
              excludeSessionId: sessionId,
              limit: OPERATOR_SESSION_SNAPSHOT_CAP,
            }),
            totalActiveOtherSessionCount: this.options.sessionsRepository.count({
              status: "active",
              excludeSessionId: sessionId,
            }),
            currentSessionId: sessionId,
            nowMs: this.options.clock.now(),
            cap: OPERATOR_SESSION_SNAPSHOT_CAP,
          })
        : null;
    const currentSenderEntityId =
      audienceEntity?.kind === "group" ? groupSpeakerEntityId : audienceEntityId;
    const currentSenderEntity =
      currentSenderEntityId === null
        ? null
        : this.options.entityRepository.get(currentSenderEntityId);
    const creator = this.options.entityRepository.getCreator();
    const creatorIdentity = creator === null ? null : { displayName: creator.canonical_name };
    const creatorContext = {
      currentSenderEntityId,
      currentSenderDisplayName: currentSenderEntity?.canonical_name ?? null,
      currentSenderBorgRole: currentSenderEntity?.borg_role ?? null,
      sessionAudienceRole,
    };
    const perceptionResult = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "perception",
      run: () =>
        turnPerception.perceive({
          sessionId,
          isSelfAudience,
          origin: turnInput.origin,
          cognitionInput,
          workingMemory,
        }),
      completedSub: summarizePerceptionResult,
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
      attachments: turnInput.attachments,
      persistAttachments: (attachmentInput) =>
        this.options.attachmentService.persistTurnAttachments({
          ...attachmentInput,
          createdTurnGlobal: input.globalTurnCounter,
        }),
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
    const currentUserContent = openingPersistence.currentUserContent;
    const persistedPerceptionEntry = openingPersistence.persistedPerceptionEntry;
    workingMemory = openingPersistence.workingMemory;

    for (const attachment of openingPersistence.persistedAttachments) {
      await this.options.imagePerceptionService?.perceiveAttachment({
        attachmentId: attachment.attachmentId,
        turnId,
      });
    }

    // Audience tracing happens before perception because that is the clean
    // resolution boundary. Group participant rostering stays here: it relies
    // on current-turn persistence and feeds the frame/extraction context.
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
        session_id: sessionId,
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
    let participantRoster = buildParticipantRosterFromRepositories({
      activeParticipants,
      audienceEntityId,
      entityRepository: this.options.entityRepository,
      relationalSlotRepository: this.options.relationalSlotRepository,
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
      currentSenderEntityId,
      currentSenderBorgRole: currentSenderEntity?.borg_role ?? null,
      sessionAudienceRole,
    });
    const frameAnomalyClassification = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "frame",
      run: () =>
        classifyFrameAnomalyPhase({
          options: this.options,
          appendHookFailureEvent: appendHookFailure,
          llmClient,
          turnId,
          sessionId,
          isUserTurn,
          userMessage: turnInput.userMessage,
          recentHistory: recencyWindow.messages,
          conversationContext: frameAnomalyConversationContext,
          persistedUserEntryId,
          streamWriter,
        }),
      completedSub: summarizeFrameClassification,
    });
    const currentTurnFrameAnomaly = isFrameAnomaly(frameAnomalyClassification)
      ? frameAnomalyClassification
      : null;

    const extraction = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "extract",
      run: () =>
        runExtractionPhase({
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
          currentSenderEntityId,
          currentSenderDisplayName: creatorContext.currentSenderDisplayName,
          currentSenderBorgRole: creatorContext.currentSenderBorgRole,
          sessionAudienceRole,
          participantRoster,
          persistedUserEntryId,
          frameAnomalyClassification,
          streamWriter,
          trackAppliedSlotNegation: (slot) => lifecycleTracker.trackAppliedSlotNegation(slot),
        }),
      completedSub: summarizeExtraction,
    });
    const correctiveCommitment = extraction.correctiveCommitment;
    const correctiveCommitmentSupersession = extraction.correctiveCommitmentSupersession;
    workingMemory = extraction.workingMemory;
    participantRoster = buildParticipantRosterFromRepositories({
      activeParticipants,
      audienceEntityId,
      entityRepository: this.options.entityRepository,
      relationalSlotRepository: this.options.relationalSlotRepository,
    });
    lifecycleTracker.trackCreatedActionIds(extraction.createdActionIds);
    lifecycleTracker.trackCreatedGoalIds(extraction.persistedPromotions.goalIds);
    lifecycleTracker.trackCreatedExecutiveStepIds(extraction.persistedPromotions.executiveStepIds);

    const closureLoopAssessment = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "closure_loop",
      run: () =>
        classifyClosureLoopPhase({
          options: this.options,
          appendHookFailureEvent: appendHookFailure,
          llmClient,
          turnId,
          sessionId,
          isUserTurn,
          userMessage: turnInput.userMessage,
          recentHistory: recencyWindow.messages,
          persistedUserEntryId,
          workingMemory,
          streamWriter,
        }),
      completedSub: (result) =>
        result === null
          ? "skipped"
          : result.closureLoopDetected === true
            ? "closure-loop detected"
            : result.currentUserSubstantive === true
              ? "substantive"
              : result.currentUserClosureShaped === true
                ? "closure-shaped"
                : "no-op",
    });

    if (closureLoopAssessment?.currentUserSubstantive === true) {
      workingMemory = this.options.discourseStateService.clearClosureLoop({
        workingMemory,
        reason: closureLoopAssessment.reason,
        turnId,
        sessionId,
      });
    } else if (
      closureLoopAssessment?.currentUserClosureShaped === true &&
      workingMemory.discourse_state?.closure_loop?.status === "named"
    ) {
      return suppressFromClosureLoopPhase({
        turnId,
        sessionId,
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
        sessionId,
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
    const gateResult = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "generation_gate",
      run: () =>
        generationGate.evaluate({
          userMessage: turnInput.userMessage,
          workingMemory,
          recencyMessages: recencyWindow.messages,
        }),
      completedSub: (result) =>
        result.action === "suppress" ? `suppress: ${result.explanation ?? ""}` : "allow",
    });

    if (gateResult.signals.hardCapDue) {
      await this.options.discourseStateService.appendHardCapEvent({
        streamWriter,
        turnId,
        sessionId,
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
        sessionId,
      });
    }

    if (gateResult.action === "suppress") {
      return suppressFromGenerationGatePhase({
        turnId,
        sessionId,
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

    const retrievalPhase = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "retrieval",
      run: () =>
        runRetrievalPhase({
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
          currentSenderBorgRole: currentSenderEntity?.borg_role ?? null,
          audienceProfile,
          sessionAudienceRole,
          perception,
          workingMemory,
          suppressionSet,
          actionLinkSelfContext: extraction.actionLinkSelfContext,
          persistedPromotions: extraction.persistedPromotions,
          correctiveCommitment,
          activeParticipants,
          participantRoster,
          participantProfiles,
          persistedUserEntry: persistedUserEntry ?? undefined,
          currentTurnFrameAnomaly,
          closureLoopAssessment,
        }),
      completedSub: summarizeRetrieval,
    });
    const deliberationPhase = await traceTurnPhase({
      tracer: this.options.tracer,
      clock: this.options.clock,
      turnId,
      sessionId,
      phase: "delib",
      run: () =>
        runDeliberationPhase({
          options: this.options,
          llmClient,
          sessionId,
          turnId,
          turnInput,
          streamWriter,
          audienceEntityId,
          participationPolicy,
          creatorIdentity,
          creatorContext,
          operatorSessionSnapshot,
          persistedUserEntryId,
          currentUserContent,
          perception,
          workingMemory,
          activeParticipants,
          participantRoster,
          participantProfiles,
          audienceProfile,
          recencyMessages: recencyWindow.messages,
          currentTurnFrameAnomaly,
          retrievalPhase,
          contradictionRoutingCooldown: this.contradictionRoutingCooldown,
        }),
      completedSub: summarizeDeliberation,
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
