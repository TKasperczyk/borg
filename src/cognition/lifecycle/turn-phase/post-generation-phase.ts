import { performAction } from "../../action/index.js";
import type { TurnActionCoordinator } from "../../action/turn-action-coordinator.js";
import type { CorrectivePreferenceTurnService } from "../../commitments/corrective-preference-service.js";
import { Deliberator } from "../../deliberation/deliberator.js";
import type { GenerationGate } from "../../generation/generation-gate.js";
import { StopCommitmentExtractor } from "../../generation/self-stop-commitment.js";
import {
  replyTargetEntityId,
  type PendingTurnEmission,
  type TurnEmission,
} from "../../generation/types.js";
import type { FrameAnomalyClassification } from "../../frame-anomaly/index.js";
import type { PerceptionResult } from "../../types.js";
import type { LLMClient } from "../../../llm/index.js";
import type { StreamEntry, StreamWriter } from "../../../stream/index.js";
import type { EntityId, SessionId } from "../../../util/ids.js";
import type { CognitiveMode } from "../../types.js";
import type { WorkingMemory } from "../../../memory/working/index.js";
import type { SharedStateEntry } from "../../../memory/decision-artifacts/index.js";
import {
  ACTION_ARCHIVE_ACTIVE_STATES,
  ACTION_ARCHIVE_SCAN_LIMIT,
  classifyActionArchiveCandidate,
  type ActionRecord,
} from "../../../memory/actions/index.js";
import { archiveStaleAction } from "../../../memory/lifecycle-ops/index.js";
import type { TurnPhaseCoordinatorOptions, TurnPhaseInput, TurnPhaseResult } from "./types.js";
import type { TurnLifecycleTracker } from "../turn-lifecycle-tracker.js";
import type { TurnDeliberationPhaseResult } from "./deliberation-phase.js";
import type { TurnRetrievalPhaseResult } from "./retrieval-phase.js";
import {
  ACTIVE_TURN_STATUS,
  type AppendHookFailureEvent,
  persistCorrectiveCommitment,
  startLiveIngestion,
} from "./utils.js";

type CorrectiveCommitment = Parameters<
  CorrectivePreferenceTurnService["persistCommitment"]
>[0]["commitment"];
type CorrectiveCommitmentSupersession = Parameters<
  CorrectivePreferenceTurnService["persistCommitment"]
>[0]["supersession"];

const DEFAULT_ACTION_ARCHIVE_AFTER_INACTIVE_TURNS = 20;

type ActionArchiveScanResult = {
  scannedCount: number;
  eligibleCount: number;
  archivedCount: number;
  skippedByReason: Record<string, number>;
  oldestInactiveTurns: number;
  oldestEligibleInactiveTurns: number;
};

function currentTurnSharedStateEntries(input: {
  retrievalPhase: TurnRetrievalPhaseResult;
  persistedUserEntryId?: StreamEntry["id"];
}): SharedStateEntry[] {
  if (input.persistedUserEntryId === undefined) {
    return [];
  }

  const persistedUserEntryId = input.persistedUserEntryId;

  return (input.retrievalPhase.evidenceLedgerContext.ledger?.sharedState?.entries ?? []).filter(
    (entry) => entry.last_updated_stream_entry_ids.includes(persistedUserEntryId),
  );
}

function actionLifecycleTurnCounter(input: TurnPhaseInput, workingMemory: WorkingMemory): number {
  return input.globalTurnCounter ?? workingMemory.turn_counter;
}

function actionArchiveAfterInactiveTurns(options: TurnPhaseCoordinatorOptions): number {
  return (
    options.config.cognition.actionLifecycle.archiveStaleAfterInactiveTurns ??
    DEFAULT_ACTION_ARCHIVE_AFTER_INACTIVE_TURNS
  );
}

function incrementSkippedReason(skippedByReason: Record<string, number>, reason: string): void {
  skippedByReason[reason] = (skippedByReason[reason] ?? 0) + 1;
}

function archiveInactiveParticipantActions(input: {
  options: TurnPhaseCoordinatorOptions;
  turnId: string;
  turnCounter: number;
}): ActionArchiveScanResult {
  const candidates = input.options.actionRepository.list({
    states: ACTION_ARCHIVE_ACTIVE_STATES,
    limit: ACTION_ARCHIVE_SCAN_LIMIT,
  });
  const archiveAfterTurns = actionArchiveAfterInactiveTurns(input.options);
  const skippedByReason: Record<string, number> = {};
  let eligibleCount = 0;
  let archivedCount = 0;
  let oldestInactiveTurns = 0;
  let oldestEligibleInactiveTurns = 0;

  for (const action of candidates) {
    const classification = classifyActionArchiveCandidate(action, {
      turnCounter: input.turnCounter,
      archiveAfterTurns,
    });

    if (classification.status === "skipped") {
      incrementSkippedReason(skippedByReason, classification.reason);
      if (classification.inactiveTurns !== undefined) {
        oldestInactiveTurns = Math.max(oldestInactiveTurns, classification.inactiveTurns);
      }
      continue;
    }

    const inactiveTurns = classification.inactiveTurns;
    oldestInactiveTurns = Math.max(oldestInactiveTurns, inactiveTurns);
    oldestEligibleInactiveTurns = Math.max(oldestEligibleInactiveTurns, inactiveTurns);
    eligibleCount += 1;

    const result = archiveStaleAction({
      actionId: action.id,
      repository: input.options.actionRepository,
      nowMs: input.options.clock.now(),
      tracer: input.options.tracer,
      turnId: input.turnId,
      traceSource: "post_generation_inactivity_scan",
    });

    if (result.status === "success") {
      archivedCount += 1;
      if (input.options.tracer.enabled) {
        input.options.tracer.emit("action_archive.completed", {
          turnId: input.turnId,
          action_id: action.id,
          source: "post_generation_inactivity_scan",
          inactive_turns: inactiveTurns,
          last_referenced_turn_counter: action.last_referenced_turn_counter,
          last_referenced_turn_global: action.last_referenced_turn_global ?? null,
          archive_after_turns: archiveAfterTurns,
        });
      }
      continue;
    }

    incrementSkippedReason(
      skippedByReason,
      result.status === "conflict" ? "archive_conflict" : `archive_no_op_${result.reason}`,
    );
  }

  const scanResult: ActionArchiveScanResult = {
    scannedCount: candidates.length,
    eligibleCount,
    archivedCount,
    skippedByReason,
    oldestInactiveTurns,
    oldestEligibleInactiveTurns,
  };

  if (input.options.tracer.enabled) {
    input.options.tracer.emit("action_archive_scan.completed", {
      turnId: input.turnId,
      scanned_count: scanResult.scannedCount,
      eligible_count: scanResult.eligibleCount,
      archived_count: scanResult.archivedCount,
      skipped_by_reason: scanResult.skippedByReason,
      oldest_inactive_turns: scanResult.oldestInactiveTurns,
      oldest_eligible_inactive_turns: scanResult.oldestEligibleInactiveTurns,
      archive_after_turns: archiveAfterTurns,
    });
  }

  return scanResult;
}

export async function runPostGenerationPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  appendHookFailureEvent: AppendHookFailureEvent;
  llmClient: LLMClient;
  sessionId: SessionId;
  turnId: string;
  turnInput: TurnPhaseInput;
  streamWriter: StreamWriter;
  lifecycleTracker: TurnLifecycleTracker;
  cognitionInput: string;
  perception: PerceptionResult;
  workingMemory: WorkingMemory;
  workingMood: Parameters<
    TurnPhaseCoordinatorOptions["turnReflectionCoordinator"]["run"]
  >[0]["workingMood"];
  persistedUserEntry?: StreamEntry;
  persistedPerceptionEntry: Parameters<
    TurnPhaseCoordinatorOptions["turnReflectionCoordinator"]["run"]
  >[0]["persistedPerceptionEntry"];
  persistedUserEntryId?: StreamEntry["id"];
  correctiveCommitment: CorrectiveCommitment;
  correctiveCommitmentSupersession: CorrectiveCommitmentSupersession;
  deliberation: TurnDeliberationPhaseResult["deliberation"];
  retrievalPhase: TurnRetrievalPhaseResult;
  origin: TurnPhaseInput["origin"];
  autonomyTrigger: TurnPhaseInput["autonomyTrigger"];
  closureLoopCurrentUserAct: Parameters<TurnActionCoordinator["run"]>[0]["currentUserClosureKind"];
  audienceEntityId: EntityId | null;
  audienceIsGroup: boolean;
  senderEntityId: EntityId | null;
  socialInteractionEntityId: EntityId | null;
  pendingSocialAttribution: Parameters<
    TurnPhaseCoordinatorOptions["turnReflectionCoordinator"]["run"]
  >[0]["pendingSocialAttribution"];
  suppressionSet: Parameters<
    TurnPhaseCoordinatorOptions["turnReflectionCoordinator"]["run"]
  >[0]["suppressionSet"];
  isUserTurn: boolean;
  frameAnomalyClassification: FrameAnomalyClassification | null;
}): Promise<TurnPhaseResult> {
  const workingMemory = {
    ...input.workingMemory,
    updated_at: input.options.clock.now(),
  };
  const lifecycleTurnCounter = actionLifecycleTurnCounter(input.turnInput, input.workingMemory);
  const actionCoordinatorResult = await input.options.turnActionCoordinator.run({
    llmClient: input.llmClient,
    turnId: input.turnId,
    sessionId: input.sessionId,
    deliberation: input.deliberation,
    workingMemory,
    userMessage: input.turnInput.userMessage,
    cognitionInput: input.cognitionInput,
    origin: input.origin,
    autonomyTrigger: input.autonomyTrigger,
    applicableCommitments: input.retrievalPhase.applicableCommitments,
    perceptionEntities: input.perception.entities,
    persistedUserEntry: input.persistedUserEntry,
    retrievedEpisodes: input.retrievalPhase.retrievedEpisodes,
    currentUserClosureKind: input.closureLoopCurrentUserAct,
    audienceEntityId: input.audienceEntityId,
  });
  const actionResult = actionCoordinatorResult.actionResult;
  const actionEmission: PendingTurnEmission = actionCoordinatorResult.actionEmission;
  const deliberation = actionCoordinatorResult.deliberation;
  input.lifecycleTracker.trackPendingActionMerges(actionResult.pending_action_merge_count ?? 0);
  const persistedAgentEntry =
    actionEmission.kind === "message"
      ? await input.streamWriter.append({
          kind: "agent_msg",
          turn_id: input.turnId,
          turn_status: ACTIVE_TURN_STATUS,
          content: actionResult.response,
          tool_calls: actionResult.tool_calls,
          reply_target_entity_id: replyTargetEntityId(actionEmission.reply_target),
          ...(actionEmission.persistence_class === undefined
            ? {}
            : { persistence_class: actionEmission.persistence_class }),
          ...(input.turnInput.audience === undefined ? {} : { audience: input.turnInput.audience }),
        })
      : actionEmission.kind === "observed"
        ? await input.options.discourseStateService.appendObservationMarker({
            streamWriter: input.streamWriter,
            reason: actionEmission.reason,
            userEntryId: input.persistedUserEntryId,
            turnId: input.turnId,
            audience: input.turnInput.audience,
          })
        : await input.options.discourseStateService.appendSuppressionMarker({
            streamWriter: input.streamWriter,
            reason: actionEmission.reason,
            userEntryId: input.persistedUserEntryId,
            turnId: input.turnId,
            audience: input.turnInput.audience,
            noOutputCategories: actionEmission.no_output_categories,
            primaryNoOutputReason: actionEmission.primary_no_output_reason,
            structuralNoOutputFlags: actionEmission.structural_no_output_flags,
          });

  if (actionEmission.kind === "suppressed") {
    return suppressFromActionPhase({
      options: input.options,
      streamWriter: input.streamWriter,
      appendHookFailureEvent: input.appendHookFailureEvent,
      turnId: input.turnId,
      turnInput: input.turnInput,
      actionResult,
      actionEmission,
      persistedAgentEntry,
      correctiveCommitment: input.correctiveCommitment,
      correctiveCommitmentSupersession: input.correctiveCommitmentSupersession,
      perceptionMode: input.perception.mode,
      deliberation,
    });
  }

  const turnEmission: TurnEmission =
    actionEmission.kind === "observed"
      ? {
          kind: "observed",
          reason: actionEmission.reason,
          markerEntryId: persistedAgentEntry.id,
        }
      : {
          kind: "message",
          content: actionResult.response,
          agentMessageId: persistedAgentEntry.id,
          ...(actionEmission.reply_target === undefined
            ? {}
            : { reply_target: actionEmission.reply_target }),
          ...(actionEmission.persistence_class === undefined
            ? {}
            : { persistence_class: actionEmission.persistence_class }),
        };
  let postActionWorkingMemory = actionResult.workingMemory;
  if (
    actionEmission.kind === "message" &&
    actionEmission.closure_pressure_history_reason !== undefined
  ) {
    postActionWorkingMemory = input.options.discourseStateService.appendClosurePressureHistory({
      workingMemory: postActionWorkingMemory,
      turnId: input.turnId,
      reason: actionEmission.closure_pressure_history_reason,
    });
  }
  if (actionEmission.kind === "message") {
    const stopCommitmentExtractor = new StopCommitmentExtractor({
      llmClient: input.llmClient,
      model: input.options.config.anthropic.models.background,
      onDegraded: (reason, error) =>
        input.appendHookFailureEvent(
          input.streamWriter,
          "self_stop_commitment_extraction",
          error ?? reason,
          {
            reason,
          },
        ),
    });
    const stopCommitment = await stopCommitmentExtractor.extract({
      userMessage: input.turnInput.userMessage,
      agentResponse: actionResult.response,
    });

    if (stopCommitment !== null) {
      postActionWorkingMemory = input.options.discourseStateService.setStopState({
        workingMemory: postActionWorkingMemory,
        provenance: "self_commitment_extractor",
        sourceStreamEntryId: persistedAgentEntry.id,
        reason: stopCommitment.reason,
        turnId: input.turnId,
      });
    }

    if (postActionWorkingMemory.discourse_state?.closure_loop?.status === "detected") {
      postActionWorkingMemory = input.options.discourseStateService.markClosureLoopNamed({
        workingMemory: postActionWorkingMemory,
        sourceStreamEntryId: persistedAgentEntry.id,
        reason: "Closure loop detected; assistant used the single allowed naming/output turn.",
        turnId: input.turnId,
      });
      postActionWorkingMemory = input.options.discourseStateService.setStopState({
        workingMemory: postActionWorkingMemory,
        provenance: "finalizer_no_output",
        sourceStreamEntryId: persistedAgentEntry.id,
        reason:
          "Closure loop was already named once; suppress further closure-only turns until substantive content.",
        turnId: input.turnId,
      });
    }
  }

  await input.options.turnReflectionCoordinator.run({
    llmClient: input.llmClient,
    sessionId: input.sessionId,
    turnId: input.turnId,
    actionLifecycleTurnCounter: lifecycleTurnCounter,
    origin: input.origin,
    userMessage: input.turnInput.userMessage,
    perception: input.perception,
    workingMood: input.workingMood,
    postActionWorkingMemory,
    selfSnapshot: input.retrievalPhase.selfSnapshot,
    deliberation,
    actionResult,
    retrievedEpisodes: deliberation.retrievedEpisodes,
    retrievalConfidence: input.retrievalPhase.retrieval.confidence,
    executiveFocus: input.retrievalPhase.executiveFocusWithStep,
    selectedSkill: input.retrievalPhase.selectedSkill,
    proceduralContext: input.retrievalPhase.proceduralContext,
    audienceEntityId: input.audienceEntityId,
    audienceIsGroup: input.audienceIsGroup,
    senderEntityId: input.senderEntityId,
    socialInteractionEntityId: input.socialInteractionEntityId,
    pendingSocialAttribution: input.pendingSocialAttribution,
    suppressionSet: input.suppressionSet,
    persistedUserEntryId: input.persistedUserEntryId,
    persistedPerceptionEntry: input.persistedPerceptionEntry,
    persistedAgentEntry,
    isUserTurn: input.isUserTurn,
    frameAnomaly: input.frameAnomalyClassification,
    streamWriter: input.streamWriter,
    onHookFailure: (hook, error) => input.appendHookFailureEvent(input.streamWriter, hook, error),
    trackReflectionEffects: (effects) => input.lifecycleTracker.trackReflectionEffects(effects),
  });
  if (actionEmission.kind === "message" && input.persistedUserEntryId !== undefined) {
    await input.options.turnActionStateService.closeBorgSelfPerformedActions({
      llmClient: input.llmClient,
      turnId: input.turnId,
      userMessage: input.turnInput.userMessage,
      persistedUserEntryId: input.persistedUserEntryId,
      persistedAgentEntryId: persistedAgentEntry.id,
      agentResponse: actionResult.response,
      recentHistory: [],
      audienceEntityId: input.audienceEntityId,
      sessionId: input.sessionId,
      speakerEntityId: input.senderEntityId,
      currentTurnSharedStateEntries: currentTurnSharedStateEntries({
        retrievalPhase: input.retrievalPhase,
        persistedUserEntryId: input.persistedUserEntryId,
      }),
      turnCounter: lifecycleTurnCounter,
    });
  }
  archiveInactiveParticipantActions({
    options: input.options,
    turnId: input.turnId,
    turnCounter: lifecycleTurnCounter,
  });
  await persistCorrectiveCommitment({
    service: input.options.correctivePreferenceTurnService,
    streamWriter: input.streamWriter,
    turnId: input.turnId,
    commitment: input.correctiveCommitment,
    supersession: input.correctiveCommitmentSupersession,
    appendHookFailureEvent: input.appendHookFailureEvent,
  });
  startLiveIngestion(input.options.streamIngestionCoordinator, input.sessionId);

  return {
    mode: input.perception.mode,
    path: deliberation.path,
    response: actionResult.response,
    emitted: actionEmission.kind === "message",
    emission: turnEmission,
    thoughts: deliberation.thoughts,
    usage: deliberation.usage,
    retrievedEpisodeIds: deliberation.retrievedEpisodes.map((result) => result.episode.id),
    referencedEpisodeIds: [...(deliberation.referencedEpisodeIds ?? [])],
    intents: actionResult.intents,
    toolCalls: [...actionResult.tool_calls],
    ...(actionEmission.kind === "message" ? { agentMessageId: persistedAgentEntry.id } : {}),
  };
}

export async function suppressFromClosureLoopPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  streamWriter: StreamWriter;
  appendHookFailureEvent: AppendHookFailureEvent;
  turnId: string;
  turnInput: TurnPhaseInput;
  workingMemory: WorkingMemory;
  persistedUserEntryId?: StreamEntry["id"];
  correctiveCommitment: CorrectiveCommitment;
  correctiveCommitmentSupersession: CorrectiveCommitmentSupersession;
  perceptionMode: CognitiveMode;
  reason: string;
}): Promise<TurnPhaseResult> {
  let workingMemory = input.options.discourseStateService.markClosureLoopNamed({
    workingMemory: input.workingMemory,
    reason: input.reason,
    turnId: input.turnId,
    sourceStreamEntryId: input.persistedUserEntryId,
  });
  workingMemory = input.options.discourseStateService.setStopState({
    workingMemory,
    provenance: "finalizer_no_output",
    sourceStreamEntryId: input.persistedUserEntryId,
    reason: "Closure loop already named; suppressing another closure-only turn.",
    turnId: input.turnId,
  });
  const suppressionActionResult = await performAction({
    response: "",
    emission: {
      kind: "suppressed",
      reason: "finalizer_no_output",
    },
    toolCalls: [],
    intents: [],
    workingMemory: {
      ...workingMemory,
      updated_at: input.options.clock.now(),
    },
  });
  const suppressionMarker = await input.options.discourseStateService.appendSuppressionMarker({
    streamWriter: input.streamWriter,
    reason: "finalizer_no_output",
    userEntryId: input.persistedUserEntryId,
    turnId: input.turnId,
    audience: input.turnInput.audience,
  });
  const suppressionEmission: TurnEmission = {
    kind: "suppressed",
    reason: "finalizer_no_output",
    markerEntryId: suppressionMarker.id,
  };
  const suppressedWorkingMemory = input.options.discourseStateService.applySuppressedEmissionState({
    workingMemory: suppressionActionResult.workingMemory,
    reason: "finalizer_no_output",
    sourceStreamEntryId: suppressionMarker.id,
    turnId: input.turnId,
  });

  if (input.options.tracer.enabled) {
    input.options.tracer.emit("post_generation.rejected", {
      turnId: input.turnId,
      reason: "finalizer_no_output",
      streamEntryId: suppressionMarker.id,
      source: "closure_loop",
      classified: true,
    });
  }

  input.options.workingMemoryStore.save({
    ...suppressedWorkingMemory,
    updated_at: input.options.clock.now(),
  });
  await persistCorrectiveCommitment({
    service: input.options.correctivePreferenceTurnService,
    streamWriter: input.streamWriter,
    turnId: input.turnId,
    commitment: input.correctiveCommitment,
    supersession: input.correctiveCommitmentSupersession,
    appendHookFailureEvent: input.appendHookFailureEvent,
  });
  archiveInactiveParticipantActions({
    options: input.options,
    turnId: input.turnId,
    turnCounter: actionLifecycleTurnCounter(input.turnInput, input.workingMemory),
  });

  return suppressedTurnPhaseResult({
    mode: input.perceptionMode,
    emission: suppressionEmission,
    thoughts: [],
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "suppressed",
    },
    referencedEpisodeIds: [],
    retrievedEpisodeIds: [],
    toolCalls: [],
  });
}

export async function suppressFromGenerationGatePhase(input: {
  options: TurnPhaseCoordinatorOptions;
  streamWriter: StreamWriter;
  appendHookFailureEvent: AppendHookFailureEvent;
  turnId: string;
  turnInput: TurnPhaseInput;
  workingMemory: WorkingMemory;
  persistedUserEntryId?: StreamEntry["id"];
  gateResult: Awaited<ReturnType<GenerationGate["evaluate"]>>;
  correctiveCommitment: CorrectiveCommitment;
  correctiveCommitmentSupersession: CorrectiveCommitmentSupersession;
  perceptionMode: CognitiveMode;
}): Promise<TurnPhaseResult> {
  let workingMemory = input.workingMemory;
  const suppressionReason = input.gateResult.reason ?? "generation_gate";
  const activeStop = workingMemory.discourse_state?.stop_until_substantive_content ?? null;

  if (activeStop === null) {
    workingMemory = input.options.discourseStateService.setStopState({
      workingMemory,
      provenance: "generation_gate",
      sourceStreamEntryId: input.persistedUserEntryId,
      reason: input.gateResult.explanation,
      turnId: input.turnId,
    });
  }

  const suppressionActionResult = await performAction({
    response: "",
    emission: {
      kind: "suppressed",
      reason: suppressionReason,
    },
    toolCalls: [],
    intents: [],
    workingMemory: {
      ...workingMemory,
      updated_at: input.options.clock.now(),
    },
  });
  const suppressionMarker = await input.options.discourseStateService.appendSuppressionMarker({
    streamWriter: input.streamWriter,
    reason: suppressionReason,
    userEntryId: input.persistedUserEntryId,
    turnId: input.turnId,
    audience: input.turnInput.audience,
  });
  const suppressionEmission: TurnEmission = {
    kind: "suppressed",
    reason: suppressionReason,
    markerEntryId: suppressionMarker.id,
  };
  const suppressedWorkingMemory = input.options.discourseStateService.applySuppressedEmissionState({
    workingMemory: suppressionActionResult.workingMemory,
    reason: suppressionReason,
    sourceStreamEntryId: suppressionMarker.id,
    turnId: input.turnId,
  });

  if (input.options.tracer.enabled) {
    input.options.tracer.emit("post_generation.rejected", {
      turnId: input.turnId,
      reason: suppressionReason,
      streamEntryId: suppressionMarker.id,
      source: "generation_gate",
      classified: input.gateResult.classified,
    });
  }

  input.options.workingMemoryStore.save({
    ...suppressedWorkingMemory,
    updated_at: input.options.clock.now(),
  });
  await persistCorrectiveCommitment({
    service: input.options.correctivePreferenceTurnService,
    streamWriter: input.streamWriter,
    turnId: input.turnId,
    commitment: input.correctiveCommitment,
    supersession: input.correctiveCommitmentSupersession,
    appendHookFailureEvent: input.appendHookFailureEvent,
  });
  archiveInactiveParticipantActions({
    options: input.options,
    turnId: input.turnId,
    turnCounter: actionLifecycleTurnCounter(input.turnInput, input.workingMemory),
  });

  return suppressedTurnPhaseResult({
    mode: input.perceptionMode,
    emission: suppressionEmission,
    thoughts: [],
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "suppressed",
    },
    referencedEpisodeIds: [],
    retrievedEpisodeIds: [],
    toolCalls: [],
  });
}

async function suppressFromActionPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  streamWriter: StreamWriter;
  appendHookFailureEvent: AppendHookFailureEvent;
  turnId: string;
  turnInput: TurnPhaseInput;
  actionResult: Awaited<ReturnType<TurnActionCoordinator["run"]>>["actionResult"];
  actionEmission: Extract<PendingTurnEmission, { kind: "suppressed" }>;
  persistedAgentEntry: StreamEntry;
  correctiveCommitment: CorrectiveCommitment;
  correctiveCommitmentSupersession: CorrectiveCommitmentSupersession;
  perceptionMode: CognitiveMode;
  deliberation: Awaited<ReturnType<Deliberator["run"]>>;
}): Promise<TurnPhaseResult> {
  const suppressionEmission: TurnEmission = {
    kind: "suppressed",
    reason: input.actionEmission.reason,
    markerEntryId: input.persistedAgentEntry.id,
    ...(input.actionEmission.no_output_categories === undefined
      ? {}
      : { no_output_categories: [...input.actionEmission.no_output_categories] }),
    ...(input.actionEmission.primary_no_output_reason === undefined
      ? {}
      : { primary_no_output_reason: input.actionEmission.primary_no_output_reason }),
    ...(input.actionEmission.structural_no_output_flags === undefined
      ? {}
      : { structural_no_output_flags: [...input.actionEmission.structural_no_output_flags] }),
  };
  let suppressedWorkingMemory = input.options.discourseStateService.applySuppressedEmissionState({
    workingMemory: input.actionResult.workingMemory,
    reason: input.actionEmission.reason,
    sourceStreamEntryId: input.persistedAgentEntry.id,
    turnId: input.turnId,
  });
  if (
    input.actionEmission.closure_pressure_history_reason !== undefined &&
    input.actionEmission.reason !== "closure_pressure_only" &&
    input.actionEmission.reason !== "closure_response_audit_failed_closed"
  ) {
    suppressedWorkingMemory = input.options.discourseStateService.appendClosurePressureHistory({
      workingMemory: suppressedWorkingMemory,
      turnId: input.turnId,
      reason: input.actionEmission.closure_pressure_history_reason,
    });
  }

  if (input.options.tracer.enabled) {
    input.options.tracer.emit("post_generation.rejected", {
      turnId: input.turnId,
      reason: input.actionEmission.reason,
      streamEntryId: input.persistedAgentEntry.id,
      ...(input.actionEmission.no_output_categories === undefined
        ? {}
        : { no_output_categories: [...input.actionEmission.no_output_categories] }),
      ...(input.actionEmission.primary_no_output_reason === undefined
        ? {}
        : { primary_no_output_reason: input.actionEmission.primary_no_output_reason }),
      ...(input.actionEmission.structural_no_output_flags === undefined
        ? {}
        : { structural_no_output_flags: [...input.actionEmission.structural_no_output_flags] }),
    });
  }

  input.options.workingMemoryStore.save({
    ...suppressedWorkingMemory,
    updated_at: input.options.clock.now(),
  });
  await persistCorrectiveCommitment({
    service: input.options.correctivePreferenceTurnService,
    streamWriter: input.streamWriter,
    turnId: input.turnId,
    commitment: input.correctiveCommitment,
    supersession: input.correctiveCommitmentSupersession,
    appendHookFailureEvent: input.appendHookFailureEvent,
  });
  archiveInactiveParticipantActions({
    options: input.options,
    turnId: input.turnId,
    turnCounter: actionLifecycleTurnCounter(input.turnInput, input.actionResult.workingMemory),
  });

  return suppressedTurnPhaseResult({
    mode: input.perceptionMode,
    emission: suppressionEmission,
    thoughts: input.deliberation.thoughts,
    usage: input.deliberation.usage,
    retrievedEpisodeIds: input.deliberation.retrievedEpisodes.map((result) => result.episode.id),
    referencedEpisodeIds: [...(input.deliberation.referencedEpisodeIds ?? [])],
    toolCalls: [...input.actionResult.tool_calls],
  });
}

function suppressedTurnPhaseResult(input: {
  mode: CognitiveMode;
  emission: TurnEmission;
  thoughts: string[];
  usage: TurnPhaseResult["usage"];
  retrievedEpisodeIds: string[];
  referencedEpisodeIds: string[];
  toolCalls: TurnPhaseResult["toolCalls"];
}): TurnPhaseResult {
  return {
    mode: input.mode,
    path: "suppressed",
    response: "",
    emitted: false,
    emission: input.emission,
    thoughts: input.thoughts,
    usage: input.usage,
    retrievedEpisodeIds: input.retrievedEpisodeIds,
    referencedEpisodeIds: input.referencedEpisodeIds,
    intents: [],
    toolCalls: input.toolCalls,
  };
}
