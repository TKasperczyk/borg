import { performAction, type ActionResult } from "../../turn-action/index.js";
import type { TurnActionCoordinator } from "../../turn-action/turn-action-coordinator.js";
import type { CorrectivePreferenceTurnService } from "../../commitments/corrective-preference-service.js";
import { Deliberator } from "../../deliberation/deliberator.js";
import type { GenerationGate } from "../../generation/generation-gate.js";
import {
  replyTargetEntityId,
  type GenerationSuppressionReason,
  type PendingTurnEmission,
  type TurnEmission,
} from "../../generation/types.js";
import type { ActualFrameAnomalyClassification } from "../../frame-anomaly/index.js";
import type { ClosureLoopAssessment } from "../../generation/closure-loop.js";
import type { ActiveParticipant } from "../../participants.js";
import { isDirectedOutboundTurnOrigin, type PerceptionResult } from "../../types.js";
import type { LLMClient } from "../../../llm/index.js";
import type {
  StreamEntry,
  StreamEntryInput,
  StreamResponseTo,
  StreamWriter,
} from "../../../stream/index.js";
import type { EntityId, SessionId, StreamEntryId } from "../../../util/ids.js";
import type { CognitiveMode } from "../../types.js";
import type { DiscourseStopProvenance, WorkingMemory } from "../../../memory/working/index.js";
import type { ActivityEventStatus } from "../../../memory/activity/index.js";
import { CognitionError } from "../../../util/errors.js";
import type { SharedStateEntry } from "../../../memory/shared-state/index.js";
import type { OutboundDeliveryReceipt, OutboundDeliveryResult } from "../../../outbound/types.js";
import {
  ACTION_ARCHIVE_ACTIVE_STATES,
  ACTION_ARCHIVE_SCAN_LIMIT,
  classifyActionArchiveCandidate,
  type ActionRecord,
} from "../../../memory/actions/index.js";
import { archiveStaleAction } from "../../../memory/lifecycle-ops/index.js";
import type { TurnPhaseCoordinatorOptions, TurnPhaseInput, TurnPhaseResult } from "./types.js";
import type { CurrentTurnUserInputSenderAttribution } from "../../turn-input.js";
import type { TurnLifecycleTracker } from "../turn-lifecycle-tracker.js";
import type { TurnDeliberationPhaseResult } from "./deliberation-phase.js";
import {
  buildCompactedEvidenceLedgerWithoutSharedState,
  compileSharedStateArtifactForEvidenceLedgerResult,
  type TurnRetrievalPhaseResult,
} from "./retrieval-phase.js";
import { traceTurnPhase } from "./phase-trace.js";
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
type CorrectiveCommitmentRetirement = Parameters<
  CorrectivePreferenceTurnService["persistCommitment"]
>[0]["retirement"];

const DEFAULT_ACTION_ARCHIVE_AFTER_INACTIVE_TURNS = 20;

type MessageStopStateApplication = {
  provenance: DiscourseStopProvenance;
  reason: string;
};

type ActionArchiveScanResult = {
  scannedCount: number;
  eligibleCount: number;
  archivedCount: number;
  skippedByReason: Record<string, number>;
  oldestInactiveTurns: number;
  oldestEligibleInactiveTurns: number;
};

type PersistedMessageEmission = {
  entry: StreamEntry;
  outboundDelivery?: OutboundDeliveryReceipt;
};

function currentTurnSharedStateEntries(input: {
  retrievalPhase: TurnRetrievalPhaseResult;
  persistedUserEntryId?: StreamEntry["id"];
  sourceUserEntryIds?: readonly StreamEntry["id"][];
}): SharedStateEntry[] {
  const sourceUserEntryIds = new Set([
    ...(input.persistedUserEntryId === undefined ? [] : [input.persistedUserEntryId]),
    ...(input.sourceUserEntryIds ?? []),
  ]);

  if (sourceUserEntryIds.size === 0) {
    return [];
  }

  return (input.retrievalPhase.evidenceLedgerContext.ledger?.sharedState?.entries ?? []).filter(
    (entry) =>
      entry.last_updated_stream_entry_ids.some((streamEntryId) =>
        sourceUserEntryIds.has(streamEntryId),
      ),
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

function activityStatusForStreamEntry(
  entry: Pick<StreamEntry, "turn_status">,
): ActivityEventStatus {
  return entry.turn_status === "aborted" ? "inactive" : "active";
}

function outboundDeliveryReceipt(delivery: OutboundDeliveryResult): OutboundDeliveryReceipt {
  return {
    status: delivery.status,
    streamEntryId: delivery.streamEntry.id,
    sourceType: delivery.sourceType,
    ...(delivery.externalMessageId === undefined
      ? {}
      : { externalMessageId: delivery.externalMessageId }),
    ...(delivery.error === undefined ? {} : { error: delivery.error }),
  };
}

async function persistMessageEmission(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  turnId: string;
  turnInput: TurnPhaseInput;
  streamWriter: StreamWriter;
  response: string;
  actionResult: Pick<ActionResult, "tool_calls">;
  actionEmission: Extract<PendingTurnEmission, { kind: "message" }>;
  responseTo?: StreamResponseTo;
}): Promise<PersistedMessageEmission> {
  const streamInput: Omit<StreamEntryInput, "kind" | "content"> = {
    turn_id: input.turnId,
    turn_status: ACTIVE_TURN_STATUS,
    tool_calls: input.actionResult.tool_calls,
    reply_target_entity_id: replyTargetEntityId(input.actionEmission.reply_target),
    ...(input.actionEmission.persistence_class === undefined
      ? {}
      : { persistence_class: input.actionEmission.persistence_class }),
    ...(input.responseTo === undefined ? {} : { response_to: input.responseTo }),
    ...(input.turnInput.audience === undefined ? {} : { audience: input.turnInput.audience }),
  };

  if (
    isDirectedOutboundTurnOrigin(input.turnInput.origin) &&
    input.options.outboundDelivery !== undefined
  ) {
    const session = input.options.sessionsRepository?.get(input.sessionId) ?? null;

    if (session !== null) {
      const delivery = await input.options.outboundDelivery.deliver({
        session,
        streamWriter: input.streamWriter,
        message: {
          content: input.response,
          streamInput,
        },
      });

      return {
        entry: delivery.streamEntry,
        outboundDelivery: outboundDeliveryReceipt(delivery),
      };
    }
  }

  return {
    entry: await input.streamWriter.append({
      kind: "agent_msg",
      content: input.response,
      ...streamInput,
    }),
  };
}

async function persistContinueThoughtEmission(input: {
  options: TurnPhaseCoordinatorOptions;
  turnId: string;
  turnInput: TurnPhaseInput;
  streamWriter: StreamWriter;
  actionEmission: Extract<PendingTurnEmission, { kind: "continue_thought" }>;
  responseTo?: StreamResponseTo;
}): Promise<PersistedMessageEmission> {
  if (input.options.trainOfThoughtRepository === undefined) {
    throw new CognitionError("Train of thought repository is not configured");
  }

  const selfEntityId = input.options.entityRepository.resolve("self", {
    kind: "self",
    provenance: "assistant_seeded",
  });
  const stored = input.options.trainOfThoughtRepository.upsert({
    text: input.actionEmission.text,
    selfEntityId,
  });
  const entry = await input.streamWriter.append({
    kind: "internal_event",
    turn_id: input.turnId,
    turn_status: ACTIVE_TURN_STATUS,
    content: {
      kind: "train_of_thought_continued",
      turn_id: input.turnId,
      self_entity_id: stored.self_entity_id,
      updated_at: stored.updated_at,
      text_length: stored.text.length,
    },
    ...(input.responseTo === undefined ? {} : { response_to: input.responseTo }),
    audience: input.turnInput.audience,
  });

  return { entry };
}

function advanceChatResponseWatermark(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  responseTo?: StreamResponseTo;
}): void {
  if (input.responseTo === undefined) {
    return;
  }

  if (input.options.chatResponseWatermarkCoordinator === undefined) {
    throw new CognitionError("Inbound batch terminal output requires watermark coordination", {
      code: "CHAT_RESPONSE_WATERMARK_COORDINATOR_REQUIRED",
    });
  }

  input.options.chatResponseWatermarkCoordinator.advanceThrough(
    input.sessionId,
    input.responseTo.through_cursor_inclusive,
  );
}

function startTerminalLiveIngestion(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  responseTo?: StreamResponseTo;
  terminalEntry: StreamEntry;
}): void {
  if (input.responseTo === undefined) {
    startLiveIngestion(input.options.streamIngestionCoordinator, input.sessionId);
    return;
  }

  startLiveIngestion(input.options.streamIngestionCoordinator, input.sessionId, {
    answeredWindow: {
      responseTo: input.responseTo,
      terminalCursor: {
        ts: input.terminalEntry.timestamp,
        entryId: input.terminalEntry.id,
      },
    },
  });
}

function activityParticipantEntityIds(input: {
  senderEntityId: EntityId | null;
  audienceEntityId: EntityId | null;
  replyTargetEntityId?: EntityId | null;
}): EntityId[] {
  return [input.senderEntityId, input.audienceEntityId, input.replyTargetEntityId ?? null].filter(
    (entityId): entityId is EntityId => entityId !== null,
  );
}

function stopStateApplicationForMessage(input: {
  emission: Extract<PendingTurnEmission, { kind: "message" }>;
  response: string;
}): MessageStopStateApplication | null {
  if (input.emission.discourse_control?.kind === "stop_until_substantive_content") {
    return {
      provenance: "finalizer_emission_metadata",
      reason: input.emission.discourse_control.reason,
    };
  }

  return null;
}

function incrementSkippedReason(skippedByReason: Record<string, number>, reason: string): void {
  skippedByReason[reason] = (skippedByReason[reason] ?? 0) + 1;
}

async function compilePostResponseSharedState(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionId: SessionId;
  turnId: string;
  turnInput: TurnPhaseInput;
  persistedAgentEntry: StreamEntry;
  agentResponse: string;
  persistedUserEntry?: StreamEntry;
  sourceUserEntries?: readonly StreamEntry[];
  workingMemory: WorkingMemory;
  perception: PerceptionResult;
  retrievalPhase: TurnRetrievalPhaseResult;
  audienceEntityId: EntityId | null;
  isUserTurn: boolean;
  currentTurnFrameAnomaly: ActualFrameAnomalyClassification | null;
  closureLoopAssessment?: ClosureLoopAssessment | null;
  activeParticipants?: readonly ActiveParticipant[];
}): Promise<void> {
  if (input.audienceEntityId === null || !input.options.config.generation.evidenceLedger.enabled) {
    return;
  }

  const currentUserEntries =
    input.sourceUserEntries === undefined || input.sourceUserEntries.length === 0
      ? undefined
      : input.sourceUserEntries;
  const currentUserEntry = input.persistedUserEntry ?? currentUserEntries?.[0];
  const ledgerInput = {
    sessionId: input.sessionId,
    turnId: input.turnId,
    audienceEntityId: input.audienceEntityId,
    currentUserMessage: input.turnInput.userMessage,
    ...(currentUserEntry === undefined ? {} : { currentUserEntry }),
    ...(currentUserEntries === undefined ? {} : { currentUserEntries }),
    ...(input.turnInput.globalTurnCounter === undefined
      ? {}
      : { globalTurnCounter: input.turnInput.globalTurnCounter }),
    workingMemory: input.workingMemory,
    applicableCommitments: input.retrievalPhase.applicableCommitments ?? [],
    retrievedEvidence: input.retrievalPhase.retrieval.evidence ?? [],
    retrievedEpisodes: input.retrievalPhase.retrievedEpisodes ?? [],
    retrievedSemantic: input.retrievalPhase.retrievedSemantic ?? null,
    openQuestions: input.retrievalPhase.retrieval.open_questions ?? [],
    pendingCorrections: input.retrievalPhase.pendingCorrections ?? [],
    frameAnomaly: input.currentTurnFrameAnomaly,
    activeParticipants: input.activeParticipants ?? [],
    participantRoster: input.retrievalPhase.participantRoster ?? null,
    isUserTurn: input.isUserTurn,
    perception: input.perception,
    closureLoopAssessment: input.closureLoopAssessment ?? null,
  };
  const compacted = await traceTurnPhase({
    tracer: input.options.tracer,
    clock: input.options.clock,
    turnId: input.turnId,
    sessionId: input.sessionId,
    phase: "ledger",
    sub: "post_response_shared_state",
    run: () =>
      buildCompactedEvidenceLedgerWithoutSharedState({
        options: input.options,
        input: ledgerInput,
      }),
    completedSub: (result) =>
      `post_response_shared_state entries=${result.ledger.sections.reduce(
        (sum, section) => sum + section.entries.length,
        0,
      )}`,
  });

  await compileSharedStateArtifactForEvidenceLedgerResult({
    options: input.options,
    input: ledgerInput,
    ledger: compacted.ledger,
    promptVisibleLedger: compacted.rendered ?? "",
    compilePass: "post_response",
    assistantResponse: {
      streamEntryId: input.persistedAgentEntry.id,
      text: input.agentResponse,
    },
    compileAnchorStreamEntryId: input.persistedAgentEntry.id,
  });
}

function archiveInactiveParticipantActions(input: {
  options: TurnPhaseCoordinatorOptions;
  turnId: string;
  sessionId: SessionId;
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
          session_id: input.sessionId,
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
      session_id: input.sessionId,
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
  sourceUserEntries?: readonly StreamEntry[];
  persistedPerceptionEntry: Parameters<
    TurnPhaseCoordinatorOptions["turnReflectionCoordinator"]["run"]
  >[0]["persistedPerceptionEntry"];
  persistedUserEntryId?: StreamEntry["id"];
  sourceUserEntryIds?: readonly StreamEntryId[];
  senderAttribution?: readonly CurrentTurnUserInputSenderAttribution[];
  responseTo?: StreamResponseTo;
  correctiveCommitment: CorrectiveCommitment;
  correctiveCommitmentSupersession: CorrectiveCommitmentSupersession;
  correctiveCommitmentRetirement: CorrectiveCommitmentRetirement;
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
  currentTurnFrameAnomaly: ActualFrameAnomalyClassification | null;
  closureLoopAssessment?: ClosureLoopAssessment | null;
  activeParticipants?: readonly ActiveParticipant[];
  knownInternalIdentifiers?: readonly string[];
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
    applicableCommitments: input.retrievalPhase.actionApplicableCommitments,
    perceptionEntities: input.perception.entities,
    persistedUserEntry: input.persistedUserEntry,
    persistedUserEntries: input.sourceUserEntries,
    retrievedEpisodes: input.retrievalPhase.retrievedEpisodes,
    currentUserClosureKind: input.closureLoopCurrentUserAct,
    audienceEntityId: input.audienceEntityId,
    knownInternalIdentifiers: input.knownInternalIdentifiers,
  });
  const actionResult = actionCoordinatorResult.actionResult;
  const actionEmission: PendingTurnEmission = actionCoordinatorResult.actionEmission;
  const deliberation = actionCoordinatorResult.deliberation;
  input.lifecycleTracker.trackPendingActionMerges(actionResult.pending_action_merge_count ?? 0);
  const persistedEmission = await traceTurnPhase({
    tracer: input.options.tracer,
    clock: input.options.clock,
    turnId: input.turnId,
    sessionId: input.sessionId,
    phase: "persist",
    sub: `emission=${actionEmission.kind}`,
    run: async (): Promise<PersistedMessageEmission> =>
      actionEmission.kind === "message"
        ? await persistMessageEmission({
            options: input.options,
            sessionId: input.sessionId,
            turnId: input.turnId,
            turnInput: input.turnInput,
            streamWriter: input.streamWriter,
            response: actionResult.response,
            actionResult,
            actionEmission,
            responseTo: input.responseTo,
          })
        : actionEmission.kind === "continue_thought"
          ? await persistContinueThoughtEmission({
              options: input.options,
              turnId: input.turnId,
              turnInput: input.turnInput,
              streamWriter: input.streamWriter,
              actionEmission,
              responseTo: input.responseTo,
            })
          : actionEmission.kind === "observed"
            ? {
                entry: await input.options.discourseStateService.appendObservationMarker({
                  streamWriter: input.streamWriter,
                  reason: actionEmission.reason,
                  userEntryId: input.persistedUserEntryId,
                  userEntryIds: input.sourceUserEntryIds,
                  responseTo: input.responseTo,
                  turnId: input.turnId,
                  audience: input.turnInput.audience,
                }),
              }
            : {
                entry: await input.options.discourseStateService.appendSuppressionMarker({
                  streamWriter: input.streamWriter,
                  reason: actionEmission.reason,
                  userEntryId: input.persistedUserEntryId,
                  userEntryIds: input.sourceUserEntryIds,
                  responseTo: input.responseTo,
                  turnId: input.turnId,
                  audience: input.turnInput.audience,
                  noOutputCategories: actionEmission.no_output_categories,
                  primaryNoOutputReason: actionEmission.primary_no_output_reason,
                  structuralNoOutputFlags: actionEmission.structural_no_output_flags,
                  finalizerInvalidTool: actionEmission.finalizer_invalid_tool,
                }),
              },
    completedSub: (result) => `entry=${result.entry.kind}`,
  });
  const persistedAgentEntry = persistedEmission.entry;
  advanceChatResponseWatermark({
    options: input.options,
    sessionId: input.sessionId,
    responseTo: input.responseTo,
  });
  const activityRepository = input.options.activityRepository;

  const shouldRecordActivity =
    persistedEmission.outboundDelivery === undefined ||
    persistedEmission.outboundDelivery.status === "transported";

  if (activityRepository !== undefined && shouldRecordActivity) {
    const status = activityStatusForStreamEntry(persistedAgentEntry);
    const participantEntityIds = activityParticipantEntityIds({
      senderEntityId: input.senderEntityId,
      audienceEntityId: input.audienceEntityId,
      replyTargetEntityId: persistedAgentEntry.reply_target_entity_id,
    });

    if (actionEmission.kind === "message") {
      activityRepository.record({
        kind: "borg_replied",
        occurredAt: persistedAgentEntry.timestamp,
        sessionId: persistedAgentEntry.session_id,
        turnId: input.turnId,
        speakerEntityId: null,
        actorEntityId: null,
        audienceEntityId: input.audienceEntityId,
        participantEntityIds,
        sourceStreamEntryIds: [persistedAgentEntry.id],
        status,
      });
    }

    activityRepository.record({
      kind: "turn_completed",
      occurredAt: persistedAgentEntry.timestamp,
      sessionId: persistedAgentEntry.session_id,
      turnId: input.turnId,
      speakerEntityId: null,
      actorEntityId: null,
      audienceEntityId: input.audienceEntityId,
      participantEntityIds,
      sourceStreamEntryIds: [persistedAgentEntry.id],
      status,
    });
  }

  if (actionEmission.kind === "suppressed") {
    return suppressFromActionPhase({
      options: input.options,
      streamWriter: input.streamWriter,
      appendHookFailureEvent: input.appendHookFailureEvent,
      turnId: input.turnId,
      sessionId: input.sessionId,
      turnInput: input.turnInput,
      actionResult,
      actionEmission,
      persistedAgentEntry,
      sourceUserEntryIds: input.sourceUserEntryIds,
      responseTo: input.responseTo,
      correctiveCommitment: input.correctiveCommitment,
      correctiveCommitmentSupersession: input.correctiveCommitmentSupersession,
      correctiveCommitmentRetirement: input.correctiveCommitmentRetirement,
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
      : actionEmission.kind === "continue_thought"
        ? {
            kind: "continue_thought",
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
            ...(actionEmission.discourse_control === undefined
              ? {}
              : { discourse_control: actionEmission.discourse_control }),
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
    const stopStateApplication = stopStateApplicationForMessage({
      emission: actionEmission,
      response: actionResult.response,
    });

    if (stopStateApplication !== null) {
      postActionWorkingMemory = input.options.discourseStateService.setStopState({
        workingMemory: postActionWorkingMemory,
        provenance: stopStateApplication.provenance,
        sourceStreamEntryId: persistedAgentEntry.id,
        reason: stopStateApplication.reason,
        turnId: input.turnId,
        sessionId: input.sessionId,
      });
    }

    if (postActionWorkingMemory.discourse_state?.closure_loop?.status === "detected") {
      postActionWorkingMemory = input.options.discourseStateService.markClosureLoopNamed({
        workingMemory: postActionWorkingMemory,
        sourceStreamEntryId: persistedAgentEntry.id,
        reason: "Closure loop detected; assistant used the single allowed naming/output turn.",
        turnId: input.turnId,
        sessionId: input.sessionId,
      });
      postActionWorkingMemory = input.options.discourseStateService.setStopState({
        workingMemory: postActionWorkingMemory,
        provenance: "finalizer_no_output",
        sourceStreamEntryId: persistedAgentEntry.id,
        reason:
          "Closure loop was already named once; suppress further closure-only turns until substantive content.",
        turnId: input.turnId,
        sessionId: input.sessionId,
      });
    }
  }

  if (actionEmission.kind === "message" && shouldRecordActivity) {
    // Bug #35: the pre-answer shared-state compile cannot cite Borg's future
    // response, and the shared-state prompt explicitly tells that pass not to
    // record facts whose truth depends on the not-yet-existing reply. Without
    // this post-generation pass, a durable decision Borg asserted in its own
    // persisted agent_msg is not compiled into the artifact until a later turn.
    // The post_response prompt pass makes the assistant stream entry visible as
    // citable evidence while preserving the same compiler and same prompt-pass
    // boundary: the fresh artifact never feeds back into this turn's deliberation.
    // This intentionally honors shouldSkipSharedStateCompile inside the shared
    // wrapper. Closure-shaped/no-delta and idle turns still do cheap anchor
    // advancement instead of a full compile, so a decision stated in a reply to
    // a closure-shaped user turn can be skipped; that accepted profile avoids
    // adding any response-text gate or deterministic semantic catch-up path.
    await compilePostResponseSharedState({
      options: input.options,
      sessionId: input.sessionId,
      turnId: input.turnId,
      turnInput: input.turnInput,
      persistedAgentEntry,
      agentResponse: actionResult.response,
      persistedUserEntry: input.persistedUserEntry,
      sourceUserEntries: input.sourceUserEntries,
      workingMemory: postActionWorkingMemory,
      perception: input.perception,
      retrievalPhase: input.retrievalPhase,
      audienceEntityId: input.audienceEntityId,
      isUserTurn: input.isUserTurn,
      currentTurnFrameAnomaly: input.currentTurnFrameAnomaly,
      closureLoopAssessment: input.closureLoopAssessment ?? null,
      activeParticipants: input.activeParticipants ?? [],
    });
  }

  await traceTurnPhase({
    tracer: input.options.tracer,
    clock: input.options.clock,
    turnId: input.turnId,
    sessionId: input.sessionId,
    phase: "reflect",
    sub: `emission=${actionEmission.kind}`,
    run: () =>
      input.options.turnReflectionCoordinator.run({
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
        sourceUserEntryIds: input.sourceUserEntryIds,
        persistedPerceptionEntry: input.persistedPerceptionEntry,
        persistedAgentEntry,
        isUserTurn: input.isUserTurn,
        frameAnomaly: input.currentTurnFrameAnomaly,
        streamWriter: input.streamWriter,
        onHookFailure: (hook, error) =>
          input.appendHookFailureEvent(input.streamWriter, hook, error),
        trackReflectionEffects: (effects) => input.lifecycleTracker.trackReflectionEffects(effects),
      }),
    completedSub: () =>
      `emission=${actionEmission.kind} retrieved=${deliberation.retrievedEpisodes.length}`,
  });
  const closeActionSourceUserEntryIds =
    input.sourceUserEntryIds === undefined || input.sourceUserEntryIds.length === 0
      ? input.persistedUserEntryId === undefined
        ? []
        : [input.persistedUserEntryId]
      : [...input.sourceUserEntryIds];

  if (actionEmission.kind === "message" && closeActionSourceUserEntryIds.length > 0) {
    await input.options.turnActionStateService.closeBorgSelfPerformedActions({
      llmClient: input.llmClient,
      turnId: input.turnId,
      userMessage: input.turnInput.userMessage,
      persistedUserEntryId: closeActionSourceUserEntryIds[0]!,
      sourceUserEntryIds: closeActionSourceUserEntryIds,
      persistedAgentEntryId: persistedAgentEntry.id,
      agentResponse: actionResult.response,
      recentHistory: [],
      audienceEntityId: input.audienceEntityId,
      sessionId: input.sessionId,
      speakerEntityId: input.senderEntityId,
      currentTurnSharedStateEntries: currentTurnSharedStateEntries({
        retrievalPhase: input.retrievalPhase,
        persistedUserEntryId: input.persistedUserEntryId,
        sourceUserEntryIds: input.sourceUserEntryIds,
      }),
      turnCounter: lifecycleTurnCounter,
    });
  }
  archiveInactiveParticipantActions({
    options: input.options,
    turnId: input.turnId,
    sessionId: input.sessionId,
    turnCounter: lifecycleTurnCounter,
  });
  await persistCorrectiveCommitment({
    service: input.options.correctivePreferenceTurnService,
    streamWriter: input.streamWriter,
    turnId: input.turnId,
    sessionId: input.sessionId,
    commitment: input.correctiveCommitment,
    supersession: input.correctiveCommitmentSupersession,
    retirement: input.correctiveCommitmentRetirement,
    appendHookFailureEvent: input.appendHookFailureEvent,
  });
  startTerminalLiveIngestion({
    options: input.options,
    sessionId: input.sessionId,
    responseTo: input.responseTo,
    terminalEntry: persistedAgentEntry,
  });

  return {
    turn_id: input.turnId,
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
    ...(persistedEmission.outboundDelivery === undefined
      ? {}
      : { outboundDelivery: persistedEmission.outboundDelivery }),
    terminalOutcome: "reflected",
  };
}

async function finalizeSuppressedTurn(input: {
  options: TurnPhaseCoordinatorOptions;
  streamWriter: StreamWriter;
  appendHookFailureEvent: AppendHookFailureEvent;
  turnId: string;
  sessionId: SessionId;
  turnInput: TurnPhaseInput;
  archiveWorkingMemory: WorkingMemory;
  sourceUserEntryIds: readonly StreamEntryId[] | undefined;
  responseTo: StreamResponseTo | undefined;
  correctiveCommitment: CorrectiveCommitment;
  correctiveCommitmentSupersession: CorrectiveCommitmentSupersession;
  correctiveCommitmentRetirement: CorrectiveCommitmentRetirement;
  perceptionMode: CognitiveMode;
  suppressionReason: GenerationSuppressionReason;
  suppressionActionResult: Pick<ActionResult, "workingMemory">;
  suppressionMarker: StreamEntry;
  suppressionEmission: TurnEmission;
  traceSource: "closure_loop" | "generation_gate";
  classified: boolean;
  terminalOutcome: TurnPhaseResult["terminalOutcome"];
}): Promise<TurnPhaseResult> {
  const suppressedWorkingMemory = input.options.discourseStateService.applySuppressedEmissionState({
    workingMemory: input.suppressionActionResult.workingMemory,
    reason: input.suppressionReason,
    origin: input.turnInput.origin,
    sourceStreamEntryId: input.suppressionMarker.id,
    sourceStreamEntryIds: input.sourceUserEntryIds,
    turnId: input.turnId,
    sessionId: input.sessionId,
  });

  if (input.options.tracer.enabled) {
    input.options.tracer.emit("post_generation.rejected", {
      turnId: input.turnId,
      session_id: input.sessionId,
      reason: input.suppressionReason,
      streamEntryId: input.suppressionMarker.id,
      source: input.traceSource,
      classified: input.classified,
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
    sessionId: input.sessionId,
    commitment: input.correctiveCommitment,
    supersession: input.correctiveCommitmentSupersession,
    retirement: input.correctiveCommitmentRetirement,
    appendHookFailureEvent: input.appendHookFailureEvent,
  });
  archiveInactiveParticipantActions({
    options: input.options,
    turnId: input.turnId,
    sessionId: input.sessionId,
    turnCounter: actionLifecycleTurnCounter(input.turnInput, input.archiveWorkingMemory),
  });
  if (input.responseTo !== undefined) {
    startTerminalLiveIngestion({
      options: input.options,
      sessionId: input.sessionId,
      responseTo: input.responseTo,
      terminalEntry: input.suppressionMarker,
    });
  }

  return suppressedTurnPhaseResult({
    turnId: input.turnId,
    mode: input.perceptionMode,
    emission: input.suppressionEmission,
    thoughts: [],
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "suppressed",
    },
    referencedEpisodeIds: [],
    retrievedEpisodeIds: [],
    toolCalls: [],
    terminalOutcome: input.terminalOutcome,
  });
}

export async function suppressFromClosureLoopPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  streamWriter: StreamWriter;
  appendHookFailureEvent: AppendHookFailureEvent;
  turnId: string;
  sessionId: SessionId;
  turnInput: TurnPhaseInput;
  workingMemory: WorkingMemory;
  persistedUserEntryId?: StreamEntry["id"];
  sourceUserEntryIds?: readonly StreamEntryId[];
  responseTo?: StreamResponseTo;
  correctiveCommitment: CorrectiveCommitment;
  correctiveCommitmentSupersession: CorrectiveCommitmentSupersession;
  correctiveCommitmentRetirement: CorrectiveCommitmentRetirement;
  perceptionMode: CognitiveMode;
  reason: string;
}): Promise<TurnPhaseResult> {
  let workingMemory = input.options.discourseStateService.markClosureLoopNamed({
    workingMemory: input.workingMemory,
    reason: input.reason,
    turnId: input.turnId,
    sourceStreamEntryId: input.persistedUserEntryId,
    sourceStreamEntryIds: input.sourceUserEntryIds,
    sessionId: input.sessionId,
  });
  workingMemory = input.options.discourseStateService.setStopState({
    workingMemory,
    provenance: "finalizer_no_output",
    sourceStreamEntryId: input.persistedUserEntryId,
    sourceStreamEntryIds: input.sourceUserEntryIds,
    reason: "Closure loop already named; suppressing another closure-only turn.",
    turnId: input.turnId,
    sessionId: input.sessionId,
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
  const suppressionMarker = await traceTurnPhase({
    tracer: input.options.tracer,
    clock: input.options.clock,
    turnId: input.turnId,
    sessionId: input.sessionId,
    phase: "persist",
    sub: "suppressed_closure",
    run: () =>
      input.options.discourseStateService.appendSuppressionMarker({
        streamWriter: input.streamWriter,
        reason: "finalizer_no_output",
        userEntryId: input.persistedUserEntryId,
        userEntryIds: input.sourceUserEntryIds,
        responseTo: input.responseTo,
        turnId: input.turnId,
        audience: input.turnInput.audience,
      }),
    completedSub: (entry) => `entry=${entry.kind}`,
  });
  const suppressionEmission: TurnEmission = {
    kind: "suppressed",
    reason: "finalizer_no_output",
    markerEntryId: suppressionMarker.id,
  };
  advanceChatResponseWatermark({
    options: input.options,
    sessionId: input.sessionId,
    responseTo: input.responseTo,
  });

  return finalizeSuppressedTurn({
    options: input.options,
    streamWriter: input.streamWriter,
    appendHookFailureEvent: input.appendHookFailureEvent,
    turnId: input.turnId,
    sessionId: input.sessionId,
    turnInput: input.turnInput,
    archiveWorkingMemory: input.workingMemory,
    sourceUserEntryIds: input.sourceUserEntryIds,
    responseTo: input.responseTo,
    correctiveCommitment: input.correctiveCommitment,
    correctiveCommitmentSupersession: input.correctiveCommitmentSupersession,
    correctiveCommitmentRetirement: input.correctiveCommitmentRetirement,
    perceptionMode: input.perceptionMode,
    suppressionReason: "finalizer_no_output",
    suppressionActionResult,
    suppressionMarker,
    suppressionEmission,
    traceSource: "closure_loop",
    classified: true,
    terminalOutcome: "suppressed_closure",
  });
}

export async function suppressFromGenerationGatePhase(input: {
  options: TurnPhaseCoordinatorOptions;
  streamWriter: StreamWriter;
  appendHookFailureEvent: AppendHookFailureEvent;
  turnId: string;
  sessionId: SessionId;
  turnInput: TurnPhaseInput;
  workingMemory: WorkingMemory;
  persistedUserEntryId?: StreamEntry["id"];
  sourceUserEntryIds?: readonly StreamEntryId[];
  responseTo?: StreamResponseTo;
  gateResult: Awaited<ReturnType<GenerationGate["evaluate"]>>;
  correctiveCommitment: CorrectiveCommitment;
  correctiveCommitmentSupersession: CorrectiveCommitmentSupersession;
  correctiveCommitmentRetirement: CorrectiveCommitmentRetirement;
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
      sourceStreamEntryIds: input.sourceUserEntryIds,
      reason: input.gateResult.explanation,
      turnId: input.turnId,
      sessionId: input.sessionId,
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
  const suppressionMarker = await traceTurnPhase({
    tracer: input.options.tracer,
    clock: input.options.clock,
    turnId: input.turnId,
    sessionId: input.sessionId,
    phase: "persist",
    sub: "suppressed_generation_gate",
    run: () =>
      input.options.discourseStateService.appendSuppressionMarker({
        streamWriter: input.streamWriter,
        reason: suppressionReason,
        userEntryId: input.persistedUserEntryId,
        userEntryIds: input.sourceUserEntryIds,
        responseTo: input.responseTo,
        turnId: input.turnId,
        audience: input.turnInput.audience,
      }),
    completedSub: (entry) => `entry=${entry.kind}`,
  });
  const suppressionEmission: TurnEmission = {
    kind: "suppressed",
    reason: suppressionReason,
    markerEntryId: suppressionMarker.id,
  };
  advanceChatResponseWatermark({
    options: input.options,
    sessionId: input.sessionId,
    responseTo: input.responseTo,
  });

  return finalizeSuppressedTurn({
    options: input.options,
    streamWriter: input.streamWriter,
    appendHookFailureEvent: input.appendHookFailureEvent,
    turnId: input.turnId,
    sessionId: input.sessionId,
    turnInput: input.turnInput,
    archiveWorkingMemory: input.workingMemory,
    sourceUserEntryIds: input.sourceUserEntryIds,
    responseTo: input.responseTo,
    correctiveCommitment: input.correctiveCommitment,
    correctiveCommitmentSupersession: input.correctiveCommitmentSupersession,
    correctiveCommitmentRetirement: input.correctiveCommitmentRetirement,
    perceptionMode: input.perceptionMode,
    suppressionReason,
    suppressionActionResult,
    suppressionMarker,
    suppressionEmission,
    traceSource: "generation_gate",
    classified: input.gateResult.classified,
    terminalOutcome: "suppressed_generation_gate",
  });
}

async function suppressFromActionPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  streamWriter: StreamWriter;
  appendHookFailureEvent: AppendHookFailureEvent;
  turnId: string;
  sessionId: SessionId;
  turnInput: TurnPhaseInput;
  actionResult: Awaited<ReturnType<TurnActionCoordinator["run"]>>["actionResult"];
  actionEmission: Extract<PendingTurnEmission, { kind: "suppressed" }>;
  persistedAgentEntry: StreamEntry;
  sourceUserEntryIds?: readonly StreamEntryId[];
  responseTo?: StreamResponseTo;
  correctiveCommitment: CorrectiveCommitment;
  correctiveCommitmentSupersession: CorrectiveCommitmentSupersession;
  correctiveCommitmentRetirement: CorrectiveCommitmentRetirement;
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
    ...(input.actionEmission.decision_rationale === undefined
      ? {}
      : { decision_rationale: input.actionEmission.decision_rationale }),
    ...(input.actionEmission.finalizer_invalid_tool === undefined
      ? {}
      : { finalizer_invalid_tool: input.actionEmission.finalizer_invalid_tool }),
  };
  let suppressedWorkingMemory = input.options.discourseStateService.applySuppressedEmissionState({
    workingMemory: input.actionResult.workingMemory,
    reason: input.actionEmission.reason,
    origin: input.turnInput.origin,
    sourceStreamEntryId: input.persistedAgentEntry.id,
    sourceStreamEntryIds: input.sourceUserEntryIds,
    turnId: input.turnId,
    sessionId: input.sessionId,
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
      session_id: input.sessionId,
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
      ...(input.actionEmission.finalizer_invalid_tool === undefined
        ? {}
        : { finalizer_invalid_tool: input.actionEmission.finalizer_invalid_tool }),
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
    sessionId: input.sessionId,
    commitment: input.correctiveCommitment,
    supersession: input.correctiveCommitmentSupersession,
    retirement: input.correctiveCommitmentRetirement,
    appendHookFailureEvent: input.appendHookFailureEvent,
  });
  archiveInactiveParticipantActions({
    options: input.options,
    turnId: input.turnId,
    sessionId: input.sessionId,
    turnCounter: actionLifecycleTurnCounter(input.turnInput, input.actionResult.workingMemory),
  });
  if (input.responseTo !== undefined) {
    startTerminalLiveIngestion({
      options: input.options,
      sessionId: input.sessionId,
      responseTo: input.responseTo,
      terminalEntry: input.persistedAgentEntry,
    });
  }

  return suppressedTurnPhaseResult({
    turnId: input.turnId,
    mode: input.perceptionMode,
    emission: suppressionEmission,
    thoughts: input.deliberation.thoughts,
    usage: input.deliberation.usage,
    retrievedEpisodeIds: input.deliberation.retrievedEpisodes.map((result) => result.episode.id),
    referencedEpisodeIds: [...(input.deliberation.referencedEpisodeIds ?? [])],
    toolCalls: [...input.actionResult.tool_calls],
    terminalOutcome: "suppressed_action",
  });
}

function suppressedTurnPhaseResult(input: {
  turnId: string;
  mode: CognitiveMode;
  emission: TurnEmission;
  thoughts: string[];
  usage: TurnPhaseResult["usage"];
  retrievedEpisodeIds: string[];
  referencedEpisodeIds: string[];
  toolCalls: TurnPhaseResult["toolCalls"];
  terminalOutcome: TurnPhaseResult["terminalOutcome"];
}): TurnPhaseResult {
  return {
    turn_id: input.turnId,
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
    terminalOutcome: input.terminalOutcome,
  };
}
