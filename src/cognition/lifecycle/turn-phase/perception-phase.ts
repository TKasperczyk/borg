import {
  FrameAnomalyClassifier,
  isFrameAnomaly,
  type ActualFrameAnomalyClassification,
  type FrameAnomalyClassification,
  type FrameAnomalyConversationContext,
} from "../../frame-anomaly/index.js";
import {
  ClosureLoopClassifier,
  assessClosureLoopClassification,
  assessDegradedClosureLoopFallback,
  buildClosureLoopMessageWindow,
  type ClosureLoopAssessment,
} from "../../generation/closure-loop.js";
import type { RecencyMessage } from "../../recency/index.js";
import type { LLMClient } from "../../../llm/index.js";
import type { BorgRole } from "../../../memory/commitments/index.js";
import type { WorkingMemory } from "../../../memory/working/index.js";
import type { SessionAudienceRole } from "../../../sessions/index.js";
import { QUARANTINED_USER_ENTRY_EVENT, type StreamWriter } from "../../../stream/index.js";
import type { SessionId, StreamEntryId } from "../../../util/ids.js";
import type { TurnPhaseCoordinatorOptions } from "./types.js";

export type FrameAnomalyDisposition = "none" | "trusted_operator_control" | "quarantine";

export type FrameAnomalyPhaseResult = {
  classification: FrameAnomalyClassification | null;
  disposition: FrameAnomalyDisposition;
  actionableFrameAnomaly: ActualFrameAnomalyClassification | null;
};

export async function classifyFrameAnomalyPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  appendHookFailureEvent: (
    streamWriter: StreamWriter,
    hook: string,
    error: unknown,
    details?: Record<string, unknown>,
  ) => Promise<void>;
  llmClient: LLMClient;
  turnId: string;
  sessionId: SessionId;
  isUserTurn: boolean;
  userMessage: string;
  recentHistory: readonly RecencyMessage[];
  conversationContext?: FrameAnomalyConversationContext;
  currentSenderBorgRole: BorgRole | null;
  sessionAudienceRole: SessionAudienceRole;
  persistedUserEntryId?: StreamEntryId;
  streamWriter: StreamWriter;
}): Promise<FrameAnomalyPhaseResult> {
  if (!input.isUserTurn || input.persistedUserEntryId === undefined) {
    return {
      classification: null,
      disposition: "none",
      actionableFrameAnomaly: null,
    };
  }

  const classifier = new FrameAnomalyClassifier({
    llmClient: input.llmClient,
    model: input.options.config.anthropic.models.recallExpansion,
    tracer: input.options.tracer,
    turnId: input.turnId,
    sessionId: input.sessionId,
    onDegraded: (reason, error) => {
      if (!input.options.tracer.enabled) {
        return;
      }

      input.options.tracer.emit("frame_anomaly.degraded", {
        turnId: input.turnId,
        session_id: input.sessionId,
        reason,
        ...(input.options.tracer.includePayloads && error !== undefined
          ? { error: error instanceof Error ? error.message : String(error) }
          : {}),
      });
    },
  });
  const classification = await classifier.classify({
    userMessage: input.userMessage,
    recentHistory: input.recentHistory,
    ...(input.conversationContext === undefined
      ? {}
      : { conversationContext: input.conversationContext }),
  });

  if (classification.status === "degraded" && input.options.tracer.enabled) {
    input.options.tracer.emit("frame_anomaly.degraded_fail_open", {
      turnId: input.turnId,
      session_id: input.sessionId,
      reason: classification.reason,
    });
  }

  const actualFrameAnomaly = isFrameAnomaly(classification) ? classification : null;
  const trustedOperatorControl =
    actualFrameAnomaly !== null &&
    input.sessionAudienceRole === "operator" &&
    input.currentSenderBorgRole === "creator";
  const disposition: FrameAnomalyDisposition =
    actualFrameAnomaly === null
      ? "none"
      : trustedOperatorControl
        ? "trusted_operator_control"
        : "quarantine";
  const actionableFrameAnomaly = disposition === "quarantine" ? actualFrameAnomaly : null;

  traceFrameAnomalyDisposition({
    options: input.options,
    turnId: input.turnId,
    sessionId: input.sessionId,
    classification,
    disposition,
    currentSenderBorgRole: input.currentSenderBorgRole,
    sessionAudienceRole: input.sessionAudienceRole,
  });

  if (disposition === "quarantine" && actionableFrameAnomaly !== null) {
    await appendFrameAnomalyEvents({
      options: input.options,
      appendHookFailureEvent: input.appendHookFailureEvent,
      streamWriter: input.streamWriter,
      turnId: input.turnId,
      sessionId: input.sessionId,
      persistedUserEntryId: input.persistedUserEntryId,
      classification: actionableFrameAnomaly,
    });
  }

  return {
    classification,
    disposition,
    actionableFrameAnomaly,
  };
}

function traceFrameAnomalyDisposition(input: {
  options: TurnPhaseCoordinatorOptions;
  turnId: string;
  sessionId: SessionId;
  classification: FrameAnomalyClassification;
  disposition: FrameAnomalyDisposition;
  currentSenderBorgRole: BorgRole | null;
  sessionAudienceRole: SessionAudienceRole;
}): void {
  if (!input.options.tracer.enabled) {
    return;
  }

  input.options.tracer.emit("frame_anomaly.disposition", {
    turnId: input.turnId,
    session_id: input.sessionId,
    disposition: input.disposition,
    status: input.classification.status,
    kind: input.classification.status === "ok" ? input.classification.kind : null,
    session_audience_role: input.sessionAudienceRole,
    current_sender_borg_role: input.currentSenderBorgRole,
  });
}

export async function classifyClosureLoopPhase(input: {
  options: TurnPhaseCoordinatorOptions;
  appendHookFailureEvent: (
    streamWriter: StreamWriter,
    hook: string,
    error: unknown,
    details?: Record<string, unknown>,
  ) => Promise<void>;
  llmClient: LLMClient;
  turnId: string;
  sessionId: SessionId;
  isUserTurn: boolean;
  userMessage: string;
  recentHistory: readonly RecencyMessage[];
  persistedUserEntryId?: StreamEntryId;
  workingMemory: WorkingMemory;
  streamWriter: StreamWriter;
}): Promise<ClosureLoopAssessment | null> {
  if (!input.isUserTurn || input.persistedUserEntryId === undefined) {
    return null;
  }

  const activeClosureLoop = input.workingMemory.discourse_state?.closure_loop ?? null;
  const closurePressureHistory =
    input.workingMemory.discourse_state?.closure_pressure_history ?? [];

  if (
    activeClosureLoop === null &&
    closurePressureHistory.length === 0 &&
    input.recentHistory.length < 4
  ) {
    return null;
  }

  const messages = buildClosureLoopMessageWindow({
    recentHistory: input.recentHistory,
    currentUserMessage: input.userMessage,
    currentUserEntryId: input.persistedUserEntryId,
  });
  const classifier = new ClosureLoopClassifier({
    llmClient: input.llmClient,
    model: input.options.config.anthropic.models.recallExpansion,
    tracer: input.options.tracer,
    turnId: input.turnId,
    sessionId: input.sessionId,
    onDegraded: (reason, error, metadata) => {
      if (!input.options.tracer.enabled) {
        return;
      }

      input.options.tracer.emit("closure_loop.degraded", {
        turnId: input.turnId,
        session_id: input.sessionId,
        label: "closure_loop_classifier",
        reason,
        stopReason: metadata?.stopReason ?? null,
        ...(input.options.tracer.includePayloads && error !== undefined
          ? { error: error instanceof Error ? error.message : String(error) }
          : {}),
      });
    },
  });
  const classification = await classifier.classify({
    messages,
  });

  if (classification.degraded) {
    await input.appendHookFailureEvent(input.streamWriter, "closure_loop_classifier", null, {
      turnId: input.turnId,
      reason: classification.rationale,
    });

    return assessDegradedClosureLoopFallback({
      suppliedMessages: messages,
      currentUserRef: input.persistedUserEntryId,
      priorClosureLoopActive: activeClosureLoop !== null,
    });
  }

  return assessClosureLoopClassification({
    classification,
    suppliedMessages: messages,
    currentUserRef: input.persistedUserEntryId,
  });
}

async function appendFrameAnomalyEvents(input: {
  options: TurnPhaseCoordinatorOptions;
  appendHookFailureEvent: (
    streamWriter: StreamWriter,
    hook: string,
    error: unknown,
    details?: Record<string, unknown>,
  ) => Promise<void>;
  streamWriter: StreamWriter;
  turnId: string;
  sessionId: SessionId;
  persistedUserEntryId: StreamEntryId;
  classification: ActualFrameAnomalyClassification;
}): Promise<void> {
  try {
    await input.streamWriter.appendMany([
      {
        kind: "internal_event",
        turn_id: input.turnId,
        content: {
          event: "frame_anomaly_gate",
          turn_id: input.turnId,
          source_stream_entry_id: input.persistedUserEntryId,
          cited_stream_entry_ids: [input.persistedUserEntryId],
          kind: input.classification.kind,
          confidence: input.classification.confidence,
          rationale: input.classification.rationale,
        },
      },
      {
        kind: "internal_event",
        turn_id: input.turnId,
        content: {
          event: QUARANTINED_USER_ENTRY_EVENT,
          turn_id: input.turnId,
          source_stream_entry_id: input.persistedUserEntryId,
          cited_stream_entry_ids: [input.persistedUserEntryId],
          kind: input.classification.kind,
          confidence: input.classification.confidence,
          rationale: input.classification.rationale,
        },
      },
    ]);

    if (input.options.tracer.enabled) {
      input.options.tracer.emit("frame_anomaly.transitioned", {
        turnId: input.turnId,
        session_id: input.sessionId,
        kind: input.classification.kind,
        sourceStreamEntryId: input.persistedUserEntryId,
      });
    }
  } catch (error) {
    await input.appendHookFailureEvent(input.streamWriter, "frame_anomaly_gate_event", error, {
      turnId: input.turnId,
    });
  }
}
