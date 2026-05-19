import {
  FrameAnomalyClassifier,
  classifyFrameAnomalyDegradedFallback,
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
import type { WorkingMemory } from "../../../memory/working/index.js";
import { QUARANTINED_USER_ENTRY_EVENT, type StreamWriter } from "../../../stream/index.js";
import type { StreamEntryId } from "../../../util/ids.js";
import type { TurnPhaseCoordinatorOptions } from "./types.js";

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
  isUserTurn: boolean;
  userMessage: string;
  recentHistory: readonly RecencyMessage[];
  conversationContext?: FrameAnomalyConversationContext;
  persistedUserEntryId?: StreamEntryId;
  streamWriter: StreamWriter;
}): Promise<FrameAnomalyClassification | null> {
  if (!input.isUserTurn || input.persistedUserEntryId === undefined) {
    return null;
  }

  const classifier = new FrameAnomalyClassifier({
    llmClient: input.llmClient,
    model: input.options.config.anthropic.models.recallExpansion,
    tracer: input.options.tracer,
    turnId: input.turnId,
    onDegraded: (reason, error) => {
      if (!input.options.tracer.enabled) {
        return;
      }

      input.options.tracer.emit("frame_anomaly_classifier_degraded", {
        turnId: input.turnId,
        reason,
        ...(input.options.tracer.includePayloads && error !== undefined
          ? { error: error instanceof Error ? error.message : String(error) }
          : {}),
      });
    },
  });
  let classification = await classifier.classify({
    userMessage: input.userMessage,
    recentHistory: input.recentHistory,
    ...(input.conversationContext === undefined
      ? {}
      : { conversationContext: input.conversationContext }),
  });

  if (classification.status === "degraded") {
    const fallback = classifyFrameAnomalyDegradedFallback(input.userMessage);

    if (fallback.matched) {
      if (input.options.tracer.enabled) {
        input.options.tracer.emit("frame_anomaly_degraded_fallback_match", {
          turnId: input.turnId,
          pattern: fallback.pattern,
          kind: fallback.kind,
        });
      }

      classification = fallback.classification;
    } else if (input.options.tracer.enabled) {
      input.options.tracer.emit("frame_anomaly_degraded_fallback_normal", {
        turnId: input.turnId,
      });
    }
  }

  if (isFrameAnomaly(classification)) {
    await appendFrameAnomalyEvents({
      options: input.options,
      appendHookFailureEvent: input.appendHookFailureEvent,
      streamWriter: input.streamWriter,
      turnId: input.turnId,
      persistedUserEntryId: input.persistedUserEntryId,
      classification,
    });
  }

  return classification;
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
    onDegraded: (reason, error) => {
      if (!input.options.tracer.enabled) {
        return;
      }

      input.options.tracer.emit("closure_loop_classifier_degraded", {
        turnId: input.turnId,
        reason,
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
      input.options.tracer.emit("frame_anomaly_quarantine_appended", {
        turnId: input.turnId,
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
