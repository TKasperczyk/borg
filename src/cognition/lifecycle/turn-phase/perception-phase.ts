import {
  FrameAnomalyClassifier,
  isFrameAnomaly,
  type ActualFrameAnomalyClassification,
  type FrameAnomalyClassification,
  type FrameAnomalyConversationContext,
  type FrameAnomalyKind,
} from "../../frame-anomaly/index.js";
import { isCreatorInOperatorContext } from "../../authority.js";
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
import {
  QUARANTINED_USER_ENTRY_EVENT,
  type StreamEntry,
  type StreamWriter,
} from "../../../stream/index.js";
import type { SessionSourceType } from "../../../sessions/index.js";
import type { SessionId, StreamEntryId } from "../../../util/ids.js";
import type { TurnPhaseCoordinatorOptions } from "./types.js";
import {
  FRAME_ANOMALY_ANNOTATION_EVENT,
  frameAnomalyAnnotationFromEntry,
} from "../../../stream/frame-anomaly.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromMetadata,
  unknownMemoryDisclosureLabel,
} from "../../../memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../../../memory/common/disclosure-serializers.js";

export type FrameAnomalyDisposition =
  | "none"
  | "trusted_operator_control"
  | "trusted_peer_channel"
  | "annotate"
  | "quarantine";

export type FrameAnomalyPhaseResult = {
  classification: FrameAnomalyClassification | null;
  disposition: FrameAnomalyDisposition;
  actionableFrameAnomaly: ActualFrameAnomalyClassification | null;
};

const TRUSTED_PEER_CHANNEL_FRAME_ANOMALY_KINDS: ReadonlySet<FrameAnomalyKind> = new Set([
  "assistant_self_claim_in_user_role",
  "roleplay_inversion",
  // A machine peer may truthfully claim authorship of earlier user-role dialogue.
  // On non-peer channels this kind still guards against gaslighting about who spoke.
  "agent_authorship_claim",
]);

function isTrustedPeerChannelFrameAnomalyKind(kind: FrameAnomalyKind): boolean {
  return TRUSTED_PEER_CHANNEL_FRAME_ANOMALY_KINDS.has(kind);
}

function isAuthorizedFrameAnomalyPeerChannel(input: {
  options: TurnPhaseCoordinatorOptions;
  sessionSourceType: SessionSourceType | null;
}): boolean {
  if (input.sessionSourceType === null) {
    return false;
  }

  return input.options.config.frameAnomaly.peerChannelSourceTypes.includes(input.sessionSourceType);
}

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
  sessionSourceType: SessionSourceType | null;
  persistedUserEntryId?: StreamEntryId;
  sourceUserEntryIds?: readonly StreamEntryId[];
  sourceUserEntries?: readonly StreamEntry[];
  streamWriter: StreamWriter;
}): Promise<FrameAnomalyPhaseResult> {
  const sourceUserEntryIds =
    input.sourceUserEntryIds === undefined || input.sourceUserEntryIds.length === 0
      ? input.persistedUserEntryId === undefined
        ? []
        : [input.persistedUserEntryId]
      : [...input.sourceUserEntryIds];

  if (!input.isUserTurn || sourceUserEntryIds.length === 0) {
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
    isCreatorInOperatorContext({
      currentSenderBorgRole: input.currentSenderBorgRole,
      sessionAudienceRole: input.sessionAudienceRole,
    });
  // Opus 5.0 verdict: IN-SCOPE -- the harness lacked the structural concept of
  // an authorized AI-peer channel, so this is a disposition policy fix rather
  // than model-judgment machinery.
  const trustedPeerChannel =
    actualFrameAnomaly !== null &&
    !trustedOperatorControl &&
    isAuthorizedFrameAnomalyPeerChannel({
      options: input.options,
      sessionSourceType: input.sessionSourceType,
    }) &&
    isTrustedPeerChannelFrameAnomalyKind(actualFrameAnomaly.kind);
  const disposition: FrameAnomalyDisposition =
    actualFrameAnomaly === null
      ? "none"
      : trustedOperatorControl
        ? "trusted_operator_control"
        : trustedPeerChannel
          ? "trusted_peer_channel"
          : isAuthorizedFrameAnomalyPeerChannel(input)
            ? "annotate"
            : "quarantine";
  // Annotation changes recall, not the existing action-authorization policy.
  const actionableFrameAnomaly =
    disposition === "quarantine" || disposition === "annotate" ? actualFrameAnomaly : null;

  traceFrameAnomalyDisposition({
    options: input.options,
    turnId: input.turnId,
    sessionId: input.sessionId,
    classification,
    disposition,
    currentSenderBorgRole: input.currentSenderBorgRole,
    sessionAudienceRole: input.sessionAudienceRole,
    sessionSourceType: input.sessionSourceType,
  });

  if (
    (disposition === "quarantine" || disposition === "annotate") &&
    actionableFrameAnomaly !== null
  ) {
    await appendFrameAnomalyEvents({
      options: input.options,
      appendHookFailureEvent: input.appendHookFailureEvent,
      streamWriter: input.streamWriter,
      turnId: input.turnId,
      sessionId: input.sessionId,
      sourceUserEntryIds,
      classification: actionableFrameAnomaly,
      annotationSources:
        disposition === "annotate"
          ? {
              sourceType: input.sessionSourceType!,
              entries: input.sourceUserEntries ?? [],
            }
          : undefined,
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
  sessionSourceType: SessionSourceType | null;
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
    session_source_type: input.sessionSourceType,
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
  sourceUserEntryIds?: readonly StreamEntryId[];
  sourceUserEntries?: readonly StreamEntry[];
  workingMemory: WorkingMemory;
  streamWriter: StreamWriter;
}): Promise<ClosureLoopAssessment | null> {
  const sourceUserEntryIds =
    input.sourceUserEntryIds === undefined || input.sourceUserEntryIds.length === 0
      ? input.persistedUserEntryId === undefined
        ? []
        : [input.persistedUserEntryId]
      : [...input.sourceUserEntryIds];

  if (!input.isUserTurn || sourceUserEntryIds.length === 0) {
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
    currentUserEntryId: sourceUserEntryIds[0]!,
    currentUserMessages: (input.sourceUserEntries ?? []).map((entry) => ({
      entryId: entry.id,
      content: typeof entry.content === "string" ? entry.content : JSON.stringify(entry.content),
      ts: entry.timestamp,
    })),
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
      currentUserRefs: sourceUserEntryIds,
      priorClosureLoopActive: activeClosureLoop !== null,
    });
  }

  return assessClosureLoopClassification({
    classification,
    suppliedMessages: messages,
    currentUserRefs: sourceUserEntryIds,
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
  sourceUserEntryIds: readonly StreamEntryId[];
  classification: ActualFrameAnomalyClassification;
  annotationSources?: { sourceType: string; entries: readonly StreamEntry[] };
}): Promise<void> {
  const scalarUserEntryId =
    input.sourceUserEntryIds.length === 1 ? input.sourceUserEntryIds[0] : undefined;

  try {
    if (input.annotationSources !== undefined) {
      const sources = new Map(input.annotationSources.entries.map((entry) => [entry.id, entry]));
      const labels = new Map(
        input.sourceUserEntryIds.map((id) => {
          const source = sources.get(id);
          return [
            id,
            memoryDisclosureLabelFromMetadata(source?.metadata?.disclosure_label) ??
              (source === undefined
                ? undefined
                : input.options.sourceStreamAudienceDisclosureResolver?.resolveStreamEntryLabel(
                    source,
                  )) ??
              unknownMemoryDisclosureLabel(),
          ] as const;
        }),
      );
      // One classifier rationale can cite several sources. Its disclosure label
      // covers the whole batch, while each target retains its original label.
      const rationaleDisclosure = memoryDisclosurePayloadFields(
        combineMemoryDisclosureLabels([...labels.values()]),
      );
      const annotations = await input.streamWriter.appendMany(
        input.sourceUserEntryIds.map((id) => {
          const source = sources.get(id);
          return {
            kind: "internal_event" as const,
            turn_id: input.turnId,
            content: {
              event: FRAME_ANOMALY_ANNOTATION_EVENT,
              version: 1,
              source_stream_entry_id: id,
              source_session_id: input.sessionId,
              source_type: input.annotationSources!.sourceType,
              source_stream_entry_ids: [...input.sourceUserEntryIds],
              kind: input.classification.kind,
              confidence: input.classification.confidence,
              rationale: input.classification.rationale,
              classifier_model: input.options.config.anthropic.models.recallExpansion,
              classifier_turn_id: input.turnId,
              classifier_source_event_id: null,
              restores_quarantine_marker_ids: [],
              audience: source?.audience ?? null,
              sender_entity_id: source?.sender_entity_id ?? null,
              source_disclosure_label: memoryDisclosurePayloadFields(labels.get(id)!)
                .disclosure_label,
              ...rationaleDisclosure,
            },
          };
        }),
      );
      // These already-persisted source objects feed the current-turn ledger. Later
      // reads obtain the identical annotation from the index (or JSONL replay).
      for (const event of annotations) {
        const annotation = frameAnomalyAnnotationFromEntry(event);
        if (annotation === null) continue;
        for (const source of input.annotationSources.entries) {
          if (source.id === annotation.source_stream_entry_id) {
            source.frame_anomalies = [...(source.frame_anomalies ?? []), annotation];
          }
        }
      }
    } else {
      // Two rows with the same payload under two event names. Only
      // QUARANTINED_USER_ENTRY_EVENT is a strike marker (src/stream/turn-status.ts), so the
      // `frame_anomaly_gate` copy stays active after the quarantine takes effect -- and with it
      // the classifier rationale, which quotes the message it struck.
      await input.streamWriter.appendMany([
        {
          kind: "internal_event",
          turn_id: input.turnId,
          content: {
            event: "frame_anomaly_gate",
            turn_id: input.turnId,
            ...(scalarUserEntryId === undefined
              ? {}
              : { source_stream_entry_id: scalarUserEntryId }),
            source_stream_entry_ids: [...input.sourceUserEntryIds],
            cited_stream_entry_ids: [...input.sourceUserEntryIds],
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
            ...(scalarUserEntryId === undefined
              ? {}
              : { source_stream_entry_id: scalarUserEntryId }),
            source_stream_entry_ids: [...input.sourceUserEntryIds],
            cited_stream_entry_ids: [...input.sourceUserEntryIds],
            kind: input.classification.kind,
            confidence: input.classification.confidence,
            rationale: input.classification.rationale,
          },
        },
      ]);
    }

    if (input.options.tracer.enabled) {
      input.options.tracer.emit("frame_anomaly.transitioned", {
        turnId: input.turnId,
        session_id: input.sessionId,
        kind: input.classification.kind,
        sourceStreamEntryIds: [...input.sourceUserEntryIds],
        disposition: input.annotationSources === undefined ? "quarantine" : "annotate",
      });
    }
  } catch (error) {
    await input.appendHookFailureEvent(input.streamWriter, "frame_anomaly_gate_event", error, {
      turnId: input.turnId,
    });
  }
}
