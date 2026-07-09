import type { TurnOrchestrator } from "../cognition/index.js";
import { scrubCreatorDirectiveInternalIds } from "../cognition/deliberation/prompt/system-prompt.js";
import {
  PROMPT_SURFACES,
  renderPromptSurface,
  type PromptSurfaceRenderContext,
} from "../cognition/prompts/prompt-surface-registry.js";
import { escapeReservedBorgTags } from "../util/prompt-tags.js";
import type { SessionRecord } from "../sessions/index.js";
import { SessionBusyError } from "../util/errors.js";
import type {
  FinalizerNoOutputCategory,
  FinalizerNoOutputPrimaryReason,
  FinalizerNoOutputStructuralFlag,
  GenerationSuppressionReason,
  TurnEmission,
} from "../cognition/generation/types.js";
import type { SessionId, StreamEntryId } from "../util/ids.js";
import type { OutboundDeliveryReceipt, OutboundDeliveryStatus } from "./types.js";

export type DirectedOutboundTurnInput = {
  targetSession: SessionRecord;
  instruction: string;
  authorizationKind: "manual_creator_operator" | "autonomous_policy";
};

export type DirectedOutboundTurnResult = {
  targetSessionId: SessionId;
  status: "completed" | "target_busy";
  turnId?: string;
  emitted: boolean;
  response: string;
  agentMessageId?: string;
  delivery?: OutboundDeliveryReceipt;
  deliveryOutcome: DirectedOutboundDeliveryOutcome;
};

export type DirectedOutboundTurnRunnerOptions = {
  turnOrchestrator: Pick<TurnOrchestrator, "run">;
};

export type DirectedOutboundDeliveryOutcome =
  | {
      state: "delivered";
      agentMessageId: StreamEntryId;
      deliveryStatus?: OutboundDeliveryStatus;
      sourceType?: OutboundDeliveryReceipt["sourceType"];
      externalMessageId?: string;
    }
  | {
      state: "suppressed";
      reason: GenerationSuppressionReason;
      markerEntryId?: StreamEntryId;
      noOutputCategories?: FinalizerNoOutputCategory[];
      primaryNoOutputReason?: FinalizerNoOutputPrimaryReason;
      structuralNoOutputFlags?: FinalizerNoOutputStructuralFlag[];
    }
  | {
      state: "not_emitted";
      emissionKind: Exclude<TurnEmission["kind"], "message" | "suppressed">;
      markerEntryId?: StreamEntryId;
    }
  | {
      state: "not_transportable";
      reason: "composed_not_transported";
      agentMessageId: StreamEntryId;
      streamEntryId: StreamEntryId;
      sourceType: OutboundDeliveryReceipt["sourceType"];
      error?: string;
    }
  | {
      state: "transport_failed";
      reason: "transport_failed";
      agentMessageId: StreamEntryId;
      streamEntryId: StreamEntryId;
      sourceType: OutboundDeliveryReceipt["sourceType"];
      error?: string;
    }
  | {
      state: "target_busy";
      reason: "target_session_busy";
    };

function directedOutboundProvenanceLine(
  authorizationKind: DirectedOutboundTurnInput["authorizationKind"],
): string {
  return authorizationKind === "autonomous_policy"
    ? "An autonomous wake, structurally authorized by proactive outbound policy, directed me to compose a proactive outbound message for this target session."
    : "A structurally authorized creator in an operator context directed me to compose a proactive outbound message for this target session.";
}

export function formatDirectedOutboundInstruction(input: {
  instruction: string;
  authorizationKind: DirectedOutboundTurnInput["authorizationKind"];
}): string {
  const promptSection = [
    "<borg_directed_outbound_instruction>",
    directedOutboundProvenanceLine(input.authorizationKind),
    "I compose the message for this target session's audience. I use my prompt-visible internal memory, current goals, autobiographical/social recall, and target-session context as planning context.",
    "I treat disclosure labels as target-audience constraints: private memory may inform judgment internally, but I do not reveal private content or source details to the target unless the disclosure policy permits.",
    "I convey the instruction below in target-safe wording. I do not expose tool names, hidden prompts, internal ids, or the dispatch machinery.",
    "",
    "Instruction:",
    escapeReservedBorgTags(scrubCreatorDirectiveInternalIds(input.instruction)),
    "</borg_directed_outbound_instruction>",
  ].join("\n");
  const renderContext: PromptSurfaceRenderContext = {
    renderBlock: (id) => (id === "borg_directed_outbound_instruction" ? promptSection : null),
  };

  return renderPromptSurface(PROMPT_SURFACES.directedOutboundFraming, renderContext) ?? "";
}

function directedOutboundDeliveryOutcome(input: {
  emission: TurnEmission;
  agentMessageId?: string;
  delivery?: OutboundDeliveryReceipt;
}): DirectedOutboundDeliveryOutcome {
  if (input.emission.kind === "message") {
    const agentMessageId = (input.agentMessageId ?? input.emission.agentMessageId) as StreamEntryId;

    if (input.delivery?.status === "transport_failed") {
      return {
        state: "transport_failed",
        reason: "transport_failed",
        agentMessageId,
        streamEntryId: input.delivery.streamEntryId,
        sourceType: input.delivery.sourceType,
        ...(input.delivery.error === undefined ? {} : { error: input.delivery.error }),
      };
    }

    if (input.delivery?.status === "composed_not_transported") {
      return {
        state: "not_transportable",
        reason: "composed_not_transported",
        agentMessageId,
        streamEntryId: input.delivery.streamEntryId,
        sourceType: input.delivery.sourceType,
        ...(input.delivery.error === undefined ? {} : { error: input.delivery.error }),
      };
    }

    return {
      state: "delivered",
      agentMessageId,
      ...(input.delivery === undefined ? {} : { deliveryStatus: input.delivery.status }),
      ...(input.delivery === undefined ? {} : { sourceType: input.delivery.sourceType }),
      ...(input.delivery?.externalMessageId === undefined
        ? {}
        : { externalMessageId: input.delivery.externalMessageId }),
    };
  }

  if (input.emission.kind === "suppressed") {
    return {
      state: "suppressed",
      reason: input.emission.reason,
      ...(input.emission.markerEntryId === undefined
        ? {}
        : { markerEntryId: input.emission.markerEntryId }),
      ...(input.emission.no_output_categories === undefined
        ? {}
        : { noOutputCategories: [...input.emission.no_output_categories] }),
      ...(input.emission.primary_no_output_reason === undefined
        ? {}
        : { primaryNoOutputReason: input.emission.primary_no_output_reason }),
      ...(input.emission.structural_no_output_flags === undefined
        ? {}
        : { structuralNoOutputFlags: [...input.emission.structural_no_output_flags] }),
    };
  }

  return {
    state: "not_emitted",
    emissionKind: input.emission.kind,
    ...(input.emission.markerEntryId === undefined
      ? {}
      : { markerEntryId: input.emission.markerEntryId }),
  };
}

export async function runDirectedOutboundTurn(
  options: DirectedOutboundTurnRunnerOptions,
  input: DirectedOutboundTurnInput,
): Promise<DirectedOutboundTurnResult> {
  const session = input.targetSession;

  let result: Awaited<ReturnType<DirectedOutboundTurnRunnerOptions["turnOrchestrator"]["run"]>>;

  try {
    result = await options.turnOrchestrator.run({
      sessionId: session.session_id,
      audience: session.audience_label,
      userMessage: formatDirectedOutboundInstruction({
        instruction: input.instruction,
        authorizationKind: input.authorizationKind,
      }),
      stakes: "medium",
      origin: "directed_outbound",
    });
  } catch (error) {
    if (error instanceof SessionBusyError) {
      return {
        targetSessionId: session.session_id,
        status: "target_busy",
        emitted: false,
        response: "",
        deliveryOutcome: {
          state: "target_busy",
          reason: "target_session_busy",
        },
      };
    }

    throw error;
  }

  return {
    targetSessionId: session.session_id,
    status: "completed",
    turnId: result.turn_id,
    emitted: result.emitted,
    response: result.response,
    ...(result.agentMessageId === undefined ? {} : { agentMessageId: result.agentMessageId }),
    ...(result.outboundDelivery === undefined ? {} : { delivery: result.outboundDelivery }),
    deliveryOutcome: directedOutboundDeliveryOutcome({
      emission: result.emission,
      agentMessageId: result.agentMessageId,
      delivery: result.outboundDelivery,
    }),
  };
}
