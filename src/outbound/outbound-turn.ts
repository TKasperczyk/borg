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
import type { SessionId } from "../util/ids.js";
import type { OutboundDeliveryReceipt } from "./types.js";

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
};

export type DirectedOutboundTurnRunnerOptions = {
  turnOrchestrator: Pick<TurnOrchestrator, "run">;
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
  };
}
