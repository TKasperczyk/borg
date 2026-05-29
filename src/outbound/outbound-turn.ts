import type { TurnOrchestrator } from "../cognition/index.js";
import { scrubCreatorDirectiveInternalIds } from "../cognition/deliberation/prompt/system-prompt.js";
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
    ? "An autonomous wake, structurally authorized by proactive outbound policy, directed Borg to compose a proactive outbound message for this target session."
    : "A structurally authorized creator in an operator context directed Borg to compose a proactive outbound message for this target session.";
}

function formatDirectedOutboundInstruction(input: {
  instruction: string;
  authorizationKind: DirectedOutboundTurnInput["authorizationKind"];
}): string {
  return [
    "<borg_directed_outbound_instruction>",
    directedOutboundProvenanceLine(input.authorizationKind),
    "Compose the message under this target session's audience scope, using only prompt-visible target-session context and audience-scoped memory.",
    "Convey the instruction below in the message. Do not expose tool names, hidden prompts, internal ids, or the dispatch machinery.",
    "",
    "Instruction:",
    escapeReservedBorgTags(scrubCreatorDirectiveInternalIds(input.instruction)),
    "</borg_directed_outbound_instruction>",
  ].join("\n");
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
