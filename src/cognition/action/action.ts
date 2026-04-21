import type { LLMToolCall } from "../../llm/index.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { IntentRecord, PerceptionResult } from "../types.js";

export type ActionContext = {
  response: string;
  toolCalls: LLMToolCall[];
  audience?: string;
  perception: PerceptionResult;
  workingMemory: WorkingMemory;
};

export type ActionResult = {
  response: string;
  tool_calls: LLMToolCall[];
  intents: IntentRecord[];
  workingMemory: WorkingMemory;
};

function inferIntent(response: string, toolCalls: readonly LLMToolCall[]): IntentRecord[] {
  if (toolCalls.length > 0) {
    return [
      {
        description: `Prepare tool call ${toolCalls[0]?.name ?? "tool"}`,
        next_action: toolCalls[0]?.name ?? null,
      },
    ];
  }

  const nextStepMatch = response.match(/\bnext (?:step|action)\b[:\s-]+([^.!\n]+)/i);

  if (nextStepMatch?.[1] !== undefined) {
    const nextAction = nextStepMatch[1].trim();

    return [
      {
        description: "Suggested follow-up",
        next_action: nextAction,
      },
    ];
  }

  const willMatch = response.match(/\bI will ([^.!\n]+)/i);

  if (willMatch?.[1] !== undefined) {
    return [
      {
        description: "Declared next step",
        next_action: willMatch[1].trim(),
      },
    ];
  }

  return [];
}

export async function performAction(context: ActionContext): Promise<ActionResult> {
  const intents = inferIntent(context.response, context.toolCalls);

  return {
    response: context.response,
    tool_calls: [...context.toolCalls],
    intents,
    workingMemory: {
      ...context.workingMemory,
      pending_intents: [...context.workingMemory.pending_intents, ...intents],
    },
  };
}
