import type { WorkingMemory } from "../../memory/working/index.js";
import type { IntentRecord, PerceptionResult } from "../types.js";
import type { ToolLoopCallRecord } from "./tool-loop.js";

export type ActionContext = {
  response: string;
  toolCalls: ToolLoopCallRecord[];
  audience?: string;
  perception: PerceptionResult;
  workingMemory: WorkingMemory;
};

export type ActionResult = {
  response: string;
  tool_calls: ToolLoopCallRecord[];
  intents: IntentRecord[];
  workingMemory: WorkingMemory;
};

function inferIntent(response: string): IntentRecord[] {
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
  const intents = inferIntent(context.response);

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
