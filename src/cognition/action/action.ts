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

export async function performAction(context: ActionContext): Promise<ActionResult> {
  const intents: IntentRecord[] = [];

  return {
    response: context.response,
    tool_calls: [...context.toolCalls],
    intents,
    workingMemory: {
      ...context.workingMemory,
      pending_intents: [...context.workingMemory.pending_intents],
    },
  };
}
