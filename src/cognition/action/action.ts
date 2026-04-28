import type { WorkingMemory } from "../../memory/working/index.js";
import type { IntentRecord } from "../types.js";
import type { ToolLoopCallRecord } from "./tool-loop.js";

// Commits the action outcome -- the commitment-checked response, the tool
// calls that produced it, and the planner's structured intents -- into the
// working-memory shape that reflection consumes. The tool-use loop itself
// runs earlier inside the deliberation finalizer; the agent_msg stream
// append happens later in the orchestrator. Intents flow from the S2
// planner only -- this stage never infers them from response prose.
export type ActionContext = {
  response: string;
  toolCalls: ToolLoopCallRecord[];
  intents: readonly IntentRecord[];
  workingMemory: WorkingMemory;
};

export type ActionResult = {
  response: string;
  tool_calls: ToolLoopCallRecord[];
  intents: IntentRecord[];
  workingMemory: WorkingMemory;
};

export async function performAction(context: ActionContext): Promise<ActionResult> {
  const intents = [...context.intents];

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
