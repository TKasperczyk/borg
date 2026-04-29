import type { WorkingMemory } from "../../memory/working/index.js";
import type { PendingTurnEmission } from "../generation/types.js";
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
  emission?: PendingTurnEmission;
  toolCalls: ToolLoopCallRecord[];
  intents: readonly IntentRecord[];
  workingMemory: WorkingMemory;
};

export type ActionResult = {
  response: string;
  emitted?: boolean;
  emission?: PendingTurnEmission;
  tool_calls: ToolLoopCallRecord[];
  intents: IntentRecord[];
  workingMemory: WorkingMemory;
};

export async function performAction(context: ActionContext): Promise<ActionResult> {
  const emission = context.emission ?? {
    kind: "message",
    content: context.response,
  };
  const emitted = emission.kind === "message";
  const intents = [...context.intents];

  return {
    response: emitted ? emission.content : "",
    emitted,
    emission,
    tool_calls: [...context.toolCalls],
    intents: emitted ? intents : [],
    workingMemory: {
      ...context.workingMemory,
      pending_intents: emitted
        ? [...context.workingMemory.pending_intents, ...intents]
        : [...context.workingMemory.pending_intents],
    },
  };
}
