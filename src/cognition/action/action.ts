import type { EmbeddingClient } from "../../embeddings/index.js";
import { mergePendingActionsBySimilarity, type WorkingMemory } from "../../memory/working/index.js";
import type { PendingTurnEmission } from "../generation/types.js";
import type { IntentRecord } from "../types.js";
import type { PendingActionJudge } from "./pending-action-judge.js";
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
  pendingActionJudge?: PendingActionJudge;
  pendingActionEmbeddingClient?: EmbeddingClient;
  pendingActionTimestamp?: number;
  pendingActionSimilarityThreshold?: number;
  onPendingActionRejected?: (event: PendingActionRejection) => void | Promise<void>;
};

export type ActionResult = {
  response: string;
  emitted?: boolean;
  emission?: PendingTurnEmission;
  tool_calls: ToolLoopCallRecord[];
  intents: IntentRecord[];
  workingMemory: WorkingMemory;
};

export type PendingActionRejection = {
  record: IntentRecord;
  reason: string;
  confidence: number;
  degraded: boolean;
};

async function notifyRejected(
  context: Pick<ActionContext, "onPendingActionRejected">,
  event: PendingActionRejection,
): Promise<void> {
  try {
    await context.onPendingActionRejected?.(event);
  } catch {
    // Best-effort observability only.
  }
}

async function filterPendingActions(context: ActionContext): Promise<IntentRecord[]> {
  const accepted: IntentRecord[] = [];

  for (const record of context.intents) {
    if (record.next_action === null || record.next_action.trim().length === 0) {
      await notifyRejected(context, {
        record,
        reason: "missing_next_action",
        confidence: 1,
        degraded: false,
      });
      continue;
    }

    if (context.pendingActionJudge === undefined) {
      accepted.push(record);
      continue;
    }

    const judgment = await context.pendingActionJudge.judge(record);

    if (judgment.accepted) {
      accepted.push(record);
      continue;
    }

    await notifyRejected(context, {
      record,
      reason: judgment.reason,
      confidence: judgment.confidence,
      degraded: judgment.degraded,
    });
  }

  return accepted;
}

export async function performAction(context: ActionContext): Promise<ActionResult> {
  const emission = context.emission ?? {
    kind: "message",
    content: context.response,
  };
  const emitted = emission.kind === "message";
  const pendingActions = emitted ? await filterPendingActions(context) : [];
  const pending_actions = emitted
    ? await mergePendingActionsBySimilarity({
        existing: context.workingMemory.pending_actions,
        incoming: pendingActions,
        embeddingClient: context.pendingActionEmbeddingClient,
        nowMs: context.pendingActionTimestamp ?? context.workingMemory.updated_at,
        threshold: context.pendingActionSimilarityThreshold,
      })
    : [...context.workingMemory.pending_actions];

  return {
    response: emitted ? emission.content : "",
    emitted,
    emission,
    tool_calls: [...context.toolCalls],
    intents: pendingActions,
    workingMemory: {
      ...context.workingMemory,
      pending_actions,
    },
  };
}
