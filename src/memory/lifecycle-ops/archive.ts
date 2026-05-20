import type { ActionRepository } from "../actions/repository.js";
import type { ActionRecord } from "../actions/types.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import type { ActionId } from "../../util/ids.js";
import { isTerminalActionState } from "./complete.js";
import type { LifecycleOperationResult, LifecycleTracer } from "./types.js";

export type ArchiveStaleActionRepository = Pick<ActionRepository, "update"> &
  Partial<Pick<ActionRepository, "get">>;

export function archiveStaleAction(input: {
  actionId: ActionId;
  repository: ArchiveStaleActionRepository;
  nowMs: number;
  tracer?: LifecycleTracer;
  turnId?: string;
  traceSource?: string;
}): LifecycleOperationResult<{ actionId: ActionId; previous: ActionRecord | null }> {
  const previous = input.repository.get?.(input.actionId);

  if (input.repository.get !== undefined && previous == null) {
    return {
      status: "no_op",
      reason: "missing",
      value: {
        actionId: input.actionId,
        previous: null,
      },
    };
  }

  if (previous !== undefined && previous !== null && isTerminalActionState(previous.state)) {
    return {
      status: "no_op",
      reason: "terminal",
      value: {
        actionId: input.actionId,
        previous,
      },
    };
  }

  try {
    input.repository.update(input.actionId, {
      state: "archived",
      archived_at: input.nowMs,
      updated_at: input.nowMs,
    });
  } catch (error) {
    if (error instanceof IdentityCasMismatchError) {
      return {
        status: "conflict",
        error,
      };
    }

    throw error;
  }

  if (input.tracer?.enabled === true && input.turnId !== undefined) {
    input.tracer.emit("action_state.archived", {
      turnId: input.turnId,
      action_id: input.actionId,
      source: input.traceSource,
    });
  }

  return {
    status: "success",
    value: {
      actionId: input.actionId,
      previous: previous ?? null,
    },
  };
}
