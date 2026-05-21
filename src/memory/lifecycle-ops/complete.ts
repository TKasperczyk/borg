import type { ActionRepository } from "../actions/repository.js";
import type { ActionRecord } from "../actions/types.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import type { ActionId, SharedStateEntryId } from "../../util/ids.js";
import type { LifecycleOperationResult, LifecycleTracer } from "./types.js";

export function isTerminalActionState(state: string): boolean {
  return (
    state === "completed" ||
    state === "not_done" ||
    state === "expired" ||
    state === "archived" ||
    state === "superseded"
  );
}

export type CompleteActionRepository = Pick<ActionRepository, "update"> &
  Partial<Pick<ActionRepository, "get">>;

export function completeAction(input: {
  actionId: ActionId;
  repository: CompleteActionRepository;
  canonicalizedByArtifactEntryId?: SharedStateEntryId;
  skipSideEffects?: boolean;
  lastReferencedAtMs?: number;
  lastReferencedTurnCounter?: number | null;
  lastReferencedTurnGlobal?: number | null;
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
    input.repository.update(
      input.actionId,
      {
        state: "completed",
        ...(input.canonicalizedByArtifactEntryId === undefined
          ? {}
          : {
              canonicalized_by_artifact_entry_id: input.canonicalizedByArtifactEntryId,
            }),
        ...(input.lastReferencedAtMs === undefined
          ? {}
          : { last_referenced_at_ms: input.lastReferencedAtMs }),
        ...(input.lastReferencedTurnCounter === undefined
          ? {}
          : { last_referenced_turn_counter: input.lastReferencedTurnCounter }),
        ...(input.lastReferencedTurnGlobal === undefined
          ? {}
          : { last_referenced_turn_global: input.lastReferencedTurnGlobal }),
      },
      {
        skipSideEffects: input.skipSideEffects,
      },
    );
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
    input.tracer.emit("action_state.transitioned", {
      turnId: input.turnId,
      action_id: input.actionId,
      terminal_state: "completed",
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
