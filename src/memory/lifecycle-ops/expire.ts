import type { ActionRepository } from "../actions/repository.js";
import type { ActionRecord } from "../actions/types.js";
import type { ActionId, SessionId } from "../../util/ids.js";
import { IdentityCasMismatchError } from "../../util/errors.js";
import type { LifecycleOperationResult, LifecycleTracer } from "./types.js";
import { isTerminalActionState } from "./complete.js";

const SESSION_SCOPED_ACTIVE_STATES = [
  "considering",
  "committed_to_do",
  "scheduled",
  "unknown",
] as const satisfies readonly ActionRecord["state"][];

export type ExpireSessionScopedActionsRepository = Pick<ActionRepository, "list" | "update">;

export type ExpireSessionScopedActionsResult = {
  sessionId: SessionId;
  expiredActionIds: ActionId[];
  conflictedActionIds: ActionId[];
  skippedActionIds: ActionId[];
};

export type RolloverNextSessionActionsResult = {
  fromSessionId: SessionId;
  toSessionId: SessionId;
  promotedActionIds: ActionId[];
  conflictedActionIds: ActionId[];
  skippedActionIds: ActionId[];
};

export function expireSessionScopedActions(input: {
  sessionId: SessionId;
  repository: ExpireSessionScopedActionsRepository;
  nowMs: number;
  tracer?: LifecycleTracer;
}): LifecycleOperationResult<ExpireSessionScopedActionsResult> {
  const candidates = input.repository.list({
    states: SESSION_SCOPED_ACTIVE_STATES,
    sessionScope: "current_session",
    sessionAnchorId: input.sessionId,
  });
  const expiredActionIds: ActionId[] = [];
  const conflictedActionIds: ActionId[] = [];
  const skippedActionIds: ActionId[] = [];
  let firstConflict: IdentityCasMismatchError | null = null;

  for (const action of candidates) {
    if (isTerminalActionState(action.state)) {
      skippedActionIds.push(action.id);
      continue;
    }

    try {
      input.repository.update(action.id, {
        state: "expired",
        expired_at: input.nowMs,
        updated_at: input.nowMs,
      });
      expiredActionIds.push(action.id);
    } catch (error) {
      if (error instanceof IdentityCasMismatchError) {
        conflictedActionIds.push(action.id);
        firstConflict ??= error;
        continue;
      }

      throw error;
    }
  }

  if (input.tracer?.enabled === true) {
    input.tracer.emit("action_session_scope.expired", {
      turnId: `session_end:${input.sessionId}`,
      session_id: input.sessionId,
      actions_expired_at_session_close: expiredActionIds.length,
      conflict_count: conflictedActionIds.length,
      skipped_count: skippedActionIds.length,
    });
  }

  const value = {
    sessionId: input.sessionId,
    expiredActionIds,
    conflictedActionIds,
    skippedActionIds,
  };

  if (firstConflict !== null) {
    return {
      status: "conflict",
      error: firstConflict,
      value,
    };
  }

  return {
    status: "success",
    value,
  };
}

export function rolloverNextSessionActions(input: {
  fromSessionId: SessionId;
  toSessionId: SessionId;
  repository: ExpireSessionScopedActionsRepository;
  nowMs: number;
  tracer?: LifecycleTracer;
}): LifecycleOperationResult<RolloverNextSessionActionsResult> {
  const candidates = input.repository.list({
    states: SESSION_SCOPED_ACTIVE_STATES,
    sessionScope: "next_session",
    sessionAnchorId: input.fromSessionId,
  });
  const promotedActionIds: ActionId[] = [];
  const conflictedActionIds: ActionId[] = [];
  const skippedActionIds: ActionId[] = [];
  let firstConflict: IdentityCasMismatchError | null = null;

  for (const action of candidates) {
    if (isTerminalActionState(action.state)) {
      skippedActionIds.push(action.id);
      continue;
    }

    try {
      input.repository.update(action.id, {
        session_scope: "current_session",
        session_anchor_id: input.toSessionId,
        updated_at: input.nowMs,
      });
      promotedActionIds.push(action.id);
    } catch (error) {
      if (error instanceof IdentityCasMismatchError) {
        conflictedActionIds.push(action.id);
        firstConflict ??= error;
        continue;
      }

      throw error;
    }
  }

  if (input.tracer?.enabled === true) {
    input.tracer.emit("action_session_scope.rolled_over", {
      turnId: `session_transition:${input.fromSessionId}:${input.toSessionId}`,
      from_session_id: input.fromSessionId,
      to_session_id: input.toSessionId,
      promoted_count: promotedActionIds.length,
      conflict_count: conflictedActionIds.length,
      skipped_count: skippedActionIds.length,
    });
  }

  const value = {
    fromSessionId: input.fromSessionId,
    toSessionId: input.toSessionId,
    promotedActionIds,
    conflictedActionIds,
    skippedActionIds,
  };

  if (firstConflict !== null) {
    return {
      status: "conflict",
      error: firstConflict,
      value,
    };
  }

  return {
    status: "success",
    value,
  };
}
