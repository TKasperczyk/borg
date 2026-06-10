import type { ActionRepository } from "../../memory/actions/index.js";
import type { SharedStateEntry } from "../../memory/shared-state/index.js";
import {
  canonicalizeActionWithSharedStateEntry,
  type LifecycleTracer,
} from "../../memory/lifecycle-ops/index.js";
import type { ActionId } from "../../util/ids.js";
import { errorMessage, type SharedStateReconciliationResult } from "./reconciliation-summary.js";

export { isTerminalActionState } from "../../memory/lifecycle-ops/index.js";

export function reconcileActionCanonicalizations(input: {
  entry: SharedStateEntry;
  actionIds: readonly ActionId[];
  repository:
    | (Pick<ActionRepository, "update"> & Partial<Pick<ActionRepository, "get">>)
    | undefined;
  retiredActions: Set<ActionId>;
  result: SharedStateReconciliationResult;
  nowMs: number;
  turnCounter?: number | null;
  tracer?: LifecycleTracer;
  turnId?: string;
}): void {
  for (const actionId of input.actionIds) {
    input.result.actions_completed_attempted += 1;

    if (input.retiredActions.has(actionId)) {
      input.result.actions_completed_skipped += 1;
      continue;
    }

    if (input.repository === undefined) {
      input.result.actions_completed_skipped += 1;
      continue;
    }

    try {
      const result = canonicalizeActionWithSharedStateEntry({
        actionId,
        entry: input.entry,
        repository: input.repository,
        nowMs: input.nowMs,
        turnCounter: input.turnCounter,
        tracer: input.tracer,
        turnId: input.turnId,
      });

      if (result.status === "no_op" && result.reason === "missing") {
        input.result.actions_completed_skipped += 1;
        input.result.errors.push({
          channel: "action",
          id: actionId,
          artifactEntryId: input.entry.id,
          message: `Unknown action record id: ${actionId}`,
        });
        continue;
      }

      if (result.status === "no_op") {
        input.result.actions_completed_skipped += 1;
        continue;
      }

      if (result.status === "conflict") {
        input.result.actions_completed_skipped += 1;
        input.result.errors.push({
          channel: "action",
          id: actionId,
          artifactEntryId: input.entry.id,
          message: errorMessage(result.error),
        });
        continue;
      }

      input.retiredActions.add(actionId);
      input.result.actions_retired += 1;
      input.result.actions_completed_succeeded += 1;
      if (result.value.previous?.actor === "borg") {
        input.result.actions_closed_by_borg_self_performance += 1;
        if (input.tracer?.enabled === true && input.turnId !== undefined) {
          input.tracer.emit("action_state.borg_self_performance.completed", {
            turnId: input.turnId,
            action_id: actionId,
            artifact_entry_id: input.entry.id,
            terminal_state: "completed",
            source: "shared_state_reconciliation",
          });
        }
      }
    } catch (error) {
      input.result.errors.push({
        channel: "action",
        id: actionId,
        artifactEntryId: input.entry.id,
        message: errorMessage(error),
      });
    }
  }
}
