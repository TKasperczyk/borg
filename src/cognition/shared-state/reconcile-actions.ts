import type { ActionRepository } from "../../memory/actions/index.js";
import type { SharedStateEntry } from "../../memory/decision-artifacts/index.js";
import type { ActionId } from "../../util/ids.js";
import { errorMessage, type SharedStateReconciliationResult } from "./reconciliation-summary.js";

export function isTerminalActionState(state: string): boolean {
  return state === "completed" || state === "not_done" || state === "superseded";
}

export function reconcileActionCanonicalizations(input: {
  entry: SharedStateEntry;
  actionIds: readonly ActionId[];
  repository:
    | (Pick<ActionRepository, "update"> & Partial<Pick<ActionRepository, "get">>)
    | undefined;
  retiredActions: Set<ActionId>;
  result: SharedStateReconciliationResult;
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
      const action = input.repository.get?.(actionId) ?? null;

      if (action !== null && isTerminalActionState(action.state)) {
        input.result.actions_completed_skipped += 1;
        continue;
      }

      input.repository.update(
        actionId,
        {
          state: "completed",
          canonicalized_by_artifact_entry_id: input.entry.id,
        },
        {
          skipSideEffects: true,
        },
      );
      input.retiredActions.add(actionId);
      input.result.actions_retired += 1;
      input.result.actions_completed_succeeded += 1;
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
