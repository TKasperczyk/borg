import type { ActionRecord, ActionState } from "./types.js";

export const ACTION_ARCHIVE_SCAN_LIMIT = 256;

export const ACTION_ARCHIVE_ACTIVE_STATES: readonly ActionState[] = [
  "considering",
  "committed_to_do",
  "scheduled",
];

export type ActionArchiveSkipReason =
  | "borg_owned"
  | "group_owned"
  | "non_participant_owned"
  | "scheduled_or_due"
  | "missing_reference_turn"
  | "below_inactive_threshold";

export type ActionArchiveCandidateClassification =
  | {
      status: "eligible";
      inactiveTurns: number;
    }
  | {
      status: "skipped";
      reason: ActionArchiveSkipReason;
      inactiveTurns?: number;
    };

export function isParticipantOwnedAction(
  action: Pick<ActionRecord, "actor" | "audience_entity_id">,
): boolean {
  return action.actor === "user" || action.actor !== action.audience_entity_id;
}

export function lastReferencedActionLifecycleTurn(
  action: Pick<ActionRecord, "last_referenced_turn_counter" | "last_referenced_turn_global">,
): number | null {
  return action.last_referenced_turn_global ?? action.last_referenced_turn_counter;
}

export function classifyActionArchiveCandidate(
  action: Pick<
    ActionRecord,
    | "actor"
    | "audience_entity_id"
    | "state"
    | "scheduled_at"
    | "last_referenced_turn_counter"
    | "last_referenced_turn_global"
  >,
  input: {
    turnCounter: number;
    archiveAfterTurns: number;
  },
): ActionArchiveCandidateClassification {
  if (action.actor === "borg") {
    return { status: "skipped", reason: "borg_owned" };
  }

  if (action.actor !== "user" && action.actor === action.audience_entity_id) {
    return { status: "skipped", reason: "group_owned" };
  }

  if (!isParticipantOwnedAction(action)) {
    return { status: "skipped", reason: "non_participant_owned" };
  }

  if (action.state === "scheduled" || action.scheduled_at !== null) {
    return { status: "skipped", reason: "scheduled_or_due" };
  }

  const lastReferencedTurn = lastReferencedActionLifecycleTurn(action);

  if (lastReferencedTurn === null) {
    return { status: "skipped", reason: "missing_reference_turn" };
  }

  const inactiveTurns = input.turnCounter - lastReferencedTurn;

  if (inactiveTurns < input.archiveAfterTurns) {
    return {
      status: "skipped",
      reason: "below_inactive_threshold",
      inactiveTurns,
    };
  }

  return {
    status: "eligible",
    inactiveTurns,
  };
}
