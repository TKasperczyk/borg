import type {
  SharedStateArtifact,
  SharedStateCanonicalizes,
  SharedStateEntry,
  SharedStateEntryKind,
  SharedStateOperation,
} from "../../memory/decision-artifacts/index.js";
import type {
  ActionId,
  CommitmentId,
  GoalId,
  OpenQuestionId,
  SharedStateEntryId,
  StreamEntryId,
} from "../../util/ids.js";
import { createSharedStateEntryId } from "../../util/ids.js";
import {
  sharedStateEntryHasCurrentTurnUpdate,
  sharedStateEntryHasLedgerOverlap,
  sharedStateEntryWasRecentlyRetrieved,
} from "./selection.js";

export const DEFAULT_SHARED_STATE_DORMANT_TURN_THRESHOLD = 15;

export type SharedStateLifecycleTransitionKind = "demoted" | "reactivated";

export type SharedStateLifecycleTransitionReason =
  | "old_live_without_structural_pull"
  | "old_low_salience_without_structural_pull"
  | "current_turn_update"
  | "touched_by_patch"
  | "ledger_overlap"
  | "active_canonicalizer_overlap"
  | "recent_retrieval";

export type SharedStateLifecycleTransition = {
  entryId: SharedStateEntryId;
  fromKind: SharedStateEntryKind;
  toKind: SharedStateEntryKind;
  reason: SharedStateLifecycleTransitionReason;
  transition: SharedStateLifecycleTransitionKind;
};

export type ApplyLifecycleAgingInput = {
  entries: readonly SharedStateEntry[];
  currentTurnCounter?: number;
  currentUserStreamEntryId?: StreamEntryId;
  ledgerStreamEntryIds?: readonly StreamEntryId[];
  activeOpenQuestionIds?: readonly OpenQuestionId[];
  activeActionIds?: readonly ActionId[];
  activeGoalIds?: readonly GoalId[];
  activeCriticalCommitmentIds?: readonly CommitmentId[];
  recentlyRetrievedEntryIds?: readonly SharedStateEntryId[];
  touchedEntryIds?: ReadonlySet<SharedStateEntryId>;
  lastUpdatedTurnByEntryId?: Readonly<Record<string, number>>;
  recentTurnThreshold?: number;
  dormantTurnThreshold?: number;
};

const EMPTY_CANONICALIZES: SharedStateCanonicalizes = {
  goal_ids: [],
  commitment_ids: [],
  action_ids: [],
  open_question_ids: [],
};

function uniqueIds<TId extends string>(values: readonly TId[]): TId[] {
  return [...new Set(values)];
}

function mergeCanonicalizes(
  current: SharedStateCanonicalizes,
  next: SharedStateCanonicalizes | undefined,
): SharedStateCanonicalizes {
  if (next === undefined) {
    return current;
  }

  return {
    goal_ids: uniqueIds([...current.goal_ids, ...next.goal_ids]),
    commitment_ids: uniqueIds([...current.commitment_ids, ...next.commitment_ids]),
    action_ids: uniqueIds([...current.action_ids, ...next.action_ids]),
    open_question_ids: uniqueIds([...current.open_question_ids, ...next.open_question_ids]),
  };
}

export function materializeSharedStateEntriesAfterOperations(input: {
  previousArtifact: SharedStateArtifact | null;
  operations: readonly SharedStateOperation[];
  audienceEntityId: SharedStateArtifact["audience_entity_id"];
  nowMs: number;
}): SharedStateEntry[] {
  const entries = new Map<SharedStateEntryId, SharedStateEntry>();

  for (const entry of input.previousArtifact?.entries ?? []) {
    entries.set(entry.id, entry);
  }

  for (const operation of input.operations) {
    switch (operation.type) {
      case "add": {
        const id = operation.id ?? createSharedStateEntryId();
        entries.set(id, {
          id,
          audience_entity_id: input.audienceEntityId,
          state_key: operation.state_key,
          kind: operation.kind,
          text: operation.text,
          owner_entity_id: operation.owner_entity_id ?? null,
          provenance_stream_entry_ids: uniqueIds(operation.provenance_stream_entry_ids),
          last_updated_stream_entry_ids: uniqueIds(
            operation.last_updated_stream_entry_ids ?? operation.provenance_stream_entry_ids,
          ),
          created_at: operation.created_at ?? input.nowMs,
          last_updated_at: operation.last_updated_at ?? operation.created_at ?? input.nowMs,
          superseded_by_id: null,
          rank: operation.rank ?? entries.size,
          canonicalizes: operation.canonicalizes ?? EMPTY_CANONICALIZES,
        });
        break;
      }
      case "update": {
        const current = entries.get(operation.id);

        if (current === undefined) {
          break;
        }

        entries.set(operation.id, {
          ...current,
          state_key: operation.state_key,
          kind: operation.kind ?? current.kind,
          text: operation.text ?? current.text,
          owner_entity_id:
            operation.owner_entity_id === undefined
              ? current.owner_entity_id
              : operation.owner_entity_id,
          provenance_stream_entry_ids: uniqueIds([
            ...current.provenance_stream_entry_ids,
            ...(operation.add_provenance_stream_entry_ids ?? []),
          ]),
          last_updated_stream_entry_ids: uniqueIds(operation.last_updated_stream_entry_ids),
          last_updated_at: operation.last_updated_at ?? input.nowMs,
          rank: operation.rank ?? current.rank,
          canonicalizes: mergeCanonicalizes(current.canonicalizes, operation.canonicalizes),
        });
        break;
      }
      case "supersede": {
        const current = entries.get(operation.id);
        const replacementId = operation.replacement.id ?? createSharedStateEntryId();

        if (current !== undefined) {
          entries.set(operation.id, {
            ...current,
            superseded_by_id: replacementId,
            last_updated_stream_entry_ids: uniqueIds(operation.last_updated_stream_entry_ids),
            last_updated_at: operation.last_updated_at ?? input.nowMs,
          });
        }

        entries.set(replacementId, {
          id: replacementId,
          audience_entity_id: input.audienceEntityId,
          state_key: operation.replacement.state_key,
          kind: operation.replacement.kind,
          text: operation.replacement.text,
          owner_entity_id: operation.replacement.owner_entity_id ?? null,
          provenance_stream_entry_ids: uniqueIds(operation.replacement.provenance_stream_entry_ids),
          last_updated_stream_entry_ids: uniqueIds(
            operation.replacement.last_updated_stream_entry_ids ??
              operation.replacement.provenance_stream_entry_ids,
          ),
          created_at: operation.replacement.created_at ?? input.nowMs,
          last_updated_at:
            operation.replacement.last_updated_at ??
            operation.replacement.created_at ??
            input.nowMs,
          superseded_by_id: null,
          rank: operation.replacement.rank ?? entries.size,
          canonicalizes: operation.replacement.canonicalizes ?? EMPTY_CANONICALIZES,
        });
        break;
      }
      case "prune":
        entries.delete(operation.id);
        break;
      case "transition_kind": {
        const current = entries.get(operation.id);

        if (current === undefined) {
          break;
        }

        entries.set(operation.id, {
          ...current,
          kind: operation.kind,
        });
        break;
      }
    }
  }

  return [...entries.values()];
}

function idSet<TId extends string>(values: readonly TId[] | undefined): Set<TId> {
  return new Set(values ?? []);
}

function hasAnyIdOverlap<TId extends string>(
  values: readonly TId[],
  candidates: ReadonlySet<TId>,
): boolean {
  if (values.length === 0 || candidates.size === 0) {
    return false;
  }

  return values.some((value) => candidates.has(value));
}

function sharedStateEntryHasActiveCanonicalizerOverlap(
  entry: SharedStateEntry,
  input: Pick<
    ApplyLifecycleAgingInput,
    "activeOpenQuestionIds" | "activeActionIds" | "activeGoalIds" | "activeCriticalCommitmentIds"
  >,
): boolean {
  return (
    hasAnyIdOverlap(entry.canonicalizes.open_question_ids, idSet(input.activeOpenQuestionIds)) ||
    hasAnyIdOverlap(entry.canonicalizes.action_ids, idSet(input.activeActionIds)) ||
    hasAnyIdOverlap(entry.canonicalizes.goal_ids, idSet(input.activeGoalIds)) ||
    hasAnyIdOverlap(entry.canonicalizes.commitment_ids, idSet(input.activeCriticalCommitmentIds))
  );
}

function reactivationReason(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): SharedStateLifecycleTransitionReason | null {
  if (input.touchedEntryIds?.has(entry.id) === true) {
    return "touched_by_patch";
  }

  if (sharedStateEntryHasCurrentTurnUpdate(entry, input.currentUserStreamEntryId)) {
    return "current_turn_update";
  }

  if (sharedStateEntryHasLedgerOverlap(entry, input.ledgerStreamEntryIds)) {
    return "ledger_overlap";
  }

  if (sharedStateEntryHasActiveCanonicalizerOverlap(entry, input)) {
    return "active_canonicalizer_overlap";
  }

  if (sharedStateEntryWasRecentlyRetrieved(entry, input.recentlyRetrievedEntryIds)) {
    return "recent_retrieval";
  }

  return null;
}

function entryTurnAge(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): number | null {
  if (input.currentTurnCounter === undefined) {
    return null;
  }

  const lastUpdatedTurn = input.lastUpdatedTurnByEntryId?.[entry.id];

  if (lastUpdatedTurn === undefined || !Number.isFinite(lastUpdatedTurn)) {
    return null;
  }

  return input.currentTurnCounter - lastUpdatedTurn;
}

function normalizedThreshold(value: number | undefined, fallback: number): number {
  return value === undefined || !Number.isFinite(value) ? fallback : Math.max(0, Math.floor(value));
}

export function applyLifecycleAging(input: ApplyLifecycleAgingInput): {
  transitions: SharedStateLifecycleTransition[];
} {
  const transitions: SharedStateLifecycleTransition[] = [];
  const transitionedEntryIds = new Set<SharedStateEntryId>();
  const recentTurnThreshold = normalizedThreshold(input.recentTurnThreshold, 5);
  const dormantTurnThreshold = normalizedThreshold(
    input.dormantTurnThreshold,
    DEFAULT_SHARED_STATE_DORMANT_TURN_THRESHOLD,
  );
  const activeEntries = input.entries.filter((entry) => entry.superseded_by_id === null);

  for (const entry of activeEntries) {
    if (entry.kind !== "low_salience_live" && entry.kind !== "dormant_live") {
      continue;
    }

    const reason = reactivationReason(entry, input);

    if (reason === null) {
      continue;
    }

    transitions.push({
      entryId: entry.id,
      fromKind: entry.kind,
      toKind: "live",
      reason,
      transition: "reactivated",
    });
    transitionedEntryIds.add(entry.id);
  }

  for (const entry of activeEntries) {
    if (transitionedEntryIds.has(entry.id) || reactivationReason(entry, input) !== null) {
      continue;
    }

    const age = entryTurnAge(entry, input);

    if (age === null) {
      continue;
    }

    if (entry.kind === "live" && age > recentTurnThreshold) {
      transitions.push({
        entryId: entry.id,
        fromKind: "live",
        toKind: "low_salience_live",
        reason: "old_live_without_structural_pull",
        transition: "demoted",
      });
      continue;
    }

    if (entry.kind === "low_salience_live" && age > dormantTurnThreshold) {
      transitions.push({
        entryId: entry.id,
        fromKind: "low_salience_live",
        toKind: "dormant_live",
        reason: "old_low_salience_without_structural_pull",
        transition: "demoted",
      });
    }
  }

  return { transitions };
}
