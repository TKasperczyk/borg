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

export type LifecycleProtectionReason =
  | "touched_by_patch"
  | "current_turn_update"
  | "ledger_overlap"
  | "active_canonicalizer_overlap"
  | "recent_retrieval";

export const LIFECYCLE_PROTECTION_REASONS = [
  "touched_by_patch",
  "current_turn_update",
  "ledger_overlap",
  "active_canonicalizer_overlap",
  "recent_retrieval",
] as const;

export type LifecycleAgingBlockerCounts = {
  demotable_count: number;
  unknown_age_count: number;
  demoted_count: number;
  blocked_by_current_turn_update: number;
  blocked_by_patch_touch: number;
  blocked_by_ledger_overlap: number;
  blocked_by_recent_retrieval: number;
  blocked_by_active_canonicalizer: number;
  blocked_by_multiple_reasons: number;
};

export type LifecycleAgingBlockedSampleEntry = {
  entry_id: SharedStateEntryId;
  state_key: string | null;
  age_turns: number | null;
  block_reasons: LifecycleProtectionReason[];
  active_canonicalizer_kinds: string[] | null;
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

function isTouchedByPatch(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): boolean {
  return input.touchedEntryIds?.has(entry.id) === true;
}

function hasCurrentTurnUpdate(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): boolean {
  return sharedStateEntryHasCurrentTurnUpdate(entry, input.currentUserStreamEntryId);
}

function hasLedgerOverlap(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): boolean {
  return sharedStateEntryHasLedgerOverlap(entry, input.ledgerStreamEntryIds);
}

function hasActiveCanonicalizerOverlap(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): boolean {
  return sharedStateEntryHasActiveCanonicalizerOverlap(entry, input);
}

function hasRecentRetrieval(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): boolean {
  return sharedStateEntryWasRecentlyRetrieved(entry, input.recentlyRetrievedEntryIds);
}

function activeCanonicalizerOverlapKinds(
  entry: SharedStateEntry,
  input: Pick<
    ApplyLifecycleAgingInput,
    "activeOpenQuestionIds" | "activeActionIds" | "activeGoalIds" | "activeCriticalCommitmentIds"
  >,
): string[] {
  const kinds: string[] = [];

  if (hasAnyIdOverlap(entry.canonicalizes.open_question_ids, idSet(input.activeOpenQuestionIds))) {
    kinds.push("oq");
  }

  if (hasAnyIdOverlap(entry.canonicalizes.action_ids, idSet(input.activeActionIds))) {
    kinds.push("action");
  }

  if (hasAnyIdOverlap(entry.canonicalizes.goal_ids, idSet(input.activeGoalIds))) {
    kinds.push("goal");
  }

  if (
    hasAnyIdOverlap(entry.canonicalizes.commitment_ids, idSet(input.activeCriticalCommitmentIds))
  ) {
    kinds.push("critical");
  }

  return kinds;
}

export function sharedStateLifecycleProtectionReasons(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): LifecycleProtectionReason[] {
  const reasons: LifecycleProtectionReason[] = [];

  if (isTouchedByPatch(entry, input)) {
    reasons.push("touched_by_patch");
  }

  if (hasCurrentTurnUpdate(entry, input)) {
    reasons.push("current_turn_update");
  }

  if (hasLedgerOverlap(entry, input)) {
    reasons.push("ledger_overlap");
  }

  if (hasActiveCanonicalizerOverlap(entry, input)) {
    reasons.push("active_canonicalizer_overlap");
  }

  if (hasRecentRetrieval(entry, input)) {
    reasons.push("recent_retrieval");
  }

  return reasons;
}

function reactivationReason(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): SharedStateLifecycleTransitionReason | null {
  if (isTouchedByPatch(entry, input)) {
    return "touched_by_patch";
  }

  if (hasCurrentTurnUpdate(entry, input)) {
    return "current_turn_update";
  }

  if (hasLedgerOverlap(entry, input)) {
    return "ledger_overlap";
  }

  if (hasActiveCanonicalizerOverlap(entry, input)) {
    return "active_canonicalizer_overlap";
  }

  if (hasRecentRetrieval(entry, input)) {
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

function emptyLifecycleAgingBlockerCounts(): LifecycleAgingBlockerCounts {
  return {
    demotable_count: 0,
    unknown_age_count: 0,
    demoted_count: 0,
    blocked_by_current_turn_update: 0,
    blocked_by_patch_touch: 0,
    blocked_by_ledger_overlap: 0,
    blocked_by_recent_retrieval: 0,
    blocked_by_active_canonicalizer: 0,
    blocked_by_multiple_reasons: 0,
  };
}

function incrementBlockerCount(
  counts: LifecycleAgingBlockerCounts,
  reason: LifecycleProtectionReason,
): void {
  switch (reason) {
    case "touched_by_patch":
      counts.blocked_by_patch_touch += 1;
      break;
    case "current_turn_update":
      counts.blocked_by_current_turn_update += 1;
      break;
    case "ledger_overlap":
      counts.blocked_by_ledger_overlap += 1;
      break;
    case "active_canonicalizer_overlap":
      counts.blocked_by_active_canonicalizer += 1;
      break;
    case "recent_retrieval":
      counts.blocked_by_recent_retrieval += 1;
      break;
  }
}

function recordDemotionCandidate(input: {
  entry: SharedStateEntry;
  age: number;
  reasons: LifecycleProtectionReason[];
  counts: LifecycleAgingBlockerCounts;
  blockedSamples: LifecycleAgingBlockedSampleEntry[];
  demoted: boolean;
  agingInput: ApplyLifecycleAgingInput;
}): void {
  input.counts.demotable_count += 1;

  if (input.demoted) {
    input.counts.demoted_count += 1;
    return;
  }

  if (input.reasons.length > 1) {
    input.counts.blocked_by_multiple_reasons += 1;
  } else if (input.reasons.length === 1) {
    incrementBlockerCount(input.counts, input.reasons[0]!);
  }

  const activeCanonicalizerKinds = activeCanonicalizerOverlapKinds(input.entry, input.agingInput);
  input.blockedSamples.push({
    entry_id: input.entry.id,
    state_key: input.entry.state_key,
    age_turns: input.age,
    block_reasons: input.reasons,
    active_canonicalizer_kinds:
      activeCanonicalizerKinds.length === 0 ? null : activeCanonicalizerKinds,
  });
}

export function applyLifecycleAging(input: ApplyLifecycleAgingInput): {
  transitions: SharedStateLifecycleTransition[];
  blockerCountsLiveToLowSalience: LifecycleAgingBlockerCounts;
  blockerCountsLowSalienceToDormant: LifecycleAgingBlockerCounts;
  blockedSample: LifecycleAgingBlockedSampleEntry[];
} {
  const transitions: SharedStateLifecycleTransition[] = [];
  const transitionedEntryIds = new Set<SharedStateEntryId>();
  const blockerCountsLiveToLowSalience = emptyLifecycleAgingBlockerCounts();
  const blockerCountsLowSalienceToDormant = emptyLifecycleAgingBlockerCounts();
  const blockedSamples: LifecycleAgingBlockedSampleEntry[] = [];
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
    if (transitionedEntryIds.has(entry.id)) {
      continue;
    }

    const age = entryTurnAge(entry, input);
    const reasons = sharedStateLifecycleProtectionReasons(entry, input);

    if (entry.kind === "live" && age === null) {
      blockerCountsLiveToLowSalience.unknown_age_count += 1;
    }

    if (entry.kind === "low_salience_live" && age === null) {
      blockerCountsLowSalienceToDormant.unknown_age_count += 1;
    }

    if (entry.kind === "live" && age !== null && age > recentTurnThreshold) {
      const demoted = reasons.length === 0;
      recordDemotionCandidate({
        entry,
        age,
        reasons,
        counts: blockerCountsLiveToLowSalience,
        blockedSamples,
        demoted,
        agingInput: input,
      });
    }

    if (entry.kind === "low_salience_live" && age !== null && age > dormantTurnThreshold) {
      const demoted = reasons.length === 0;
      recordDemotionCandidate({
        entry,
        age,
        reasons,
        counts: blockerCountsLowSalienceToDormant,
        blockedSamples,
        demoted,
        agingInput: input,
      });
    }

    if (reactivationReason(entry, input) !== null) {
      continue;
    }

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

  return {
    transitions,
    blockerCountsLiveToLowSalience,
    blockerCountsLowSalienceToDormant,
    blockedSample: blockedSamples
      .filter((sample) => sample.block_reasons.length > 0)
      .sort((left, right) => (right.age_turns ?? -1) - (left.age_turns ?? -1))
      .slice(0, 10),
  };
}
