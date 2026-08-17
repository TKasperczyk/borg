import type {
  SharedStateArtifact,
  SharedStateCanonicalizes,
  SharedStateEntry,
  SharedStateEntryKind,
  SharedStateOperation,
} from "../../memory/shared-state/index.js";
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

// DELIBERATE: shared-state lifecycle aging is a separate integer-turn clock; see the episodic heat note in src/memory/episodic/decay.ts (Tier-3 review).

// Tunes when live shared-state entries become eligible for low-salience demotion.
export const DEFAULT_SHARED_STATE_RECENT_TURN_THRESHOLD = 5;

// Tunes when low-salience shared-state entries become eligible for dormant demotion.
export const DEFAULT_SHARED_STATE_DORMANT_TURN_THRESHOLD = 15;

export type SharedStateLifecycleTransitionKind = "demoted" | "reactivated";

export type SharedStateLifecycleTransitionReason =
  | "old_live_without_structural_pull"
  | "old_low_salience_without_structural_pull"
  | "current_turn_update"
  | "touched_by_patch"
  | "ledger_overlap"
  | "active_canonicalizer_critical"
  | "active_canonicalizer_operational"
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
  | "active_canonicalizer_critical"
  | "active_canonicalizer_operational"
  | "recent_retrieval";

export type LifecycleProtectionStrength = "hard" | "soft";

export type LifecycleProtectionEntry = {
  reason: LifecycleProtectionReason;
  strength: LifecycleProtectionStrength;
};

export type EntryProtectionState = {
  hard: LifecycleProtectionReason[];
  soft: LifecycleProtectionReason[];
};

export const LIFECYCLE_HARD_PROTECTION_REASONS = [
  "touched_by_patch",
  "current_turn_update",
  "ledger_overlap",
  "active_canonicalizer_critical",
] as const satisfies readonly LifecycleProtectionReason[];

export const LIFECYCLE_SOFT_PROTECTION_REASONS = [
  "active_canonicalizer_operational",
  "recent_retrieval",
] as const satisfies readonly LifecycleProtectionReason[];

export const LIFECYCLE_PROTECTION_REASONS = [
  ...LIFECYCLE_HARD_PROTECTION_REASONS,
  ...LIFECYCLE_SOFT_PROTECTION_REASONS,
] as const satisfies readonly LifecycleProtectionReason[];

export type LifecycleAgingBlockerCounts = {
  demotable_count: number;
  unknown_age_count: number;
  demoted_count: number;
  blocked_by_current_turn_update: number;
  blocked_by_patch_touch: number;
  blocked_by_ledger_overlap: number;
  blocked_by_recent_retrieval: number;
  blocked_by_active_canonicalizer_critical: number;
  blocked_by_active_canonicalizer_operational: number;
  blocked_by_hard_total: number;
  blocked_by_soft_total: number;
  blocked_by_multiple_reasons: number;
};

export type LifecycleAgingBlockedSampleEntry = {
  entry_id: SharedStateEntryId;
  state_key: string | null;
  age_turns: number | null;
  block_reasons: LifecycleProtectionReason[];
  block_strengths: LifecycleProtectionStrength[];
  block_reasons_with_strength: LifecycleProtectionEntry[];
  active_canonicalizer_kinds: string[] | null;
};

export type LifecycleAgingUnknownAgeSampleEntry = {
  entry_id: SharedStateEntryId;
  state_key: string | null;
  kind: SharedStateEntryKind;
  last_updated_stream_entry_ids_count: number;
  last_updated_turn_global: number | null;
  rank: number;
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
  activeOperationalCommitmentIds?: readonly CommitmentId[];
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
  lastUpdatedTurnGlobal?: number | null;
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
          last_updated_turn_global:
            operation.last_updated_turn_global ?? input.lastUpdatedTurnGlobal ?? null,
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
          last_updated_turn_global:
            operation.last_updated_turn_global ?? input.lastUpdatedTurnGlobal ?? null,
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
            last_updated_turn_global:
              operation.last_updated_turn_global ?? input.lastUpdatedTurnGlobal ?? null,
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
          last_updated_turn_global:
            operation.replacement.last_updated_turn_global ?? input.lastUpdatedTurnGlobal ?? null,
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

function isTouchedByPatch(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): boolean {
  return input.touchedEntryIds?.has(entry.id) === true;
}

function hasCurrentTurnUpdate(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): boolean {
  return sharedStateEntryHasCurrentTurnUpdate(entry, input.currentUserStreamEntryId);
}

function hasLedgerOverlap(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): boolean {
  return sharedStateEntryHasLedgerOverlap(entry, input.ledgerStreamEntryIds);
}

function hasActiveCriticalCanonicalizerOverlap(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): boolean {
  return hasAnyIdOverlap(
    entry.canonicalizes.commitment_ids,
    idSet(input.activeCriticalCommitmentIds),
  );
}

function hasActiveOperationalCanonicalizerOverlap(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): boolean {
  return (
    hasAnyIdOverlap(entry.canonicalizes.open_question_ids, idSet(input.activeOpenQuestionIds)) ||
    hasAnyIdOverlap(entry.canonicalizes.action_ids, idSet(input.activeActionIds)) ||
    hasAnyIdOverlap(entry.canonicalizes.goal_ids, idSet(input.activeGoalIds)) ||
    hasAnyIdOverlap(entry.canonicalizes.commitment_ids, idSet(input.activeOperationalCommitmentIds))
  );
}

function hasRecentRetrieval(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): boolean {
  return sharedStateEntryWasRecentlyRetrieved(entry, input.recentlyRetrievedEntryIds);
}

function activeCanonicalizerOverlapKinds(
  entry: SharedStateEntry,
  input: Pick<
    ApplyLifecycleAgingInput,
    | "activeOpenQuestionIds"
    | "activeActionIds"
    | "activeGoalIds"
    | "activeCriticalCommitmentIds"
    | "activeOperationalCommitmentIds"
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

  if (
    hasAnyIdOverlap(entry.canonicalizes.commitment_ids, idSet(input.activeOperationalCommitmentIds))
  ) {
    kinds.push("operational_commitment");
  }

  return kinds;
}

export function entryProtectionState(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): EntryProtectionState {
  const hard: LifecycleProtectionReason[] = [];
  const soft: LifecycleProtectionReason[] = [];

  if (isTouchedByPatch(entry, input)) {
    hard.push("touched_by_patch");
  }

  if (hasCurrentTurnUpdate(entry, input)) {
    hard.push("current_turn_update");
  }

  if (hasLedgerOverlap(entry, input)) {
    hard.push("ledger_overlap");
  }

  if (hasActiveCriticalCanonicalizerOverlap(entry, input)) {
    hard.push("active_canonicalizer_critical");
  }

  if (hasActiveOperationalCanonicalizerOverlap(entry, input)) {
    soft.push("active_canonicalizer_operational");
  }

  if (hasRecentRetrieval(entry, input)) {
    soft.push("recent_retrieval");
  }

  return { hard, soft };
}

export function sharedStateLifecycleProtectionReasons(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): LifecycleProtectionEntry[] {
  const protection = entryProtectionState(entry, input);

  return [
    ...protection.hard.map((reason) => ({ reason, strength: "hard" as const })),
    ...protection.soft.map((reason) => ({ reason, strength: "soft" as const })),
  ];
}

export function blocksLiveToLowSalienceDemotion(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): LifecycleProtectionReason | null {
  return entryProtectionState(entry, input).hard[0] ?? null;
}

export function blocksLowSalienceToDormantDemotion(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): LifecycleProtectionReason | null {
  const protection = entryProtectionState(entry, input);
  return protection.hard[0] ?? protection.soft[0] ?? null;
}

export function reactivatesDemoted(
  entry: SharedStateEntry,
  input: ApplyLifecycleAgingInput,
): SharedStateLifecycleTransitionReason | null {
  return entryProtectionState(entry, input).hard[0] ?? null;
}

function entryTurnAge(entry: SharedStateEntry, input: ApplyLifecycleAgingInput): number | null {
  if (input.currentTurnCounter === undefined) {
    return null;
  }

  const lastUpdatedTurn =
    entry.last_updated_turn_global ?? input.lastUpdatedTurnByEntryId?.[entry.id];

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
    blocked_by_active_canonicalizer_critical: 0,
    blocked_by_active_canonicalizer_operational: 0,
    blocked_by_hard_total: 0,
    blocked_by_soft_total: 0,
    blocked_by_multiple_reasons: 0,
  };
}

function lifecycleProtectionStrength(
  reason: LifecycleProtectionReason,
): LifecycleProtectionStrength {
  return LIFECYCLE_HARD_PROTECTION_REASONS.includes(
    reason as (typeof LIFECYCLE_HARD_PROTECTION_REASONS)[number],
  )
    ? "hard"
    : "soft";
}

function incrementBlockerReasonCount(
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
    case "active_canonicalizer_critical":
      counts.blocked_by_active_canonicalizer_critical += 1;
      break;
    case "active_canonicalizer_operational":
      counts.blocked_by_active_canonicalizer_operational += 1;
      break;
    case "recent_retrieval":
      counts.blocked_by_recent_retrieval += 1;
      break;
  }
}

export function recordBlockerCounts(
  counts: LifecycleAgingBlockerCounts,
  reasons: readonly LifecycleProtectionReason[],
): void {
  for (const reason of reasons) {
    incrementBlockerReasonCount(counts, reason);
  }

  if (reasons.some((reason) => lifecycleProtectionStrength(reason) === "hard")) {
    counts.blocked_by_hard_total += 1;
  }

  if (reasons.some((reason) => lifecycleProtectionStrength(reason) === "soft")) {
    counts.blocked_by_soft_total += 1;
  }

  if (reasons.length > 1) {
    counts.blocked_by_multiple_reasons += 1;
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

  recordBlockerCounts(input.counts, input.reasons);

  const activeCanonicalizerKinds = activeCanonicalizerOverlapKinds(input.entry, input.agingInput);
  const blockReasonsWithStrength = input.reasons.map((reason) => ({
    reason,
    strength: lifecycleProtectionStrength(reason),
  }));
  input.blockedSamples.push({
    entry_id: input.entry.id,
    state_key: input.entry.state_key,
    age_turns: input.age,
    block_reasons: input.reasons,
    block_strengths: blockReasonsWithStrength.map((reason) => reason.strength),
    block_reasons_with_strength: blockReasonsWithStrength,
    active_canonicalizer_kinds:
      activeCanonicalizerKinds.length === 0 ? null : activeCanonicalizerKinds,
  });
}

function recordUnknownAgeSample(
  entry: SharedStateEntry,
  unknownAgeSamples: LifecycleAgingUnknownAgeSampleEntry[],
): void {
  unknownAgeSamples.push({
    entry_id: entry.id,
    state_key: entry.state_key,
    kind: entry.kind,
    last_updated_stream_entry_ids_count: entry.last_updated_stream_entry_ids.length,
    last_updated_turn_global: entry.last_updated_turn_global,
    rank: entry.rank,
  });
}

// The ladder is one-way and narrow: live -> low_salience_live -> dormant_live, with reactivation
// only ever back to `live`. `locked` and `tentative` are outside it entirely -- they are never
// demotion candidates and are never produced, so an entry that reads as locked was written locked by
// the model on an add/update, never aged into it.
//
// `currentTurnCounter` is the host-wide action_lifecycle_turn_counter, not a per-audience one: every
// session on the host advances the same integer once per turn. Measured on the live demo data dir
// 2026-08-17, that clock ran ~3.05 turns/hour over the preceding 39 hours, so the 5/15-turn defaults
// above are roughly 1.6h and 5h of wall clock for an audience that is not itself talking. A quiet
// room ages at the whole host's cadence.
//
// Demotion is not a soft shelf, and what it costs is decided in lifecycle-cap.ts: `dormant_live` is
// both the first kind SHARED_STATE_LIFECYCLE_PRUNE_ORDER scans and the kind with the smallest
// default soft cap (1, against 24 for `locked`), so the bottom rung of this ladder is the head of
// the eviction queue. Observed on the live demo data dir 2026-08-17: one over-cap draw took two
// `dormant_live` rows -- the younger of them last written five hours earlier -- while 15 `locked`
// rows staler than it survived the same draw, because kind chooses the pool before staleness orders
// it. So an entry nobody touches for ~15 host turns is not shelved, it is queued: the band it lands
// in holds one.
export function applyLifecycleAging(input: ApplyLifecycleAgingInput): {
  transitions: SharedStateLifecycleTransition[];
  blockerCountsLiveToLowSalience: LifecycleAgingBlockerCounts;
  blockerCountsLowSalienceToDormant: LifecycleAgingBlockerCounts;
  blockedSample: LifecycleAgingBlockedSampleEntry[];
  unknownAgeSample: LifecycleAgingUnknownAgeSampleEntry[];
} {
  const transitions: SharedStateLifecycleTransition[] = [];
  const transitionedEntryIds = new Set<SharedStateEntryId>();
  const blockerCountsLiveToLowSalience = emptyLifecycleAgingBlockerCounts();
  const blockerCountsLowSalienceToDormant = emptyLifecycleAgingBlockerCounts();
  const blockedSamples: LifecycleAgingBlockedSampleEntry[] = [];
  const unknownAgeSamples: LifecycleAgingUnknownAgeSampleEntry[] = [];
  const recentTurnThreshold = normalizedThreshold(
    input.recentTurnThreshold,
    DEFAULT_SHARED_STATE_RECENT_TURN_THRESHOLD,
  );
  const dormantTurnThreshold = normalizedThreshold(
    input.dormantTurnThreshold,
    DEFAULT_SHARED_STATE_DORMANT_TURN_THRESHOLD,
  );
  const activeEntries = input.entries.filter((entry) => entry.superseded_by_id === null);

  for (const entry of activeEntries) {
    if (entry.kind !== "low_salience_live" && entry.kind !== "dormant_live") {
      continue;
    }

    const reason = reactivatesDemoted(entry, input);

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
    const protection = entryProtectionState(entry, input);
    const liveToLowSalienceBlockReasons = protection.hard;
    const lowSalienceToDormantBlockReasons = [...protection.hard, ...protection.soft];

    if (entry.kind === "live" && age === null) {
      blockerCountsLiveToLowSalience.unknown_age_count += 1;
      recordUnknownAgeSample(entry, unknownAgeSamples);
    }

    if (entry.kind === "low_salience_live" && age === null) {
      blockerCountsLowSalienceToDormant.unknown_age_count += 1;
    }

    if (entry.kind === "live" && age !== null && age > recentTurnThreshold) {
      const demoted = liveToLowSalienceBlockReasons.length === 0;
      recordDemotionCandidate({
        entry,
        age,
        reasons: liveToLowSalienceBlockReasons,
        counts: blockerCountsLiveToLowSalience,
        blockedSamples,
        demoted,
        agingInput: input,
      });
    }

    if (entry.kind === "low_salience_live" && age !== null && age > dormantTurnThreshold) {
      const demoted = lowSalienceToDormantBlockReasons.length === 0;
      recordDemotionCandidate({
        entry,
        age,
        reasons: lowSalienceToDormantBlockReasons,
        counts: blockerCountsLowSalienceToDormant,
        blockedSamples,
        demoted,
        agingInput: input,
      });
    }

    if (age === null) {
      continue;
    }

    if (
      entry.kind === "live" &&
      age > recentTurnThreshold &&
      blocksLiveToLowSalienceDemotion(entry, input) === null
    ) {
      transitions.push({
        entryId: entry.id,
        fromKind: "live",
        toKind: "low_salience_live",
        reason: "old_live_without_structural_pull",
        transition: "demoted",
      });
      continue;
    }

    if (
      entry.kind === "low_salience_live" &&
      age > dormantTurnThreshold &&
      blocksLowSalienceToDormantDemotion(entry, input) === null
    ) {
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
    unknownAgeSample: unknownAgeSamples
      .sort((left, right) => left.rank - right.rank || left.entry_id.localeCompare(right.entry_id))
      .slice(0, 10),
  };
}
