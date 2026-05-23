import {
  SHARED_STATE_ENTRY_KINDS,
  type SharedStateArtifact,
  type SharedStateEntry,
  type SharedStateEntryKind,
} from "../../memory/decision-artifacts/index.js";
import type {
  ActionId,
  CommitmentId,
  GoalId,
  OpenQuestionId,
  StreamEntryId,
} from "../../util/ids.js";

export const SHARED_STATE_RESERVED_KINDS = [
  "live",
  "invalidated",
  "pending",
] as const satisfies readonly SharedStateEntryKind[];

export const SHARED_STATE_RENDER_FILL_ORDER = [
  "live",
  "pending",
  "invalidated",
  "locked",
  "low_salience_live",
  "dormant_live",
  "tentative",
] as const satisfies readonly SharedStateEntryKind[];

export type SharedStateKindCounts = Record<SharedStateEntryKind, number>;

export function emptySharedStateKindCounts(): SharedStateKindCounts {
  return Object.fromEntries(
    SHARED_STATE_ENTRY_KINDS.map((kind) => [kind, 0]),
  ) as SharedStateKindCounts;
}

export function countSharedStateArtifactEntriesByKind(
  entries: readonly SharedStateEntry[],
): SharedStateKindCounts {
  const counts = emptySharedStateKindCounts();

  for (const entry of entries) {
    counts[entry.kind] += 1;
  }

  return counts;
}

export function subtractSharedStateKindCounts(
  left: SharedStateKindCounts,
  right: SharedStateKindCounts,
): SharedStateKindCounts {
  const counts = emptySharedStateKindCounts();

  for (const kind of SHARED_STATE_ENTRY_KINDS) {
    counts[kind] = Math.max(0, left[kind] - right[kind]);
  }

  return counts;
}

export function activeSharedStateArtifactEntries(
  artifact: SharedStateArtifact | null | undefined,
): SharedStateEntry[] {
  return (artifact?.entries ?? []).filter((entry) => entry.superseded_by_id === null);
}

export function compareSharedStateArtifactEntriesByRecency(
  left: SharedStateEntry,
  right: SharedStateEntry,
): number {
  return (
    right.last_updated_at - left.last_updated_at ||
    left.rank - right.rank ||
    right.created_at - left.created_at ||
    left.id.localeCompare(right.id)
  );
}

function compareSharedStateArtifactEntriesByNewestStateChange(
  left: SharedStateEntry,
  right: SharedStateEntry,
): number {
  return (
    right.last_updated_at - left.last_updated_at ||
    right.created_at - left.created_at ||
    left.rank - right.rank ||
    left.id.localeCompare(right.id)
  );
}

function newestStateChangeReservedIds(input: {
  entries: readonly SharedStateEntry[];
  limit: number;
}): Set<SharedStateEntry["id"]> {
  if (input.limit <= 0) {
    return new Set<SharedStateEntry["id"]>();
  }

  return new Set(
    input.entries
      .filter((entry) => entry.kind === "live" || entry.kind === "pending")
      .sort(compareSharedStateArtifactEntriesByNewestStateChange)
      .slice(0, input.limit)
      .map((entry) => entry.id),
  );
}

export type SharedStateRenderSalienceOptions = {
  currentUserStreamEntryId?: StreamEntryId;
  ledgerStreamEntryIds?: readonly StreamEntryId[];
  recentlyRetrievedEntryIds?: readonly SharedStateEntry["id"][];
  activeOpenQuestionIds?: readonly OpenQuestionId[];
  activeActionIds?: readonly ActionId[];
  activeGoalIds?: readonly GoalId[];
  activeCriticalCommitmentIds?: readonly CommitmentId[];
  activeOperationalCommitmentIds?: readonly CommitmentId[];
};

export type SharedStateRenderSelection = {
  entries: SharedStateEntry[];
  newestReservedIds: Set<SharedStateEntry["id"]>;
  salienceReservedIds: Set<SharedStateEntry["id"]>;
  dropTiers: Map<SharedStateEntry["id"], SharedStateTokenDropTier>;
};

export type SharedStateTokenDropTier = 1 | 2 | 3 | 4 | 5 | 6 | 7;

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

export function sharedStateEntryHasCurrentTurnUpdate(
  entry: SharedStateEntry,
  currentUserStreamEntryId: StreamEntryId | undefined,
): boolean {
  return (
    currentUserStreamEntryId !== undefined &&
    entry.last_updated_stream_entry_ids.includes(currentUserStreamEntryId)
  );
}

export function sharedStateEntryHasLedgerOverlap(
  entry: SharedStateEntry,
  ledgerStreamEntryIds: readonly StreamEntryId[] | undefined,
): boolean {
  const ledgerIds = idSet(ledgerStreamEntryIds);

  return (
    hasAnyIdOverlap(entry.provenance_stream_entry_ids, ledgerIds) ||
    hasAnyIdOverlap(entry.last_updated_stream_entry_ids, ledgerIds)
  );
}

export function sharedStateEntryWasRecentlyRetrieved(
  entry: SharedStateEntry,
  recentlyRetrievedEntryIds: readonly SharedStateEntry["id"][] | undefined,
): boolean {
  return idSet(recentlyRetrievedEntryIds).has(entry.id);
}

export function sharedStateEntryHasOperationalCanonicalizer(
  entry: SharedStateEntry,
  options: Pick<
    SharedStateRenderSalienceOptions,
    "activeOpenQuestionIds" | "activeActionIds" | "activeGoalIds"
  >,
): boolean {
  return (
    hasAnyIdOverlap(entry.canonicalizes.open_question_ids, idSet(options.activeOpenQuestionIds)) ||
    hasAnyIdOverlap(entry.canonicalizes.action_ids, idSet(options.activeActionIds)) ||
    hasAnyIdOverlap(entry.canonicalizes.goal_ids, idSet(options.activeGoalIds))
  );
}

export function sharedStateEntryHasAnyOperationalCanonicalizer(entry: SharedStateEntry): boolean {
  return (
    entry.canonicalizes.open_question_ids.length > 0 ||
    entry.canonicalizes.action_ids.length > 0 ||
    entry.canonicalizes.goal_ids.length > 0
  );
}

export function sharedStateEntryHasCriticalCommitmentCanonicalizer(
  entry: SharedStateEntry,
  activeCriticalCommitmentIds: readonly CommitmentId[] | undefined,
): boolean {
  return (
    entry.kind === "locked" &&
    hasAnyIdOverlap(entry.canonicalizes.commitment_ids, idSet(activeCriticalCommitmentIds))
  );
}

function sharedStateEntrySalienceScore(
  entry: SharedStateEntry,
  options: SharedStateRenderSalienceOptions,
  newestReservedIds: ReadonlySet<SharedStateEntry["id"]> = new Set(),
): number {
  if (sharedStateEntryHasCurrentTurnUpdate(entry, options.currentUserStreamEntryId)) {
    return 600;
  }

  if (sharedStateEntryHasLedgerOverlap(entry, options.ledgerStreamEntryIds)) {
    return 500;
  }

  if (sharedStateEntryWasRecentlyRetrieved(entry, options.recentlyRetrievedEntryIds)) {
    return 500;
  }

  if (newestReservedIds.has(entry.id)) {
    return 400;
  }

  if (entry.kind === "pending" || entry.kind === "invalidated") {
    return 300;
  }

  if (
    sharedStateEntryHasCriticalCommitmentCanonicalizer(entry, options.activeCriticalCommitmentIds)
  ) {
    return 250;
  }

  if (sharedStateEntryHasOperationalCanonicalizer(entry, options)) {
    return 200;
  }

  return 0;
}

function salienceReservedIds(input: {
  entries: readonly SharedStateEntry[];
  options: SharedStateRenderSalienceOptions;
  newestReservedIds?: ReadonlySet<SharedStateEntry["id"]>;
}): Set<SharedStateEntry["id"]> {
  return new Set(
    input.entries
      .map((entry) => ({
        entry,
        score: sharedStateEntrySalienceScore(entry, input.options, input.newestReservedIds),
      }))
      .filter((candidate) => candidate.score > 0)
      .sort(
        (left, right) =>
          right.score - left.score ||
          compareSharedStateArtifactEntriesByRecency(left.entry, right.entry),
      )
      .map((candidate) => candidate.entry.id),
  );
}

function reservedKindSlotIds(input: {
  entries: readonly SharedStateEntry[];
  reservedSlots: Partial<Record<SharedStateEntryKind, number>>;
}): Set<SharedStateEntry["id"]> {
  const ids = new Set<SharedStateEntry["id"]>();

  for (const kind of SHARED_STATE_RESERVED_KINDS) {
    const limit = input.reservedSlots[kind] ?? 0;

    if (limit <= 0) {
      continue;
    }

    for (const entry of input.entries
      .filter((candidate) => candidate.kind === kind)
      .sort(compareSharedStateArtifactEntriesByRecency)
      .slice(0, Math.floor(limit))) {
      ids.add(entry.id);
    }
  }

  return ids;
}

function sharedStateEntryTokenDropTier(input: {
  entry: SharedStateEntry;
  options: SharedStateRenderSalienceOptions;
  newestReservedIds: ReadonlySet<SharedStateEntry["id"]>;
  reservedKindSlotIds: ReadonlySet<SharedStateEntry["id"]>;
}): SharedStateTokenDropTier {
  if (sharedStateEntryHasCurrentTurnUpdate(input.entry, input.options.currentUserStreamEntryId)) {
    return 1;
  }

  if (sharedStateEntryHasLedgerOverlap(input.entry, input.options.ledgerStreamEntryIds)) {
    return 2;
  }

  if (input.newestReservedIds.has(input.entry.id)) {
    return 3;
  }

  if (input.reservedKindSlotIds.has(input.entry.id)) {
    return 4;
  }

  if (
    sharedStateEntryHasCriticalCommitmentCanonicalizer(
      input.entry,
      input.options.activeCriticalCommitmentIds,
    )
  ) {
    return 5;
  }

  if (sharedStateEntryHasOperationalCanonicalizer(input.entry, input.options)) {
    return 6;
  }

  return 7;
}

function sharedStateEntryTokenDropTiers(input: {
  entries: readonly SharedStateEntry[];
  options: SharedStateRenderSalienceOptions;
  newestReservedIds: ReadonlySet<SharedStateEntry["id"]>;
  reservedKindSlotIds: ReadonlySet<SharedStateEntry["id"]>;
}): Map<SharedStateEntry["id"], SharedStateTokenDropTier> {
  return new Map(
    input.entries.map((entry) => [
      entry.id,
      sharedStateEntryTokenDropTier({
        entry,
        options: input.options,
        newestReservedIds: input.newestReservedIds,
        reservedKindSlotIds: input.reservedKindSlotIds,
      }),
    ]),
  );
}

function tokenDropTier(
  entry: SharedStateEntry,
  dropTiers: ReadonlyMap<SharedStateEntry["id"], SharedStateTokenDropTier> | undefined,
): SharedStateTokenDropTier {
  return dropTiers?.get(entry.id) ?? 7;
}

function sharedStateTokenDropTierOrder(
  entries: readonly SharedStateEntry[],
  dropTiers: ReadonlyMap<SharedStateEntry["id"], SharedStateTokenDropTier> | undefined,
): SharedStateTokenDropTier[] {
  return [
    ...new Set(
      entries.map((entry) => tokenDropTier(entry, dropTiers)).sort((left, right) => right - left),
    ),
  ];
}

export function selectSharedStateArtifactEntriesForRenderWithSummary(input: {
  entries: readonly SharedStateEntry[];
  maxEntries: number;
  reservedSlots: Partial<Record<SharedStateEntryKind, number>>;
  lockedMaxEntries: number;
  newestStateChangeReservedSlots?: number;
  salience?: SharedStateRenderSalienceOptions;
}): SharedStateRenderSelection {
  const byKind = new Map<SharedStateEntryKind, SharedStateEntry[]>();

  for (const kind of SHARED_STATE_ENTRY_KINDS) {
    byKind.set(
      kind,
      input.entries
        .filter((entry) => entry.kind === kind)
        .sort(compareSharedStateArtifactEntriesByRecency),
    );
  }

  const selected: SharedStateEntry[] = [];
  const selectedIds = new Set<SharedStateEntry["id"]>();
  const selectedByKind = emptySharedStateKindCounts();
  const newestReservedIds = newestStateChangeReservedIds({
    entries: input.entries,
    limit: input.newestStateChangeReservedSlots ?? 0,
  });
  const salienceIds = salienceReservedIds({
    entries: input.entries,
    options: input.salience ?? {},
    newestReservedIds,
  });
  const reservedSlotIds = reservedKindSlotIds({
    entries: input.entries,
    reservedSlots: input.reservedSlots,
  });
  const dropTiers = sharedStateEntryTokenDropTiers({
    entries: input.entries,
    options: input.salience ?? {},
    newestReservedIds,
    reservedKindSlotIds: reservedSlotIds,
  });

  const takeEntry = (entry: SharedStateEntry, options: { countBudget: boolean }): void => {
    selected.push(entry);
    selectedIds.add(entry.id);

    if (options.countBudget) {
      selectedByKind[entry.kind] += 1;
    }
  };

  const takeFromKind = (
    kind: SharedStateEntryKind,
    limit: number,
    options: { countBudget?: boolean } = {},
  ): void => {
    if (limit <= 0 || selected.length >= input.maxEntries) {
      return;
    }

    const candidates = byKind.get(kind) ?? [];

    for (const candidate of candidates) {
      if (selected.length >= input.maxEntries || selectedByKind[kind] >= limit) {
        return;
      }

      if (kind === "locked" && selectedByKind.locked >= input.lockedMaxEntries) {
        return;
      }

      if (selectedIds.has(candidate.id)) {
        continue;
      }

      takeEntry(candidate, { countBudget: options.countBudget !== false });
    }
  };

  for (const candidate of input.entries
    .filter((entry) => salienceIds.has(entry.id))
    .sort((left, right) => {
      const salience = input.salience ?? {};
      const leftScore = sharedStateEntrySalienceScore(left, salience, newestReservedIds);
      const rightScore = sharedStateEntrySalienceScore(right, salience, newestReservedIds);

      return rightScore - leftScore || compareSharedStateArtifactEntriesByRecency(left, right);
    })) {
    if (selected.length >= input.maxEntries) {
      break;
    }

    if (!selectedIds.has(candidate.id)) {
      takeEntry(candidate, { countBudget: false });
    }
  }

  for (const candidate of input.entries
    .filter((entry) => newestReservedIds.has(entry.id))
    .sort(compareSharedStateArtifactEntriesByNewestStateChange)) {
    if (selected.length >= input.maxEntries) {
      break;
    }

    if (!selectedIds.has(candidate.id)) {
      takeEntry(candidate, { countBudget: false });
    }
  }

  for (const kind of SHARED_STATE_RESERVED_KINDS) {
    takeFromKind(kind, input.reservedSlots[kind] ?? 0);
  }

  for (const kind of SHARED_STATE_RENDER_FILL_ORDER) {
    const categoryLimit = kind === "locked" ? input.lockedMaxEntries : Number.POSITIVE_INFINITY;
    takeFromKind(kind, categoryLimit);
  }

  const orderByKind = new Map(SHARED_STATE_RENDER_FILL_ORDER.map((kind, index) => [kind, index]));

  return {
    entries: selected.sort(
      (left, right) =>
        (orderByKind.get(left.kind) ?? Number.MAX_SAFE_INTEGER) -
          (orderByKind.get(right.kind) ?? Number.MAX_SAFE_INTEGER) ||
        compareSharedStateArtifactEntriesByRecency(left, right),
    ),
    newestReservedIds,
    salienceReservedIds: salienceIds,
    dropTiers,
  };
}

export function selectSharedStateArtifactEntriesForRender(input: {
  entries: readonly SharedStateEntry[];
  maxEntries: number;
  reservedSlots: Partial<Record<SharedStateEntryKind, number>>;
  lockedMaxEntries: number;
  newestStateChangeReservedSlots?: number;
  salience?: SharedStateRenderSalienceOptions;
}): SharedStateEntry[] {
  return selectSharedStateArtifactEntriesForRenderWithSummary(input).entries;
}

export function onePerKindTokenDropFloor(
  kind: SharedStateEntryKind,
  activeCounts: SharedStateKindCounts,
): number {
  if (activeCounts[kind] <= 0) {
    return 0;
  }

  if (kind === "tentative" || kind === "low_salience_live" || kind === "dormant_live") {
    return 0;
  }

  return 1;
}

export function reservedTokenDropMinimum(input: {
  kind: SharedStateEntryKind;
  activeCounts: SharedStateKindCounts;
  reservedSlots: Partial<Record<SharedStateEntryKind, number>>;
}): number {
  const floor = onePerKindTokenDropFloor(input.kind, input.activeCounts);
  const reserved = input.reservedSlots[input.kind] ?? 0;

  if (reserved <= 0) {
    return floor;
  }

  return Math.max(floor, Math.min(input.activeCounts[input.kind], Math.floor(reserved)));
}

export function latestSharedStateArtifactDropIndex(
  entries: readonly SharedStateEntry[],
  kind: SharedStateEntryKind,
  dropTier?: SharedStateTokenDropTier,
  dropTiers?: ReadonlyMap<SharedStateEntry["id"], SharedStateTokenDropTier>,
): number | null {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];

    if (
      entry?.kind === kind &&
      (dropTier === undefined || tokenDropTier(entry, dropTiers) === dropTier)
    ) {
      return index;
    }
  }

  return null;
}

export function tokenDropIndexForKinds(input: {
  entries: readonly SharedStateEntry[];
  kinds: readonly SharedStateEntryKind[];
  minimumForKind: (kind: SharedStateEntryKind) => number;
  dropTier?: SharedStateTokenDropTier;
  dropTiers?: ReadonlyMap<SharedStateEntry["id"], SharedStateTokenDropTier>;
}): number | null {
  const renderedCounts = countSharedStateArtifactEntriesByKind(input.entries);
  let selectedIndex: number | null = null;
  let selectedSurplus = 0;

  for (const kind of input.kinds) {
    const surplus = renderedCounts[kind] - input.minimumForKind(kind);

    if (surplus > selectedSurplus) {
      const dropIndex = latestSharedStateArtifactDropIndex(
        input.entries,
        kind,
        input.dropTier,
        input.dropTiers,
      );

      if (dropIndex === null) {
        continue;
      }

      selectedIndex = dropIndex;
      selectedSurplus = surplus;
    }
  }

  return selectedIndex;
}

function latestSharedStateArtifactDropIndexForTier(
  entries: readonly SharedStateEntry[],
  dropTier: SharedStateTokenDropTier,
  dropTiers: ReadonlyMap<SharedStateEntry["id"], SharedStateTokenDropTier> | undefined,
): number | null {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];

    if (entry !== undefined && tokenDropTier(entry, dropTiers) === dropTier) {
      return index;
    }
  }

  return null;
}

export function tokenDropIndex(input: {
  entries: readonly SharedStateEntry[];
  activeCounts: SharedStateKindCounts;
  reservedSlots: Partial<Record<SharedStateEntryKind, number>>;
  lockedMaxEntries: number;
  dropTiers?: ReadonlyMap<SharedStateEntry["id"], SharedStateTokenDropTier>;
}): number | null {
  for (const dropTier of sharedStateTokenDropTierOrder(input.entries, input.dropTiers)) {
    const dropTentative = tokenDropIndexForKinds({
      entries: input.entries,
      kinds: ["tentative", "dormant_live", "low_salience_live"],
      minimumForKind: () => 0,
      dropTier,
      dropTiers: input.dropTiers,
    });

    if (dropTentative !== null) {
      return dropTentative;
    }

    const dropLockedAboveCap = tokenDropIndexForKinds({
      entries: input.entries,
      kinds: ["locked"],
      minimumForKind: () => input.lockedMaxEntries,
      dropTier,
      dropTiers: input.dropTiers,
    });

    if (dropLockedAboveCap !== null) {
      return dropLockedAboveCap;
    }

    const dropReservedAboveMinimum = tokenDropIndexForKinds({
      entries: input.entries,
      kinds: SHARED_STATE_RESERVED_KINDS,
      minimumForKind: (kind) =>
        reservedTokenDropMinimum({
          kind,
          activeCounts: input.activeCounts,
          reservedSlots: input.reservedSlots,
        }),
      dropTier,
      dropTiers: input.dropTiers,
    });

    if (dropReservedAboveMinimum !== null) {
      return dropReservedAboveMinimum;
    }

    const dropLockedAboveFloor = tokenDropIndexForKinds({
      entries: input.entries,
      kinds: ["locked"],
      minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
      dropTier,
      dropTiers: input.dropTiers,
    });

    if (dropLockedAboveFloor !== null) {
      return dropLockedAboveFloor;
    }

    const dropReservedAboveFloor = tokenDropIndexForKinds({
      entries: input.entries,
      kinds: SHARED_STATE_RESERVED_KINDS,
      minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
      dropTier,
      dropTiers: input.dropTiers,
    });

    if (dropReservedAboveFloor !== null) {
      return dropReservedAboveFloor;
    }

    const dropAnyAboveFloor = tokenDropIndexForKinds({
      entries: input.entries,
      kinds: SHARED_STATE_ENTRY_KINDS,
      minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
      dropTier,
      dropTiers: input.dropTiers,
    });

    if (dropAnyAboveFloor !== null) {
      return dropAnyAboveFloor;
    }

    const dropAnyInTier =
      dropTier >= 5
        ? latestSharedStateArtifactDropIndexForTier(input.entries, dropTier, input.dropTiers)
        : null;

    if (dropAnyInTier !== null) {
      return dropAnyInTier;
    }
  }

  return null;
}
