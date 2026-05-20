import {
  SHARED_STATE_ENTRY_KINDS,
  type SharedStateArtifact,
  type SharedStateEntry,
  type SharedStateEntryKind,
} from "../../memory/decision-artifacts/index.js";

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

export type SharedStateRenderSelection = {
  entries: SharedStateEntry[];
  newestReservedIds: Set<SharedStateEntry["id"]>;
};

export function selectSharedStateArtifactEntriesForRenderWithSummary(input: {
  entries: readonly SharedStateEntry[];
  maxEntries: number;
  reservedSlots: Partial<Record<SharedStateEntryKind, number>>;
  lockedMaxEntries: number;
  newestStateChangeReservedSlots?: number;
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
  };
}

export function selectSharedStateArtifactEntriesForRender(input: {
  entries: readonly SharedStateEntry[];
  maxEntries: number;
  reservedSlots: Partial<Record<SharedStateEntryKind, number>>;
  lockedMaxEntries: number;
  newestStateChangeReservedSlots?: number;
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

  if (kind === "tentative") {
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
): number | null {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    if (entries[index]?.kind === kind) {
      return index;
    }
  }

  return null;
}

export function tokenDropIndexForKinds(input: {
  entries: readonly SharedStateEntry[];
  kinds: readonly SharedStateEntryKind[];
  minimumForKind: (kind: SharedStateEntryKind) => number;
}): number | null {
  const renderedCounts = countSharedStateArtifactEntriesByKind(input.entries);
  let selectedKind: SharedStateEntryKind | null = null;
  let selectedSurplus = 0;

  for (const kind of input.kinds) {
    const surplus = renderedCounts[kind] - input.minimumForKind(kind);

    if (surplus > selectedSurplus) {
      selectedKind = kind;
      selectedSurplus = surplus;
    }
  }

  return selectedKind === null
    ? null
    : latestSharedStateArtifactDropIndex(input.entries, selectedKind);
}

export function tokenDropIndex(input: {
  entries: readonly SharedStateEntry[];
  activeCounts: SharedStateKindCounts;
  reservedSlots: Partial<Record<SharedStateEntryKind, number>>;
  lockedMaxEntries: number;
}): number {
  const dropTentative = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["tentative"],
    minimumForKind: () => 0,
  });

  if (dropTentative !== null) {
    return dropTentative;
  }

  const dropLockedAboveCap = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["locked"],
    minimumForKind: () => input.lockedMaxEntries,
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
  });

  if (dropReservedAboveMinimum !== null) {
    return dropReservedAboveMinimum;
  }

  const dropLockedAboveFloor = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: ["locked"],
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });

  if (dropLockedAboveFloor !== null) {
    return dropLockedAboveFloor;
  }

  const dropReservedAboveFloor = tokenDropIndexForKinds({
    entries: input.entries,
    kinds: SHARED_STATE_RESERVED_KINDS,
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });

  if (dropReservedAboveFloor !== null) {
    return dropReservedAboveFloor;
  }

  return (
    tokenDropIndexForKinds({
      entries: input.entries,
      kinds: SHARED_STATE_ENTRY_KINDS,
      minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
    }) ?? Math.max(0, input.entries.length - 1)
  );
}
