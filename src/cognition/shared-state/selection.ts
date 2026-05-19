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

export function selectSharedStateArtifactEntriesForRender(input: {
  entries: readonly SharedStateEntry[];
  maxEntries: number;
  reservedSlots: Partial<Record<SharedStateEntryKind, number>>;
  lockedMaxEntries: number;
}): SharedStateEntry[] {
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

  const takeFromKind = (kind: SharedStateEntryKind, limit: number): void => {
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

      selected.push(candidate);
      selectedIds.add(candidate.id);
      selectedByKind[kind] += 1;
    }
  };

  for (const kind of SHARED_STATE_RESERVED_KINDS) {
    takeFromKind(kind, input.reservedSlots[kind] ?? 0);
  }

  for (const kind of SHARED_STATE_RENDER_FILL_ORDER) {
    const categoryLimit = kind === "locked" ? input.lockedMaxEntries : Number.POSITIVE_INFINITY;
    takeFromKind(kind, categoryLimit);
  }

  const orderByKind = new Map(SHARED_STATE_RENDER_FILL_ORDER.map((kind, index) => [kind, index]));

  return selected.sort(
    (left, right) =>
      (orderByKind.get(left.kind) ?? Number.MAX_SAFE_INTEGER) -
        (orderByKind.get(right.kind) ?? Number.MAX_SAFE_INTEGER) ||
      compareSharedStateArtifactEntriesByRecency(left, right),
  );
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
