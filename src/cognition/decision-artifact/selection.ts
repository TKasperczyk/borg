import {
  DECISION_ARTIFACT_ENTRY_KINDS,
  type DecisionArtifact,
  type DecisionArtifactEntry,
  type DecisionArtifactEntryKind,
} from "../../memory/decision-artifacts/index.js";

export const DECISION_ARTIFACT_RESERVED_KINDS = [
  "live",
  "invalidated",
  "pending",
] as const satisfies readonly DecisionArtifactEntryKind[];

export const DECISION_ARTIFACT_RENDER_FILL_ORDER = [
  "live",
  "pending",
  "invalidated",
  "locked",
  "tentative",
] as const satisfies readonly DecisionArtifactEntryKind[];

export type DecisionArtifactKindCounts = Record<DecisionArtifactEntryKind, number>;

export function emptyDecisionArtifactKindCounts(): DecisionArtifactKindCounts {
  return Object.fromEntries(
    DECISION_ARTIFACT_ENTRY_KINDS.map((kind) => [kind, 0]),
  ) as DecisionArtifactKindCounts;
}

export function countDecisionArtifactEntriesByKind(
  entries: readonly DecisionArtifactEntry[],
): DecisionArtifactKindCounts {
  const counts = emptyDecisionArtifactKindCounts();

  for (const entry of entries) {
    counts[entry.kind] += 1;
  }

  return counts;
}

export function subtractDecisionArtifactKindCounts(
  left: DecisionArtifactKindCounts,
  right: DecisionArtifactKindCounts,
): DecisionArtifactKindCounts {
  const counts = emptyDecisionArtifactKindCounts();

  for (const kind of DECISION_ARTIFACT_ENTRY_KINDS) {
    counts[kind] = Math.max(0, left[kind] - right[kind]);
  }

  return counts;
}

export function activeDecisionArtifactEntries(
  artifact: DecisionArtifact | null | undefined,
): DecisionArtifactEntry[] {
  return (artifact?.entries ?? []).filter((entry) => entry.superseded_by_id === null);
}

export function compareDecisionArtifactEntriesByRecency(
  left: DecisionArtifactEntry,
  right: DecisionArtifactEntry,
): number {
  return (
    right.last_updated_at - left.last_updated_at ||
    left.rank - right.rank ||
    right.created_at - left.created_at ||
    left.id.localeCompare(right.id)
  );
}

export function selectDecisionArtifactEntriesForRender(input: {
  entries: readonly DecisionArtifactEntry[];
  maxEntries: number;
  reservedSlots: Partial<Record<DecisionArtifactEntryKind, number>>;
  lockedMaxEntries: number;
}): DecisionArtifactEntry[] {
  const byKind = new Map<DecisionArtifactEntryKind, DecisionArtifactEntry[]>();

  for (const kind of DECISION_ARTIFACT_ENTRY_KINDS) {
    byKind.set(
      kind,
      input.entries
        .filter((entry) => entry.kind === kind)
        .sort(compareDecisionArtifactEntriesByRecency),
    );
  }

  const selected: DecisionArtifactEntry[] = [];
  const selectedIds = new Set<DecisionArtifactEntry["id"]>();
  const selectedByKind = emptyDecisionArtifactKindCounts();

  const takeFromKind = (kind: DecisionArtifactEntryKind, limit: number): void => {
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

  for (const kind of DECISION_ARTIFACT_RESERVED_KINDS) {
    takeFromKind(kind, input.reservedSlots[kind] ?? 0);
  }

  for (const kind of DECISION_ARTIFACT_RENDER_FILL_ORDER) {
    const categoryLimit = kind === "locked" ? input.lockedMaxEntries : Number.POSITIVE_INFINITY;
    takeFromKind(kind, categoryLimit);
  }

  const orderByKind = new Map(
    DECISION_ARTIFACT_RENDER_FILL_ORDER.map((kind, index) => [kind, index]),
  );

  return selected.sort(
    (left, right) =>
      (orderByKind.get(left.kind) ?? Number.MAX_SAFE_INTEGER) -
        (orderByKind.get(right.kind) ?? Number.MAX_SAFE_INTEGER) ||
      compareDecisionArtifactEntriesByRecency(left, right),
  );
}

export function onePerKindTokenDropFloor(
  kind: DecisionArtifactEntryKind,
  activeCounts: DecisionArtifactKindCounts,
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
  kind: DecisionArtifactEntryKind;
  activeCounts: DecisionArtifactKindCounts;
  reservedSlots: Partial<Record<DecisionArtifactEntryKind, number>>;
}): number {
  const floor = onePerKindTokenDropFloor(input.kind, input.activeCounts);
  const reserved = input.reservedSlots[input.kind] ?? 0;

  if (reserved <= 0) {
    return floor;
  }

  return Math.max(floor, Math.min(input.activeCounts[input.kind], Math.floor(reserved)));
}

export function latestDecisionArtifactDropIndex(
  entries: readonly DecisionArtifactEntry[],
  kind: DecisionArtifactEntryKind,
): number | null {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    if (entries[index]?.kind === kind) {
      return index;
    }
  }

  return null;
}

export function tokenDropIndexForKinds(input: {
  entries: readonly DecisionArtifactEntry[];
  kinds: readonly DecisionArtifactEntryKind[];
  minimumForKind: (kind: DecisionArtifactEntryKind) => number;
}): number | null {
  const renderedCounts = countDecisionArtifactEntriesByKind(input.entries);
  let selectedKind: DecisionArtifactEntryKind | null = null;
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
    : latestDecisionArtifactDropIndex(input.entries, selectedKind);
}

export function tokenDropIndex(input: {
  entries: readonly DecisionArtifactEntry[];
  activeCounts: DecisionArtifactKindCounts;
  reservedSlots: Partial<Record<DecisionArtifactEntryKind, number>>;
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
    kinds: DECISION_ARTIFACT_RESERVED_KINDS,
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
    kinds: DECISION_ARTIFACT_RESERVED_KINDS,
    minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
  });

  if (dropReservedAboveFloor !== null) {
    return dropReservedAboveFloor;
  }

  return (
    tokenDropIndexForKinds({
      entries: input.entries,
      kinds: DECISION_ARTIFACT_ENTRY_KINDS,
      minimumForKind: (kind) => onePerKindTokenDropFloor(kind, input.activeCounts),
    }) ?? Math.max(0, input.entries.length - 1)
  );
}
