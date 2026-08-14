import {
  SHARED_STATE_ENTRY_KINDS,
  type SharedStateArtifact,
  type SharedStateEntry,
  type SharedStateEntryKind,
  type SharedStateOperation,
} from "../../memory/shared-state/index.js";
import { createSharedStateEntryId, type SharedStateEntryId } from "../../util/ids.js";
import type { SharedStateLifecycleOptions } from "./types.js";

const DEFAULT_MAX_ACTIVE_SHARED_STATE_ENTRIES = 40;
const DEFAULT_SHARED_STATE_KIND_SOFT_CAPS = {
  locked: 24,
  live: 10,
  low_salience_live: 4,
  dormant_live: 1,
  invalidated: 4,
  tentative: 2,
} as const satisfies Partial<Record<SharedStateEntryKind, number>>;
const SHARED_STATE_LIFECYCLE_CAPPED_KINDS = [
  "locked",
  "live",
  "low_salience_live",
  "dormant_live",
  "invalidated",
  "tentative",
] as const satisfies readonly SharedStateEntryKind[];
const DEFAULT_NEWEST_STATE_CHANGE_RESERVED_SLOTS = 3;
const SHARED_STATE_LIFECYCLE_PRUNE_ORDER = [
  "dormant_live",
  "low_salience_live",
  "live",
  "tentative",
  "invalidated",
  "locked",
] as const satisfies readonly SharedStateEntryKind[];

type LifecycleEntry = Pick<
  SharedStateEntry,
  "id" | "kind" | "created_at" | "last_updated_at" | "superseded_by_id" | "rank"
>;

function normalizeLifecycleKindSoftCaps(
  options: SharedStateLifecycleOptions | undefined,
): Record<SharedStateEntryKind, number> {
  const caps = Object.fromEntries(SHARED_STATE_ENTRY_KINDS.map((kind) => [kind, 0])) as Record<
    SharedStateEntryKind,
    number
  >;

  for (const kind of SHARED_STATE_LIFECYCLE_CAPPED_KINDS) {
    caps[kind] = options?.kindSoftCaps?.[kind] ?? DEFAULT_SHARED_STATE_KIND_SOFT_CAPS[kind] ?? 0;
  }

  return caps;
}

function lifecycleMaxActiveEntries(options: SharedStateLifecycleOptions | undefined): number {
  const value = options?.maxActiveEntries ?? DEFAULT_MAX_ACTIVE_SHARED_STATE_ENTRIES;

  return Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : DEFAULT_MAX_ACTIVE_SHARED_STATE_ENTRIES;
}

function newestStateChangeReservedSlots(options: SharedStateLifecycleOptions | undefined): number {
  const value =
    options?.newestStateChangeReservedSlots ?? DEFAULT_NEWEST_STATE_CHANGE_RESERVED_SLOTS;

  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
}

export function materializeSharedStateOperationIds(
  operations: readonly SharedStateOperation[],
): SharedStateOperation[] {
  return operations.map((operation) => {
    switch (operation.type) {
      case "add":
        return {
          ...operation,
          id: operation.id ?? createSharedStateEntryId(),
        };
      case "supersede":
        return {
          ...operation,
          replacement: {
            ...operation.replacement,
            id: operation.replacement.id ?? createSharedStateEntryId(),
          },
        };
      case "update":
      case "prune":
      case "transition_kind":
        return operation;
    }
  });
}

function lifecycleEntryFromSharedStateEntry(entry: SharedStateEntry): LifecycleEntry {
  return {
    id: entry.id,
    kind: entry.kind,
    created_at: entry.created_at,
    last_updated_at: entry.last_updated_at,
    superseded_by_id: entry.superseded_by_id,
    rank: entry.rank,
  };
}

function materializePostPatchLifecycleEntries(input: {
  previousArtifact: SharedStateArtifact | null;
  operations: readonly SharedStateOperation[];
  nowMs: number;
  applyPrunes?: boolean;
}): LifecycleEntry[] {
  const entries = new Map<SharedStateEntryId, LifecycleEntry>();

  for (const entry of input.previousArtifact?.entries ?? []) {
    entries.set(entry.id, lifecycleEntryFromSharedStateEntry(entry));
  }

  for (const operation of input.operations) {
    switch (operation.type) {
      case "add": {
        const id = operation.id ?? createSharedStateEntryId();
        entries.set(id, {
          id,
          kind: operation.kind,
          created_at: operation.created_at ?? input.nowMs,
          last_updated_at: operation.last_updated_at ?? operation.created_at ?? input.nowMs,
          superseded_by_id: null,
          rank: operation.rank ?? entries.size,
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
          kind: operation.kind ?? current.kind,
          last_updated_at: operation.last_updated_at ?? input.nowMs,
          rank: operation.rank ?? current.rank,
        });
        break;
      }
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
      case "supersede": {
        const current = entries.get(operation.id);
        const replacementId = operation.replacement.id ?? createSharedStateEntryId();

        if (current !== undefined) {
          entries.set(operation.id, {
            ...current,
            superseded_by_id: replacementId,
            last_updated_at: operation.last_updated_at ?? input.nowMs,
          });
        }

        entries.set(replacementId, {
          id: replacementId,
          kind: operation.replacement.kind,
          created_at: operation.replacement.created_at ?? input.nowMs,
          last_updated_at:
            operation.replacement.last_updated_at ??
            operation.replacement.created_at ??
            input.nowMs,
          superseded_by_id: null,
          rank: operation.replacement.rank ?? entries.size,
        });
        break;
      }
      case "prune":
        if (input.applyPrunes !== false) {
          entries.delete(operation.id);
        }
        break;
    }
  }

  return [...entries.values()];
}

function activeLifecycleEntries(entries: readonly LifecycleEntry[]): LifecycleEntry[] {
  return entries.filter((entry) => entry.superseded_by_id === null);
}

function lifecycleKindCounts(
  entries: readonly LifecycleEntry[],
): Record<SharedStateEntryKind, number> {
  const counts = Object.fromEntries(SHARED_STATE_ENTRY_KINDS.map((kind) => [kind, 0])) as Record<
    SharedStateEntryKind,
    number
  >;

  for (const entry of entries) {
    counts[entry.kind] += 1;
  }

  return counts;
}

/**
 * Orders entries prune-first. Staleness dominates: the oldest `last_updated_at`
 * goes first. Ties on that stamp are the normal case rather than the edge case,
 * because every entry a single compile pass writes carries that pass's stamp --
 * so `rank` decides most real evictions, and it is read ascending here just as
 * the repository reads it ascending when listing. The member of a tied round
 * that renders at the top of the index is therefore the first one pruned.
 *
 * That direction is load-bearing in both directions and the two readings of
 * `rank` are not reconciled: as a salience position it argues for descending
 * (keep what renders first), as a within-patch emission sequence it argues for
 * ascending (a later `add` supersedes an earlier duplicate, which is what
 * compiler.test.ts's canonicalization case depends on). Do not flip it without
 * settling which `rank` means.
 */
function compareLifecyclePrunePriority(left: LifecycleEntry, right: LifecycleEntry): number {
  return (
    left.last_updated_at - right.last_updated_at ||
    left.rank - right.rank ||
    left.created_at - right.created_at ||
    left.id.localeCompare(right.id)
  );
}

function compareLifecycleNewestStateChangePriority(
  left: LifecycleEntry,
  right: LifecycleEntry,
): number {
  return (
    right.last_updated_at - left.last_updated_at ||
    right.created_at - left.created_at ||
    left.rank - right.rank ||
    left.id.localeCompare(right.id)
  );
}

function newestStateChangeReservedIds(
  entries: readonly LifecycleEntry[],
  limit: number,
): Set<SharedStateEntryId> {
  if (limit <= 0) {
    return new Set<SharedStateEntryId>();
  }

  return new Set(
    activeLifecycleEntries(entries)
      .filter((entry) => entry.kind === "live")
      .sort(compareLifecycleNewestStateChangePriority)
      .slice(0, limit)
      .map((entry) => entry.id),
  );
}

function lifecycleKindCountsForIds(
  entries: readonly LifecycleEntry[],
  ids: ReadonlySet<SharedStateEntryId>,
): Record<SharedStateEntryKind, number> {
  const counts = Object.fromEntries(SHARED_STATE_ENTRY_KINDS.map((kind) => [kind, 0])) as Record<
    SharedStateEntryKind,
    number
  >;

  for (const entry of entries) {
    if (ids.has(entry.id)) {
      counts[entry.kind] += 1;
    }
  }

  return counts;
}

function nextLifecyclePruneCandidate(input: {
  entries: readonly LifecycleEntry[];
  kind: SharedStateEntryKind;
  prunedIds: ReadonlySet<SharedStateEntryId>;
  reservedIds?: ReadonlySet<SharedStateEntryId>;
  allowReserved?: boolean;
}): LifecycleEntry | null {
  return (
    activeLifecycleEntries(input.entries)
      .filter(
        (entry) =>
          entry.kind === input.kind &&
          !input.prunedIds.has(entry.id) &&
          (input.allowReserved === true || input.reservedIds?.has(entry.id) !== true),
      )
      .sort(compareLifecyclePrunePriority)[0] ?? null
  );
}

function lifecycleEntriesById(
  entries: readonly LifecycleEntry[],
): Map<SharedStateEntryId, LifecycleEntry> {
  const byId = new Map<SharedStateEntryId, LifecycleEntry>();

  for (const entry of entries) {
    byId.set(entry.id, entry);
  }

  return byId;
}

function lifecycleReferrersByReplacement(
  entries: readonly LifecycleEntry[],
): Map<SharedStateEntryId, LifecycleEntry[]> {
  const byReplacement = new Map<SharedStateEntryId, LifecycleEntry[]>();

  for (const entry of entries) {
    if (entry.superseded_by_id === null) {
      continue;
    }

    const referrers = byReplacement.get(entry.superseded_by_id) ?? [];
    referrers.push(entry);
    byReplacement.set(entry.superseded_by_id, referrers);
  }

  return byReplacement;
}

function appendPruneWithDependencies(input: {
  entryId: SharedStateEntryId;
  entriesById: ReadonlyMap<SharedStateEntryId, LifecycleEntry>;
  referrersByReplacement: ReadonlyMap<SharedStateEntryId, readonly LifecycleEntry[]>;
  prunedIds: Set<SharedStateEntryId>;
  visitingIds: Set<SharedStateEntryId>;
  pruneOperations: SharedStateOperation[];
}): boolean {
  if (input.prunedIds.has(input.entryId)) {
    return true;
  }

  if (input.visitingIds.has(input.entryId)) {
    return false;
  }

  if (!input.entriesById.has(input.entryId)) {
    return true;
  }

  input.visitingIds.add(input.entryId);

  for (const referrer of input.referrersByReplacement.get(input.entryId) ?? []) {
    if (input.prunedIds.has(referrer.id)) {
      continue;
    }

    if (
      !appendPruneWithDependencies({
        ...input,
        entryId: referrer.id,
      })
    ) {
      input.visitingIds.delete(input.entryId);
      return false;
    }
  }

  input.visitingIds.delete(input.entryId);
  input.prunedIds.add(input.entryId);
  input.pruneOperations.push({
    type: "prune",
    id: input.entryId,
  });

  return true;
}

export function expandPruneDependencies(input: {
  previousArtifact: SharedStateArtifact | null;
  operations: readonly SharedStateOperation[];
  nowMs: number;
}): SharedStateOperation[] {
  const entries = materializePostPatchLifecycleEntries({
    previousArtifact: input.previousArtifact,
    operations: input.operations,
    nowMs: input.nowMs,
    applyPrunes: false,
  });
  const entriesById = lifecycleEntriesById(entries);
  const referrersByReplacement = lifecycleReferrersByReplacement(entries);
  const expandedOperations: SharedStateOperation[] = [];
  const prunedIds = new Set<SharedStateEntryId>();

  for (const operation of input.operations) {
    if (operation.type !== "prune") {
      expandedOperations.push(operation);
      continue;
    }

    const previousPrunedIds = new Set(prunedIds);
    const previousOperationCount = expandedOperations.length;
    const appended = appendPruneWithDependencies({
      entryId: operation.id,
      entriesById,
      referrersByReplacement,
      prunedIds,
      visitingIds: new Set<SharedStateEntryId>(),
      pruneOperations: expandedOperations,
    });

    if (!appended) {
      expandedOperations.splice(previousOperationCount);
      prunedIds.clear();
      for (const prunedId of previousPrunedIds) {
        prunedIds.add(prunedId);
      }
      expandedOperations.push(operation);
    }
  }

  return expandedOperations;
}

export function applySharedStateArtifactLifecycleCap(input: {
  previousArtifact: SharedStateArtifact | null;
  operations: readonly SharedStateOperation[];
  options?: SharedStateLifecycleOptions;
  nowMs: number;
}): {
  operations: SharedStateOperation[];
  maxActiveEntries: number;
  postPlanActiveEntryCount: number;
  overCapDelta: number;
  newestReservedEntryCount: number;
} {
  const operations = materializeSharedStateOperationIds(input.operations);
  const entries = materializePostPatchLifecycleEntries({
    previousArtifact: input.previousArtifact,
    operations,
    nowMs: input.nowMs,
  });
  const maxActiveEntries = lifecycleMaxActiveEntries(input.options);
  const kindSoftCaps = normalizeLifecycleKindSoftCaps(input.options);
  const reservedIds = newestStateChangeReservedIds(
    entries,
    newestStateChangeReservedSlots(input.options),
  );
  const prunedIds = new Set<SharedStateEntryId>();
  const pruneOperations: SharedStateOperation[] = [];
  let activeEntries = activeLifecycleEntries(entries);
  let activeCounts = lifecycleKindCounts(activeEntries);
  let reservedCounts = lifecycleKindCountsForIds(activeEntries, reservedIds);

  const selectFromKind = (kind: SharedStateEntryKind, allowReserved = false): boolean => {
    const candidate = nextLifecyclePruneCandidate({
      entries,
      kind,
      prunedIds,
      reservedIds,
      allowReserved,
    });

    if (candidate === null) {
      return false;
    }

    prunedIds.add(candidate.id);
    pruneOperations.push({
      type: "prune",
      id: candidate.id,
    });
    activeCounts[candidate.kind] -= 1;
    if (reservedIds.has(candidate.id)) {
      reservedCounts[candidate.kind] -= 1;
    }
    activeEntries = activeEntries.filter((entry) => entry.id !== candidate.id);
    return true;
  };

  while (activeEntries.length > maxActiveEntries) {
    let pruned = false;

    for (const kind of SHARED_STATE_LIFECYCLE_PRUNE_ORDER) {
      if (activeCounts[kind] - reservedCounts[kind] <= kindSoftCaps[kind]) {
        continue;
      }

      pruned = selectFromKind(kind);

      if (pruned) {
        break;
      }
    }

    if (pruned) {
      continue;
    }

    for (const kind of SHARED_STATE_LIFECYCLE_PRUNE_ORDER) {
      pruned = selectFromKind(kind);

      if (pruned) {
        break;
      }
    }

    if (pruned) {
      continue;
    }

    for (const kind of SHARED_STATE_LIFECYCLE_PRUNE_ORDER) {
      pruned = selectFromKind(kind, true);

      if (pruned) {
        break;
      }
    }

    if (!pruned) {
      break;
    }
  }

  const postPlanActiveEntryCount = activeEntries.length;

  return {
    operations: [...operations, ...pruneOperations],
    maxActiveEntries,
    postPlanActiveEntryCount,
    overCapDelta: Math.max(0, postPlanActiveEntryCount - maxActiveEntries),
    newestReservedEntryCount: [...reservedIds].filter((id) => !prunedIds.has(id)).length,
  };
}
