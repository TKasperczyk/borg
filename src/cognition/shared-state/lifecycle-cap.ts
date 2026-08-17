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
  "id" | "state_key" | "kind" | "created_at" | "last_updated_at" | "superseded_by_id" | "rank"
>;

/**
 * An entry the cap deleted to hold the artifact at `maxActiveEntries`. The
 * entity never asked for these: they are emitted as ordinary `prune`
 * operations, so without this record the trace cannot tell a deletion the
 * compiler requested from one the store forced. Carries the state key because
 * the id is gone from the artifact by the time anyone reads the trace.
 *
 * `selection_pass` names which of the loop's three scans drew it, because the
 * first scan only considers kinds above their own soft cap. Without it an
 * eviction record read against the rendered index looks like the comparator
 * skipped the oldest row: on a live 40-entry artifact the globally-oldest
 * entry was a `tentative` row three days staler than anything else, sitting
 * at its cap of 2 and therefore never a candidate, while the draw ran inside
 * `locked` (38 against a cap of 24) and took a much newer entry. Staleness
 * orders candidates; it does not choose the pool they come from.
 */
export type SharedStateLifecycleCapEviction = {
  id: SharedStateEntryId;
  state_key: string | null;
  kind: SharedStateEntryKind;
  last_updated_at: number;
  rank: number;
  selection_pass: SharedStateLifecycleCapSelectionPass;
};

/**
 * Which scan of the eviction loop drew an entry: the kind was over its own
 * soft cap, or nothing was and the prune order alone decided, or the pool had
 * to be widened to entries the newest-state-change reservation was holding.
 */
export type SharedStateLifecycleCapSelectionPass =
  | "over_soft_cap"
  | "any_kind"
  | "any_kind_reserved_allowed";

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
    state_key: entry.state_key,
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
          state_key: operation.state_key,
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
          state_key: operation.state_key,
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
          state_key: operation.replacement.state_key,
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
 * the repository reads it ascending when listing. That order is not visible on
 * the rendered index: `renderSharedStateArtifact` groups by state key and sorts
 * the groups with `localeCompare`, and `rank` is not a field on an index line at
 * all -- so index position reports alphabet, never prune position. Do not read
 * one off the other.
 *
 * The direction is settled by the only writer. Nothing in production supplies a
 * salience `rank`: `patch-validation.ts` computes it as `baseRank +
 * operations.length` -- the previous artifact's entry count plus the operation's
 * position in the accepted list -- and `update` preserves whatever the row
 * already had. It is a within-patch emission index, which is why ascending is
 * correct (a later `add` supersedes an earlier duplicate, which is what
 * compiler.test.ts's canonicalization case depends on) and why the descending
 * "keep what renders first" reading has no writer behind it. Two consequences
 * worth keeping: on an artifact pinned at cap `baseRank` is constant, so `rank`
 * carries no age information across passes; and updates keep their old value
 * while same-pass adds take fresh ones, so `rank` is not unique and ties fall
 * through to `created_at`.
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
  capEvictions: SharedStateLifecycleCapEviction[];
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
  const capEvictions: SharedStateLifecycleCapEviction[] = [];
  let activeEntries = activeLifecycleEntries(entries);
  let activeCounts = lifecycleKindCounts(activeEntries);
  let reservedCounts = lifecycleKindCountsForIds(activeEntries, reservedIds);

  const selectFromKind = (
    kind: SharedStateEntryKind,
    selectionPass: SharedStateLifecycleCapSelectionPass,
    allowReserved = false,
  ): boolean => {
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
    capEvictions.push({
      id: candidate.id,
      state_key: candidate.state_key,
      kind: candidate.kind,
      last_updated_at: candidate.last_updated_at,
      rank: candidate.rank,
      selection_pass: selectionPass,
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

      pruned = selectFromKind(kind, "over_soft_cap");

      if (pruned) {
        break;
      }
    }

    if (pruned) {
      continue;
    }

    for (const kind of SHARED_STATE_LIFECYCLE_PRUNE_ORDER) {
      pruned = selectFromKind(kind, "any_kind");

      if (pruned) {
        break;
      }
    }

    if (pruned) {
      continue;
    }

    for (const kind of SHARED_STATE_LIFECYCLE_PRUNE_ORDER) {
      pruned = selectFromKind(kind, "any_kind_reserved_allowed", true);

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
    capEvictions,
  };
}
