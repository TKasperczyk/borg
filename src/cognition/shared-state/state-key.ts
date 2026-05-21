import type { SharedStateEntry } from "../../memory/decision-artifacts/index.js";

export const LEGACY_SHARED_STATE_KEY = "legacy";

export function sharedStateKeyBucket(stateKey: string | null | undefined): string {
  return stateKey ?? LEGACY_SHARED_STATE_KEY;
}

export function countSharedStateEntriesByKey(
  entries: readonly Pick<SharedStateEntry, "state_key">[],
): Record<string, number> {
  const counts: Record<string, number> = {};

  for (const entry of entries) {
    const key = sharedStateKeyBucket(entry.state_key);
    counts[key] = (counts[key] ?? 0) + 1;
  }

  return Object.fromEntries(
    Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)),
  );
}

export function topSharedStateEntryKeysByCount(
  counts: Record<string, number>,
  limit: number,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(counts)
      .sort(([leftKey, leftCount], [rightKey, rightCount]) => {
        return rightCount - leftCount || leftKey.localeCompare(rightKey);
      })
      .slice(0, Math.max(0, Math.floor(limit))),
  );
}
