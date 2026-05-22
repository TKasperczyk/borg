import type { SharedStateEntry } from "../../memory/decision-artifacts/index.js";

export const LEGACY_SHARED_STATE_KEY = "legacy";
// state_key values are machine-generated handles; this is structural parsing, not language interpretation.
const STATE_KEY_TOKEN_SEPARATOR = /[^a-z0-9]+/;

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

export function tokenizeStateKey(key: string): string[] {
  return key
    .toLowerCase()
    .split(STATE_KEY_TOKEN_SEPARATOR)
    .filter((token) => token.length > 0);
}

function uniqueStateKeyTokens(tokens: readonly string[]): string[] {
  return [...new Set(tokens)];
}

export function sharedStateKeyTokens(leftKey: string, rightKey: string): string[] {
  const rightTokens = new Set(uniqueStateKeyTokens(tokenizeStateKey(rightKey)));

  return uniqueStateKeyTokens(tokenizeStateKey(leftKey)).filter((token) => rightTokens.has(token));
}

export function stateKeysAreNearDuplicate(leftKey: string, rightKey: string): boolean {
  if (leftKey === rightKey) {
    return false;
  }

  const leftTokens = tokenizeStateKey(leftKey);
  const rightTokens = tokenizeStateKey(rightKey);

  if (leftTokens.length <= 2 || rightTokens.length <= 2) {
    return false;
  }

  if (leftTokens[0] !== rightTokens[0]) {
    return false;
  }

  const leftSet = new Set(leftTokens);
  const rightSet = new Set(rightTokens);
  const intersection = [...leftSet].filter((token) => rightSet.has(token)).length;
  const union = new Set([...leftTokens, ...rightTokens]).size;
  const containment = intersection / Math.min(leftSet.size, rightSet.size);
  const jaccard = intersection / union;

  return (containment >= 0.75 && intersection >= 3) || jaccard >= 0.6;
}

export function similarStateKeyClusterCount(keys: readonly string[]): number {
  const uniqueKeys = [...new Set(keys)].sort((left, right) => left.localeCompare(right));
  const visited = new Set<string>();
  let clusterCount = 0;

  for (const key of uniqueKeys) {
    if (visited.has(key)) {
      continue;
    }

    const cluster = new Set<string>([key]);
    const pending = [key];
    visited.add(key);

    while (pending.length > 0) {
      const current = pending.pop()!;

      for (const candidate of uniqueKeys) {
        if (visited.has(candidate) || !stateKeysAreNearDuplicate(current, candidate)) {
          continue;
        }

        visited.add(candidate);
        cluster.add(candidate);
        pending.push(candidate);
      }
    }

    if (cluster.size >= 2) {
      clusterCount += 1;
    }
  }

  return clusterCount;
}
