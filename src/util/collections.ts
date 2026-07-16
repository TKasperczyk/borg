export function dedupePreservingOrder<T extends string>(values: readonly T[]): T[] {
  return [...new Set(values)];
}

export function uniqueStrings(values: readonly string[]): string[] {
  return dedupePreservingOrder(values);
}

export function sortStrings<T extends string>(values: readonly T[]): T[] {
  return [...values].sort((left, right) => left.localeCompare(right));
}

export async function mapWithConcurrency<T, U>(
  items: readonly T[],
  limit: number,
  mapper: (item: T, index: number) => Promise<U>,
): Promise<U[]> {
  const normalizedLimit = Math.max(1, Math.floor(limit));
  const results: U[] = [];

  for (let start = 0; start < items.length; start += normalizedLimit) {
    const batch = items.slice(start, start + normalizedLimit);
    results.push(...(await Promise.all(batch.map((item, index) => mapper(item, start + index)))));
  }

  return results;
}
