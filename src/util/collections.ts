export function dedupePreservingOrder<T extends string>(values: readonly T[]): T[] {
  return [...new Set(values)];
}

export function uniqueStrings(values: readonly string[]): string[] {
  return dedupePreservingOrder(values);
}

export function sortStrings<T extends string>(values: readonly T[]): T[] {
  return [...values].sort((left, right) => left.localeCompare(right));
}
