export function getDistance(row: Record<string, unknown>): number | undefined {
  const value = row._distance;
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function toSimilarity(distance: number | undefined): number {
  if (distance === undefined) {
    return 0;
  }

  return Math.max(0, Math.min(1, 1 - distance));
}
