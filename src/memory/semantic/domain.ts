export function canonicalizeDomain(raw: string | null | undefined): string | null {
  const normalized = raw?.trim().toLowerCase() ?? "";

  if (normalized.length === 0) {
    return null;
  }

  return normalized;
}
