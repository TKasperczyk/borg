export const DOMAIN_SYNONYMS = {
  technology: "tech",
  computing: "tech",
  software: "tech",
  tech_and_engineering: "tech",
  person: "people",
  persons: "people",
  human: "people",
  humans: "people",
  place: "places",
  location: "places",
  locations: "places",
  culinary: "food",
  cooking: "food",
  cuisine: "food",
} as const satisfies Readonly<Record<string, string>>;

export function canonicalizeDomain(raw: string | null | undefined): string | null {
  const normalized = raw?.trim().toLowerCase() ?? "";

  if (normalized.length === 0) {
    return null;
  }

  const mapped = DOMAIN_SYNONYMS[normalized as keyof typeof DOMAIN_SYNONYMS];
  return mapped ?? normalized;
}
