export function dateLabel(timestamp: number | null | undefined): string {
  if (timestamp === null || timestamp === undefined) {
    return "—";
  }

  return new Date(timestamp).toLocaleDateString("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric",
  });
}

export function shortId(id: string | null | undefined): string {
  if (id === null || id === undefined || id.length <= 14) {
    return id ?? "—";
  }

  return `${id.slice(0, 8)}…${id.slice(-4)}`;
}

const INTERNAL_ID_SHAPE = /^[a-z]+_[a-z0-9]{6,}$/;
const VECTOR_ARRAY_LENGTH = 16;

export function isInternalId(value: string): boolean {
  return INTERNAL_ID_SHAPE.test(value);
}

export function fieldLabel(key: string): string {
  const spaced = key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replaceAll("_", " ")
    .toLowerCase();
  return spaced.replace(/\bids?\b$/u, "").trim() || spaced;
}

export function displayValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "-";
  }

  if (typeof value === "string") {
    if (value.length === 0) {
      return "-";
    }
    return isInternalId(value) ? shortId(value) : value;
  }

  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return "-";
    }

    if (value.length > VECTOR_ARRAY_LENGTH && value.every((item) => typeof item === "number")) {
      return `vector(${value.length})`;
    }

    return value.map(displayValue).join(", ");
  }

  if (isRecord(value)) {
    return `{${Object.keys(value).length} fields}`;
  }

  return String(value);
}

export function displayTargetSummary(targets: Record<string, unknown>): string {
  const entries = Object.entries(targets);
  if (entries.length === 0) {
    return "-";
  }

  const idEntry = entries.find(([, value]) => typeof value === "string" && isInternalId(value));
  const idArrayEntry = entries.find(
    ([, value]) =>
      Array.isArray(value) &&
      value.length > 0 &&
      value.every((item) => typeof item === "string" && isInternalId(item)),
  );

  if (idEntry !== undefined && idArrayEntry !== undefined) {
    const [idKey, idValue] = idEntry;
    const [arrayKey, arrayValue] = idArrayEntry;
    const arrayLength = Array.isArray(arrayValue) ? arrayValue.length : 0;
    return `${fieldLabel(idKey)} ${displayValue(idValue)}, ${arrayLength} ${fieldLabel(arrayKey)}`;
  }

  return entries.map(([key, value]) => `${fieldLabel(key)} ${displayValue(value)}`).join(", ");
}

export function jsonText(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function parseJsonPatch(text: string): Record<string, unknown> {
  const parsed = JSON.parse(text) as unknown;
  if (!isRecord(parsed)) {
    throw new Error("patch must be a JSON object");
  }
  return parsed;
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function contentField(value: unknown, key: string): string | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const field = value[key];
  return typeof field === "string" ? field : undefined;
}

export function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
