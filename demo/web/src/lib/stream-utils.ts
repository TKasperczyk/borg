import type { StreamEntry } from "../api/types";

export type StreamSortDirection = "asc" | "desc";

const MONTH_LABELS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
] as const;
const MIN_EPOCH_MS = Date.UTC(2000, 0, 1);
const MAX_FUTURE_MS = 1000 * 60 * 60 * 24 * 366 * 10;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function jsonText(value: unknown): string {
  try {
    return JSON.stringify(value ?? null);
  } catch {
    return String(value);
  }
}

export function compactStreamText(text: string, maxLength = 220): string {
  let compacted = "";
  let pendingSpace = false;

  for (const char of text.trim()) {
    if (char === " " || char === "\n" || char === "\r" || char === "\t") {
      pendingSpace = compacted.length > 0;
      continue;
    }

    if (pendingSpace) {
      compacted += " ";
      pendingSpace = false;
    }
    compacted += char;
  }

  if (compacted.length <= maxLength) {
    return compacted;
  }
  if (maxLength <= 3) {
    return compacted.slice(0, Math.max(0, maxLength));
  }
  return `${compacted.slice(0, maxLength - 3).trimEnd()}...`;
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function sameLocalDate(left: Date, right: Date): boolean {
  return (
    left.getFullYear() === right.getFullYear() &&
    left.getMonth() === right.getMonth() &&
    left.getDate() === right.getDate()
  );
}

function timeLabel(date: Date, withSeconds: boolean): string {
  const base = `${pad2(date.getHours())}:${pad2(date.getMinutes())}`;
  return withSeconds ? `${base}:${pad2(date.getSeconds())}` : base;
}

function compactDateTimeLabel(date: Date): string {
  return `${MONTH_LABELS[date.getMonth()]} ${date.getDate()} ${timeLabel(date, false)}`;
}

function fullDateTimeLabel(date: Date): string {
  return `${MONTH_LABELS[date.getMonth()]} ${date.getDate()} ${date.getFullYear()} ${timeLabel(date, false)}`;
}

function isValidTimestamp(timestamp: number): boolean {
  return Number.isFinite(timestamp) && !Number.isNaN(new Date(timestamp).getTime());
}

export function formatTimestamp(timestamp: number | null | undefined, fallback = "never"): string {
  if (timestamp === null || timestamp === undefined) {
    return fallback;
  }

  if (!isValidTimestamp(timestamp)) {
    return fallback;
  }

  const date = new Date(timestamp);
  const now = new Date();

  if (sameLocalDate(date, now)) {
    return timeLabel(date, true);
  }

  if (date.getFullYear() === now.getFullYear()) {
    return compactDateTimeLabel(date);
  }

  return fullDateTimeLabel(date);
}

export function formatTimestampRange(
  start: number | null | undefined,
  end: number | null | undefined,
  fallback = "never",
): string {
  if (start === null || start === undefined) {
    return formatTimestamp(end, fallback);
  }
  if (end === null || end === undefined) {
    return formatTimestamp(start, fallback);
  }
  if (!isValidTimestamp(start) || !isValidTimestamp(end)) {
    return fallback;
  }

  const [from, to] = start <= end ? [start, end] : [end, start];
  if (from === to) {
    return formatTimestamp(from, fallback);
  }

  return `${formatTimestamp(from, fallback)} - ${formatTimestamp(to, fallback)}`;
}

function normalizedKey(key: string): string {
  return key.trim().toLowerCase();
}

function keyLooksLikeTimestamp(key: string): boolean {
  const normalized = normalizedKey(key);
  return (
    normalized === "ts" ||
    normalized === "timestamp" ||
    normalized === "created" ||
    normalized === "updated" ||
    normalized.endsWith("_at") ||
    normalized.endsWith("_ts") ||
    normalized.endsWith("_time") ||
    normalized.endsWith("timestamp")
  );
}

function valueLooksLikeEpochMs(value: number): boolean {
  return Number.isInteger(value) && value >= MIN_EPOCH_MS && value <= Date.now() + MAX_FUTURE_MS;
}

export function formatTimestampForKey(key: string, value: unknown): string | null {
  if (typeof value !== "number" || !keyLooksLikeTimestamp(key) || !valueLooksLikeEpochMs(value)) {
    return null;
  }

  return formatTimestamp(value);
}

export function formatTime(timestamp: number | null | undefined, fallback = "never"): string {
  return formatTimestamp(timestamp, fallback);
}

export function streamContentText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (isRecord(block) && typeof block.text === "string") {
          return block.text;
        }
        if (isRecord(block) && typeof block.attachment_id === "string") {
          return `[att:${block.attachment_id}]`;
        }
        return jsonText(block);
      })
      .join("\n");
  }

  if (isRecord(content) && typeof content.text === "string") {
    return content.text;
  }

  return jsonText(content);
}

export function compareStreamEntries(
  left: StreamEntry,
  right: StreamEntry,
  direction: StreamSortDirection = "asc",
): number {
  const multiplier = direction === "asc" ? 1 : -1;

  if (left.timestamp !== right.timestamp) {
    return (left.timestamp - right.timestamp) * multiplier;
  }

  if (
    left.entry_index !== undefined &&
    right.entry_index !== undefined &&
    left.entry_index !== right.entry_index
  ) {
    return (left.entry_index - right.entry_index) * multiplier;
  }

  return left.id.localeCompare(right.id) * multiplier;
}

export function sortStreamEntries(
  entries: readonly StreamEntry[],
  direction: StreamSortDirection = "asc",
): StreamEntry[] {
  return [...entries].sort((left, right) => compareStreamEntries(left, right, direction));
}

export function mergeEntries(
  current: readonly StreamEntry[],
  incoming: readonly StreamEntry[],
  direction: StreamSortDirection = "asc",
): StreamEntry[] {
  const byId = new Map(current.map((entry) => [entry.id, entry]));

  for (const entry of incoming) {
    byId.set(entry.id, entry);
  }

  return sortStreamEntries([...byId.values()], direction);
}
