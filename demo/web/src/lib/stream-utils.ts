import type { StreamEntry } from "../api/types";

export type StreamSortDirection = "asc" | "desc";

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

export function formatTime(timestamp: number | null | undefined, fallback = "never"): string {
  if (timestamp === null || timestamp === undefined) {
    return fallback;
  }

  return new Date(timestamp).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
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
