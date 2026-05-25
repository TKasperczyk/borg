import type { StreamEntry } from "../api/types";

export function timeLabel(timestamp: number | null | undefined): string {
  if (timestamp === null || timestamp === undefined) {
    return "never";
  }

  return new Date(timestamp).toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

export function dateLabel(timestamp: number | null | undefined): string {
  if (timestamp === null || timestamp === undefined) {
    return "—";
  }

  return new Date(timestamp).toLocaleDateString("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric"
  });
}

export function shortId(id: string | null | undefined): string {
  if (id === null || id === undefined || id.length <= 14) {
    return id ?? "—";
  }

  return `${id.slice(0, 8)}…${id.slice(-4)}`;
}

export function jsonText(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function streamContentText(entry: StreamEntry): string {
  if (typeof entry.content === "string") {
    return entry.content;
  }

  if (Array.isArray(entry.content)) {
    return entry.content
      .map((block) => {
        if (isRecord(block) && typeof block.text === "string") {
          return block.text;
        }
        if (isRecord(block) && typeof block.attachment_id === "string") {
          return `[att:${block.attachment_id}]`;
        }
        return jsonText(block);
      })
      .join(" ");
  }

  if (isRecord(entry.content) && typeof entry.content.text === "string") {
    return entry.content.text;
  }

  return jsonText(entry.content ?? null);
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
