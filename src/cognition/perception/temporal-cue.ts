import type { TemporalCue } from "../types.js";

const DAY_MS = 24 * 60 * 60 * 1_000;

function startOfDay(timestamp: number): number {
  const date = new Date(timestamp);
  date.setHours(0, 0, 0, 0);
  return date.getTime();
}

export function detectTemporalCue(text: string, nowMs = Date.now()): TemporalCue | null {
  const normalized = text.toLowerCase();

  if (/\byesterday\b/.test(normalized)) {
    const today = startOfDay(nowMs);
    return {
      sinceTs: today - DAY_MS,
      untilTs: today,
      label: "yesterday",
    };
  }

  if (/\blast week\b/.test(normalized)) {
    return {
      sinceTs: nowMs - 7 * DAY_MS,
      untilTs: nowMs,
      label: "last week",
    };
  }

  if (/\bthis morning\b/.test(normalized)) {
    const today = startOfDay(nowMs);
    return {
      sinceTs: today,
      untilTs: today + 12 * 60 * 60 * 1_000,
      label: "this morning",
    };
  }

  if (/\bthis week\b/.test(normalized)) {
    return {
      sinceTs: nowMs - 7 * DAY_MS,
      untilTs: nowMs,
      label: "this week",
    };
  }

  if (/\btoday\b/.test(normalized) || /\btonight\b/.test(normalized)) {
    const today = startOfDay(nowMs);
    return {
      sinceTs: today,
      untilTs: today + DAY_MS,
      label: /\btonight\b/.test(normalized) ? "tonight" : "today",
    };
  }

  return null;
}
