const DAY_MS = 24 * 60 * 60 * 1_000;

const DAY_BOUNDARY_FORMATTER = new Intl.DateTimeFormat("en-US", {
  weekday: "short",
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

const MONTH_DAY_FORMATTER = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

export function utcDayKey(timestampMs: number): string {
  return new Date(timestampMs).toISOString().slice(0, 10);
}

export function utcDayStartMs(timestampMs: number): number {
  const date = new Date(timestampMs);

  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

export function timestampFromUtcDayKey(dayKey: string): number {
  return Date.parse(`${dayKey}T00:00:00.000Z`);
}

export function isUtcDayBefore(timestampMs: number, cutoffMs: number): boolean {
  return utcDayStartMs(timestampMs) + DAY_MS <= cutoffMs;
}

export function formatUtcDayBoundary(timestampMs: number): string {
  return DAY_BOUNDARY_FORMATTER.format(new Date(timestampMs)).replaceAll(",", "");
}

export function formatUtcDaySpanLabel(firstMs: number, lastMs: number): string {
  const firstDayStart = utcDayStartMs(firstMs);
  const lastDayStart = utcDayStartMs(lastMs);

  if (firstDayStart === lastDayStart) {
    return MONTH_DAY_FORMATTER.format(new Date(firstDayStart));
  }

  const first = new Date(firstDayStart);
  const last = new Date(lastDayStart);

  if (
    first.getUTCFullYear() === last.getUTCFullYear() &&
    first.getUTCMonth() === last.getUTCMonth()
  ) {
    return `${first.toLocaleString("en-US", {
      month: "short",
      timeZone: "UTC",
    })} ${first.getUTCDate()}-${last.getUTCDate()}`;
  }

  return `${MONTH_DAY_FORMATTER.format(first)}-${MONTH_DAY_FORMATTER.format(last)}`;
}

export function formatUtcTimeSpan(firstMs: number, lastMs: number): string {
  const first = new Date(firstMs);
  const last = new Date(lastMs);
  const firstTime = `${String(first.getUTCHours()).padStart(2, "0")}:${String(
    first.getUTCMinutes(),
  ).padStart(2, "0")}`;
  const lastTime = `${String(last.getUTCHours()).padStart(2, "0")}:${String(
    last.getUTCMinutes(),
  ).padStart(2, "0")}`;

  return firstTime === lastTime ? `${firstTime} UTC` : `${firstTime}-${lastTime} UTC`;
}
