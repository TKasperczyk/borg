export function formatRelativeAge(timestampMs: number, nowMs: number): string {
  const elapsedMs = Math.max(0, nowMs - timestampMs);
  const duration = formatRelativeDuration(elapsedMs);

  return duration === "1d" ? "yesterday" : `${duration} ago`;
}

export function formatRelativeUntil(timestampMs: number, nowMs: number): string {
  if (timestampMs <= nowMs) {
    return formatRelativeAge(timestampMs, nowMs);
  }

  return `in ${formatRelativeDuration(timestampMs - nowMs)}`;
}

export function formatRelativeDuration(durationMs: number): string {
  const elapsedMs = Math.max(0, durationMs);
  const elapsedSeconds = Math.floor(elapsedMs / 1_000);

  if (elapsedSeconds < 60) {
    return `~${elapsedSeconds}s`;
  }

  const elapsedMinutes = Math.floor(elapsedMs / 60_000);

  if (elapsedMinutes < 60) {
    return `${elapsedMinutes}m`;
  }

  const elapsedHours = Math.floor(elapsedMinutes / 60);

  if (elapsedHours < 24) {
    return `${elapsedHours}h`;
  }

  const elapsedDays = Math.floor(elapsedHours / 24);

  return `${Math.max(1, elapsedDays)}d`;
}
