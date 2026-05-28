export function formatRelativeAge(timestampMs: number, nowMs: number): string {
  const elapsedMs = Math.max(0, nowMs - timestampMs);
  const elapsedSeconds = Math.floor(elapsedMs / 1_000);

  if (elapsedSeconds < 60) {
    return `~${elapsedSeconds}s ago`;
  }

  const elapsedMinutes = Math.floor(elapsedMs / 60_000);

  if (elapsedMinutes < 60) {
    return `${elapsedMinutes}m ago`;
  }

  const elapsedHours = Math.floor(elapsedMinutes / 60);

  if (elapsedHours < 24) {
    return `${elapsedHours}h ago`;
  }

  const elapsedDays = Math.floor(elapsedHours / 24);

  if (elapsedDays < 2) {
    return "yesterday";
  }

  return `${elapsedDays}d ago`;
}
