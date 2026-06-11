function pad(value: number, width = 2): string {
  return String(value).padStart(width, "0");
}

const MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

export function hm(date: Date): string {
  return `${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

export function hms(date: Date): string {
  return `${hm(date)}:${pad(date.getSeconds())}.${pad(date.getMilliseconds(), 3)}`;
}

export function dayLabel(date: Date): string {
  return `${MONTHS[date.getMonth()]} ${date.getDate()}`;
}

export function relativeAge(date: Date, now = new Date()): string {
  const diffMs = Math.max(0, now.getTime() - date.getTime());
  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) {
    return "now";
  }
  if (minutes < 60) {
    return `${minutes}m`;
  }

  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    return `${hours}h`;
  }

  const days = Math.floor(hours / 24);
  return `${days}d`;
}

export function humanMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) {
    return "—";
  }
  if (ms < 1_000) {
    return `${Math.round(ms)}ms`;
  }

  const seconds = ms / 1_000;
  if (seconds < 60) {
    return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  }

  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  if (minutes < 60) {
    return remainder === 0 ? `${minutes}m` : `${minutes}m ${remainder}s`;
  }

  const hours = Math.floor(minutes / 60);
  const minuteRemainder = minutes % 60;
  return minuteRemainder === 0 ? `${hours}h` : `${hours}h ${minuteRemainder}m`;
}
