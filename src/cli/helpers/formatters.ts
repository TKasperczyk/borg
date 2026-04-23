// Output formatting helpers shared by CLI command handlers.
import type { Output } from "../types.js";

export function writeLine(output: Output, line: string): void {
  output.write(`${line}\n`);
}

export function maskSecretTail(value: string): string {
  const trimmed = value.trim();
  const tail = trimmed.slice(-4);
  return `***${tail}`;
}

export function formatExpiryDelta(ms: number): string {
  if (ms <= 0) {
    return "expired";
  }

  const totalMinutes = Math.floor(ms / 60_000);
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;

  if (hours <= 0) {
    return `${minutes}m`;
  }

  return `${hours}h ${minutes}m`;
}
