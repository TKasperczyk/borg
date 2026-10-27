import { toWellFormedUtf16 } from "./utf16-boundary.js";

/** Transport-safe prefix; the first excluded byte must start a code point. */
export function utf8SafePrefix(value: string, maxBytes: number): string {
  const bytes = Buffer.from(toWellFormedUtf16(value), "utf8");
  let end = Math.min(bytes.length, maxBytes);
  while (end < bytes.length && (bytes[end]! & 0xc0) === 0x80) end -= 1;
  return bytes.subarray(0, end).toString("utf8");
}

/** Retain the newest suffix without starting on a UTF-8 continuation byte. */
export function utf8SafeSuffix(value: string, maxBytes: number): string {
  const bytes = Buffer.from(toWellFormedUtf16(value), "utf8");
  let start = Math.max(0, bytes.length - maxBytes);
  while (start < bytes.length && (bytes[start]! & 0xc0) === 0x80) start += 1;
  return bytes.subarray(start).toString("utf8");
}
