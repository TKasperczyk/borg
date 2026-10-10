function boundedBoundary(value: string, requestedBoundary: number): number {
  return Math.max(0, Math.min(value.length, Math.floor(requestedBoundary)));
}

function boundarySplitsSurrogatePair(value: string, boundary: number): boolean {
  if (boundary === 0 || boundary === value.length) {
    return false;
  }

  const prior = value.charCodeAt(boundary - 1);
  const current = value.charCodeAt(boundary);
  return prior >= 0xd800 && prior <= 0xdbff && current >= 0xdc00 && current <= 0xdfff;
}

/** Returns an end boundary that never retains only the high half of an astral character. */
export function utf16SafePrefixEnd(value: string, requestedEnd: number): number {
  const end = boundedBoundary(value, requestedEnd);
  return boundarySplitsSurrogatePair(value, end) ? end - 1 : end;
}

/**
 * Slices a UTF-16 prefix of at most `maxChars` code units without ever ending on
 * the high half of an astral character. A plain `slice(0, n)` through an emoji
 * leaves a lone surrogate that JSON serialises as `"\ud83c"`, which UTF-8
 * consumers (the embedding gateway, Python's httpx) reject for the whole body.
 */
export function utf16SafePrefix(value: string, maxChars: number): string {
  return value.slice(0, utf16SafePrefixEnd(value, maxChars));
}

/** Returns a start boundary that never retains only the low half of an astral character. */
export function utf16SafeSuffixStart(value: string, requestedStart: number): number {
  const start = boundedBoundary(value, requestedStart);
  return boundarySplitsSurrogatePair(value, start) ? start + 1 : start;
}

const LONE_SURROGATE = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/gu;

/** Replaces unpaired UTF-16 surrogates with U+FFFD so the string survives UTF-8 encoding. */
export function toWellFormedUtf16(value: string): string {
  return value.replace(LONE_SURROGATE, "\uFFFD");
}
