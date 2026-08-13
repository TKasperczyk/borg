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

/** Returns a start boundary that never retains only the low half of an astral character. */
export function utf16SafeSuffixStart(value: string, requestedStart: number): number {
  const start = boundedBoundary(value, requestedStart);
  return boundarySplitsSurrogatePair(value, start) ? start + 1 : start;
}
