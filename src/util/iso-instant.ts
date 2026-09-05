// Parses an ISO-8601 instant emitted by a model into epoch milliseconds. Anything Date.parse
// cannot turn into a finite number (missing, malformed, or a bare word) is undefined, so callers
// drop the value instead of storing NaN. With requireOffset the value must also spell out its zone
// (Z or ±hh:mm), which keeps host-time-zone interpretation and textual dates out.
const ISO_INSTANT_WITH_OFFSET =
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?(?:Z|[+-]\d{2}:?\d{2})$/;

export function parseIsoInstant(
  value: string | null | undefined,
  options: { requireOffset?: boolean } = {},
): number | undefined {
  if (value === null || value === undefined) {
    return undefined;
  }
  const trimmed = value.trim();
  if (trimmed.length === 0) {
    return undefined;
  }
  if (options.requireOffset === true && !ISO_INSTANT_WITH_OFFSET.test(trimmed)) {
    return undefined;
  }
  const parsed = Date.parse(trimmed);
  return Number.isFinite(parsed) ? parsed : undefined;
}
