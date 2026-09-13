// Parses an ISO-8601 instant emitted by a model into epoch milliseconds. Anything Date.parse
// cannot turn into a finite number (missing, malformed, or a bare word) is undefined, so callers
// drop the value instead of storing NaN. With requireOffset the value must also spell out its zone
// (Z or ±hh:mm), which keeps host-time-zone interpretation and textual dates out.
import { z } from "zod";

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

// Date's representable range; outside it toISOString() throws rather than
// returning a value, so this bounds the conversion's actual domain. Non-numbers
// and non-finite values render as null so callers can skip the field.
const MAX_EPOCH_MS = 8_640_000_000_000_000;

export function epochMsToIso(value: unknown): string | null {
  if (typeof value !== "number" || !Number.isFinite(value) || Math.abs(value) > MAX_EPOCH_MS) {
    return null;
  }

  return new Date(value).toISOString();
}

// Model-facing schema for a time the model must PRODUCE. The wire type is a
// string (an ISO-8601 date-time with an explicit zone), parsed to epoch
// milliseconds here so every downstream consumer keeps numeric time. A raw
// epoch-millisecond number is the wrong thing to ask a model to type: it is a
// 13-digit run with no delimiter and no natural end, and prod 2026-09-13 showed
// qwen3 degenerating into thousands of zeros on exactly such a field until the
// output limit cut the whole tool call off.
export function isoInstantEpochMsSchema(description: string) {
  return z
    .string()
    .trim()
    .min(1)
    .transform((value, ctx) => {
      const parsed = parseIsoInstant(value, { requireOffset: true });

      if (parsed === undefined) {
        ctx.addIssue({
          code: "custom",
          message:
            "Must be an ISO-8601 date-time with a zone offset, like 2026-01-31T09:30:00Z or 2026-01-31T09:30:00+02:00.",
        });

        return z.NEVER;
      }

      return parsed;
    })
    .describe(description);
}
