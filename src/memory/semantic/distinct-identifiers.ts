import { dedupePreservingOrder } from "../../util/collections.js";

const TICKET_KEY_PATTERN = /[A-Z]{2,}-\d+/giu;
const RUN_ID_PATTERN = /(?<![0-9A-F])[0-9A-F]{32}(?![0-9A-F])/giu;
const URL_PATTERN = /https?:\/\/[^\s"'<>]+/giu;
const LONG_DIGIT_RUN_PATTERN = /\d{9,}/gu;
const TRAILING_URL_PROSE_PUNCTUATION = new Set([".", ",", ";", ":", "!", "?"]);
const URL_CLOSING_DELIMITERS = new Map([
  [")", "("],
  ["]", "["],
  ["}", "{"],
]);

export type DistinctIdentifierConflict = {
  left: string[];
  right: string[];
};

type IdentifierSpan = {
  start: number;
  end: number;
};

function countCharacter(value: string, character: string): number {
  let count = 0;

  for (const candidate of value) {
    if (candidate === character) {
      count += 1;
    }
  }

  return count;
}

function stripTrailingUrlProsePunctuation(value: string): string {
  let stripped = value;

  while (stripped.length > 0) {
    const trailing = stripped.at(-1);

    if (trailing !== undefined && TRAILING_URL_PROSE_PUNCTUATION.has(trailing)) {
      stripped = stripped.slice(0, -1);
      continue;
    }

    if (trailing !== undefined) {
      const opening = URL_CLOSING_DELIMITERS.get(trailing);

      if (
        opening !== undefined &&
        countCharacter(stripped, trailing) > countCharacter(stripped, opening)
      ) {
        stripped = stripped.slice(0, -1);
        continue;
      }
    }

    break;
  }

  return stripped;
}

function normalizedUrlPath(value: string): string | null {
  try {
    return new URL(value).pathname.replace(/\/+$/u, "") || "/";
  } catch {
    return null;
  }
}

function matchSpan(
  match: RegExpExecArray | RegExpMatchArray,
  length = match[0].length,
): IdentifierSpan {
  const start = match.index;

  if (start === undefined) {
    throw new TypeError("Identifier match is missing its source index");
  }

  return { start, end: start + length };
}

function nestedInside(span: IdentifierSpan, containers: readonly IdentifierSpan[]): boolean {
  return containers.some((container) => span.start >= container.start && span.end <= container.end);
}

export function distinctIdentifiersFromLabel(label: string): string[] {
  const identifiers: string[] = [];
  const recognizedSpans: IdentifierSpan[] = [];

  for (const match of label.matchAll(TICKET_KEY_PATTERN)) {
    identifiers.push(`ticket:${match[0].toUpperCase()}`);
    recognizedSpans.push(matchSpan(match));
  }

  for (const match of label.matchAll(RUN_ID_PATTERN)) {
    identifiers.push(`run:${match[0].toUpperCase()}`);
    recognizedSpans.push(matchSpan(match));
  }

  for (const match of label.matchAll(URL_PATTERN)) {
    const token = stripTrailingUrlProsePunctuation(match[0]);
    const path = normalizedUrlPath(token);

    if (path !== null) {
      identifiers.push(`url_path:${path}`);
      recognizedSpans.push(matchSpan(match, token.length));
    }
  }

  for (const match of label.matchAll(LONG_DIGIT_RUN_PATTERN)) {
    if (!nestedInside(matchSpan(match), recognizedSpans)) {
      identifiers.push(`digits:${match[0]}`);
    }
  }

  return dedupePreservingOrder(identifiers);
}

export function disjointDistinctIdentifiers(
  leftLabel: string,
  rightLabel: string,
): DistinctIdentifierConflict | null {
  const left = distinctIdentifiersFromLabel(leftLabel);
  const right = distinctIdentifiersFromLabel(rightLabel);

  if (left.length === 0 || right.length === 0) {
    return null;
  }

  const rightSet = new Set(right);

  return left.some((identifier) => rightSet.has(identifier)) ? null : { left, right };
}
