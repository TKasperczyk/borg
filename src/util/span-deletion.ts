import { valueAppearsIn } from "./text-presence.js";

const STRUCTURALLY_EMPTY_TEXT_PATTERN = /^[\s\p{P}\p{S}]*$/u;
const SPAN_WHITESPACE_PATTERN = /\s/u;
const TRAILING_REMOVAL_JUNK_PATTERN = /[\s,;:]+$/u;

const SMART_QUOTE_REPLACEMENTS: ReadonlyArray<readonly [string, string]> = [
  ["\u2018", "'"],
  ["\u2019", "'"],
  ["\u201a", "'"],
  ["\u201b", "'"],
  ["\u2032", "'"],
  ["\u201c", '"'],
  ["\u201d", '"'],
  ["\u201e", '"'],
  ["\u201f", '"'],
  ["\u2033", '"'],
];

type TextRange = {
  start: number;
  end: number;
};

type NormalizedTextIndex = {
  text: string;
  ranges: TextRange[];
};

export type DeleteSpansResult = {
  result: string;
  removedSpans: string[];
  allRemoved: boolean;
};

function normalizeSpanCharacter(value: string): string {
  let normalized = value;

  for (const [from, to] of SMART_QUOTE_REPLACEMENTS) {
    normalized = normalized.replaceAll(from, to);
  }

  return normalized.normalize("NFC").toLocaleLowerCase();
}

function buildNormalizedTextIndex(value: string): NormalizedTextIndex {
  let text = "";
  const ranges: TextRange[] = [];
  let offset = 0;

  for (const character of value) {
    const start = offset;
    offset += character.length;

    if (SPAN_WHITESPACE_PATTERN.test(character)) {
      if (text[text.length - 1] === " ") {
        const lastRange = ranges[ranges.length - 1];

        if (lastRange !== undefined) {
          lastRange.end = offset;
        }
      } else {
        text += " ";
        ranges.push({ start, end: offset });
      }

      continue;
    }

    for (const normalizedCharacter of normalizeSpanCharacter(character)) {
      text += normalizedCharacter;
      ranges.push({ start, end: offset });
    }
  }

  return { text, ranges };
}

function normalizedSpanText(value: string): string {
  return buildNormalizedTextIndex(value).text.trim();
}

function findNormalizedSpanRanges(text: string, spanText: string): TextRange[] {
  const normalized = buildNormalizedTextIndex(text);
  const needle = normalizedSpanText(spanText);
  const matches: TextRange[] = [];

  if (needle.length === 0) {
    return matches;
  }

  let index = normalized.text.indexOf(needle);

  while (index >= 0) {
    const first = normalized.ranges[index];
    const last = normalized.ranges[index + needle.length - 1];

    if (first !== undefined && last !== undefined) {
      matches.push({
        start: first.start,
        end: last.end,
      });
    }

    index = normalized.text.indexOf(needle, index + 1);
  }

  return matches;
}

function rangesOverlap(left: TextRange, right: TextRange): boolean {
  return left.start < right.end && right.start < left.end;
}

function findUniqueRemovalRange(input: {
  text: string;
  span: string;
  selectedRanges: readonly TextRange[];
}): TextRange | null {
  if (!valueAppearsIn(input.text, input.span)) {
    return null;
  }

  const matches = findNormalizedSpanRanges(input.text, input.span).filter(
    (range) => !input.selectedRanges.some((selected) => rangesOverlap(range, selected)),
  );

  return matches.length === 1 ? (matches[0] ?? null) : null;
}

function cleanRemovedSpanText(value: string): string {
  return value.replace(TRAILING_REMOVAL_JUNK_PATTERN, "").trim();
}

export function isStructurallyEmptyText(value: string): boolean {
  return STRUCTURALLY_EMPTY_TEXT_PATTERN.test(value);
}

export function deleteSpans(text: string, spans: readonly string[]): DeleteSpansResult {
  const ranges: TextRange[] = [];
  const removedSpans: string[] = [];

  for (const span of spans) {
    const range = findUniqueRemovalRange({
      text,
      span,
      selectedRanges: ranges,
    });

    if (range === null) {
      return {
        result: text,
        removedSpans,
        allRemoved: false,
      };
    }

    ranges.push(range);
    removedSpans.push(span);
  }

  let result = text;

  for (const range of [...ranges].sort((left, right) => right.start - left.start)) {
    result = `${result.slice(0, range.start)}${result.slice(range.end)}`;
  }

  return {
    result: cleanRemovedSpanText(result),
    removedSpans,
    allRemoved: true,
  };
}
