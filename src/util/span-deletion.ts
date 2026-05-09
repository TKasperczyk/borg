import { valueAppearsIn } from "./text-presence.js";

const STRUCTURALLY_EMPTY_TEXT_PATTERN = /^[\s\p{P}\p{S}]*$/u;
const SPAN_WHITESPACE_PATTERN = /\s/u;
const TRAILING_REMOVAL_JUNK_PATTERN = /[\s,;:]+$/u;
const SUBSTANTIVE_TEXT_PATTERN = /[\p{L}\p{N}]/u;
const EMPTY_PARENS_PATTERN = /\(\s*\)/u;
const EMPTY_QUOTES_PATTERN = /['"‘’‚‛′“”„‟″]\s*['"‘’‚‛′“”„‟″]/u;
const ORPHAN_DASH_DOT_PATTERN = /(?:^|\s)(?:--|[-—–])\s*\.(?=\s|$)/u;
const ORPHAN_SPACE_PERIOD_PATTERN = /\s+\.\s*$/u;
const TRAILING_ORPHAN_PUNCTUATION_PATTERN = /[:;,—–-]\s*\.?$/u;
const INITIAL_CHAIN_SENTENCE_PATTERN = /^(?:\p{Lu}\.\s*)+$/u;
const LONE_CONNECTIVE_SENTENCE_PATTERN =
  /^(?:and|but|or|so|because|though|although|however|also|then|the point is|it(?:'|’)?s just|it is just)\s*\.?$/iu;
const COMMON_ABBREVIATION_SENTENCE_PATTERN =
  /^(?:mr|mrs|ms|dr|prof|sr|jr|vs|etc|e\.g|i\.e|u\.s|u\.k)\.$/iu;

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

type SentenceSegment = {
  segment: string;
  index: number;
};

type SentenceSegmenter = {
  segment(input: string): Iterable<SentenceSegment>;
};

const sentenceSegmenter = new (Intl as typeof Intl & {
  Segmenter: new (
    locales?: Intl.LocalesArgument,
    options?: { granularity: "sentence" },
  ) => SentenceSegmenter;
}).Segmenter(undefined, { granularity: "sentence" });

type NormalizedTextIndex = {
  text: string;
  ranges: TextRange[];
};

export type DeleteSpansResult = {
  result: string;
  removedSpans: string[];
  allRemoved: boolean;
};

export type SentenceAwareDeleteSpansOutcome = "clean" | "malformed" | "empty";

export type SentenceAwareDeleteSpansResult = {
  rewrittenText: string;
  outcome: SentenceAwareDeleteSpansOutcome;
  removedSpans: string[];
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

function trimRange(text: string, range: TextRange): TextRange | null {
  let start = range.start;
  let end = range.end;

  while (start < end && /\s/u.test(text[start] ?? "")) {
    start += 1;
  }

  while (end > start && /\s/u.test(text[end - 1] ?? "")) {
    end -= 1;
  }

  return start < end ? { start, end } : null;
}

function sentenceRanges(text: string): TextRange[] {
  const ranges: TextRange[] = [];

  for (const segment of sentenceSegmenter.segment(text)) {
    const range = trimRange(text, {
      start: segment.index,
      end: segment.index + segment.segment.length,
    });

    if (range !== null) {
      ranges.push(range);
    }
  }

  return mergeAbbreviationSentenceRanges(text, ranges);
}

// Node's Intl.Segmenter still splits common abbreviations like "Dr.", so stitch those first.
function mergeAbbreviationSentenceRanges(
  text: string,
  ranges: readonly TextRange[],
): TextRange[] {
  const merged: TextRange[] = [];

  for (const range of ranges) {
    const previous = merged[merged.length - 1];

    if (
      previous !== undefined &&
      COMMON_ABBREVIATION_SENTENCE_PATTERN.test(text.slice(previous.start, previous.end).trim())
    ) {
      previous.end = range.end;
      continue;
    }

    merged.push({ ...range });
  }

  return merged;
}

function paragraphRanges(text: string): TextRange[] {
  const ranges: TextRange[] = [];
  const paragraphBreakPattern = /\n\s*\n/gu;
  let start = 0;
  let match: RegExpExecArray | null;

  while ((match = paragraphBreakPattern.exec(text)) !== null) {
    const range = trimRange(text, { start, end: match.index });

    if (range !== null) {
      ranges.push(range);
    }

    start = paragraphBreakPattern.lastIndex;
  }

  const range = trimRange(text, { start, end: text.length });

  if (range !== null) {
    ranges.push(range);
  }

  return ranges;
}

function rangeContainsPosition(range: TextRange, position: number): boolean {
  return range.start <= position && position < range.end;
}

function unitRangeForRemovalRange(units: readonly TextRange[], range: TextRange): TextRange {
  if (units.length === 0) {
    return range;
  }

  const startUnit = units.find((unit) => rangeContainsPosition(unit, range.start)) ?? units[0];
  const endPosition = Math.max(range.start, range.end - 1);
  const endUnit =
    units.find((unit) => rangeContainsPosition(unit, endPosition)) ?? units[units.length - 1];

  return {
    start: startUnit?.start ?? range.start,
    end: endUnit?.end ?? range.end,
  };
}

function mergeRanges(text: string, ranges: readonly TextRange[]): TextRange[] {
  const sorted = [...ranges].sort((left, right) => left.start - right.start);
  const merged: TextRange[] = [];

  for (const range of sorted) {
    const previous = merged[merged.length - 1];

    if (previous === undefined) {
      merged.push({ ...range });
      continue;
    }

    if (range.start <= previous.end || text.slice(previous.end, range.start).trim().length === 0) {
      previous.end = Math.max(previous.end, range.end);
      continue;
    }

    merged.push({ ...range });
  }

  return merged;
}

function cleanSentenceAwareText(value: string): string {
  return value
    .replace(/[ \t]+\n/gu, "\n")
    .replace(/(?<=\S)[ \t]{2,}(?=\S)/gu, " ")
    .replace(/\n{3,}/gu, "\n\n")
    .replace(/^(?:[ \t]*\n)+/u, "")
    .replace(/[ \t]*(?:\n[ \t]*)*$/u, "");
}

function deleteRanges(text: string, ranges: readonly TextRange[]): string {
  let result = text;

  for (const range of [...ranges].sort((left, right) => right.start - left.start)) {
    result = `${result.slice(0, range.start)}${result.slice(range.end)}`;
  }

  return cleanSentenceAwareText(result);
}

function malformedSentenceResidue(text: string): boolean {
  if (EMPTY_PARENS_PATTERN.test(text) || EMPTY_QUOTES_PATTERN.test(text)) {
    return true;
  }

  if (ORPHAN_DASH_DOT_PATTERN.test(text)) {
    return true;
  }

  const ranges = sentenceRanges(text);
  const sentences =
    ranges.length === 0 ? [text] : ranges.map((range) => text.slice(range.start, range.end));

  return sentences.some((sentence) => {
    const trimmed = sentence.trim();

    if (trimmed.length === 0) {
      return false;
    }

    return (
      ORPHAN_SPACE_PERIOD_PATTERN.test(trimmed) ||
      TRAILING_ORPHAN_PUNCTUATION_PATTERN.test(trimmed) ||
      INITIAL_CHAIN_SENTENCE_PATTERN.test(trimmed) ||
      LONE_CONNECTIVE_SENTENCE_PATTERN.test(trimmed)
    );
  });
}

function sentenceAwareOutcome(text: string): SentenceAwareDeleteSpansOutcome {
  if (
    text.trim().length === 0 ||
    !SUBSTANTIVE_TEXT_PATTERN.test(text) ||
    isStructurallyEmptyText(text)
  ) {
    return "empty";
  }

  return malformedSentenceResidue(text) ? "malformed" : "clean";
}

function expandedDeletion(input: {
  text: string;
  spans: readonly string[];
  units: readonly TextRange[];
}): SentenceAwareDeleteSpansResult {
  const rawRanges: TextRange[] = [];
  const expandedRanges: TextRange[] = [];
  const removedSpans: string[] = [];

  for (const span of input.spans) {
    const range = findUniqueRemovalRange({
      text: input.text,
      span,
      selectedRanges: rawRanges,
    });

    if (range === null) {
      return {
        rewrittenText: input.text,
        outcome: "malformed",
        removedSpans,
      };
    }

    rawRanges.push(range);
    expandedRanges.push(unitRangeForRemovalRange(input.units, range));
    removedSpans.push(span);
  }

  const rewrittenText = deleteRanges(input.text, mergeRanges(input.text, expandedRanges));

  return {
    rewrittenText,
    outcome: sentenceAwareOutcome(rewrittenText),
    removedSpans,
  };
}

export function deleteSpansSentenceAware(
  text: string,
  spans: readonly string[],
): SentenceAwareDeleteSpansResult {
  const sentenceDeletion = expandedDeletion({
    text,
    spans,
    units: sentenceRanges(text),
  });

  if (sentenceDeletion.outcome !== "malformed") {
    return sentenceDeletion;
  }

  const paragraphDeletion = expandedDeletion({
    text,
    spans,
    units: paragraphRanges(text),
  });

  if (paragraphDeletion.outcome === "clean" || paragraphDeletion.outcome === "empty") {
    return paragraphDeletion;
  }

  return sentenceDeletion;
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
