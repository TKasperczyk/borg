const WHITESPACE_PATTERN = /\s+/gu;
const LEADING_SEARCH_PUNCTUATION_PATTERN = /^[\s"'`]+/gu;
const TRAILING_SEARCH_PUNCTUATION_PATTERN = /[\s"'`.,!?;:]+$/gu;
const ALPHANUMERIC_PATTERN = /[\p{L}\p{N}]/u;

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

function normalizeSmartQuotes(value: string): string {
  let normalized = value;

  for (const [from, to] of SMART_QUOTE_REPLACEMENTS) {
    normalized = normalized.replaceAll(from, to);
  }

  return normalized;
}

function normalizePresenceText(value: string): string {
  return normalizeSmartQuotes(value)
    .normalize("NFC")
    .toLocaleLowerCase()
    .replace(WHITESPACE_PATTERN, " ")
    .trim();
}

function normalizePresenceNeedle(value: string): string {
  return normalizePresenceText(value)
    .replace(LEADING_SEARCH_PUNCTUATION_PATTERN, "")
    .replace(TRAILING_SEARCH_PUNCTUATION_PATTERN, "")
    .trim();
}

function isAlphaNumeric(value: string): boolean {
  return ALPHANUMERIC_PATTERN.test(value);
}

function hasAlphaNumericBoundary(input: { text: string; start: number; length: number }): boolean {
  const before = input.start === 0 ? "" : (input.text[input.start - 1] ?? "");
  const after = input.text[input.start + input.length] ?? "";

  return !isAlphaNumeric(before) && !isAlphaNumeric(after);
}

export function valueAppearsIn(text: string, value: string): boolean {
  const haystack = normalizePresenceText(text);
  const needle = normalizePresenceNeedle(value);

  if (haystack.length === 0 || needle.length === 0) {
    return false;
  }

  let start = haystack.indexOf(needle);

  while (start >= 0) {
    if (
      hasAlphaNumericBoundary({
        text: haystack,
        start,
        length: needle.length,
      })
    ) {
      return true;
    }

    start = haystack.indexOf(needle, start + 1);
  }

  return false;
}
