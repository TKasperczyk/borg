import { utf16SafePrefixEnd, utf16SafeSuffixStart } from "./utf16-boundary.js";

export type HeadTailTextExcerpt = {
  text: string;
  head: string;
  tail: string;
  exact: boolean;
  renderedChars: number;
  totalChars: number;
};

/**
 * Applies a mechanical character budget while preserving both ends of the
 * source. Callers own the surrounding truncation metadata so the same source
 * cut does not need to repeat explanatory prose in every row.
 */
export function headTailTextExcerpt(value: string, maxChars: number): HeadTailTextExcerpt {
  const budget = Math.max(0, Math.floor(maxChars));

  if (value.length <= budget) {
    return {
      text: value,
      head: value,
      tail: "",
      exact: true,
      renderedChars: value.length,
      totalChars: value.length,
    };
  }

  const requestedHeadChars = Math.ceil(budget / 2);
  const headEnd = utf16SafePrefixEnd(value, requestedHeadChars);
  const requestedTailChars = Math.max(0, budget - headEnd);
  const tailStart = utf16SafeSuffixStart(value, value.length - requestedTailChars);
  const head = value.slice(0, headEnd);
  const tail = value.slice(tailStart);

  return {
    text: `${head}${tail}`,
    head,
    tail,
    exact: false,
    renderedChars: head.length + tail.length,
    totalChars: value.length,
  };
}
