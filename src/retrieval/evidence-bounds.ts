import { utf16SafePrefix } from "../util/utf16-boundary.js";

// Shared bounds for verbatim recalled-message evidence exposed to bounded consumers.
export const MAX_RECALLED_SOURCE_MESSAGES_PER_EPISODE = 3;
export const MAX_RECALLED_EVIDENCE_TEXT_CHARS = 180;

// Code-unit safe: these excerpts are embedded (activity ranking) and re-serialised
// by the Team Agent, so a cut through an emoji must not leave a lone surrogate.
export function clipRecalledEvidenceText(text: string): string {
  return utf16SafePrefix(text, MAX_RECALLED_EVIDENCE_TEXT_CHARS);
}
