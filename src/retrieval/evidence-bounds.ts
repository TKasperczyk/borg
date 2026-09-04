// Shared bounds for verbatim recalled-message evidence exposed to bounded consumers.
export const MAX_RECALLED_SOURCE_MESSAGES_PER_EPISODE = 3;
export const MAX_RECALLED_EVIDENCE_TEXT_CHARS = 180;

export function clipRecalledEvidenceText(text: string): string {
  return text.slice(0, MAX_RECALLED_EVIDENCE_TEXT_CHARS);
}
