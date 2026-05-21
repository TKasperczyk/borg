// Mechanical escaping for substrate-owned prompt tags. This only prevents
// untrusted content from forging borg_* prompt blocks; it does not interpret
// natural-language content.
export function escapeReservedBorgTags(content: string): string {
  return content.replace(/<(\/?)borg_/gi, "<$1-borg_");
}
