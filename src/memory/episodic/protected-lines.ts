const PROTECTED_EPISODE_TOKEN_PATTERNS = [/OUTCOME fp=\S+/u, /decision=\S+/u] as const;
const COMPLETE_LINE_PATTERN = /\r\n|\n|\r/u;

export function collectProtectedEpisodeTokenLines(sourceTexts: readonly string[]): string[] {
  const lines: string[] = [];
  const seen = new Set<string>();

  for (const sourceText of sourceTexts) {
    for (const line of sourceText.split(COMPLETE_LINE_PATTERN)) {
      if (
        seen.has(line) ||
        !PROTECTED_EPISODE_TOKEN_PATTERNS.some((pattern) => pattern.test(line))
      ) {
        continue;
      }

      seen.add(line);
      lines.push(line);
    }
  }

  return lines;
}

export function preserveProtectedEpisodeTokenLines(
  narrative: string,
  sourceTexts: readonly string[],
): string {
  let protectedNarrative = narrative.trim();
  const narrativeLines = new Set(protectedNarrative.split(COMPLETE_LINE_PATTERN));

  for (const line of collectProtectedEpisodeTokenLines(sourceTexts)) {
    if (narrativeLines.has(line)) {
      continue;
    }

    protectedNarrative = protectedNarrative.length === 0 ? line : `${protectedNarrative}\n${line}`;
    narrativeLines.add(line);
  }

  return protectedNarrative;
}
