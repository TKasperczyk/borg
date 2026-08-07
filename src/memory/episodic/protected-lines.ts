const OUTCOME_EPISODE_TOKEN_PATTERN = /OUTCOME fp=\S+/u;
const PROTECTED_EPISODE_TOKEN_PATTERNS = [
  OUTCOME_EPISODE_TOKEN_PATTERN,
  /decision=\S+/u,
  /^[\t ]*ticket=\S+ action=\S+(?:[\t ]+.*)?$/u,
  /^[\t ]*action=teams_card[\t ]*$/u,
] as const;
const COMPLETE_LINE_PATTERN = /\r\n|\n|\r/u;
const OUTCOME_HEADER_PREFIX_PATTERN = /^\[[^\]\r\n]+\]$/u;
const OUTCOME_HEADER_FIELD_PATTERN = /^(?:role|tenant)=\S+$/u;

function protectedTokenStart(line: string): number | null {
  let start = Number.POSITIVE_INFINITY;

  for (const pattern of PROTECTED_EPISODE_TOKEN_PATTERNS) {
    const match = pattern.exec(line);

    if (match !== null) {
      start = Math.min(start, match.index);
    }
  }

  return Number.isFinite(start) ? start : null;
}

function isStandaloneOutcomeHeaderLine(line: string, outcomeMatch: RegExpExecArray): boolean {
  const prefix = line.slice(0, outcomeMatch.index).trim();
  const suffix = line.slice(outcomeMatch.index + outcomeMatch[0].length).trim();
  const prefixIsHeaderLabel = prefix.length === 0 || OUTCOME_HEADER_PREFIX_PATTERN.test(prefix);
  const suffixFields = suffix.length === 0 ? [] : suffix.split(/[\t ]+/u);

  return (
    prefixIsHeaderLabel && suffixFields.every((field) => OUTCOME_HEADER_FIELD_PATTERN.test(field))
  );
}

function collectOutcomeEmbeddingHeaderLines(sourceTexts: readonly string[]): string[] {
  const order: string[] = [];
  const standaloneLinesByToken = new Map<string, string[]>();

  for (const line of collectProtectedEpisodeTokenLines(sourceTexts)) {
    const outcomeMatch = OUTCOME_EPISODE_TOKEN_PATTERN.exec(line);

    if (outcomeMatch === null) {
      continue;
    }

    const token = outcomeMatch[0];

    if (!standaloneLinesByToken.has(token)) {
      order.push(token);
      standaloneLinesByToken.set(token, []);
    }

    if (!isStandaloneOutcomeHeaderLine(line, outcomeMatch)) {
      continue;
    }

    const standaloneLines = standaloneLinesByToken.get(token)!;

    if (!standaloneLines.includes(line)) {
      standaloneLines.push(line);
    }
  }

  return order.flatMap((token) => {
    const standaloneLines = standaloneLinesByToken.get(token) ?? [];
    return standaloneLines.length > 0 ? standaloneLines : [token];
  });
}

function proseWithoutProtectedEpisodeTokens(narrative: string): string {
  const protectedLines = new Set(collectProtectedEpisodeTokenLines([narrative]));

  return narrative
    .split(COMPLETE_LINE_PATTERN)
    .flatMap((line) => {
      if (!protectedLines.has(line)) {
        return [line];
      }

      const tokenStart = protectedTokenStart(line);

      if (tokenStart === null) {
        return [line];
      }

      const prefix = line.slice(0, tokenStart).trim();

      return prefix.length === 0 || OUTCOME_HEADER_PREFIX_PATTERN.test(prefix) ? [] : [prefix];
    })
    .join("\n")
    .trim();
}

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

export function buildConsolidationEpisodeEmbeddingText(input: {
  title: string;
  synthesizedNarrative: string;
  protectedSourceTexts: readonly string[];
  tags: readonly string[];
  participants: readonly string[];
}): string {
  const prose = proseWithoutProtectedEpisodeTokens(input.synthesizedNarrative);
  const narrativeParts = [
    ...(prose.length === 0 ? [] : [prose]),
    ...collectOutcomeEmbeddingHeaderLines(input.protectedSourceTexts),
  ];

  return `${input.title.trim()}\n${narrativeParts.join("\n")}\n${input.tags.join(" ")}\n${input.participants.join(" ")}`;
}
