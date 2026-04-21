export type TokenizeTextOptions = {
  minLength?: number;
  stopwords?: readonly string[];
};

export function tokenizeText(text: string, options: TokenizeTextOptions = {}): Set<string> {
  const minLength = options.minLength ?? 3;
  const stopwords = new Set((options.stopwords ?? []).map((value) => value.toLowerCase()));

  return new Set(
    text
      .toLowerCase()
      .split(/[^a-z0-9]+/i)
      .filter((token) => token.length >= minLength && !stopwords.has(token)),
  );
}
