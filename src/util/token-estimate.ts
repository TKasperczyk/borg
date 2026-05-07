export function estimatePromptTokens(text: string): number {
  return estimatePromptTokensFromLength(text.length);
}

export function estimatePromptTokensFromLength(length: number): number {
  return Math.max(1, Math.ceil(length / 4));
}

export function stringifyPromptContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  try {
    return JSON.stringify(content ?? null);
  } catch {
    return String(content);
  }
}
