// "Estimated tokens" everywhere in borg means this chars/4 estimate, and the real
// provider input it stands for depends on the surface. Measured from trace usage
// (uncached + cache-written + cache-read input) over 2026-08-25..09-04:
//   compact planner call   median 1.98x the estimate (n=317, 1.75..2.03)
//   finalizer call         median 1.76x the estimate (n=508, 1.71..2.01)
// An estimated figure from one surface is not comparable to another's without
// that surface's ratio; a budget denominated here is a ceiling on the estimate,
// not on provider tokens.
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
