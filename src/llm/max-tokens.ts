function matchesModelFamily(model: string, family: RegExp): boolean {
  return family.test(model.trim().toLowerCase());
}

// Family matches are version-generic: a model bump must not silently drop the
// ceiling to the 8_192 fallback below. (The Opus/Sonnet families now advertise
// 128_000; 64_000 is the ceiling this harness deliberately runs at.)
export function getModelMaxOutputTokens(model: string): number {
  if (matchesModelFamily(model, /^claude-(opus|sonnet)-\d(?:[-._].+)?$/)) {
    return 64_000;
  }

  if (matchesModelFamily(model, /^claude-haiku-\d(?:[-._].+)?$/)) {
    return 32_000;
  }

  if (matchesModelFamily(model, /^(?:[^/]+\/)?qwen3(?:[-._].+)?$/)) {
    return 16_384;
  }

  return 8_192;
}

export function clampMaxOutputTokens(model: string, requested: number): number {
  return Math.min(requested, getModelMaxOutputTokens(model));
}
