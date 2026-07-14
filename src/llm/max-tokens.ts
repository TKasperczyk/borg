function matchesModelFamily(model: string, family: RegExp): boolean {
  return family.test(model.trim().toLowerCase());
}

export function getModelMaxOutputTokens(model: string): number {
  if (matchesModelFamily(model, /^claude-(opus|sonnet)-4(?:[-._].+)?$/)) {
    return 64_000;
  }

  if (matchesModelFamily(model, /^claude-haiku-4(?:[-._].+)?$/)) {
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
