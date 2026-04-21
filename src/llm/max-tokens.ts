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

  return 8_192;
}
