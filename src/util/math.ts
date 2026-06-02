export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function halfLifeDecay(elapsed: number, halfLife: number): number {
  return Math.pow(0.5, elapsed / halfLife);
}
