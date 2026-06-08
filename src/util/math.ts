import { positiveIntegerValue } from "./parse.js";

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function coercePositiveIntegerOrFallback(
  value: number | undefined,
  fallback: number,
): number {
  return value === undefined || !Number.isFinite(value) || value <= 0
    ? fallback
    : Math.floor(value);
}

export function clampPositiveIntegerOrFallback(
  value: number | undefined,
  fallback: number,
): number {
  if (value === undefined || !Number.isFinite(value)) {
    return fallback;
  }

  return Math.max(1, Math.floor(value));
}

export function requirePositiveInteger(value: number, label = "value"): number {
  if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
    throw new TypeError(`${label} must be a positive integer`);
  }

  return value;
}

export function positiveIntegerOptionOrFallback(
  value: number | undefined,
  fallback: number,
  label: string,
): number {
  return value === undefined ? fallback : requirePositiveInteger(value, label);
}

export function optionalPositiveIntegerOption(
  value: number | undefined,
  label: string,
): number | undefined {
  return value === undefined ? undefined : requirePositiveInteger(value, label);
}

export function positiveIntegerRecordParamOrFallback(
  params: Record<string, unknown> | undefined,
  key: string,
  fallback: number,
): number {
  return positiveIntegerValue(params?.[key]) ?? fallback;
}

export function coerceUnitIntervalOrFallback(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value)) {
    return fallback;
  }

  return clamp(value, 0, 1);
}

export function requireUnitInterval(value: number, label = "value"): number {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new TypeError(`${label} must be between 0 and 1`);
  }

  return value;
}

export function unitIntervalOptionOrFallback(
  value: number | undefined,
  fallback: number,
  label: string,
): number {
  return value === undefined ? fallback : requireUnitInterval(value, label);
}

export function halfLifeDecay(elapsed: number, halfLife: number): number {
  return Math.pow(0.5, elapsed / halfLife);
}
