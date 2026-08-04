import type { StreamWatermark } from "../stream/index.js";
import { StorageError } from "../util/errors.js";

export const FLEET_BRAKE_PROCESS_NAME = "autonomy:fleet:operational-empty-streak";

export type FleetBrakeMetadata = {
  empty_streak: number;
  streak_anchor_ts: number;
  last_wake_ts: number;
  error_streak: number;
  last_error_ts: number;
  bypass_count: number;
};

export type FleetBrakeOptions = {
  enabled: boolean;
  emptyStreakThreshold: number;
  baseCooldownMs: number;
  cooldownMultiplier: number;
  maxCooldownMs: number;
  errorStreakThreshold: number;
  errorBasePauseMs: number;
  errorMaxPauseMs: number;
  freshnessBypassCap: number;
};

export const DEFAULT_FLEET_BRAKE_OPTIONS: FleetBrakeOptions = {
  enabled: true,
  emptyStreakThreshold: 5,
  baseCooldownMs: 30 * 60 * 1_000,
  cooldownMultiplier: 2,
  maxCooldownMs: 6 * 60 * 60 * 1_000,
  errorStreakThreshold: 3,
  errorBasePauseMs: 5 * 60 * 1_000,
  errorMaxPauseMs: 30 * 60 * 1_000,
  freshnessBypassCap: 3,
};

const METADATA_KEYS = [
  "empty_streak",
  "streak_anchor_ts",
  "last_wake_ts",
  "error_streak",
  "last_error_ts",
  "bypass_count",
] as const satisfies readonly (keyof FleetBrakeMetadata)[];

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isNonnegativeInteger(value: unknown): value is number {
  return (
    typeof value === "number" && Number.isFinite(value) && Number.isInteger(value) && value >= 0
  );
}

export function emptyFleetBrakeMetadata(): FleetBrakeMetadata {
  return {
    empty_streak: 0,
    streak_anchor_ts: 0,
    last_wake_ts: 0,
    error_streak: 0,
    last_error_ts: 0,
    bypass_count: 0,
  };
}

/**
 * The fleet singleton sits on the scheduler hot path, so it intentionally
 * differs from the strict per-goal reader. Missing fields are treated as zero
 * for forward/backward tolerance. A wrong-typed value reports one error and
 * returns zeros; the next bookkeeping write replaces the malformed shape.
 */
export function readFleetBrakeMetadata(
  watermark: Pick<StreamWatermark, "metadata"> | null,
  notifyError?: (error: StorageError) => void,
): FleetBrakeMetadata {
  if (watermark === null || watermark.metadata === null) {
    return emptyFleetBrakeMetadata();
  }

  const metadata = watermark.metadata;

  if (!isRecord(metadata)) {
    notifyFleetBrakeMetadataError(notifyError);
    return emptyFleetBrakeMetadata();
  }

  const parsed = emptyFleetBrakeMetadata();

  for (const key of METADATA_KEYS) {
    const value = metadata[key];

    if (value === undefined) {
      continue;
    }

    if (!isNonnegativeInteger(value)) {
      notifyFleetBrakeMetadataError(notifyError);
      return emptyFleetBrakeMetadata();
    }

    parsed[key] = value;
  }

  return parsed;
}

function notifyFleetBrakeMetadataError(
  notifyError: ((error: StorageError) => void) | undefined,
): void {
  if (notifyError === undefined) {
    return;
  }

  try {
    notifyError(
      new StorageError("Autonomy fleet-brake watermark metadata is invalid", {
        code: "AUTONOMY_FLEET_BRAKE_METADATA_INVALID",
      }),
    );
  } catch {
    // Observer failures must never turn a tolerant scheduler-state read into a
    // global autonomy failure.
  }
}

function cappedExponentialMs(input: {
  baseMs: number;
  multiplier: number;
  exponent: number;
  maxMs: number;
}): number {
  const scaled = input.baseMs * Math.pow(input.multiplier, input.exponent);
  const finiteScaled = Number.isFinite(scaled) ? scaled : input.maxMs;

  return Math.min(Math.max(input.baseMs, finiteScaled), input.maxMs);
}

export function fleetBrakeCooldownUntilMs(
  metadata: FleetBrakeMetadata,
  options: Pick<
    FleetBrakeOptions,
    "emptyStreakThreshold" | "baseCooldownMs" | "cooldownMultiplier" | "maxCooldownMs"
  >,
): number | null {
  if (metadata.empty_streak < options.emptyStreakThreshold) {
    return null;
  }

  const cooldownMs = cappedExponentialMs({
    baseMs: options.baseCooldownMs,
    multiplier: options.cooldownMultiplier,
    exponent: metadata.empty_streak - options.emptyStreakThreshold,
    maxMs: options.maxCooldownMs,
  });

  return metadata.last_wake_ts + cooldownMs;
}

export function fleetBrakeErrorPausedUntilMs(
  metadata: FleetBrakeMetadata,
  options: Pick<FleetBrakeOptions, "errorStreakThreshold" | "errorBasePauseMs" | "errorMaxPauseMs">,
): number | null {
  if (metadata.error_streak < options.errorStreakThreshold) {
    return null;
  }

  const pauseMs = cappedExponentialMs({
    baseMs: options.errorBasePauseMs,
    multiplier: 2,
    exponent: metadata.error_streak - options.errorStreakThreshold,
    maxMs: options.errorMaxPauseMs,
  });

  return metadata.last_error_ts + pauseMs;
}
