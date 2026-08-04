import type { StreamWatermark } from "../stream/index.js";
import { StorageError } from "../util/errors.js";

// The name is historical: this is the per-goal empty-wake brake shared by
// executive_focus_due and goal_followup_due. Changing it would orphan durable
// dormancy state written under the original executive-focus name.
const EXECUTIVE_FOCUS_GOAL_STALE_BACKOFF_PREFIX = "autonomy:executive-focus-due:goal-stale-backoff";

export type ExecutiveFocusGoalStaleBackoffMetadata = {
  empty_count: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function getExecutiveFocusGoalStaleBackoffProcessName(goalId: string): string {
  return `${EXECUTIVE_FOCUS_GOAL_STALE_BACKOFF_PREFIX}:${goalId}`;
}

export function readExecutiveFocusGoalStaleBackoffMetadata(
  watermark: Pick<StreamWatermark, "metadata"> | null,
): ExecutiveFocusGoalStaleBackoffMetadata {
  if (watermark === null || watermark.metadata === null) {
    return { empty_count: 0 };
  }

  const metadata = watermark.metadata;

  if (
    !isRecord(metadata) ||
    typeof metadata.empty_count !== "number" ||
    !Number.isInteger(metadata.empty_count) ||
    metadata.empty_count < 0
  ) {
    throw new StorageError("Executive focus stale-backoff watermark metadata is invalid", {
      code: "EXECUTIVE_FOCUS_STALE_BACKOFF_METADATA_INVALID",
    });
  }

  return {
    empty_count: metadata.empty_count,
  };
}

export function executiveFocusGoalStaleBackoffCooldownMs(input: {
  baseCooldownMs: number;
  multiplier: number;
  maxCooldownMs: number;
  emptyCount: number;
}): number {
  const scaled = input.baseCooldownMs * Math.pow(input.multiplier, input.emptyCount);
  const capped = Math.min(
    Number.isFinite(scaled) ? scaled : input.maxCooldownMs,
    input.maxCooldownMs,
  );

  return Math.max(input.baseCooldownMs, capped);
}

export function goalStaleBackoffEndMs(input: {
  watermark: Pick<StreamWatermark, "metadata" | "updatedAt"> | null;
  lastProgressTs: number | null;
  baseCooldownMs: number;
  multiplier: number;
  maxCooldownMs: number;
  dormancyCount: number;
}): number | null {
  if (input.watermark === null) {
    return null;
  }

  if (input.lastProgressTs !== null && input.lastProgressTs >= input.watermark.updatedAt) {
    return null;
  }

  const metadata = readExecutiveFocusGoalStaleBackoffMetadata(input.watermark);

  if (metadata.empty_count <= 0) {
    return null;
  }

  if (metadata.empty_count >= input.dormancyCount) {
    return Number.POSITIVE_INFINITY;
  }

  return (
    input.watermark.updatedAt +
    executiveFocusGoalStaleBackoffCooldownMs({
      baseCooldownMs: input.baseCooldownMs,
      multiplier: input.multiplier,
      maxCooldownMs: input.maxCooldownMs,
      emptyCount: metadata.empty_count,
    })
  );
}
