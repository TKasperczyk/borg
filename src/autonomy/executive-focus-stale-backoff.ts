import type { StreamWatermark } from "../stream/index.js";
import { StorageError } from "../util/errors.js";

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
