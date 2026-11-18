import { ACTION_STATE_METADATA, type ActionRecord } from "./types.js";

export const STALE_OPEN_ROW_DAYS = 30;
const DAY_MS = 24 * 60 * 60 * 1_000;

type ActionActivity = Pick<ActionRecord, "state" | "updated_at" | "last_referenced_at_ms">;

export function actionActivityCutoff(nowMs: number): number {
  return nowMs - STALE_OPEN_ROW_DAYS * DAY_MS;
}

export function lastActionActivityAt(action: ActionActivity): number {
  return Math.max(action.updated_at, action.last_referenced_at_ms ?? action.updated_at);
}

/** Page priority only: age never gates similarity recall or changes a stored state. */
export function actionActivityTier(action: ActionActivity, nowMs: number): 0 | 1 | 2 | 3 {
  const open = ACTION_STATE_METADATA[action.state].active;
  const recent = lastActionActivityAt(action) >= actionActivityCutoff(nowMs);
  return recent ? (open ? 0 : 1) : open ? 2 : 3;
}

export function isStaleOpenAction(action: ActionActivity, nowMs: number): boolean {
  return actionActivityTier(action, nowMs) === 2;
}
