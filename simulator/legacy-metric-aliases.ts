import type { MetricsRow } from "./types.js";

// Legacy alias keys preserved in raw metrics JSONL for one-sprint backward compat
// with external dashboards and fixtures. Filtered from overseer-facing context
// so the LLM doesn't misread cumulative counters as final-compile snapshots
// (see v82 review).
export const LEGACY_METRIC_ALIAS_KEYS = [
  "shared_state_omitted_recent_entries",
  "shared_state_omitted_live_recent_operational",
  "shared_state_omitted_live_recent_low_salience",
  "shared_state_omitted_live_old",
  "shared_state_omitted_live_unknown_age",
  "shared_state_omitted_locked",
  "shared_state_omitted_pending",
  "shared_state_omitted_low_salience_live",
  "shared_state_omitted_dormant_live",
  "shared_state_active_low_salience_live",
  "shared_state_active_dormant_live",
  "shared_state_empty_update_attempted_total",
  "finalizer_no_output_by_category",
  "borg_aborted_turns",
] as const satisfies readonly (keyof MetricsRow)[];

export type LegacyMetricAlias = (typeof LEGACY_METRIC_ALIAS_KEYS)[number];
export type OverseerMetricsRow = Omit<MetricsRow, LegacyMetricAlias>;

export const LEGACY_METRIC_ALIASES: ReadonlySet<LegacyMetricAlias> = new Set(
  LEGACY_METRIC_ALIAS_KEYS,
);

export function stripLegacyAliases<T extends object>(
  row: T,
): Omit<T, Extract<keyof T, LegacyMetricAlias>> {
  const filtered = { ...row };

  for (const key of Object.keys(filtered)) {
    if (LEGACY_METRIC_ALIASES.has(key as LegacyMetricAlias)) {
      delete filtered[key as keyof T];
    }
  }

  return filtered as Omit<T, Extract<keyof T, LegacyMetricAlias>>;
}
