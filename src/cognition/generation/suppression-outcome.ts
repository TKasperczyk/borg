import type { GenerationSuppressionReason } from "./types.js";

export type SuppressionOutcomeClass =
  | "deliberate-silence"
  | "emission-failed"
  | "guard-blocked"
  | "observed"
  | "unknown";

type MappedSuppressionOutcomeClass = Exclude<
  SuppressionOutcomeClass,
  "observed" | "unknown"
>;

const SUPPRESSION_REASON_OUTCOME_CLASS = {
  generation_gate: "deliberate-silence",
  active_discourse_stop: "deliberate-silence",
  empty_finalizer: "emission-failed",
  finalizer_failed: "emission-failed",
  finalizer_no_output: "deliberate-silence",
  invalid_tool_after_regenerate: "emission-failed",
  no_output_tool: "deliberate-silence",
  s2_planner_no_output: "deliberate-silence",
  closure_pressure_only: "deliberate-silence",
  closure_response_audit_failed_closed: "guard-blocked",
  commitment_violation: "guard-blocked",
  commitment_violation_after_regenerate: "guard-blocked",
  commitment_revision_failed: "guard-blocked",
  internal_identifier_leak: "guard-blocked",
  rewrite_unsupported_or_empty: "guard-blocked",
} as const satisfies Record<GenerationSuppressionReason, MappedSuppressionOutcomeClass>;

export function classifySuppressionReason(reason: unknown): SuppressionOutcomeClass {
  if (typeof reason !== "string" || !Object.hasOwn(SUPPRESSION_REASON_OUTCOME_CLASS, reason)) {
    return "unknown";
  }

  return SUPPRESSION_REASON_OUTCOME_CLASS[reason as GenerationSuppressionReason];
}
