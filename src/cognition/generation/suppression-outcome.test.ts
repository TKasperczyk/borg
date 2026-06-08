import { describe, expect, it } from "vitest";

import type { GenerationSuppressionReason } from "./types.js";
import { classifySuppressionReason, type SuppressionOutcomeClass } from "./suppression-outcome.js";

const EXPECTED_OUTCOMES = {
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
} as const satisfies Record<GenerationSuppressionReason, SuppressionOutcomeClass>;

describe("classifySuppressionReason", () => {
  it.each(Object.entries(EXPECTED_OUTCOMES))("maps %s to %s", (reason, outcome) => {
    expect(classifySuppressionReason(reason)).toBe(outcome);
  });

  it("returns unknown for non-members", () => {
    expect(classifySuppressionReason("future_reason")).toBe("unknown");
    expect(classifySuppressionReason(null)).toBe("unknown");
  });
});
