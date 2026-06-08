import { describe, expect, it } from "vitest";
import { classifySuppressionReason } from "borg/suppression-outcome";

import type { StreamEntry } from "../api/types";
import { streamOutcomeSummary } from "./stream-outcomes";

function streamEntry(
  input: Pick<StreamEntry, "kind" | "content">,
): Pick<StreamEntry, "kind" | "content"> {
  return input;
}

describe("stream outcome classifier", () => {
  it.each([
    ["generation_gate", "deliberate-silence"],
    ["active_discourse_stop", "deliberate-silence"],
    ["finalizer_no_output", "deliberate-silence"],
    ["no_output_tool", "deliberate-silence"],
    ["s2_planner_no_output", "deliberate-silence"],
    ["closure_pressure_only", "deliberate-silence"],
    ["empty_finalizer", "emission-failed"],
    ["finalizer_failed", "emission-failed"],
    ["invalid_tool_after_regenerate", "emission-failed"],
    ["closure_response_audit_failed_closed", "guard-blocked"],
    ["commitment_violation", "guard-blocked"],
    ["commitment_violation_after_regenerate", "guard-blocked"],
    ["commitment_revision_failed", "guard-blocked"],
    ["internal_identifier_leak", "guard-blocked"],
    ["rewrite_unsupported_or_empty", "guard-blocked"],
  ] as const)("maps %s to %s", (reason, outcomeClass) => {
    expect(classifySuppressionReason(reason)).toBe(outcomeClass);
    expect(
      streamOutcomeSummary(streamEntry({ kind: "agent_suppressed", content: { reason } }))?.outcome
        .outcomeClass,
    ).toBe(outcomeClass);
  });

  it("keeps observed classification based on kind only", () => {
    const summary = streamOutcomeSummary(
      streamEntry({
        kind: "agent_observed",
        content: { reason: "finalizer_failed" },
      }),
    );

    expect(summary?.outcome.outcomeClass).toBe("observed");
    expect(summary?.outcome.tagKind).toBe("info");
    expect(summary?.reason).toBe("finalizer_failed");
  });

  it("uses neutral fallback for unknown suppression reasons", () => {
    const summary = streamOutcomeSummary(
      streamEntry({
        kind: "agent_suppressed",
        content: { reason: "future_reason_code" },
      }),
    );

    expect(summary?.outcome.outcomeClass).toBe("unknown");
    expect(summary?.outcome.tagKind).toBe("");
    expect(summary?.reason).toBe("future_reason_code");
  });

  it("copies structured no-output fields into the summary", () => {
    const summary = streamOutcomeSummary(
      streamEntry({
        kind: "agent_suppressed",
        content: {
          reason: "finalizer_no_output",
          primary_no_output_reason: "low_value_echo",
          no_output_categories: ["closure", "with_open_question"],
          structural_no_output_flags: ["with_open_question", "open_question_rendered"],
          finalizer_invalid_tool: {
            tool_name: "EmitAnswer",
            reason: "invalid schema",
            attempt: "regenerate",
          },
        },
      }),
    );

    expect(summary?.primaryNoOutputReason).toBe("low_value_echo");
    expect(summary?.noOutputCategories).toEqual(["closure", "with_open_question"]);
    expect(summary?.structuralNoOutputFlags).toEqual([
      "with_open_question",
      "open_question_rendered",
    ]);
    expect(summary?.finalizerInvalidTool).toEqual({
      tool_name: "EmitAnswer",
      reason: "invalid schema",
      attempt: "regenerate",
    });
  });
});
