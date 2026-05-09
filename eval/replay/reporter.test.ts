import { describe, expect, it } from "vitest";

import { buildReplayReport, formatReplayMarkdown, type ReplayScenarioRecord } from "./reporter.js";

function pipeline(overrides: Partial<ReplayScenarioRecord["pipelines"]["A"]> = {}) {
  return {
    pipelineId: "A" as const,
    safe: true,
    safeWithUsefulOutput: true,
    guardCaught: false,
    shadowSevereRemaining: null,
    emittedText: "safe",
    emissionKind: "message",
    guardCategories: [],
    error: null,
    ...overrides,
  };
}

describe("replay reporter", () => {
  it("formats the replay pass/fail table", () => {
    const scenario: ReplayScenarioRecord = {
      id: "13-synthetic",
      failureClass: "Synthetic failure",
      description: "Synthetic description.",
      notes: [],
      pipelines: {
        A: pipeline(),
        B: pipeline({ pipelineId: "B", safe: false, guardCaught: true }),
        C: pipeline({
          pipelineId: "C",
          safe: true,
          shadowSevereRemaining: false,
        }),
        Cdoubleprime: pipeline({
          pipelineId: "Cdoubleprime",
          safe: true,
          safeWithUsefulOutput: true,
          shadowSevereRemaining: true,
        }),
      },
    };
    const report = buildReplayReport([scenario], "2026-05-06T00:00:00.000Z");
    const markdown = formatReplayMarkdown(report);

    expect(markdown).toContain("| Failure class | Pipeline A safe | Pipeline A guard caught |");
    expect(markdown).toContain("| Synthetic failure | yes | no | yes | no | yes | yes | yes |");
    expect(markdown).toContain("- Pipeline C″ safe_with_useful_output: 1 / 1");
    expect(markdown).not.toContain("validator caught");
    expect(markdown).not.toContain("Manifest Validation");
  });
});
