import { describe, expect, it } from "vitest";

import { formatAssessorReport } from "../report.js";
import { runScenarios } from "../runner.js";
import { getScenario, SCENARIOS } from "./index.js";

describe("assessor scenarios", () => {
  it("registers the ten shipped scenarios", () => {
    expect(SCENARIOS.map((scenario) => scenario.name)).toEqual([
      "recall",
      "commitment-respect",
      "contradiction-handling",
      "goal-progress-tracking",
      "autonomous-wake-machinery",
      "identity-guard-refusal",
      "tool-use-correctness",
      "open-question-creation",
      "mood-persistence",
      "multi-session-continuity",
    ]);
    expect(getScenario("recall")?.name).toBe("recall");
  });

  it.each(SCENARIOS)("imports %s with required structure", (scenario) => {
    expect(scenario.name).toMatch(/^[a-z0-9-]+$/);
    expect(scenario.description.length).toBeGreaterThan(10);
    expect(scenario.systemPrompt.length).toBeGreaterThan(20);
    expect(scenario.maxTurns).toBeGreaterThan(0);
  });

  it("runs every scenario through the scripted mock path and produces a report", async () => {
    const report = await runScenarios({
      runId: "all-scenarios-test",
      scenarios: SCENARIOS,
      mock: true,
    });
    const markdown = formatAssessorReport(report);

    expect(report.results).toHaveLength(10);
    expect(markdown).toContain("Scenarios: 10");
    expect(markdown.length).toBeGreaterThan(1_000);
  });
});
