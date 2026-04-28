import { describe, expect, it } from "vitest";

import { formatAssessorReport, hasAssessorRunFailure } from "./report.js";
import { runScenarios, ScenarioRunner } from "./runner.js";
import { failingMockFixtureScenario } from "./scenarios/failing-mock-fixture.js";
import { recallScenario } from "./scenarios/recall.js";

describe("ScenarioRunner", () => {
  it("orchestrates a mock scenario end-to-end", async () => {
    const result = await new ScenarioRunner({
      runId: "runner-test",
      scenario: recallScenario,
      mock: true,
    }).run();

    expect(result.verdict.status).toBe("pass");
    expect(result.turns.length).toBeGreaterThan(0);
    expect(result.traceAssertions.some((assertion) => assertion.passed)).toBe(true);
    expect(result.coveredPhases).toContain("action");
  });

  it("formats a non-empty mock report", async () => {
    const report = await runScenarios({
      runId: "report-test",
      scenarios: [recallScenario],
      mock: true,
    });
    const markdown = formatAssessorReport(report);

    expect(markdown).toContain("# Borg Assessor Run report-test");
    expect(markdown).toContain("Scenario: recall");
  });

  it("fails a mock scenario when required trace assertions fail", async () => {
    const result = await new ScenarioRunner({
      runId: "failing-fixture-test",
      scenario: failingMockFixtureScenario,
      mock: true,
    }).run();

    expect(result.verdict.status).toBe("fail");
    expect(result.traceAssertions.some((assertion) => !assertion.passed)).toBe(true);
    expect(result.verdict.evidence.join("\n")).toContain("UNREACHABLE_FIXTURE_TOKEN");
  });

  it("marks reports with failed mock assertions as failing", async () => {
    const report = await runScenarios({
      runId: "failing-report-test",
      scenarios: [failingMockFixtureScenario],
      mock: true,
    });

    expect(hasAssessorRunFailure(report)).toBe(true);
  });
});
