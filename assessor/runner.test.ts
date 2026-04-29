import { describe, expect, it } from "vitest";

import { formatAssessorReport, hasAssessorRunFailure } from "./report.js";
import { reconcileVerdictWithAssertions, runScenarios, ScenarioRunner } from "./runner.js";
import { failingMockFixtureScenario } from "./scenarios/failing-mock-fixture.js";
import { recallScenario } from "./scenarios/recall.js";
import type { AssessorVerdict, TraceAssertionResult } from "./types.js";

function makeAssertion(passed: boolean, description = "test assertion"): TraceAssertionResult {
  return { description, passed, evidence: passed ? "ok" : "no" };
}

function makeVerdict(status: AssessorVerdict["status"]): AssessorVerdict {
  return { status, reasoning: "stub", evidence: ["stub-evidence"] };
}

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
    expect(result.verdict.evidence.join("\n")).toContain("fixture.never.called");
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

describe("reconcileVerdictWithAssertions", () => {
  it("upgrades inconclusive to pass when all assertions passed", () => {
    const reconciled = reconcileVerdictWithAssertions(makeVerdict("inconclusive"), [
      makeAssertion(true),
      makeAssertion(true),
    ]);

    expect(reconciled.status).toBe("pass");
    expect(reconciled.reasoning).toContain("Upgraded to pass");
    expect(reconciled.evidence).toEqual(["stub-evidence"]);
  });

  it("downgrades pass to fail when any assertion failed", () => {
    const reconciled = reconcileVerdictWithAssertions(makeVerdict("pass"), [
      makeAssertion(true),
      makeAssertion(false, "memory continuity"),
    ]);

    expect(reconciled.status).toBe("fail");
    expect(reconciled.reasoning).toContain("Downgraded to fail");
    expect(reconciled.reasoning).toContain("memory continuity");
  });

  it("downgrades inconclusive to fail when any assertion failed", () => {
    const reconciled = reconcileVerdictWithAssertions(makeVerdict("inconclusive"), [
      makeAssertion(false, "wake event"),
    ]);

    expect(reconciled.status).toBe("fail");
    expect(reconciled.reasoning).toContain("wake event");
  });

  it("keeps inconclusive when no assertions ran (e.g. runner aborted)", () => {
    const reconciled = reconcileVerdictWithAssertions(makeVerdict("inconclusive"), []);

    expect(reconciled.status).toBe("inconclusive");
    expect(reconciled.reasoning).toBe("stub");
  });

  it("keeps pass verdict when all assertions also passed", () => {
    const reconciled = reconcileVerdictWithAssertions(makeVerdict("pass"), [makeAssertion(true)]);

    expect(reconciled.status).toBe("pass");
    expect(reconciled.reasoning).toBe("stub");
  });

  it("keeps fail verdict regardless of assertion state", () => {
    const reconciled = reconcileVerdictWithAssertions(makeVerdict("fail"), [makeAssertion(true)]);

    expect(reconciled.status).toBe("fail");
    expect(reconciled.reasoning).toBe("stub");
  });
});
