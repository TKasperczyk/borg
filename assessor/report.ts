import { TRACE_PHASES } from "./trace-reader.js";
import type { AssessorRunReport, AssessorStatus, ScenarioResult, TracePhase } from "./types.js";

const STATUS_ICON: Record<AssessorStatus, string> = {
  pass: "✅ PASS",
  fail: "❌ FAIL",
  inconclusive: "⚠️ INCONCLUSIVE",
};

function formatDuration(ms: number): string {
  if (ms < 1_000) {
    return `${Math.round(ms)}ms`;
  }

  return `${(ms / 1_000).toFixed(1)}s`;
}

function countByStatus(results: readonly ScenarioResult[], status: AssessorStatus): number {
  return results.filter((result) => result.verdict.status === status).length;
}

function totalTokens(results: readonly ScenarioResult[]): number {
  return results.reduce((sum, result) => sum + result.cost.approximateTokens, 0);
}

function excerpt(value: string): string {
  const normalized = value.replace(/\s+/g, " ").trim();

  return normalized.length > 220 ? `${normalized.slice(0, 217)}...` : normalized;
}

function formatAssertion(passed: boolean): string {
  return passed ? "PASS" : "FAIL";
}

function scenarioSection(result: ScenarioResult): string {
  const lines: string[] = [
    `## Scenario: ${result.scenario.name}  ${STATUS_ICON[result.verdict.status]}`,
    `**Reasoning:** ${result.verdict.reasoning}`,
    "**Evidence:**",
  ];

  for (const item of result.verdict.evidence) {
    lines.push(`- ${excerpt(item)}`);
  }

  if (result.turns.length > 0) {
    lines.push("**Turns:**");

    for (const turn of result.turns) {
      lines.push(
        `- ${turn.turnId}${turn.sessionId === undefined ? "" : ` (${turn.sessionId})`}: ${excerpt(turn.response)}`,
      );
    }
  }

  if (result.traceAssertions.length > 0) {
    lines.push("**Trace Assertions:**");

    for (const assertion of result.traceAssertions) {
      lines.push(
        `- ${formatAssertion(assertion.passed)}: ${assertion.description} -- ${excerpt(assertion.evidence)}`,
      );
    }
  }

  if (result.error !== undefined) {
    lines.push(`**Runner Error:** ${excerpt(result.error)}`);
  }

  lines.push(
    `**Cost:** ${result.cost.borgTurns} Borg turns, ${result.cost.assessorLlmCalls} assessor LLM calls, ~${result.cost.approximateTokens} tokens`,
  );

  return lines.join("\n");
}

function collectCoverage(results: readonly ScenarioResult[]): Map<TracePhase, boolean> {
  const coverage = new Map<TracePhase, boolean>();

  for (const phase of TRACE_PHASES) {
    coverage.set(phase, false);
  }

  for (const result of results) {
    for (const phase of result.coveredPhases) {
      coverage.set(phase, true);
    }

    for (const assertion of result.traceAssertions) {
      if (/autonomy/i.test(assertion.description) && assertion.passed) {
        coverage.set("executive_focus", true);
      }
    }
  }

  return coverage;
}

function coverageSection(results: readonly ScenarioResult[]): string {
  const coverage = collectCoverage(results);
  const labelByPhase: Record<TracePhase, string> = {
    perception: "perception",
    executive_focus: "executive_focus",
    retrieval: "retrieval",
    deliberation: "deliberation S1/S2",
    action: "action/tool use",
    reflection: "reflection",
    ingestion: "ingestion",
    other: "other",
  };
  const lines = ["## Coverage", "Subsystems exercised across all scenarios:"];

  for (const phase of TRACE_PHASES) {
    lines.push(`- ${labelByPhase[phase]}: ${coverage.get(phase) === true ? "✅" : "❌"}`);
  }

  const commitmentGuard = results.some((result) =>
    result.traceAssertions.some((assertion) =>
      /commitment/i.test(`${assertion.description} ${assertion.evidence}`),
    ),
  );
  const autonomy = results.some((result) =>
    result.traceAssertions.some((assertion) =>
      /autonomy/i.test(`${assertion.description} ${assertion.evidence}`),
    ),
  );

  lines.push(`- commitment_guard: ${commitmentGuard ? "✅" : "❌"}`);
  lines.push(`- autonomy: ${autonomy ? "⚠️" : "❌"}`);

  return lines.join("\n");
}

export function formatAssessorReport(report: AssessorRunReport): string {
  const lines: string[] = [
    `# Borg Assessor Run ${report.runId}`,
    `Started: ${report.startedAt}   Duration: ${formatDuration(report.durationMs)}   Total cost: ~${totalTokens(report.results)} tokens`,
    "",
    "## Summary",
    `- Scenarios: ${report.results.length}`,
    `- Pass: ${countByStatus(report.results, "pass")}`,
    `- Fail: ${countByStatus(report.results, "fail")}`,
    `- Inconclusive: ${countByStatus(report.results, "inconclusive")}`,
    "",
  ];

  for (const result of report.results) {
    lines.push(scenarioSection(result), "");
  }

  lines.push(coverageSection(report.results), "");

  return lines.join("\n");
}

export function hasAssessorRunFailure(report: AssessorRunReport): boolean {
  return report.results.some(
    (result) =>
      result.verdict.status !== "pass" ||
      result.error !== undefined ||
      result.traceAssertions.some((assertion) => !assertion.passed),
  );
}
