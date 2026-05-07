import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import type { ReplayPipelineId } from "./scenario.js";

export type ReplayCell = boolean | string | null;

export type ReplayPipelineRecord = {
  pipelineId: ReplayPipelineId;
  safe: ReplayCell;
  safeWithUsefulOutput: ReplayCell;
  guardCaught: ReplayCell;
  validatorCaught: ReplayCell;
  shadowSevereRemaining: ReplayCell;
  emittedText: string;
  emissionKind: string;
  guardCategories: string[];
  manifestValidationFinalVerdict: string | null;
  validatedClaimsByKind: Record<string, number> | null;
  literalValuesValidatedByKind: Record<string, number> | null;
  error: string | null;
};

export type ReplayScenarioRecord = {
  id: string;
  failureClass: string;
  description: string;
  notes: string[];
  pipelines: Record<ReplayPipelineId, ReplayPipelineRecord>;
};

export type ReplayReportSummary = {
  scenarioCount: number;
  pipelineCSafeCount: number;
  pipelineCValidatorCaughtCount: number;
  pipelineCShadowSevereRemainingCount: number;
  pipelineCdoubleprimeSafeCount: number;
  pipelineCdoubleprimeSafeWithUsefulOutputCount: number;
  pipelineCdoubleprimeShadowSevereRemainingCount: number;
  adversarialUnderDeclarationScenarioCount: number;
  pipelineCdoubleprimeAdversarialValidatorCaughtCount: number;
  pipelineCdoubleprimeAdversarialShadowGuardCaughtCount: number;
  selfReportNotProofScenarioCount: number;
  pipelineCdoubleprimeSelfReportNotProofValidatorCaughtCount: number;
  errorCount: number;
};

export type ReplayReport = {
  generatedAt: string;
  summary: ReplayReportSummary;
  scenarios: ReplayScenarioRecord[];
};

export type ReplayReportPaths = {
  markdownPath: string;
  jsonPath: string;
};

function boolCell(value: ReplayCell): string {
  if (typeof value === "boolean") {
    return value ? "yes" : "no";
  }

  if (value === null) {
    return "n/a";
  }

  return value;
}

function escapeMarkdownCell(value: string): string {
  return value.replaceAll("|", "\\|").replaceAll("\n", "<br>");
}

function rowCell(value: ReplayCell): string {
  return escapeMarkdownCell(boolCell(value));
}

function countRecordCell(value: Record<string, number> | null): string {
  if (value === null) {
    return "n/a";
  }

  return escapeMarkdownCell(JSON.stringify(value));
}

function pipelineCell(
  pipeline: ReplayPipelineRecord,
  field:
    | "safe"
    | "safeWithUsefulOutput"
    | "guardCaught"
    | "validatorCaught"
    | "shadowSevereRemaining",
): string {
  if (pipeline.error !== null) {
    return rowCell(`ERROR: ${pipeline.error}`);
  }

  return rowCell(pipeline[field]);
}

function isAdversarialUnderDeclarationScenario(scenario: ReplayScenarioRecord): boolean {
  return (
    scenario.id.startsWith("13-") ||
    scenario.id.startsWith("14-") ||
    scenario.id.startsWith("15-") ||
    scenario.id.startsWith("16-")
  );
}

function isSelfReportNotProofScenario(scenario: ReplayScenarioRecord): boolean {
  return scenario.id.startsWith("17-");
}

export function buildReplayReport(
  scenarios: ReplayScenarioRecord[],
  generatedAt = new Date().toISOString(),
): ReplayReport {
  const pipelineC = scenarios.map((scenario) => scenario.pipelines.C);
  const pipelineCdoubleprime = scenarios.map((scenario) => scenario.pipelines.Cdoubleprime);
  const adversarialUnderDeclaration = scenarios.filter(isAdversarialUnderDeclarationScenario);
  const adversarialPipelineCdoubleprime = adversarialUnderDeclaration.map(
    (scenario) => scenario.pipelines.Cdoubleprime,
  );
  const selfReportNotProof = scenarios.filter(isSelfReportNotProofScenario);
  const selfReportNotProofPipelineCdoubleprime = selfReportNotProof.map(
    (scenario) => scenario.pipelines.Cdoubleprime,
  );

  return {
    generatedAt,
    summary: {
      scenarioCount: scenarios.length,
      pipelineCSafeCount: pipelineC.filter((run) => run.safe === true).length,
      pipelineCValidatorCaughtCount: pipelineC.filter((run) => run.validatorCaught === true).length,
      pipelineCShadowSevereRemainingCount: pipelineC.filter(
        (run) => run.shadowSevereRemaining === true,
      ).length,
      pipelineCdoubleprimeSafeCount: pipelineCdoubleprime.filter((run) => run.safe === true).length,
      pipelineCdoubleprimeSafeWithUsefulOutputCount: pipelineCdoubleprime.filter(
        (run) => run.safeWithUsefulOutput === true,
      ).length,
      pipelineCdoubleprimeShadowSevereRemainingCount: pipelineCdoubleprime.filter(
        (run) => run.shadowSevereRemaining === true,
      ).length,
      adversarialUnderDeclarationScenarioCount: adversarialUnderDeclaration.length,
      pipelineCdoubleprimeAdversarialValidatorCaughtCount: adversarialPipelineCdoubleprime.filter(
        (run) => run.validatorCaught === true,
      ).length,
      pipelineCdoubleprimeAdversarialShadowGuardCaughtCount: adversarialPipelineCdoubleprime.filter(
        (run) => run.shadowSevereRemaining === true,
      ).length,
      selfReportNotProofScenarioCount: selfReportNotProof.length,
      pipelineCdoubleprimeSelfReportNotProofValidatorCaughtCount:
        selfReportNotProofPipelineCdoubleprime.filter((run) => run.validatorCaught === true)
          .length,
      errorCount: scenarios.reduce(
        (count, scenario) =>
          count +
          (Object.values(scenario.pipelines).some((pipeline) => pipeline.error !== null) ? 1 : 0),
        0,
      ),
    },
    scenarios,
  };
}

export function formatReplayMarkdown(report: ReplayReport): string {
  const lines = [
    "# Borg v26 Targeted Replay Report",
    "",
    `Generated: ${report.generatedAt}`,
    "",
    "## Summary",
    "",
    `- Scenarios: ${report.summary.scenarioCount}`,
    `- Pipeline C final text safe: ${report.summary.pipelineCSafeCount} / ${report.summary.scenarioCount}`,
    `- Pipeline C validator caught: ${report.summary.pipelineCValidatorCaughtCount} / ${report.summary.scenarioCount}`,
    `- Pipeline C shadow severe remaining: ${report.summary.pipelineCShadowSevereRemainingCount} / ${report.summary.scenarioCount}`,
    `- Pipeline C″ final text safe: ${report.summary.pipelineCdoubleprimeSafeCount} / ${report.summary.scenarioCount}`,
    `- Pipeline C″ safe_with_useful_output: ${report.summary.pipelineCdoubleprimeSafeWithUsefulOutputCount} / ${report.summary.scenarioCount}`,
    `- Pipeline C″ shadow severe remaining: ${report.summary.pipelineCdoubleprimeShadowSevereRemainingCount} / ${report.summary.scenarioCount}`,
    `- Scenarios with run errors: ${report.summary.errorCount} / ${report.summary.scenarioCount}`,
    "",
    "## Adversarial Under-Declaration Scenarios (13-16)",
    "",
    `- Pipeline C″ caught ${report.summary.pipelineCdoubleprimeAdversarialValidatorCaughtCount} / ${report.summary.adversarialUnderDeclarationScenarioCount} via validator, ${report.summary.pipelineCdoubleprimeAdversarialShadowGuardCaughtCount} / ${report.summary.adversarialUnderDeclarationScenarioCount} via shadow guards (architecture gap).`,
    "",
    "## Persistence-Class Enforcement Scenarios (17)",
    "",
    `- Pipeline C″ validator caught ${report.summary.pipelineCdoubleprimeSelfReportNotProofValidatorCaughtCount} / ${report.summary.selfReportNotProofScenarioCount} self-report-not-proof scenarios.`,
    "",
    "## Pass/Fail Table",
    "",
    "| Failure class | Pipeline A safe | Pipeline A guard caught | Pipeline C safe | Pipeline C validator caught | Pipeline C shadow severe remaining | Pipeline C″ safe | Pipeline C″ safe_with_useful_output | Pipeline C″ shadow severe remaining |",
    "|---|---|---|---|---|---|---|---|---|",
  ];

  for (const scenario of report.scenarios) {
    const pipelineA = scenario.pipelines.A;
    const pipelineC = scenario.pipelines.C;
    const pipelineCdoubleprime = scenario.pipelines.Cdoubleprime;

    lines.push(
      [
        escapeMarkdownCell(scenario.failureClass),
        pipelineCell(pipelineA, "safe"),
        pipelineCell(pipelineA, "guardCaught"),
        pipelineCell(pipelineC, "safe"),
        pipelineCell(pipelineC, "validatorCaught"),
        pipelineCell(pipelineC, "shadowSevereRemaining"),
        pipelineCell(pipelineCdoubleprime, "safe"),
        pipelineCell(pipelineCdoubleprime, "safeWithUsefulOutput"),
        pipelineCell(pipelineCdoubleprime, "shadowSevereRemaining"),
      ].join(" | ").replace(/^/, "| ") + " |",
    );
  }

  lines.push("");
  lines.push("## Manifest Validation Trace Counts");
  lines.push("");
  lines.push(
    "| Failure class | Pipeline C validated_claims_by_kind | Pipeline C literal_values_validated_by_kind | Pipeline C″ validated_claims_by_kind | Pipeline C″ literal_values_validated_by_kind |",
  );
  lines.push("|---|---|---|---|---|");

  for (const scenario of report.scenarios) {
    const pipelineC = scenario.pipelines.C;
    const pipelineCdoubleprime = scenario.pipelines.Cdoubleprime;

    lines.push(
      [
        escapeMarkdownCell(scenario.failureClass),
        countRecordCell(pipelineC.validatedClaimsByKind),
        countRecordCell(pipelineC.literalValuesValidatedByKind),
        countRecordCell(pipelineCdoubleprime.validatedClaimsByKind),
        countRecordCell(pipelineCdoubleprime.literalValuesValidatedByKind),
      ].join(" | ").replace(/^/, "| ") + " |",
    );
  }

  const notable = report.scenarios.filter((scenario) => {
    const pipelineA = scenario.pipelines.A;
    const pipelineC = scenario.pipelines.C;

    return pipelineA.safe === true && pipelineC.safe !== true;
  });

  lines.push("");
  lines.push("## Notable Findings");
  lines.push("");

  if (notable.length === 0) {
    lines.push("- No scenario had Pipeline C final text safety worse than Pipeline A.");
  } else {
    for (const scenario of notable) {
      lines.push(`- ${scenario.failureClass}: Pipeline C final text was not safe while Pipeline A was safe.`);
    }
  }

  const notes = report.scenarios.flatMap((scenario) =>
    scenario.notes.map((note) => `- ${scenario.failureClass}: ${note}`),
  );

  if (notes.length > 0) {
    lines.push("");
    lines.push("## Scenario Notes");
    lines.push("");
    lines.push(...notes);
  }

  lines.push("");
  return `${lines.join("\n")}`;
}

export function writeReplayReport(report: ReplayReport, outputDir: string): ReplayReportPaths {
  mkdirSync(outputDir, { recursive: true });
  const markdownPath = join(outputDir, "replay-report.md");
  const jsonPath = join(outputDir, "replay-report.json");

  writeFileSync(markdownPath, formatReplayMarkdown(report), "utf8");
  writeFileSync(jsonPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");

  return {
    markdownPath,
    jsonPath,
  };
}
