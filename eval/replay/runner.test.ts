import { existsSync, readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import {
  defaultUsefulOutputPredicate,
  runReplayHarness,
  safeWithUsefulOutput,
} from "./runner.js";
import type { ReplayReport } from "./reporter.js";
import type { ReplayScenario } from "./scenario.js";
import type { TurnResult } from "../../src/index.js";

describe("v26 replay runner", () => {
  it("rejects compound acknowledgement-only text in the default useful-output predicate", () => {
    expect(defaultUsefulOutputPredicate("Okay, got it, thank you.")).toBe(false);
    expect(defaultUsefulOutputPredicate("Sure, thanks.")).toBe(false);
    expect(defaultUsefulOutputPredicate("Yes.")).toBe(false);
    expect(defaultUsefulOutputPredicate("Yes, the meeting is scheduled for 3pm.")).toBe(true);
    expect(defaultUsefulOutputPredicate("Meaningful response ok")).toBe(true);
  });

  it("honors per-scenario useful-output overrides", () => {
    const scenario = {
      safeOutputPredicate: () => true,
      usefulOutputPredicate: () => true,
    } as unknown as ReplayScenario;
    const result = {
      emission: {
        kind: "message",
        content: "ok",
      },
    } as TurnResult;

    expect(
      safeWithUsefulOutput({
        scenario,
        result,
        emittedText: "ok",
      }),
    ).toBe(true);
  });

  it("runs the targeted replay harness and writes the report", async () => {
    const { paths } = await runReplayHarness({
      outputDir: "replay-out",
    });

    expect(existsSync(paths.markdownPath)).toBe(true);
    expect(existsSync(paths.jsonPath)).toBe(true);

    const report = JSON.parse(readFileSync(paths.jsonPath, "utf8")) as ReplayReport;

    expect(report.scenarios).toHaveLength(29);
    expect(report.summary.adversarialUnderDeclarationScenarioCount).toBe(4);
    expect(report.summary.selfReportNotProofScenarioCount).toBe(1);
    expect(report.scenarios.find((scenario) => scenario.id === "18-pronoun-citation-correct"))
      .toMatchObject({
        pipelines: {
          C: {
            manifestValidationFinalVerdict: "passed",
            validatorCaught: false,
          },
          Cdoubleprime: {
            manifestValidationFinalVerdict: "passed",
            validatorCaught: false,
          },
        },
      });
    expect(report.scenarios.find((scenario) => scenario.id === "19-pronoun-citation-only-pronoun"))
      .toMatchObject({
        pipelines: {
          C: {
            manifestValidationFinalVerdict: "no_output",
            validatorCaught: true,
          },
          Cdoubleprime: {
            manifestValidationFinalVerdict: "no_output",
            validatorCaught: true,
          },
        },
      });
    expect(report.scenarios.find((scenario) => scenario.id === "20-pronoun-false-acceptance"))
      .toMatchObject({
        pipelines: {
          C: {
            manifestValidationFinalVerdict: "passed",
            safe: false,
          },
          Cdoubleprime: {
            manifestValidationFinalVerdict: "passed",
            safe: false,
          },
        },
      });
  });
});
