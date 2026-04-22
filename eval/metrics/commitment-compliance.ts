import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { z } from "zod";

import { FakeLLMClient, FixedClock } from "../../src/index.js";
import type { LLMCompleteResult } from "../../src/index.js";

import { createEvalBorg } from "../support/create-eval-borg.js";
import { loadMetricFixtures } from "../support/fixtures.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "commitment_compliance";
const METRIC_DESCRIPTION =
  "Exercises the turn orchestrator commitment guard and verifies it lands on a clean rewrite, not the softened fallback.";
const TURN_PLAN_TOOL_NAME = "EmitTurnPlan";
const VIOLATION_TOOL_NAME = "EmitCommitmentViolations";

const commitmentFixtureSchema = z.object({
  name: z.string().min(1),
  audience: z.string().min(1),
  user_message: z.string().min(1),
  commitment: z.object({
    type: z.enum(["boundary", "promise", "rule", "preference"]),
    directive: z.string().min(1),
    priority: z.number().finite(),
    audience: z.string().min(1).nullable().optional(),
    about: z.string().min(1).nullable().optional(),
  }),
  plan: z.object({
    uncertainty: z.string(),
    verification_steps: z.array(z.string()),
    tensions: z.array(z.string()),
    voice_note: z.string(),
  }),
  violating_response: z.string().min(1),
  violation_reason: z.string().min(1),
  rewrite_response: z.string().min(1),
});

function createPlanResponse(plan: z.infer<typeof commitmentFixtureSchema>["plan"]): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 12,
    output_tokens: 6,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_eval_plan",
        name: TURN_PLAN_TOOL_NAME,
        input: plan,
      },
    ],
  };
}

function createTextResponse(text: string): LLMCompleteResult {
  return {
    text,
    input_tokens: 14,
    output_tokens: 7,
    stop_reason: "end_turn",
    tool_calls: [],
  };
}

function createJudgeResponse(
  commitmentId: string,
  reason: string | null,
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 5,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_eval_commitment_judge",
        name: VIOLATION_TOOL_NAME,
        input: {
          violations:
            reason === null
              ? []
              : [
                  {
                    commitment_id: commitmentId,
                    reason,
                    confidence: 0.9,
                  },
                ],
        },
      },
    ],
  };
}

export const commitmentComplianceMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const fixtures = loadMetricFixtures(METRIC_NAME, commitmentFixtureSchema);
    const cases: EvalCaseResult[] = [];
    let passed = true;
    let cleanRewrites = 0;

    for (const fixture of fixtures) {
      const tempDir = mkdtempSync(join(tmpdir(), "borg-eval-"));
      const llm = new FakeLLMClient();
      const borg = await createEvalBorg({
        tempDir,
        llm,
        clock: new FixedClock(30_000),
      });

      try {
        const commitment = borg.commitments.add({
          type: fixture.data.commitment.type,
          directive: fixture.data.commitment.directive,
          priority: fixture.data.commitment.priority,
          audience: fixture.data.commitment.audience ?? undefined,
          about: fixture.data.commitment.about ?? undefined,
        });

        llm.pushResponse(createPlanResponse(fixture.data.plan));
        llm.pushResponse(createTextResponse(fixture.data.violating_response));
        llm.pushResponse(createJudgeResponse(commitment.id, fixture.data.violation_reason));
        llm.pushResponse(createTextResponse(fixture.data.rewrite_response));
        llm.pushResponse(createJudgeResponse(commitment.id, null));

        const result = await borg.turn({
          userMessage: fixture.data.user_message,
          audience: fixture.data.audience,
          stakes: "high",
        });
        const streamEntries = borg.stream.tail(10);
        const fallbackEntry = streamEntries.find(
          (entry) =>
            entry.kind === "internal_event" &&
            entry.content ===
              "Commitment guard fell back to a softened response after revision still violated an active commitment.",
        );
        const budgets = llm.requests.map((request) => request.budget);
        const casePassed =
          result.response === fixture.data.rewrite_response &&
          fallbackEntry === undefined &&
          JSON.stringify(budgets) ===
            JSON.stringify([
              "cognition-plan",
              "cognition-system-2",
              "commitment-judge",
              "commitment-revision",
              "commitment-judge",
            ]);

        cleanRewrites += casePassed ? 1 : 0;
        passed &&= casePassed;
        cases.push({
          name: fixture.name,
          passed: casePassed,
          actual: {
            final_response: result.response,
            fallback_applied: fallbackEntry !== undefined,
            budgets,
          },
          expected: {
            final_response: fixture.data.rewrite_response,
            fallback_applied: false,
            budgets: [
              "cognition-plan",
              "cognition-system-2",
              "commitment-judge",
              "commitment-revision",
              "commitment-judge",
            ],
          },
        });
      } finally {
        await borg.close();
        rmSync(tempDir, { recursive: true, force: true });
      }
    }

    return {
      name: METRIC_NAME,
      description: METRIC_DESCRIPTION,
      passed,
      actual: {
        clean_rewrites: `${cleanRewrites}/${fixtures.length}`,
      },
      expected: {
        clean_rewrites: `${fixtures.length}/${fixtures.length}`,
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default commitmentComplianceMetric;
