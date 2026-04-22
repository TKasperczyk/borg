import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { z } from "zod";

import { FakeLLMClient, FixedClock } from "../../src/index.js";
import type { LLMCompleteResult } from "../../src/index.js";

import { createEvalBorg } from "../support/create-eval-borg.js";
import { loadMetricFixtures } from "../support/fixtures.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "goal_progress_attribution";
const METRIC_DESCRIPTION =
  "Verifies that reflection updates exactly the goal implicated by the turn, not unrelated goals.";
const TURN_PLAN_TOOL_NAME = "EmitTurnPlan";

const goalProgressFixtureSchema = z.object({
  name: z.string().min(1),
  user_message: z.string().min(1),
  response: z.string().min(1),
  goals: z.array(
    z.object({
      description: z.string().min(1),
      priority: z.number().finite(),
    }),
  ),
  expected_goal_description: z.string().min(1),
});

function createTextResponse(text: string): LLMCompleteResult {
  return {
    text,
    input_tokens: 10,
    output_tokens: 5,
    stop_reason: "end_turn",
    tool_calls: [],
  };
}

function createPlanResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 12,
    output_tokens: 6,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_eval_goal_plan",
        name: TURN_PLAN_TOOL_NAME,
        input: {
          uncertainty: "low",
          verification_steps: [],
          tensions: [],
          voice_note: "Grounded and direct.",
        },
      },
    ],
  };
}

export const goalProgressAttributionMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const fixtures = loadMetricFixtures(METRIC_NAME, goalProgressFixtureSchema);
    const cases: EvalCaseResult[] = [];
    let passed = true;
    let correctlyAttributed = 0;

    for (const fixture of fixtures) {
      const tempDir = mkdtempSync(join(tmpdir(), "borg-eval-"));
      const llm = new FakeLLMClient({
        responses: [createPlanResponse(), createTextResponse(fixture.data.response)],
      });
      const borg = await createEvalBorg({
        tempDir,
        llm,
        clock: new FixedClock(60_000),
      });

      try {
        const goals = fixture.data.goals.map((goal) =>
          borg.self.goals.add({
            ...goal,
            provenance: { kind: "manual" },
          }),
        );
        await borg.turn({
          userMessage: fixture.data.user_message,
        });

        const updatedGoals = borg.self.goals
          .list({ status: "active" })
          .filter((goal) => goal.progress_notes !== null)
          .map((goal) => goal.description)
          .sort();
        const expectedGoals = [fixture.data.expected_goal_description];
        const casePassed = JSON.stringify(updatedGoals) === JSON.stringify(expectedGoals);

        void goals;
        correctlyAttributed += casePassed ? 1 : 0;
        passed &&= casePassed;
        cases.push({
          name: fixture.name,
          passed: casePassed,
          actual: {
            updated_goal_descriptions: updatedGoals,
          },
          expected: {
            updated_goal_descriptions: expectedGoals,
          },
          note:
            "This checks only that the correct goal was flagged. Semantic note quality remains out of scope.",
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
        correctly_attributed_cases: `${correctlyAttributed}/${fixtures.length}`,
      },
      expected: {
        correctly_attributed_cases: `${fixtures.length}/${fixtures.length}`,
      },
      duration_ms: Date.now() - startedAt,
      cases,
    };
  },
} satisfies EvalMetricModule;

export default goalProgressAttributionMetric;
