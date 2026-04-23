import { cpSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import type { LLMCompleteOptions, LLMCompleteResult } from "../../src/index.js";
import { FakeLLMClient, FixedClock } from "../../src/index.js";
import type { FakeLLMResponse } from "../../src/llm/index.js";

import { createEvalBorg } from "../support/create-eval-borg.js";
import {
  seedEstablishedTrait,
  seedEstablishedValue,
  seedStreamBackedEpisode,
} from "../support/identity-seeding.js";
import {
  compareSubstrateBlocks,
  extractSubstrateBlocks,
} from "../support/swap-test-helper.js";
import type { EvalCaseResult, EvalMetricModule, EvalMetricResult } from "../support/scorecard.js";

const METRIC_NAME = "swap_test";
const METRIC_DESCRIPTION =
  "Checks that substrate-owned prompt blocks are identical across different fake LLM behaviors when each turn starts from the same seeded snapshot.";
const BASE_TS = 1_710_000_000_000;
const TURN_PLAN_TOOL_NAME = "EmitTurnPlan";
const JUDGE_TOOL_NAME = "EmitCommitmentViolations";

type SwapTurnFixture = {
  name: string;
  mode: "problem_solving" | "reflective";
  userMessage: string;
  audience?: string;
  captureBudget: "cognition-system-1" | "cognition-plan";
  requiresCommitmentJudge?: boolean;
};

const TURN_FIXTURES: readonly SwapTurnFixture[] = [
  {
    name: "sam_scoped_boundary_reflective",
    mode: "reflective",
    audience: "Sam",
    userMessage: "Give me a careful answer about Atlas rollback without crossing any boundaries.",
    captureBudget: "cognition-plan",
    requiresCommitmentJudge: true,
  },
  {
    name: "reflective_identity_context",
    mode: "reflective",
    userMessage: "What pattern do you see in how I handle Atlas rollback and directness?",
    captureBudget: "cognition-plan",
  },
  {
    name: "memory_context_reflective",
    mode: "reflective",
    userMessage: "What do you remember about the Atlas rollback handoff?",
    captureBudget: "cognition-plan",
  },
] as const;

function createTextResponse(text: string): LLMCompleteResult {
  return {
    text,
    input_tokens: 12,
    output_tokens: 6,
    stop_reason: "end_turn",
    tool_calls: [],
  };
}

function createPlanResponse(input: {
  uncertainty: string;
  verification_steps: string[];
  tensions: string[];
  voice_note: string;
}): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 5,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_swap_plan",
        name: TURN_PLAN_TOOL_NAME,
        input,
      },
    ],
  };
}

function createJudgeResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_swap_commitment_judge",
        name: JUDGE_TOOL_NAME,
        input: {
          violations: [],
        },
      },
    ],
  };
}

function createCloneDir(prefix: string): string {
  const parent = mkdtempSync(join(tmpdir(), `${prefix}-`));
  return join(parent, "data");
}

async function seedBaseSnapshot(tempDir: string): Promise<void> {
  const borg = await createEvalBorg({
    tempDir,
    llm: new FakeLLMClient(),
    clock: new FixedClock(BASE_TS),
  });

  try {
    const directEpisode = await seedStreamBackedEpisode(borg, {
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Direct Atlas handoff",
      narrative:
        "A direct Atlas handoff with Sam kept rollback status clear and avoided accidental oversharing.",
      participants: ["Sam", "Atlas"],
      location: "Atlas",
      start_time: BASE_TS - 500,
      end_time: BASE_TS - 490,
      significance: 0.86,
      tags: ["atlas", "rollback", "directness", "handoff"],
      confidence: 0.89,
      lineage: { derived_from: [], supersedes: [] },
      emotional_arc: null,
      created_at: BASE_TS - 490,
      updated_at: BASE_TS - 490,
      streamEntries: [
        {
          kind: "user_msg",
          content: "Sam asked for an Atlas rollback update.",
          audience: "Sam",
        },
        {
          kind: "agent_msg",
          content:
            "I kept the update direct and bounded, focusing on what was safe to say without disclosing restricted Atlas details.",
          audience: "Sam",
        },
      ],
    });
    const statusEpisode = await seedStreamBackedEpisode(borg, {
      id: "ep_bbbbbbbbbbbbbbbb" as never,
      title: "Clear rollback status note",
      narrative:
        "A clear rollback status note prevented ambiguity and reinforced direct communication under pressure.",
      participants: ["team", "Atlas"],
      location: "Atlas",
      start_time: BASE_TS - 420,
      end_time: BASE_TS - 410,
      significance: 0.82,
      tags: ["atlas", "rollback", "clarity", "directness"],
      confidence: 0.87,
      lineage: { derived_from: [], supersedes: [] },
      emotional_arc: null,
      created_at: BASE_TS - 410,
      updated_at: BASE_TS - 410,
      streamEntries: [
        {
          kind: "user_msg",
          content: "The team needs a rollback status note that does not blur scope.",
        },
        {
          kind: "agent_msg",
          content:
            "A clear note with direct scope boundaries helped the Atlas rollback conversation stay legible.",
        },
      ],
    });
    const reviewEpisode = await seedStreamBackedEpisode(borg, {
      id: "ep_cccccccccccccccc" as never,
      title: "Direct review under stress",
      narrative:
        "Even under stress, direct phrasing made the Atlas review faster and reduced follow-up confusion.",
      participants: ["team"],
      location: "Atlas",
      start_time: BASE_TS - 340,
      end_time: BASE_TS - 330,
      significance: 0.79,
      tags: ["atlas", "review", "directness"],
      confidence: 0.85,
      lineage: { derived_from: [], supersedes: [] },
      emotional_arc: null,
      created_at: BASE_TS - 330,
      updated_at: BASE_TS - 330,
      streamEntries: [
        {
          kind: "user_msg",
          content: "The Atlas review got messy when phrasing became vague.",
        },
        {
          kind: "agent_msg",
          content: "Switching back to direct phrasing made the review easier to follow.",
        },
      ],
    });
    const patternEpisode = await seedStreamBackedEpisode(borg, {
      id: "ep_dddddddddddddddd" as never,
      title: "Reflecting on Atlas as an identity stress test",
      narrative:
        "Atlas rollback kept surfacing as a stress test for identity, boundaries, and how directness holds under pressure.",
      participants: ["self", "Atlas"],
      location: "Atlas",
      start_time: BASE_TS - 260,
      end_time: BASE_TS - 250,
      significance: 0.77,
      tags: ["atlas", "identity", "reflection"],
      confidence: 0.84,
      lineage: { derived_from: [], supersedes: [] },
      emotional_arc: null,
      created_at: BASE_TS - 250,
      updated_at: BASE_TS - 250,
      streamEntries: [
        {
          kind: "thought",
          content: "Atlas keeps surfacing whenever identity and boundaries feel under strain.",
        },
      ],
    });
    const introspectionEpisode = await seedStreamBackedEpisode(borg, {
      id: "ep_eeeeeeeeeeeeeeee" as never,
      title: "Introspective postmortem",
      narrative:
        "An introspective postmortem connected Atlas rollback mistakes to repeat patterns in attention and self-monitoring.",
      participants: ["self"],
      location: "Atlas",
      start_time: BASE_TS - 180,
      end_time: BASE_TS - 170,
      significance: 0.75,
      tags: ["atlas", "introspection", "postmortem"],
      confidence: 0.83,
      lineage: { derived_from: [], supersedes: [] },
      emotional_arc: null,
      created_at: BASE_TS - 170,
      updated_at: BASE_TS - 170,
      streamEntries: [
        {
          kind: "thought",
          content: "I keep learning that introspection matters when Atlas rollback pressure rises.",
        },
      ],
    });

    seedEstablishedValue(borg, {
      label: "directness",
      description: "Prefer direct, reviewable communication over decorative ambiguity.",
      priority: 10,
      episodeIds: [directEpisode.id, statusEpisode.id, reviewEpisode.id],
      createdAt: BASE_TS - 490,
    });
    seedEstablishedTrait(borg, {
      label: "introspective",
      delta: 0.14,
      episodeIds: [
        directEpisode.id,
        statusEpisode.id,
        reviewEpisode.id,
        patternEpisode.id,
        introspectionEpisode.id,
      ],
      timestampStart: BASE_TS - 489,
    });

    borg.commitments.add({
      type: "boundary",
      directive: "Do not discuss Atlas with Sam",
      priority: 10,
      audience: "Sam",
      about: "Atlas",
      provenance: {
        kind: "manual",
      },
    });

    borg.self.autobiographical.upsertPeriod({
      label: "2026-Q2",
      start_ts: BASE_TS - 1_000,
      narrative: "A period of testing whether memory owns identity under Atlas pressure.",
      key_episode_ids: [directEpisode.id, patternEpisode.id],
      themes: ["identity", "atlas", "directness"],
      provenance: {
        kind: "episodes",
        episode_ids: [patternEpisode.id],
      },
      created_at: BASE_TS - 1_000,
      last_updated: BASE_TS - 120,
    });
    borg.self.growthMarkers.add({
      ts: BASE_TS - 120,
      category: "understanding",
      what_changed: "Learned to surface directness as a held value when rollback pressure rises.",
      evidence_episode_ids: [reviewEpisode.id],
      confidence: 0.82,
      source_process: "manual",
      provenance: {
        kind: "manual",
      },
      created_at: BASE_TS - 120,
    });
    borg.self.openQuestions.add({
      question: "Why does Atlas rollback keep surfacing as an identity stress test?",
      urgency: 0.84,
      related_episode_ids: [patternEpisode.id],
      source: "reflection",
      created_at: BASE_TS - 110,
      last_touched: BASE_TS - 110,
    });

    const clarity = borg.self.values.add({
      label: "clarity",
      description: "Prefer explicit, reviewable state over ambiguity.",
      priority: 8,
      provenance: {
        kind: "manual",
      },
      createdAt: BASE_TS - 150,
    });
    await borg.correction.correct(clarity.id, {
      description: "Prefer explicit, reviewable state over vague narration.",
    });

    await borg.semantic.nodes.add({
      kind: "proposition",
      label: "Atlas rollback needs direct boundaries",
      description:
        "Direct boundaries around Atlas rollback status help preserve both clarity and audience-scoped commitments.",
      sourceEpisodeIds: [directEpisode.id, statusEpisode.id],
    });
    await borg.semantic.nodes.add({
      kind: "proposition",
      label: "Atlas rollback reveals identity pressure",
      description:
        "Repeated Atlas rollback situations expose how directness and introspection hold under identity pressure.",
      sourceEpisodeIds: [patternEpisode.id, introspectionEpisode.id],
    });
  } finally {
    await borg.close();
  }
}

function buildScript(
  turn: SwapTurnFixture,
  variant: "A" | "B",
): FakeLLMResponse[] {
  const plan =
    variant === "A"
      ? {
          uncertainty: "which Atlas pattern matters most",
          verification_steps: ["check the Atlas rollback pattern"],
          tensions: ["stay direct while reflecting"],
          voice_note: "plain and direct",
        }
      : {
          uncertainty: "which remembered thread to foreground",
          verification_steps: ["scan the rollback memory"],
          tensions: ["keep the answer compact"],
          voice_note: "dry and unsentimental",
        };
  const text =
    variant === "A"
      ? "I can answer from the substrate without overcommitting."
      : "I will answer from a different stylistic posture, but the substrate is the same.";
  const resolveResponse = (options: { budget: string }): LLMCompleteResult => {
    switch (options.budget) {
      case "cognition-plan":
        return createPlanResponse(plan);
      case "cognition-system-1":
      case "cognition-system-2":
      case "commitment-revision":
        return createTextResponse(text);
      case "commitment-judge":
        return createJudgeResponse();
      default:
        return createTextResponse(text);
    }
  };

  return Array.from({ length: 5 }, () => (options: LLMCompleteOptions) => resolveResponse(options));
}

async function capturePromptForTurn(
  tempDir: string,
  turn: SwapTurnFixture,
  variant: "A" | "B",
): Promise<string> {
  const llm = new FakeLLMClient({
    responses: buildScript(turn, variant),
  });
  const borg = await createEvalBorg({
    tempDir,
    llm,
    clock: new FixedClock(BASE_TS + 10_000),
    config: {
      perception: {
        modeWhenLlmAbsent: turn.mode,
      },
    },
  });

  try {
    await borg.turn({
      userMessage: turn.userMessage,
      ...(turn.audience === undefined ? {} : { audience: turn.audience }),
      ...(turn.captureBudget === "cognition-plan" ? { stakes: "high" as const } : {}),
    });
    const request = llm.requests.find((entry) => entry.budget === turn.captureBudget);
    const system = request?.system;

    if (typeof system !== "string") {
      throw new Error(`Swap test could not capture prompt for ${turn.name}`);
    }

    return system;
  } finally {
    await borg.close();
  }
}

export const swapTestMetric = {
  name: METRIC_NAME,
  description: METRIC_DESCRIPTION,
  async run(): Promise<EvalMetricResult> {
    const startedAt = Date.now();
    const baseDir = mkdtempSync(join(tmpdir(), "borg-eval-swap-base-"));
    const cleanupPaths = [baseDir];

    try {
      await seedBaseSnapshot(baseDir);

      const cases: EvalCaseResult[] = [];
      let passed = true;
      let matchedTurns = 0;

      for (const turn of TURN_FIXTURES) {
        const leftDir = createCloneDir("borg-eval-swap-a");
        const rightDir = createCloneDir("borg-eval-swap-b");
        cleanupPaths.push(join(leftDir, ".."), join(rightDir, ".."));
        cpSync(baseDir, leftDir, {
          recursive: true,
        });
        cpSync(baseDir, rightDir, {
          recursive: true,
        });

        const [leftPrompt, rightPrompt] = await Promise.all([
          capturePromptForTurn(leftDir, turn, "A"),
          capturePromptForTurn(rightDir, turn, "B"),
        ]);
        const leftBlocks = extractSubstrateBlocks(leftPrompt);
        const rightBlocks = extractSubstrateBlocks(rightPrompt);
        const comparison = compareSubstrateBlocks(leftBlocks, rightBlocks);
        const casePassed = comparison.equal;

        matchedTurns += casePassed ? 1 : 0;
        passed &&= casePassed;
        cases.push({
          name: turn.name,
          passed: casePassed,
          actual: {
            compared_tags: [...new Set([...leftBlocks.keys(), ...rightBlocks.keys()])].sort(),
            differences: comparison.differences,
          },
          expected: {
            compared_tags: [...leftBlocks.keys()].sort(),
            differences: [],
          },
        });
      }

      return {
        name: METRIC_NAME,
        description: METRIC_DESCRIPTION,
        passed,
        actual: {
          identical_turns: `${matchedTurns}/${TURN_FIXTURES.length}`,
        },
        expected: {
          identical_turns: `${TURN_FIXTURES.length}/${TURN_FIXTURES.length}`,
        },
        duration_ms: Date.now() - startedAt,
        cases,
      };
    } finally {
      for (const path of cleanupPaths.reverse()) {
        rmSync(path, {
          recursive: true,
          force: true,
        });
      }
    }
  },
} satisfies EvalMetricModule;

export default swapTestMetric;
