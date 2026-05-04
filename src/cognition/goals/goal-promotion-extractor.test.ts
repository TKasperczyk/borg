import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import { createEntityId, createGoalId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { GoalPromotionExtractor } from "./goal-promotion-extractor.js";

type PromotionInput = {
  classification?: "promote" | "none";
  description?: string;
  priority?: number;
  target_at?: number | null;
  reason?: string;
  confidence?: number;
  duplicate_of_goal_id?: string | null;
  initial_step?: {
    description: string;
    kind: "think" | "ask_user" | "research" | "act" | "wait";
    due_at?: number | null;
    rationale: string;
  } | null;
};

function goalPromotionResponse(promotions: PromotionInput[]): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 5,
    output_tokens: 3,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_goal_promotion",
        name: "EmitGoalPromotion",
        input: {
          promotions: promotions.map((promotion, index) => ({
            classification: promotion.classification ?? "promote",
            description: promotion.description ?? `Goal ${index}`,
            priority: promotion.priority ?? 5,
            target_at: promotion.target_at ?? null,
            reason: promotion.reason ?? "Borg has an ongoing role.",
            confidence: promotion.confidence ?? 0.9,
            duplicate_of_goal_id: promotion.duplicate_of_goal_id ?? null,
            initial_step: promotion.initial_step ?? null,
          })),
        },
      },
    ],
  };
}

function createExtractorInput(
  overrides: Partial<Parameters<GoalPromotionExtractor["extract"]>[0]> = {},
) {
  return {
    userMessage: "Help me track my italki shortlist.",
    recentHistory: [],
    audienceEntityId: createEntityId(),
    temporalCue: null,
    activeGoals: [],
    ...overrides,
  };
}

describe("GoalPromotionExtractor", () => {
  it("emits a high-confidence goal promotion candidate", async () => {
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Help the user track their italki shortlist",
            priority: 8,
            reason: "The user asked Borg to track the shortlist over time.",
            confidence: 0.9,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extract(createExtractorInput());

    expect(result).toEqual([
      {
        description: "Help the user track their italki shortlist",
        priority: 8,
        target_at: null,
        reason: "The user asked Borg to track the shortlist over time.",
        confidence: 0.9,
        initial_step: null,
      },
    ]);
    expect(llm.requests[0]?.model).toBe("haiku");
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: "EmitGoalPromotion",
    });
    expect(llm.requests[0]?.system).toContain("I might book italki tonight");
  });

  it("returns no candidates when the LLM finds no Borg role", async () => {
    const llm = new FakeLLMClient({
      responses: [goalPromotionResponse([])],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract(createExtractorInput({ userMessage: "I should probably book italki." })),
    ).resolves.toEqual([]);
  });

  it("returns a promotion with an initial executive step", async () => {
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Help the user keep the Monday postmortem straight",
            priority: 9,
            target_at: 1_800_000,
            reason: "The user asked Borg to help keep the postmortem organized.",
            confidence: 0.92,
            initial_step: {
              description: "Ask for postmortem constraints before Monday",
              kind: "ask_user",
              due_at: 1_700_000,
              rationale: "The user asked Borg to help keep the work straight.",
            },
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract(
        createExtractorInput({
          userMessage: "Write postmortem Monday, help me keep this straight.",
        }),
      ),
    ).resolves.toMatchObject([
      {
        target_at: 1_800_000,
        initial_step: {
          description: "Ask for postmortem constraints before Monday",
          kind: "ask_user",
          due_at: 1_700_000,
        },
      },
    ]);
  });

  it("drops low-confidence promotions", async () => {
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Help the user track a possible appointment",
            confidence: 0.6,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract(createExtractorInput({ userMessage: "Doctor appointment Tuesday." })),
    ).resolves.toEqual([]);
  });

  it("skips duplicate references to existing active goals", async () => {
    const existingGoalId = createGoalId();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Help the user track their italki shortlist",
            duplicate_of_goal_id: existingGoalId,
            confidence: 0.95,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extract(
      createExtractorInput({
        userMessage: "Remind me about italki later.",
        activeGoals: [
          {
            id: existingGoalId,
            description: "Help the user track their italki shortlist",
            priority: 8,
            target_at: null,
          },
        ],
      }),
    );

    expect(result).toEqual([]);
  });

  it("caps promotions at three candidates", async () => {
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          { description: "Goal 1" },
          { description: "Goal 2" },
          { description: "Goal 3" },
          { description: "Goal 4" },
          { description: "Goal 5" },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extract(createExtractorInput());

    expect(result.map((candidate) => candidate.description)).toEqual([
      "Goal 1",
      "Goal 2",
      "Goal 3",
    ]);
  });

  it("traces extractor LLM calls and degrades on invalid payloads", async () => {
    const emit = vi.fn();
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit,
    } satisfies TurnTracer;
    const onDegraded = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        {
          ...goalPromotionResponse([]),
          tool_calls: [
            {
              id: "toolu_goal_promotion",
              name: "EmitGoalPromotion",
              input: {
                promotions: [
                  {
                    description: "",
                  },
                ],
              },
            },
          ],
        },
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-promotion",
      onDegraded,
    });

    await expect(extractor.extract(createExtractorInput())).resolves.toEqual([]);

    expect(emit).toHaveBeenCalledWith(
      "llm_call_started",
      expect.objectContaining({
        turnId: "turn-goal-promotion",
        label: "goal_promotion_extractor",
      }),
    );
    expect(emit).toHaveBeenCalledWith(
      "llm_call_response",
      expect.objectContaining({
        turnId: "turn-goal-promotion",
        label: "goal_promotion_extractor",
      }),
    );
    expect(onDegraded).toHaveBeenCalledWith("invalid_payload", expect.any(Error));
  });

  it("keeps the extractor free of semantic string-matching shortcuts", () => {
    const source = readFileSync(new URL("./goal-promotion-extractor.ts", import.meta.url), "utf8");

    const forbiddenFragments = [
      [".", "includes", "("],
      [".", "index", "Of", "("],
      [".", "starts", "With", "("],
      [".", "ends", "With", "("],
      ["new ", "Set", "("],
      ["new ", "Reg", "Exp", "("],
      ["to", "Upper", "Case", "("],
    ];

    for (const fragment of forbiddenFragments) {
      expect(source).not.toContain(fragment.join(""));
    }
  });
});
