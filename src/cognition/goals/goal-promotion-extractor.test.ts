import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { createEntityId, createGoalId, createSessionId } from "../../util/ids.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import {
  GOAL_PROMOTION_CLASSIFICATIONS,
  GoalPromotionExtractor,
  type GoalPromotionClassification,
} from "./goal-promotion-extractor.js";

type PromotionInput = {
  classification?: GoalPromotionClassification | string;
  omitClassification?: boolean;
  description?: string;
  terminal_condition?: string | null;
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

type GoalPromotionResponseOptions = {
  durableGoalBatch?: "single" | "explicit_multiple";
};

function promotionPayload(promotion: PromotionInput, index: number): Record<string, unknown> {
  const classification = promotion.classification ?? "durable_borg_goal";
  const payload: Record<string, unknown> = {
    description: promotion.description ?? `Goal ${index}`,
    terminal_condition:
      promotion.terminal_condition ??
      (classification === "durable_borg_goal"
        ? `Goal ${index} reaches its stated completion condition`
        : null),
    priority: promotion.priority ?? 5,
    target_at: promotion.target_at ?? null,
    reason: promotion.reason ?? "Borg has an ongoing role.",
    confidence: promotion.confidence ?? 0.9,
    duplicate_of_goal_id: promotion.duplicate_of_goal_id ?? null,
    initial_step: promotion.initial_step ?? null,
  };

  if (promotion.omitClassification !== true) {
    payload.classification = classification;
  }

  return payload;
}

function goalPromotionResponse(
  promotions: PromotionInput[],
  options: GoalPromotionResponseOptions = {},
): LLMCompleteResult {
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
          durable_goal_batch: options.durableGoalBatch ?? "single",
          promotions: promotions.map((promotion, index) => promotionPayload(promotion, index)),
        },
      },
    ],
  };
}

function createExtractorInput(
  overrides: Partial<Parameters<GoalPromotionExtractor["extract"]>[0]> = {},
) {
  return {
    userMessage: "Help me track the refactor across sessions.",
    recentHistory: [],
    audienceEntityId: createEntityId(),
    temporalCue: null,
    activeGoals: [],
    ...overrides,
  };
}

function tracingHarness() {
  const emit = vi.fn();
  const tracer = {
    enabled: true,
    includePayloads: false,
    emit,
  } satisfies TurnTracer;

  return { emit, tracer };
}

describe("GoalPromotionExtractor", () => {
  it("emits a high-confidence durable goal promotion candidate", async () => {
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Help the user track the refactor across sessions",
            priority: 8,
            reason: "The user asked Borg to track the refactor over time.",
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
        description: "Help the user track the refactor across sessions",
        terminal_condition: "Goal 0 reaches its stated completion condition",
        priority: 8,
        target_at: null,
        reason: "The user asked Borg to track the refactor over time.",
        confidence: 0.9,
        duplicate_of_goal_id: null,
        initial_step: null,
      },
    ]);
    expect(llm.requests[0]?.model).toBe("haiku");
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: "EmitGoalPromotion",
    });
    expect(llm.requests[0]?.max_tokens).toBe(EXTRACTOR_MAX_TOKENS_DEFAULT);
    expect(llm.requests[0]?.system).toContain("not_borg_responsibility");
    expect(llm.requests[0]?.system).toContain("terminal_condition");
    expect(llm.requests[0]?.system).toContain("structural completion condition");
    expect(llm.requests[0]?.system).toContain(SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE);
  });

  it("emits extractor completion traces with session scope", async () => {
    const sessionId = createSessionId();
    const { emit, tracer } = tracingHarness();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Track the session-scoped observability sweep",
            priority: 7,
            reason: "The user asked Borg to keep the sweep moving.",
            confidence: 0.91,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-session-trace",
      sessionId,
    });

    await expect(extractor.extract(createExtractorInput())).resolves.toHaveLength(1);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        turnId: "turn-goal-session-trace",
        session_id: sessionId,
        degraded: false,
      }),
    );
  });

  it("rejects impossible-for-Borg capability classifications distinctly", async () => {
    const { emit, tracer } = tracingHarness();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            classification: "impossible_for_borg_without_capability",
            description: "Borg will monitor p95 and send the incident note later",
            reason: "The candidate asks Borg to perform external monitoring and later messaging.",
            confidence: 0.96,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-impossible-capability",
    });

    await expect(
      extractor.extract(
        createExtractorInput({
          userMessage: "You monitor p95 and send the incident note later.",
        }),
      ),
    ).resolves.toEqual([]);

    expect(GOAL_PROMOTION_CLASSIFICATIONS).toEqual([
      "durable_borg_goal",
      "one_off",
      "not_borg_responsibility",
      "impossible_for_borg_without_capability",
      "already_represented",
      "none",
    ]);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.rejected",
      expect.objectContaining({
        turnId: "turn-goal-impossible-capability",
        classification: "impossible_for_borg_without_capability",
        reason: "non_durable_classification",
      }),
    );
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        classification_counts: expect.objectContaining({
          impossible_for_borg_without_capability: 1,
        }),
        rejected_by_classification: expect.objectContaining({
          impossible_for_borg_without_capability: 1,
        }),
      }),
    );
    expect(llm.requests[0]?.system).toContain("monitoring p95");
    expect(llm.requests[0]?.system).toContain("scheduled document edits");
    expect(llm.requests[0]?.system).toContain(
      "Durable goals are about Borg's durable conversation/memory responsibility",
    );
    expect(llm.requests[0]?.system).toContain(
      'A user saying "my goal is to..." is usually participant-side context',
    );
    expect(llm.requests[0]?.system).toContain('my goal is to deploy", "friend will respond"');
    expect(llm.requests[0]?.system).not.toContain("treat that speaker as the goal owner");
    expect(llm.requests[0]?.system).toContain("user will deploy -> not_borg_responsibility");
    expect(llm.requests[0]?.system).toContain(
      "Borg will monitor p95 -> impossible_for_borg_without_capability",
    );
    expect(llm.requests[0]?.system).toContain("My host capability boundary");
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
      extractor.extract(
        createExtractorInput({ userMessage: "My friend will respond when she can." }),
      ),
    ).resolves.toEqual([]);
  });

  it("returns a durable goal promotion with an initial executive step", async () => {
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Help the user keep the Monday planning review organized",
            priority: 9,
            target_at: 1_800_000,
            reason: "The user asked Borg to keep the planning review organized.",
            confidence: 0.92,
            initial_step: {
              description: "Ask for review constraints before Monday",
              kind: "ask_user",
              due_at: 1_700_000,
              rationale: "Borg needs the constraints to track the review well.",
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
          userMessage: "Monday's planning review matters; help me keep it organized.",
        }),
      ),
    ).resolves.toMatchObject([
      {
        target_at: 1_800_000,
        initial_step: {
          description: "Ask for review constraints before Monday",
          kind: "ask_user",
          due_at: 1_700_000,
        },
      },
    ]);
  });

  it.each(
    GOAL_PROMOTION_CLASSIFICATIONS.filter(
      (classification) => classification !== "durable_borg_goal",
    ),
  )("rejects non-durable taxonomy classification %s", async (classification) => {
    const { emit, tracer } = tracingHarness();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            classification,
            description: `Candidate classified as ${classification}`,
            reason: `The current turn belongs in ${classification}.`,
            confidence: 0.94,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-taxonomy-reject",
    });

    await expect(extractor.extract(createExtractorInput())).resolves.toEqual([]);

    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.rejected",
      expect.objectContaining({
        turnId: "turn-goal-taxonomy-reject",
        classification,
        reason: "non_durable_classification",
      }),
    );
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        turnId: "turn-goal-taxonomy-reject",
        candidates_emitted: 0,
        classification_counts: expect.objectContaining({
          [classification]: 1,
        }),
        rejected_by_classification: expect.objectContaining({
          [classification]: 1,
        }),
      }),
    );
  });

  it("rejects low-confidence durable goals separately from taxonomy rejection", async () => {
    const { emit, tracer } = tracingHarness();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Help the user keep supporting their sibling through a job search",
            confidence: 0.6,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-low-confidence",
    });

    await expect(
      extractor.extract(
        createExtractorInput({
          userMessage: "Maybe keep an eye on how I support my sibling's job search.",
        }),
      ),
    ).resolves.toEqual([]);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.rejected",
      expect.objectContaining({
        turnId: "turn-goal-low-confidence",
        classification: "durable_borg_goal",
        reason: "low_confidence",
      }),
    );
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        rejected_low_confidence: 1,
        rejected_by_classification: expect.objectContaining({
          durable_borg_goal: 0,
        }),
      }),
    );
  });

  it("preserves duplicate references for persistence-time dedup", async () => {
    const existingGoalId = createGoalId();
    const owner = createEntityId();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Help the user track their release checklist",
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
        userMessage: "Keep tracking the release checklist.",
        activeGoals: [
          {
            id: existingGoalId,
            description: "Help the user track their release checklist",
            terminal_condition: "The release checklist is settled",
            priority: 8,
            target_at: null,
            owner_entity_id: owner,
          },
        ],
      }),
    );

    expect(result).toEqual([
      expect.objectContaining({
        description: "Help the user track their release checklist",
        duplicate_of_goal_id: existingGoalId,
      }),
    ]);
    const payload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      active_goals?: Array<{
        owner_entity_id?: string | null;
        terminal_condition?: string | null;
        disclosure?: string;
        disclosure_label?: { disclosure_class?: string; private_to_entity_ids?: string[] };
      }>;
    };
    const activeGoal = payload.active_goals?.[0];

    expect(activeGoal?.owner_entity_id).toBe(owner);
    expect(activeGoal?.terminal_condition).toBe("The release checklist is settled");
    expect(activeGoal?.disclosure).toContain("disclosure_class=relationship_private");
    expect(activeGoal?.disclosure).toContain(`private-to=${owner}`);
    expect(activeGoal?.disclosure_label).toMatchObject({
      disclosure_class: "relationship_private",
      private_to_entity_ids: [owner],
    });
  });

  it("caps single-mode durable promotions to the highest-confidence candidate", async () => {
    const { emit, tracer } = tracingHarness();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          { description: "Track the relationship check-in rhythm", confidence: 0.86 },
          { description: "Track the API migration", confidence: 0.97 },
          { description: "Track the planning review follow-up", confidence: 0.91 },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-single-cap",
    });

    const result = await extractor.extract(createExtractorInput());

    expect(result.map((candidate) => candidate.description)).toEqual(["Track the API migration"]);
    expect(
      emit.mock.calls.filter(
        ([event, data]) =>
          event === "extraction.goals.rejected" &&
          typeof data === "object" &&
          data !== null &&
          "reason" in data &&
          data.reason === "cap_exceeded",
      ),
    ).toHaveLength(2);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        candidates_emitted: 1,
        rejected_by_cap: 2,
      }),
    );
  });

  it("caps explicit-multiple durable promotions at three candidates", async () => {
    const { emit, tracer } = tracingHarness();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse(
          [
            { description: "Track the plan across sessions", confidence: 0.9 },
            { description: "Track the job-search support", confidence: 0.89 },
            { description: "Track the refactor cleanup", confidence: 0.88 },
            { description: "Track the API migration", confidence: 0.99 },
            { description: "Track the relationship check-in", confidence: 0.98 },
          ],
          { durableGoalBatch: "explicit_multiple" },
        ),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-explicit-multiple",
    });

    const result = await extractor.extract(createExtractorInput());

    expect(result.map((candidate) => candidate.description)).toEqual([
      "Track the plan across sessions",
      "Track the job-search support",
      "Track the refactor cleanup",
    ]);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        candidates_emitted: 3,
        rejected_by_cap: 2,
      }),
    );
  });

  it("salvages valid promotions when another promotion fails item validation", async () => {
    const emit = vi.fn();
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit,
    } satisfies TurnTracer;
    const onDegraded = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Help the user maintain the onboarding checklist",
            priority: 6,
            reason: "The user asked Borg to keep the onboarding work organized.",
            confidence: 0.91,
          },
          {
            description: "Help the user track a duplicate learning objective",
            duplicate_of_goal_id: "not-a-goal-id",
            reason: "This promotion has an invalid duplicate reference.",
            confidence: 0.94,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-salvage",
      onDegraded,
    });

    const result = await extractor.extract(
      createExtractorInput({
        userMessage: "Keep the onboarding checklist moving this week.",
      }),
    );

    expect(result).toHaveLength(1);
    expect(result[0]?.description).toBe("Help the user maintain the onboarding checklist");
    expect(onDegraded).not.toHaveBeenCalled();
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        turnId: "turn-goal-salvage",
        candidates_emitted: 1,
        valid_promotion_count: 1,
        skipped_promotion_count: 1,
        salvaged_promotion_count: 1,
        skipped_promotions: [
          {
            candidate_index: 1,
            reason: "invalid_duplicate_of_goal_id",
          },
        ],
      }),
    );
  });

  it("counts invalid enum classifications as item-level invalid classifications", async () => {
    const { emit, tracer } = tracingHarness();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            classification: "legacy_goal",
            description: "Track the deprecated classification fixture",
            confidence: 0.95,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-invalid-enum",
    });

    await expect(extractor.extract(createExtractorInput())).resolves.toEqual([]);
    expect(emit).not.toHaveBeenCalledWith("extraction.goals.rejected", expect.anything());
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        candidates_emitted: 0,
        skipped_promotion_count: 1,
        skipped_promotions: [
          {
            candidate_index: 0,
            reason: "invalid_classification",
          },
        ],
        classification_counts: expect.objectContaining({
          invalid_classification: 1,
        }),
        rejected_invalid_enum: 1,
      }),
    );
  });

  it("requires classification and salvages missing-classification items as invalid classifications", async () => {
    const { emit, tracer } = tracingHarness();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            omitClassification: true,
            description: "Track the missing classification fixture",
            confidence: 0.95,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-goal-missing-classification",
    });

    await expect(extractor.extract(createExtractorInput())).resolves.toEqual([]);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        skipped_promotions: [
          {
            candidate_index: 0,
            reason: "invalid_classification",
          },
        ],
        classification_counts: expect.objectContaining({
          invalid_classification: 1,
        }),
      }),
    );
  });

  it("traces extractor LLM calls and degrades on invalid payloads", async () => {
    const emit = vi.fn();
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit,
    } satisfies TurnTracer;
    const onDegraded = vi.fn();
    const invalidResponse = {
      ...goalPromotionResponse([]),
      tool_calls: [
        {
          id: "toolu_goal_promotion",
          name: "EmitGoalPromotion",
          input: {},
        },
      ],
    };
    const llm = new FakeLLMClient({
      responses: [invalidResponse, invalidResponse],
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
      "llm_call.started",
      expect.objectContaining({
        turnId: "turn-goal-promotion",
        label: "goal_promotion_extractor",
      }),
    );
    expect(emit).toHaveBeenCalledWith(
      "llm_call.completed",
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
