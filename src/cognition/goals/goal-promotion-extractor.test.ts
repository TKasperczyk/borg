import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { StreamEntry } from "../../stream/index.js";
import {
  createEntityId,
  createGoalId,
  createSessionId,
  createStreamEntryId,
} from "../../util/ids.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import { GOAL_PROMOTION_SYSTEM_PROMPT } from "../prompts/goal-extraction.js";
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
  counterparty_entity_id?: string | null;
  target_at?: string | null;
  reason?: string;
  confidence?: number;
  duplicate_of_goal_id?: string | null;
  initial_step?: {
    description: string;
    kind: "think" | "ask_user" | "research" | "act" | "wait";
    due_at?: string | null;
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
    counterparty_entity_id: promotion.counterparty_entity_id ?? null,
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
    nowMs: 1_700_000_000_000,
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
        counterparty_entity_id: null,
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
    expect(llm.requests[0]?.system).toBe(GOAL_PROMOTION_SYSTEM_PROMPT);
    expect(llm.requests[0]?.tools?.some((tool) => tool.cache_control !== undefined)).toBe(false);
    expect(GOAL_PROMOTION_SYSTEM_PROMPT).toContain("not_borg_responsibility");
    expect(GOAL_PROMOTION_SYSTEM_PROMPT).toContain("terminal_condition");
    expect(GOAL_PROMOTION_SYSTEM_PROMPT).toContain("meaningful future completion condition");
    expect(GOAL_PROMOTION_SYSTEM_PROMPT).toContain(SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE);
  });

  it("describes global duplicate coverage and prospective completion in the prompt and tool schema", async () => {
    const llm = new FakeLLMClient({ responses: [goalPromotionResponse([])] });
    const extractor = new GoalPromotionExtractor({ llmClient: llm, model: "haiku" });

    await extractor.extract(createExtractorInput());

    const systemPrompt = String(llm.requests[0]?.system ?? "");
    const properties = llm.requests[0]?.tools?.[0]?.inputSchema.properties as {
      promotions?: {
        items?: {
          properties?: Record<string, { description?: string }>;
        };
      };
    };
    const promotionProperties = properties.promotions?.items?.properties ?? {};

    expect(systemPrompt).toContain(
      "active_goals list is the complete global set of active goals across audiences and owners",
    );
    expect(systemPrompt).toContain("same underlying Borg responsibility or completion outcome");
    expect(systemPrompt).toContain("rephrased, translated, renewed later");
    expect(systemPrompt).toContain("the current turn need not refer to the prior goal");
    expect(systemPrompt).toContain(
      "A description of the current conversation, topic, exchange, or what Borg is presently tracking is context, not a goal",
    );
    expect(systemPrompt).toContain(
      "do not invent a terminal condition merely to qualify the candidate",
    );
    expect(systemPrompt).not.toContain("genuinely open-ended but actionable");
    expect(promotionProperties.duplicate_of_goal_id?.description).toContain(
      "same underlying Borg responsibility or completion outcome",
    );
    expect(promotionProperties.duplicate_of_goal_id?.description).toContain(
      "the current turn need not refer to the prior goal",
    );
    expect(promotionProperties.terminal_condition?.description).toContain(
      "meaningful future completion condition",
    );
    expect(promotionProperties.terminal_condition?.description).toContain(
      "null for non-durable classifications",
    );
    expect(promotionProperties.terminal_condition?.description).not.toContain(
      "genuinely open-ended",
    );
    expect(promotionProperties.counterparty_entity_id?.description).toContain(
      "presented_entity_ids",
    );
    expect(systemPrompt).toContain("responsibility carried to a completion condition");
    expect(systemPrompt).toContain("A priority of 9 or 10 asserts");
  });

  it("presents self identity, attributed history, and current structural addressing", async () => {
    const selfEntityId = createEntityId();
    const aliceEntityId = createEntityId();
    const bobEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const priorAgentEntryId = createStreamEntryId();
    const priorUserEntryId = createStreamEntryId();
    const currentEntryId = createStreamEntryId();
    const sessionId = createSessionId();
    const currentEntry = {
      id: currentEntryId,
      timestamp: 1_700_000_000_300,
      kind: "user_msg",
      content: "Which responsibility should continue?",
      turn_status: "active",
      audience: "Arena",
      sender_entity_id: bobEntityId,
      reply_target_entity_id: selfEntityId,
      session_id: sessionId,
      compressed: false,
    } satisfies StreamEntry;
    const displayNames = new Map([
      [aliceEntityId, "Alice"],
      [bobEntityId, "Bob"],
    ]);
    const llm = new FakeLLMClient({ responses: [goalPromotionResponse([])] });
    const extractor = new GoalPromotionExtractor({ llmClient: llm, model: "haiku" });

    await extractor.extract(
      createExtractorInput({
        selfIdentity: {
          id: selfEntityId,
          canonical_name: "Borg",
          aliases: ["B", "Memory Keeper"],
        },
        recentHistory: [
          {
            role: "assistant",
            content: "I can carry the responsibility.",
            stream_entry_id: priorAgentEntryId,
            sender_entity_id: null,
            ts: 1_700_000_000_100,
            kind: "agent_msg",
          },
          {
            role: "user",
            content: "Alice described the prior choice.",
            stream_entry_id: priorUserEntryId,
            sender_entity_id: aliceEntityId,
            ts: 1_700_000_000_200,
            kind: "user_msg",
          },
        ],
        currentMessageEntries: [currentEntry],
        currentMessageSenderAttribution: [
          {
            entryId: currentEntryId,
            senderEntityId: bobEntityId,
            senderDisplayName: "Bob",
          },
        ],
        audienceEntityId,
        speakerEntityId: bobEntityId,
        speakerDisplayName: "Bob",
        senderDisplayNameById: (entityId) => displayNames.get(entityId) ?? null,
      }),
    );

    const payload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as any;
    expect(payload.self_identity).toEqual({
      entity_id: selfEntityId,
      canonical_name: "Borg",
      handles: ["B", "Memory Keeper"],
    });
    expect(payload.recent_history).toEqual([
      expect.objectContaining({
        stream_entry_id: priorAgentEntryId,
        kind: "agent_msg",
        sender_entity_id: selfEntityId,
        sender_display_name: "Borg",
        sender_is_self: true,
      }),
      expect.objectContaining({
        stream_entry_id: priorUserEntryId,
        kind: "user_msg",
        sender_entity_id: aliceEntityId,
        sender_display_name: "Alice",
        sender_is_self: false,
      }),
    ]);
    expect(payload.current_message_entries).toEqual([
      expect.objectContaining({
        stream_entry_id: currentEntryId,
        sender_entity_id: bobEntityId,
        sender_display_name: "Bob",
        sender_is_self: false,
        audience_routing_label: "Arena",
        audience_entity_id: audienceEntityId,
        reply_target_entity_id: selfEntityId,
      }),
    ]);
    expect(payload.presented_entity_ids).toHaveLength(4);
    expect(payload.presented_entity_ids).toEqual(
      expect.arrayContaining([selfEntityId, aliceEntityId, bobEntityId, audienceEntityId]),
    );
  });

  it("presents a complete active-goal priority distribution", async () => {
    const llm = new FakeLLMClient({ responses: [goalPromotionResponse([])] });
    const extractor = new GoalPromotionExtractor({ llmClient: llm, model: "haiku" });

    await extractor.extract(
      createExtractorInput({
        activeGoals: [-1, 2, 5, 7, 9, 11].map((priority) => ({
          id: createGoalId(),
          description: `Goal at priority ${priority}`,
          terminal_condition: null,
          priority,
          target_at: null,
          audience_entity_id: null,
          owner_entity_id: null,
        })),
      }),
    );

    const payload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as any;
    expect(payload.active_goal_priority_distribution).toEqual({
      below_0: 1,
      "0_to_under_4": 1,
      "4_to_under_7": 1,
      "7_to_under_9": 1,
      "9_to_10": 1,
      above_10: 1,
    });
  });

  it("accepts only counterparty ids from the presented manifest", async () => {
    const { emit, tracer } = tracingHarness();
    const presentedEntityId = createEntityId();
    const unpresentedEntityId = createEntityId();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse(
          [
            {
              description: "Carry the presented participant's review responsibility",
              counterparty_entity_id: presentedEntityId,
              confidence: 0.95,
            },
            {
              description: "Carry a responsibility toward an unpresented participant",
              counterparty_entity_id: unpresentedEntityId,
              confidence: 0.96,
            },
          ],
          { durableGoalBatch: "explicit_multiple" },
        ),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-counterparty-reference",
    });

    const result = await extractor.extract(
      createExtractorInput({ audienceEntityId: presentedEntityId }),
    );

    expect(result).toHaveLength(1);
    expect(result[0]?.counterparty_entity_id).toBe(presentedEntityId);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        skipped_promotions: [{ candidate_index: 1, reason: "invalid_counterparty_entity_id" }],
      }),
    );
  });

  it("anchors the prompt with the current time so deadlines resolve to the right year", async () => {
    // A year-less date in the user turn ("before August 14") is unresolvable
    // without this anchor and produced a live goal dated two years in the past.
    const llm = new FakeLLMClient({ responses: [goalPromotionResponse([])] });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
    });
    const nowMs = 1_700_000_000_000;

    await extractor.extract(createExtractorInput({ nowMs }));

    const payload = JSON.parse(String(llm.requests[0]?.messages[0]?.content ?? "{}")) as {
      current_time?: { epoch_ms?: number; iso?: string };
      self_identity?: unknown;
    };
    expect(payload.current_time?.epoch_ms).toBe(nowMs);
    expect(payload.current_time?.iso).toBe(new Date(nowMs).toISOString());
    expect(payload.self_identity).toBeNull();
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
    const systemPrompt = llm.requests[0]?.system as string;

    expect(systemPrompt).toContain("monitoring p95");
    expect(systemPrompt).toContain("scheduled document edits");
    expect(systemPrompt).toContain(
      "Durable goals are about Borg's durable conversation/memory responsibility",
    );
    expect(systemPrompt).toContain(
      'A user saying "my goal is to..." is usually participant-side context',
    );
    expect(systemPrompt).toContain('my goal is to deploy", "friend will respond"');
    expect(systemPrompt).not.toContain("treat that speaker as the goal owner");
    expect(systemPrompt).toContain("user will deploy -> not_borg_responsibility");
    expect(systemPrompt).toContain(
      "Borg will monitor p95 -> impossible_for_borg_without_capability",
    );
    expect(systemPrompt).toContain("My host capability boundary");
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
            target_at: "2026-11-30",
            reason: "The user asked Borg to keep the planning review organized.",
            confidence: 0.92,
            initial_step: {
              description: "Ask for review constraints before Monday",
              kind: "ask_user",
              due_at: "2026-11-23T17:00:00Z",
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
        target_at: Date.parse("2026-11-30T23:59:59.999Z"),
        initial_step: {
          description: "Ask for review constraints before Monday",
          kind: "ask_user",
          due_at: Date.parse("2026-11-23T17:00:00Z"),
        },
      },
    ]);
  });

  it("asks the model for calendar dates rather than epoch integers", async () => {
    const llm = new FakeLLMClient({ responses: [goalPromotionResponse([])] });
    const extractor = new GoalPromotionExtractor({ llmClient: llm, model: "haiku" });

    await extractor.extract(createExtractorInput({}));

    const properties = llm.requests[0]?.tools?.[0]?.inputSchema.properties as {
      promotions?: { items?: { properties?: Record<string, { type?: string }> } };
    };
    const promotionProperties = properties.promotions?.items?.properties ?? {};

    // A 13-digit epoch has to be written digit by digit, and one wrong digit is a
    // silent year-scale error. The wire type is what keeps that off the record.
    expect(promotionProperties.target_at?.type).not.toBe("number");
    expect(JSON.stringify(promotionProperties.target_at)).toContain("string");
  });

  it("resolves a year-forward calendar date to that year rather than to the current one", async () => {
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Track the one-year verification the participants agreed on",
            target_at: "2027-08-11",
            confidence: 0.95,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({ llmClient: llm, model: "haiku" });

    const candidates = await extractor.extract(
      createExtractorInput({ nowMs: Date.parse("2026-08-11T14:40:44Z") }),
    );

    expect(candidates[0]?.target_at).toBe(Date.parse("2027-08-11T23:59:59.999Z"));
    expect(candidates[0]?.target_at).toBeGreaterThan(Date.parse("2026-08-11T14:40:44Z"));
  });

  it("skips a promotion whose deadline is not a parseable calendar date", async () => {
    const { emit, tracer } = tracingHarness();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse([
          {
            description: "Track something with an unusable deadline",
            target_at: "next August sometime",
            confidence: 0.95,
          },
        ]),
      ],
    });
    const extractor = new GoalPromotionExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn_deadline",
    });

    await expect(extractor.extract(createExtractorInput({}))).resolves.toEqual([]);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.completed",
      expect.objectContaining({
        skipped_promotions: [{ candidate_index: 0, reason: "invalid_target_at" }],
      }),
    );
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
    const audience = createEntityId();
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
            audience_entity_id: audience,
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
        audience_entity_id?: string | null;
        owner_entity_id?: string | null;
        terminal_condition?: string | null;
        disclosure?: string;
        disclosure_label?: { disclosure_class?: string; private_to_entity_ids?: string[] };
      }>;
    };
    const activeGoal = payload.active_goals?.[0];

    expect(activeGoal?.audience_entity_id).toBe(audience);
    expect(activeGoal?.owner_entity_id).toBe(owner);
    expect(activeGoal?.terminal_condition).toBe("The release checklist is settled");
    expect(activeGoal?.disclosure).toContain("disclosure_class=relationship_private");
    expect(activeGoal?.disclosure).toContain(`private-to=${audience},${owner}`);
    expect(activeGoal?.disclosure_label).toMatchObject({
      disclosure_class: "relationship_private",
      private_to_entity_ids: [audience, owner],
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
