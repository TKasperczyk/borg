import { describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { FixedClock } from "../../util/clock.js";
import { createEntityId, createGoalId, createStreamEntryId } from "../../util/ids.js";
import type { GoalRecord, GoalTreeNode } from "../../memory/self/index.js";
import type { GoalPromotionClassification } from "./goal-promotion-extractor.js";
import { TurnGoalPromotionService } from "./turn-goal-promotion-service.js";

type GoalPromotionFixture = {
  classification?: GoalPromotionClassification;
  description?: string;
  terminal_condition?: string | null;
  priority?: number;
  target_at?: number | null;
  reason?: string;
  confidence?: number;
  duplicate_of_goal_id?: GoalRecord["id"] | null;
  initial_step?: {
    description: string;
    kind: "think" | "ask_user" | "research" | "act" | "wait";
    due_at: number | null;
    rationale: string;
  } | null;
};

function goalPromotionResponse(
  overrides: GoalPromotionFixture | readonly GoalPromotionFixture[] = {},
  options: { durableGoalBatch?: "single" | "explicit_multiple" } = {},
) {
  const promotions = Array.isArray(overrides) ? overrides : [overrides];

  return {
    text: "",
    input_tokens: 6,
    output_tokens: 3,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_goal",
        name: "EmitGoalPromotion",
        input: {
          durable_goal_batch: options.durableGoalBatch ?? "single",
          promotions: promotions.map((promotion) => ({
            classification: promotion.classification ?? "durable_borg_goal",
            description: promotion.description ?? "Track Alice drafting the launch brief",
            terminal_condition:
              promotion.terminal_condition ?? "Alice's launch brief tracking reaches handoff",
            priority: promotion.priority ?? 6,
            target_at: promotion.target_at ?? null,
            reason: promotion.reason ?? "Borg was asked to keep the work task organized.",
            confidence: promotion.confidence ?? 0.93,
            duplicate_of_goal_id: promotion.duplicate_of_goal_id ?? null,
            initial_step: promotion.initial_step ?? null,
          })),
        },
      },
    ],
  };
}

function goalRecord(
  overrides: Partial<GoalRecord> & { id?: GoalRecord["id"]; description?: string } = {},
): GoalTreeNode {
  return {
    id: overrides.id ?? createGoalId(),
    record_version: overrides.record_version ?? 1,
    description: overrides.description ?? "Track code review follow-up",
    terminal_condition: overrides.terminal_condition ?? null,
    priority: overrides.priority ?? 6,
    parent_goal_id: overrides.parent_goal_id ?? null,
    status: overrides.status ?? "active",
    progress_notes: overrides.progress_notes ?? null,
    last_progress_ts: overrides.last_progress_ts ?? null,
    created_at: overrides.created_at ?? 2_000,
    target_at: overrides.target_at ?? null,
    audience_entity_id: overrides.audience_entity_id ?? null,
    owner_entity_id: overrides.owner_entity_id ?? null,
    source_stream_entry_ids: overrides.source_stream_entry_ids,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
    provenance: overrides.provenance ?? {
      kind: "online",
      process: "test",
    },
    children: [],
  };
}

class ScriptedEmbeddingClient implements EmbeddingClient {
  readonly embeddedTexts: string[] = [];
  readonly embeddedBatchTexts: string[][] = [];

  constructor(
    private readonly vectors: ReadonlyMap<string, readonly number[]> = new Map(),
    private readonly defaultVector: readonly number[] = [0, 1],
  ) {}

  async embed(text: string): Promise<Float32Array> {
    this.embeddedTexts.push(text);
    return Float32Array.from(this.vectors.get(text) ?? this.defaultVector);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    this.embeddedBatchTexts.push([...texts]);
    return texts.map((text) => Float32Array.from(this.vectors.get(text) ?? this.defaultVector));
  }
}

describe("TurnGoalPromotionService", () => {
  it("rejects group-chat participant goals instead of persisting them as speaker-owned goals", async () => {
    const group = createEntityId();
    const alice = createEntityId();
    const userEntryId = createStreamEntryId();
    const addGoal = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse({
          classification: "not_borg_responsibility",
          description: "Alice will draft the launch brief",
          reason: "The durable responsibility belongs to Alice, not Borg.",
        }),
      ],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: { list: () => [] },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient: new ScriptedEmbeddingClient(),
      clock: new FixedClock(2_000),
      tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-group-goal",
      isUserTurn: true,
      userMessage: "Help track this: I'll draft the launch brief.",
      recentHistory: [],
      audienceEntityId: group,
      ownerEntityId: alice,
      speakerDisplayName: "Alice",
      temporalCue: null,
      activeGoals: [],
      persistedUserEntryId: userEntryId,
      onHookFailure: vi.fn(),
    });

    expect(result).toEqual({ goalIds: [], executiveStepIds: [] });
    expect(addGoal).not.toHaveBeenCalled();
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
      `"speaker_entity_id":"${alice}"`,
    );
  });

  it("drops wait initial steps without due_at while keeping the promoted goal", async () => {
    const audience = createEntityId();
    const goalId = createGoalId();
    const addGoal = vi.fn(
      (input): GoalRecord => ({
        id: goalId,
        record_version: 1,
        description: input.description,
        terminal_condition: input.terminalCondition ?? null,
        priority: input.priority,
        parent_goal_id: null,
        status: "active",
        progress_notes: null,
        last_progress_ts: null,
        created_at: 2_000,
        target_at: input.targetAt,
        audience_entity_id: input.audienceEntityId,
        owner_entity_id: input.ownerEntityId,
        source_stream_entry_ids: input.sourceStreamEntryIds,
        provenance: input.provenance,
      }),
    );
    const addStep = vi.fn();
    const emit = vi.fn();
    const onHookFailure = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse({
          description: "Track the user preparing the quarterly review packet",
          terminal_condition: "The quarterly review packet is ready for review",
          reason: "Borg was asked to keep the review preparation organized.",
          initial_step: {
            description: "Wait for the finance team update",
            kind: "wait",
            due_at: null,
            rationale: "The next useful move depends on the external update.",
          },
        }),
      ],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: { list: () => [] },
      executiveStepsRepository: { add: addStep },
      embeddingClient: new ScriptedEmbeddingClient(),
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-wait-step",
      isUserTurn: true,
      userMessage: "Keep the quarterly review packet moving.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: null,
      temporalCue: null,
      activeGoals: [],
      onHookFailure,
    });

    expect(result).toEqual({
      goalIds: [goalId],
      executiveStepIds: [],
    });
    expect(addGoal).toHaveBeenCalledOnce();
    expect(addGoal).toHaveBeenCalledWith(
      expect.objectContaining({
        terminalCondition: "The quarterly review packet is ready for review",
      }),
    );
    expect(addStep).not.toHaveBeenCalled();
    expect(onHookFailure).not.toHaveBeenCalled();
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.transitioned",
      expect.objectContaining({
        turnId: "turn-goal-wait-step",
        reason: "wait_without_due_at",
        goalId,
      }),
    );
  });

  it("skips extractor-flagged duplicates on the same active axis without returning their ids", async () => {
    const audience = createEntityId();
    const owner = createEntityId();
    const existingGoalId = createGoalId();
    const emit = vi.fn();
    const addGoal = vi.fn();
    const embeddingClient = new ScriptedEmbeddingClient();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse({
          description: "Track the deployment checklist review",
          duplicate_of_goal_id: existingGoalId,
        }),
      ],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: {
        list: () => [
          goalRecord({
            id: existingGoalId,
            description: "Track the deployment checklist review",
            audience_entity_id: audience,
            owner_entity_id: owner,
          }),
        ],
      },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient,
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-extractor-duplicate",
      isUserTurn: true,
      userMessage: "Keep tracking the deployment checklist review.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: owner,
      temporalCue: null,
      activeGoals: [],
      onHookFailure: vi.fn(),
    });

    expect(result).toEqual({ goalIds: [], executiveStepIds: [] });
    expect(addGoal).not.toHaveBeenCalled();
    expect(embeddingClient.embeddedTexts).toEqual([]);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.skipped",
      expect.objectContaining({
        turnId: "turn-goal-extractor-duplicate",
        candidate_description: "Track the deployment checklist review",
        duplicate_of_goal_id: existingGoalId,
        reason: "extractor_signal",
      }),
    );
  });

  it("treats unknown extractor duplicate references as advisory and persists when embeddings clear", async () => {
    const audience = createEntityId();
    const owner = createEntityId();
    const unknownGoalId = createGoalId();
    const persistedGoalId = createGoalId();
    const addGoal = vi.fn(
      (input): GoalRecord =>
        goalRecord({
          id: persistedGoalId,
          description: input.description,
          terminal_condition: input.terminalCondition ?? null,
          priority: input.priority,
          target_at: input.targetAt,
          audience_entity_id: input.audienceEntityId,
          owner_entity_id: input.ownerEntityId,
          source_stream_entry_ids: input.sourceStreamEntryIds,
          provenance: input.provenance,
        }),
    );
    const embeddingClient = new ScriptedEmbeddingClient();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse({
          description: "Track the documentation handoff",
          duplicate_of_goal_id: unknownGoalId,
        }),
      ],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: { list: () => [] },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient,
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-unknown-duplicate",
      isUserTurn: true,
      userMessage: "Track the documentation handoff.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: owner,
      temporalCue: null,
      activeGoals: [],
      onHookFailure: vi.fn(),
    });

    expect(result.goalIds).toEqual([persistedGoalId]);
    expect(addGoal).toHaveBeenCalledOnce();
    expect(embeddingClient.embeddedTexts).toEqual(["Track the documentation handoff"]);
  });

  it("treats inactive extractor duplicate references as advisory and persists", async () => {
    const audience = createEntityId();
    const owner = createEntityId();
    const doneGoalId = createGoalId();
    const persistedGoalId = createGoalId();
    const description = "Track the onboarding cleanup";
    const emit = vi.fn();
    const addGoal = vi.fn(
      (input): GoalRecord =>
        goalRecord({
          id: persistedGoalId,
          description: input.description,
          terminal_condition: input.terminalCondition ?? null,
          priority: input.priority,
          target_at: input.targetAt,
          audience_entity_id: input.audienceEntityId,
          owner_entity_id: input.ownerEntityId,
          source_stream_entry_ids: input.sourceStreamEntryIds,
          provenance: input.provenance,
        }),
    );
    const llm = new FakeLLMClient({
      responses: [goalPromotionResponse({ description, duplicate_of_goal_id: doneGoalId })],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: {
        list: () => [
          goalRecord({
            id: doneGoalId,
            description,
            status: "done",
            audience_entity_id: audience,
            owner_entity_id: owner,
          }),
        ],
      },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient: new ScriptedEmbeddingClient(),
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-inactive-duplicate-reference",
      isUserTurn: true,
      userMessage: "Track the onboarding cleanup.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: owner,
      temporalCue: null,
      activeGoals: [],
      onHookFailure: vi.fn(),
    });

    expect(result.goalIds).toEqual([persistedGoalId]);
    expect(addGoal).toHaveBeenCalledOnce();
    expect(emit).not.toHaveBeenCalledWith(
      "extraction.goals.skipped",
      expect.objectContaining({
        candidate_description: description,
      }),
    );
  });

  it("treats different-owner extractor duplicate references as advisory and persists", async () => {
    const audience = createEntityId();
    const owner = createEntityId();
    const otherOwner = createEntityId();
    const referencedGoalId = createGoalId();
    const persistedGoalId = createGoalId();
    const description = "Track the build pipeline cleanup";
    const emit = vi.fn();
    const addGoal = vi.fn(
      (input): GoalRecord =>
        goalRecord({
          id: persistedGoalId,
          description: input.description,
          terminal_condition: input.terminalCondition ?? null,
          priority: input.priority,
          target_at: input.targetAt,
          audience_entity_id: input.audienceEntityId,
          owner_entity_id: input.ownerEntityId,
          source_stream_entry_ids: input.sourceStreamEntryIds,
          provenance: input.provenance,
        }),
    );
    const llm = new FakeLLMClient({
      responses: [goalPromotionResponse({ description, duplicate_of_goal_id: referencedGoalId })],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: {
        list: () => [
          goalRecord({
            id: referencedGoalId,
            description,
            audience_entity_id: audience,
            owner_entity_id: otherOwner,
          }),
        ],
      },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient: new ScriptedEmbeddingClient(),
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-owner-mismatch-duplicate-reference",
      isUserTurn: true,
      userMessage: "Track the build pipeline cleanup.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: owner,
      temporalCue: null,
      activeGoals: [],
      onHookFailure: vi.fn(),
    });

    expect(result.goalIds).toEqual([persistedGoalId]);
    expect(addGoal).toHaveBeenCalledOnce();
    expect(emit).not.toHaveBeenCalledWith(
      "extraction.goals.skipped",
      expect.objectContaining({
        candidate_description: description,
      }),
    );
  });

  it("treats different-audience extractor duplicate references as advisory and persists", async () => {
    const audience = createEntityId();
    const otherAudience = createEntityId();
    const owner = createEntityId();
    const referencedGoalId = createGoalId();
    const persistedGoalId = createGoalId();
    const description = "Track the dashboard polish pass";
    const emit = vi.fn();
    const addGoal = vi.fn(
      (input): GoalRecord =>
        goalRecord({
          id: persistedGoalId,
          description: input.description,
          terminal_condition: input.terminalCondition ?? null,
          priority: input.priority,
          target_at: input.targetAt,
          audience_entity_id: input.audienceEntityId,
          owner_entity_id: input.ownerEntityId,
          source_stream_entry_ids: input.sourceStreamEntryIds,
          provenance: input.provenance,
        }),
    );
    const llm = new FakeLLMClient({
      responses: [goalPromotionResponse({ description, duplicate_of_goal_id: referencedGoalId })],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: {
        list: () => [
          goalRecord({
            id: referencedGoalId,
            description,
            audience_entity_id: otherAudience,
            owner_entity_id: owner,
          }),
        ],
      },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient: new ScriptedEmbeddingClient(),
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-audience-mismatch-duplicate-reference",
      isUserTurn: true,
      userMessage: "Track the dashboard polish pass.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: owner,
      temporalCue: null,
      activeGoals: [],
      onHookFailure: vi.fn(),
    });

    expect(result.goalIds).toEqual([persistedGoalId]);
    expect(addGoal).toHaveBeenCalledOnce();
    expect(emit).not.toHaveBeenCalledWith(
      "extraction.goals.skipped",
      expect.objectContaining({
        candidate_description: description,
      }),
    );
  });

  it("skips embedding duplicates against active same-axis goals", async () => {
    const audience = createEntityId();
    const owner = createEntityId();
    const existingGoalId = createGoalId();
    const emit = vi.fn();
    const addGoal = vi.fn();
    const activeDescription = "Track the API migration plan";
    const candidateDescription = "Keep tracking the API migration plan";
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [activeDescription, [1, 0]],
        [candidateDescription, [0.95, 0.05]],
      ]),
    );
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse({
          description: candidateDescription,
        }),
      ],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: {
        list: () => [
          goalRecord({
            id: existingGoalId,
            description: activeDescription,
            audience_entity_id: audience,
            owner_entity_id: owner,
          }),
        ],
      },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient,
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-existing-embedding-duplicate",
      isUserTurn: true,
      userMessage: "Keep tracking the API migration plan.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: owner,
      temporalCue: null,
      activeGoals: [],
      onHookFailure: vi.fn(),
    });

    expect(result.goalIds).toEqual([]);
    expect(addGoal).not.toHaveBeenCalled();
    expect(embeddingClient.embeddedBatchTexts).toEqual([[activeDescription]]);
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.skipped",
      expect.objectContaining({
        turnId: "turn-goal-existing-embedding-duplicate",
        candidate_description: candidateDescription,
        matched_existing_id: existingGoalId,
        reason: "embedding",
        similarity: expect.any(Number),
      }),
    );
  });

  it("skips embedding duplicates within the same promotion batch", async () => {
    const audience = createEntityId();
    const owner = createEntityId();
    const firstGoalId = createGoalId();
    const addGoal = vi.fn(
      (input): GoalRecord =>
        goalRecord({
          id: firstGoalId,
          description: input.description,
          terminal_condition: input.terminalCondition ?? null,
          priority: input.priority,
          target_at: input.targetAt,
          audience_entity_id: input.audienceEntityId,
          owner_entity_id: input.ownerEntityId,
          source_stream_entry_ids: input.sourceStreamEntryIds,
          provenance: input.provenance,
        }),
    );
    const firstDescription = "Track the accessibility audit follow-up";
    const secondDescription = "Keep tracking accessibility audit follow-up";
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [firstDescription, [1, 0]],
        [secondDescription, [0.98, 0.02]],
      ]),
    );
    const emit = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        goalPromotionResponse(
          [{ description: firstDescription }, { description: secondDescription }],
          { durableGoalBatch: "explicit_multiple" },
        ),
      ],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: { list: () => [] },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient,
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-batch-embedding-duplicate",
      isUserTurn: true,
      userMessage: "Track the accessibility audit follow-up.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: owner,
      temporalCue: null,
      activeGoals: [],
      onHookFailure: vi.fn(),
    });

    expect(result.goalIds).toEqual([firstGoalId]);
    expect(addGoal).toHaveBeenCalledOnce();
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.skipped",
      expect.objectContaining({
        turnId: "turn-goal-batch-embedding-duplicate",
        candidate_description: secondDescription,
        matched_existing_id: firstGoalId,
        reason: "embedding",
      }),
    );
  });

  it("does not embedding-dedup goals across different owners or audiences", async () => {
    const audience = createEntityId();
    const otherAudience = createEntityId();
    const currentOwner = createEntityId();
    const otherOwner = createEntityId();
    const persistedGoalId = createGoalId();
    const description = "Track the release notes checklist";
    const addGoal = vi.fn(
      (input): GoalRecord =>
        goalRecord({
          id: persistedGoalId,
          description: input.description,
          terminal_condition: input.terminalCondition ?? null,
          priority: input.priority,
          target_at: input.targetAt,
          audience_entity_id: input.audienceEntityId,
          owner_entity_id: input.ownerEntityId,
          source_stream_entry_ids: input.sourceStreamEntryIds,
          provenance: input.provenance,
        }),
    );
    const embeddingClient = new ScriptedEmbeddingClient(new Map([[description, [1, 0]]]));
    const llm = new FakeLLMClient({
      responses: [goalPromotionResponse({ description })],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: {
        list: () => [
          goalRecord({
            description,
            audience_entity_id: audience,
            owner_entity_id: otherOwner,
          }),
          goalRecord({
            description,
            audience_entity_id: otherAudience,
            owner_entity_id: currentOwner,
          }),
        ],
      },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient,
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-different-owner",
      isUserTurn: true,
      userMessage: "Track the release notes checklist.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: currentOwner,
      temporalCue: null,
      activeGoals: [],
      onHookFailure: vi.fn(),
    });

    expect(result.goalIds).toEqual([persistedGoalId]);
    expect(addGoal).toHaveBeenCalledOnce();
    expect(embeddingClient.embeddedBatchTexts).toEqual([]);
  });

  it("fails open when embedding dedup is unavailable", async () => {
    const audience = createEntityId();
    const owner = createEntityId();
    const persistedGoalId = createGoalId();
    const description = "Track the incident follow-up checklist";
    const emit = vi.fn();
    const addGoal = vi.fn(
      (input): GoalRecord =>
        goalRecord({
          id: persistedGoalId,
          description: input.description,
          terminal_condition: input.terminalCondition ?? null,
          priority: input.priority,
          target_at: input.targetAt,
          audience_entity_id: input.audienceEntityId,
          owner_entity_id: input.ownerEntityId,
          source_stream_entry_ids: input.sourceStreamEntryIds,
          provenance: input.provenance,
        }),
    );
    const embeddingClient = {
      embed: vi.fn(async () => {
        throw new Error("embedding service unavailable");
      }),
      embedBatch: vi.fn(async () => []),
    } satisfies EmbeddingClient;
    const llm = new FakeLLMClient({
      responses: [goalPromotionResponse({ description })],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      goalsRepository: { list: () => [] },
      executiveStepsRepository: { add: vi.fn() },
      embeddingClient,
      clock: new FixedClock(2_000),
      tracer: { enabled: true, includePayloads: false, emit },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-goal-embedding-outage",
      isUserTurn: true,
      userMessage: "Track the incident follow-up checklist.",
      recentHistory: [],
      audienceEntityId: audience,
      ownerEntityId: owner,
      temporalCue: null,
      activeGoals: [],
      onHookFailure: vi.fn(),
    });

    expect(result.goalIds).toEqual([persistedGoalId]);
    expect(addGoal).toHaveBeenCalledOnce();
    expect(emit).toHaveBeenCalledWith(
      "extraction.goals.dedup.degraded",
      expect.objectContaining({
        turnId: "turn-goal-embedding-outage",
        reason: "candidate_embedding_failed",
        candidate_description: description,
      }),
    );
  });
});
