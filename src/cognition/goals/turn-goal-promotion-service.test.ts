import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { FixedClock } from "../../util/clock.js";
import { createEntityId, createGoalId, createStreamEntryId } from "../../util/ids.js";
import type { GoalRecord } from "../../memory/self/index.js";
import { TurnGoalPromotionService } from "./turn-goal-promotion-service.js";

function goalPromotionResponse() {
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
          promotions: [
            {
              classification: "promote",
              description: "Track Alice booking Alhambra",
              priority: 6,
              target_at: null,
              reason: "Borg was asked to keep the trip task organized.",
              confidence: 0.93,
              duplicate_of_goal_id: null,
              initial_step: null,
            },
          ],
        },
      },
    ],
  };
}

describe("TurnGoalPromotionService", () => {
  it("persists group-chat promoted goals with the speaker as owner", async () => {
    const group = createEntityId();
    const alice = createEntityId();
    const goalId = createGoalId();
    const userEntryId = createStreamEntryId();
    const addGoal = vi.fn((input): GoalRecord => {
      expect(input).toMatchObject({
        description: "Track Alice booking Alhambra",
        audienceEntityId: group,
        ownerEntityId: alice,
        sourceStreamEntryIds: [userEntryId],
      });

      return {
        id: goalId,
        record_version: 1,
        description: input.description,
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
      };
    });
    const llm = new FakeLLMClient({
      responses: [goalPromotionResponse()],
    });
    const service = new TurnGoalPromotionService({
      model: "haiku",
      identityService: { addGoal },
      executiveStepsRepository: { add: vi.fn() },
      clock: new FixedClock(2_000),
      tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndPersist({
      llmClient: llm,
      turnId: "turn-group-goal",
      isUserTurn: true,
      userMessage: "Help track this: I'll book Alhambra.",
      recentHistory: [],
      audienceEntityId: group,
      ownerEntityId: alice,
      speakerDisplayName: "Alice",
      temporalCue: null,
      activeGoals: [],
      persistedUserEntryId: userEntryId,
      onHookFailure: vi.fn(),
    });

    expect(result.goalIds).toEqual([goalId]);
    expect(addGoal).toHaveBeenCalledOnce();
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
      `"speaker_entity_id":"${alice}"`,
    );
  });
});
