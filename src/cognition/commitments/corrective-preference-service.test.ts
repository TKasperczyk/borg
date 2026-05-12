import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, createEntityId, createStreamEntryId } from "../../util/ids.js";
import { CorrectivePreferenceTurnService } from "./corrective-preference-service.js";

function correctivePreferenceResponse() {
  return {
    text: "",
    input_tokens: 6,
    output_tokens: 3,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_corrective",
        name: "EmitCorrectivePreference",
        input: {
          classification: "corrective_preference",
          type: "preference",
          directive: "Keep Alice's trip tasks separate from the group channel.",
          directive_family: "separate_trip_tasks",
          closure_pressure_relevance: "neutral",
          priority: 8,
          reason: "The current speaker made a durable correction.",
          confidence: 0.91,
          supersedes_commitment_id: null,
          slot_negations: [],
        },
      },
    ],
  };
}

describe("CorrectivePreferenceTurnService", () => {
  it("builds group-chat corrective commitments with the speaker as committer", async () => {
    const group = createEntityId();
    const alice = createEntityId();
    const userEntryId = createStreamEntryId();
    const addCommitment = vi.fn();
    const llm = new FakeLLMClient({
      responses: [correctivePreferenceResponse()],
    });
    const service = new CorrectivePreferenceTurnService({
      model: "haiku",
      commitmentRepository: { getApplicable: () => [] },
      identityService: { addCommitment },
      relationalSlotRepository: {
        list: () => [],
        applyNegation: vi.fn(),
      },
      workingMemoryStore: {
        load: () => createWorkingMemory(DEFAULT_SESSION_ID, 2_000),
        sanitizePendingActionsForRelationalSlot: vi.fn(),
      },
      clock: new FixedClock(2_000),
      tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
    });

    const result = await service.extractAndApply({
      llmClient: llm,
      turnId: "turn-group-commitment",
      userMessage: "For me, keep my trip tasks separate from the channel.",
      persistedUserEntryId: userEntryId,
      recentHistory: [],
      audienceEntityId: group,
      committedByEntityId: alice,
      speakerDisplayName: "Alice",
      sessionId: DEFAULT_SESSION_ID,
      onHookFailure: vi.fn(),
      trackAppliedSlotNegation: vi.fn(),
    });

    expect(result.commitment).toMatchObject({
      restricted_audience: group,
      committed_by_entity_id: alice,
      source_stream_entry_ids: [userEntryId],
    });

    await service.persistCommitment({
      commitment: result.commitment,
      onHookFailure: vi.fn(),
    });

    expect(addCommitment).toHaveBeenCalledWith(
      expect.objectContaining({
        restrictedAudience: group,
        committedByEntityId: alice,
      }),
    );
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
      `"speaker_entity_id":"${alice}"`,
    );
  });
});
