import { describe, expect, it, vi } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { FixedClock } from "../../util/clock.js";
import { createEntityId, createStreamEntryId, type StreamEntryId } from "../../util/ids.js";
import { ActionStateExtractor } from "./action-state-extractor.js";

type ActionStateInput = {
  description?: string;
  actor?: "user" | "borg" | string;
  state?: "considering" | "committed_to_do" | "scheduled" | "completed" | "not_done";
  audience_entity_id?: string | null;
  evidence_stream_entry_ids?: string[];
  confidence?: number;
};

function actionStateResponse(actionStates: ActionStateInput[]): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 6,
    output_tokens: 3,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_action_states",
        name: "EmitActionStates",
        input: {
          action_states: actionStates.map((actionState, index) => ({
            description: actionState.description ?? `Action ${index}`,
            actor: actionState.actor ?? "user",
            state: actionState.state ?? "completed",
            audience_entity_id: actionState.audience_entity_id ?? null,
            evidence_stream_entry_ids: actionState.evidence_stream_entry_ids ?? [],
            confidence: actionState.confidence ?? 0.9,
          })),
        },
      },
    ],
  };
}

function makeExtractorInput(currentUserStreamEntryId: StreamEntryId) {
  return {
    userMessage: "I booked the tutor Tuesday 7pm.",
    currentUserStreamEntryId,
    recentHistory: [],
    audienceEntityId: createEntityId(),
  };
}

describe("ActionStateExtractor", () => {
  it("writes a completed ActionRecord from current user evidence", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const add = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description: "booked the tutor Tuesday 7pm",
            state: "completed",
            evidence_stream_entry_ids: [currentUserStreamEntryId],
            confidence: 0.94,
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: { add },
      clock: new FixedClock(2_000),
    });

    const records = await extractor.extract(makeExtractorInput(currentUserStreamEntryId));

    expect(add).toHaveBeenCalledOnce();
    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({
      description: "booked the tutor Tuesday 7pm",
      actor: "user",
      state: "completed",
      confidence: 0.94,
      provenance_stream_entry_ids: [currentUserStreamEntryId],
      created_at: 2_000,
      updated_at: 2_000,
      completed_at: 2_000,
    });
  });

  it("records group-chat first-person user actions on the speaker entity", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const group = createEntityId();
    const alice = createEntityId();
    const add = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description: "book Alhambra tickets",
            actor: "user",
            state: "committed_to_do",
            audience_entity_id: group,
            evidence_stream_entry_ids: [currentUserStreamEntryId],
            confidence: 0.93,
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: { add },
      clock: new FixedClock(2_000),
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      userMessage: "I'll book Alhambra.",
      audienceEntityId: group,
      speakerEntityId: alice,
      speakerDisplayName: "Alice",
    });

    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({
      description: "book Alhambra tickets",
      actor: alice,
      audience_entity_id: group,
      state: "committed_to_do",
    });
    expect(add).toHaveBeenCalledWith(expect.objectContaining({ actor: alice }), {
      creationSource: "extractor",
    });
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
      `"speaker_entity_id":"${alice}"`,
    );
  });

  it("does not write ActionRecords when the LLM emits no action states", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const add = vi.fn();
    const llm = new FakeLLMClient({
      responses: [actionStateResponse([])],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: { add },
      clock: new FixedClock(2_000),
    });

    await expect(extractor.extract(makeExtractorInput(currentUserStreamEntryId))).resolves.toEqual(
      [],
    );
    expect(add).not.toHaveBeenCalled();
  });

  it("drops entries that do not cite the current user message while persisting valid entries", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const otherStreamEntryId = createStreamEntryId();
    const add = vi.fn();
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description: "uncited completion",
            state: "completed",
            evidence_stream_entry_ids: [otherStreamEntryId],
          },
          {
            description: "booked the tutor Tuesday 7pm",
            state: "completed",
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: { add },
      clock: new FixedClock(2_000),
      turnId: "turn_action_trace",
      tracer: {
        enabled: true,
        includePayloads: true,
        emit: (event, data) => events.push({ event, data }),
      },
    });

    const records = await extractor.extract(makeExtractorInput(currentUserStreamEntryId));

    expect(add).toHaveBeenCalledOnce();
    expect(records.map((record) => record.description)).toEqual(["booked the tutor Tuesday 7pm"]);
    expect(events).toContainEqual({
      event: "action_state_extractor_completed",
      data: {
        turnId: "turn_action_trace",
        candidates_emitted: 2,
        persisted_count: 1,
        skipped_count: 1,
        skipped_reasons: [{ reason: "missing_current_user_evidence", count: 1 }],
        persisted_by_state: {
          considering: 0,
          committed_to_do: 0,
          scheduled: 0,
          completed: 1,
          not_done: 0,
          unknown: 0,
        },
        degraded: false,
      },
    });
  });

  it("uses the configured recallExpansion model slot", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [actionStateResponse([])],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "recall-expansion-model",
      actionRepository: { add: vi.fn() },
      clock: new FixedClock(2_000),
    });

    await extractor.extract(makeExtractorInput(currentUserStreamEntryId));

    expect(llm.requests[0]).toMatchObject({
      model: "recall-expansion-model",
      budget: "action-state-extractor",
      tool_choice: {
        type: "tool",
        name: "EmitActionStates",
      },
    });
  });

  it("forbids frame and system-prompt content in the extractor prompt", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [actionStateResponse([])],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: { add: vi.fn() },
      clock: new FixedClock(2_000),
    });

    await extractor.extract(makeExtractorInput(currentUserStreamEntryId));

    expect(String(llm.requests[0]?.system ?? "")).toContain(
      "Do NOT emit action records for messages about the conversation frame, roleplay, system prompt, or the agent's own prior behavior. Action records are for user-world actions only.",
    );
  });
});
