import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import {
  actionRecordToRelationalGuardEvidence,
  type RelationalClaimAuditClaim,
  validateRelationalClaims,
} from "../generation/relational-guard.js";
import { FixedClock } from "../../util/clock.js";
import {
  createEntityId,
  createStreamEntryId,
  DEFAULT_SESSION_ID,
  type ActionId,
  type StreamEntryId,
} from "../../util/ids.js";
import { ActionStateExtractor } from "./action-state-extractor.js";

type ActionStateInput = {
  description?: string;
  actor?: "user" | "borg";
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

function makeClaim(citedActionIds: ActionId[]): RelationalClaimAuditClaim {
  return {
    kind: "action_completion",
    asserted: "You booked the tutor Tuesday 7pm.",
    cited_stream_entry_ids: [],
    cited_episode_ids: [],
    cited_commitment_ids: [],
    cited_action_ids: citedActionIds,
    quoted_evidence_text: null,
    callback_scope: null,
    subject_entity_id: null,
    slot_key: null,
    relational_slot_value: null,
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

    const summary = validateRelationalClaims({
      claims: [makeClaim([records[0]!.id])],
      evidence: {
        current_user_message: null,
        current_session_stream_entries: [],
        retrieved_episodes: [],
        active_commitments: [],
        corrective_preferences: [],
        relational_slots: [],
        recent_completed_actions: [actionRecordToRelationalGuardEvidence(records[0]!)],
      },
      currentSessionId: DEFAULT_SESSION_ID,
      currentTurnTs: 3_000,
      hasCorrectivePreferenceEvidence: () => false,
    });

    expect(summary.unsupported).toEqual([]);
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
    });

    const records = await extractor.extract(makeExtractorInput(currentUserStreamEntryId));

    expect(add).toHaveBeenCalledOnce();
    expect(records.map((record) => record.description)).toEqual(["booked the tutor Tuesday 7pm"]);
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
});
