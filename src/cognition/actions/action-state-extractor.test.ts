import { describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type {
  ActionRecord,
  ActionRecordListFilter,
  ActionRecordPatch,
} from "../../memory/actions/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createActionId,
  createEntityId,
  createGoalId,
  createOpenQuestionId,
  createStreamEntryId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  ActionStateExtractor,
  type ActionCandidateClassification,
} from "./action-state-extractor.js";

type ActionStateInput = {
  classification?: ActionCandidateClassification | string;
  description?: string;
  actor?: "user" | "borg" | string;
  state?: "considering" | "committed_to_do" | "scheduled" | "completed" | "not_done";
  audience_entity_id?: string | null;
  evidence_stream_entry_ids?: string[];
  confidence?: number;
};

function actionStateResponse(actionStates: ActionStateInput[]): LLMCompleteResult {
  return rawActionStateResponse(
    actionStates.map((actionState, index) => ({
      classification: actionState.classification ?? "concrete_action",
      description: actionState.description ?? `Action ${index}`,
      actor: actionState.actor ?? "user",
      state: actionState.state ?? "completed",
      audience_entity_id: actionState.audience_entity_id ?? null,
      evidence_stream_entry_ids: actionState.evidence_stream_entry_ids ?? [],
      confidence: actionState.confidence ?? 0.9,
    })),
  );
}

function rawActionStateResponse(actionStates: unknown[]): LLMCompleteResult {
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
          action_states: actionStates,
        },
      },
    ],
  };
}

function makeExtractorInput(currentUserStreamEntryId: StreamEntryId) {
  return {
    userMessage: "I reviewed the API patch.",
    currentUserStreamEntryId,
    recentHistory: [],
    audienceEntityId: createEntityId(),
  };
}

function makeActionRecord(
  overrides: Partial<ActionRecord> & { description: string },
): ActionRecord {
  const nowMs = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createActionId(),
    description: overrides.description,
    actor: overrides.actor ?? "user",
    audience_entity_id: overrides.audience_entity_id ?? null,
    goal_id: overrides.goal_id ?? null,
    open_question_id: overrides.open_question_id ?? null,
    state: overrides.state ?? "committed_to_do",
    confidence: overrides.confidence ?? 0.9,
    provenance_episode_ids: overrides.provenance_episode_ids ?? [],
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [createStreamEntryId()],
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
    considering_at: overrides.considering_at ?? null,
    committed_at: overrides.committed_at ?? nowMs,
    scheduled_at: overrides.scheduled_at ?? null,
    completed_at: overrides.completed_at ?? null,
    not_done_at: overrides.not_done_at ?? null,
    unknown_at: overrides.unknown_at ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
  };
}

function makeActionRepository(records: ActionRecord[] = []) {
  const add = vi.fn((record: ActionRecord) => {
    records.push(record);
  });
  const update = vi.fn((id: ActionRecord["id"], patch: ActionRecordPatch) => {
    const index = records.findIndex((record) => record.id === id);
    const existing = records[index];

    if (existing === undefined) {
      throw new Error(`Missing action ${id}`);
    }

    records[index] = {
      ...existing,
      ...patch,
    };
  });
  const list = vi.fn((filter: ActionRecordListFilter = {}) =>
    records.filter((record) => {
      if (filter.state !== undefined && record.state !== filter.state) {
        return false;
      }

      if (filter.states !== undefined && !filter.states.includes(record.state)) {
        return false;
      }

      if (filter.actor !== undefined && record.actor !== filter.actor) {
        return false;
      }

      if ("audienceEntityId" in filter && record.audience_entity_id !== filter.audienceEntityId) {
        return false;
      }

      if (filter.goalId !== undefined && record.goal_id !== filter.goalId) {
        return false;
      }

      if (
        filter.openQuestionId !== undefined &&
        record.open_question_id !== filter.openQuestionId
      ) {
        return false;
      }

      return true;
    }),
  );

  return {
    add,
    update,
    list,
    records,
  };
}

class ScriptedEmbeddingClient implements EmbeddingClient {
  constructor(
    private readonly vectors = new Map<string, readonly number[]>(),
    private readonly fail = false,
  ) {}

  async embed(text: string): Promise<Float32Array> {
    const [embedding] = await this.embedBatch([text]);

    if (embedding === undefined) {
      throw new Error("Missing embedding");
    }

    return embedding;
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    if (this.fail) {
      throw new Error("embedding offline");
    }

    return texts.map((text) => Float32Array.from(this.vectors.get(text) ?? [0, 1]));
  }
}

describe("ActionStateExtractor", () => {
  it("writes a completed ActionRecord from current user evidence", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const add = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description: "reviewed the API patch",
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
      description: "reviewed the API patch",
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
            description: "update the team checklist",
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
      userMessage: "I'll update the team checklist.",
      audienceEntityId: group,
      speakerEntityId: alice,
      speakerDisplayName: "Alice",
    });

    expect(records).toHaveLength(1);
    expect(records[0]).toMatchObject({
      description: "update the team checklist",
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
            description: "reviewed the API patch",
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
    expect(records.map((record) => record.description)).toEqual(["reviewed the API patch"]);
    expect(events).toContainEqual({
      event: "extraction.actions.completed",
      data: expect.objectContaining({
        turnId: "turn_action_trace",
        candidates_emitted: 2,
        valid_candidate_count: 2,
        persisted_count: 1,
        skipped_count: 1,
        skipped_reasons: [{ reason: "missing_current_user_evidence", count: 1 }],
        skipped_candidates: [{ candidate_index: 0, reason: "missing_current_user_evidence" }],
        persisted_by_state: {
          considering: 0,
          committed_to_do: 0,
          scheduled: 0,
          completed: 1,
          not_done: 0,
          unknown: 0,
        },
        classification_counts: expect.objectContaining({
          concrete_action: 2,
        }),
        degraded: false,
      }),
    });
  });

  it.each([
    {
      classification: "conversational_acknowledgment",
      description: "heading back to the office",
    },
    {
      classification: "decision_or_preference",
      description: "prefer morning review windows",
    },
    {
      classification: "already_represented",
      description: "review the open API patch",
    },
    {
      classification: "outside_borg_capability",
      description: "seed the postmortem doc by morning",
    },
    {
      classification: "none",
      description: "not relevant to memory",
    },
  ] satisfies Array<{ classification: ActionCandidateClassification; description: string }>)(
    "rejects %s candidates before persistence",
    async ({ classification, description }) => {
      const currentUserStreamEntryId = createStreamEntryId();
      const add = vi.fn();
      const events: Array<{ event: string; data: Record<string, unknown> }> = [];
      const llm = new FakeLLMClient({
        responses: [
          actionStateResponse([
            {
              classification,
              description,
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
        turnId: "turn_action_classification",
        tracer: {
          enabled: true,
          includePayloads: true,
          emit: (event, data) => events.push({ event, data }),
        },
      });

      const records = await extractor.extract(makeExtractorInput(currentUserStreamEntryId));

      expect(records).toEqual([]);
      expect(add).not.toHaveBeenCalled();
      expect(events).toContainEqual({
        event: "extraction.actions.rejected",
        data: {
          turnId: "turn_action_classification",
          classification,
          description_excerpt: description,
          reason: "non_concrete_classification",
        },
      });
      expect(events).toContainEqual({
        event: "extraction.actions.completed",
        data: expect.objectContaining({
          turnId: "turn_action_classification",
          candidates_emitted: 1,
          valid_candidate_count: 1,
          persisted_count: 0,
          skipped_count: 1,
          skipped_reasons: [{ reason: "non_concrete_classification", count: 1 }],
          classification_counts: expect.objectContaining({
            [classification]: 1,
          }),
          rejected_by_classification: expect.objectContaining({
            [classification]: 1,
          }),
          degraded: false,
        }),
      });
    },
  );

  it("rejects Borg-owned actions outside host capability as observable taxonomy rejects", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const add = vi.fn();
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            classification: "outside_borg_capability",
            description: "seed the postmortem doc by morning",
            actor: "borg",
            state: "committed_to_do",
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
      turnId: "turn_action_capability_boundary",
      tracer: {
        enabled: true,
        includePayloads: true,
        emit: (event, data) => events.push({ event, data }),
      },
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      userMessage: "I'll seed the postmortem doc by morning.",
    });

    expect(records).toEqual([]);
    expect(add).not.toHaveBeenCalled();
    expect(llm.requests[0]?.system).toContain("outside_borg_capability");
    expect(llm.requests[0]?.system).toContain("external_document_editing");
    expect(events).toContainEqual({
      event: "extraction.actions.rejected",
      data: {
        turnId: "turn_action_capability_boundary",
        classification: "outside_borg_capability",
        description_excerpt: "seed the postmortem doc by morning",
        reason: "non_concrete_classification",
      },
    });
  });

  it("rejects items missing classification with invalid_classification telemetry", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const add = vi.fn();
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const llm = new FakeLLMClient({
      responses: [
        rawActionStateResponse([
          {
            description: "review the release checklist",
            actor: "user",
            state: "committed_to_do",
            audience_entity_id: null,
            evidence_stream_entry_ids: [currentUserStreamEntryId],
            confidence: 0.9,
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: { add },
      clock: new FixedClock(2_000),
      turnId: "turn_missing_classification",
      tracer: {
        enabled: true,
        includePayloads: true,
        emit: (event, data) => events.push({ event, data }),
      },
    });

    const records = await extractor.extract(makeExtractorInput(currentUserStreamEntryId));

    expect(records).toEqual([]);
    expect(add).not.toHaveBeenCalled();
    expect(events).toContainEqual({
      event: "extraction.actions.completed",
      data: expect.objectContaining({
        turnId: "turn_missing_classification",
        candidates_emitted: 1,
        valid_candidate_count: 0,
        persisted_count: 0,
        skipped_count: 1,
        skipped_reasons: [{ reason: "invalid_classification", count: 1 }],
        skipped_candidates: [{ candidate_index: 0, reason: "invalid_classification" }],
        classification_counts: expect.objectContaining({
          invalid_classification: 1,
        }),
        rejected_invalid_enum: 1,
        degraded: false,
      }),
    });
  });

  it("rejects invalid classification values with invalid_classification telemetry", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const add = vi.fn();
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            classification: "maybe_action",
            description: "review the release checklist",
            state: "committed_to_do",
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
      turnId: "turn_invalid_classification",
      tracer: {
        enabled: true,
        includePayloads: true,
        emit: (event, data) => events.push({ event, data }),
      },
    });

    const records = await extractor.extract(makeExtractorInput(currentUserStreamEntryId));

    expect(records).toEqual([]);
    expect(add).not.toHaveBeenCalled();
    expect(events).toContainEqual({
      event: "extraction.actions.completed",
      data: expect.objectContaining({
        turnId: "turn_invalid_classification",
        candidates_emitted: 1,
        valid_candidate_count: 0,
        persisted_count: 0,
        skipped_count: 1,
        skipped_reasons: [{ reason: "invalid_classification", count: 1 }],
        skipped_candidates: [{ candidate_index: 0, reason: "invalid_classification" }],
        classification_counts: expect.objectContaining({
          invalid_classification: 1,
        }),
        rejected_invalid_enum: 1,
        degraded: false,
      }),
    });
  });

  it("skips a concrete action that matches an existing active same-axis action", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const audience = createEntityId();
    const goalId = createGoalId();
    const openQuestionId = createOpenQuestionId();
    const existingDescription = "review the auth service patch";
    const candidateDescription = "review the authentication service patch";
    const repository = makeActionRepository([
      makeActionRecord({
        description: existingDescription,
        actor: "user",
        audience_entity_id: audience,
        goal_id: goalId,
        open_question_id: openQuestionId,
      }),
    ]);
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [existingDescription, [1, 0]],
        [candidateDescription, [1, 0]],
      ]),
    );
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description: candidateDescription,
            state: "committed_to_do",
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: repository,
      embeddingClient,
      clock: new FixedClock(2_000),
      turnId: "turn_action_dedup",
      tracer: {
        enabled: true,
        includePayloads: true,
        emit: (event, data) => events.push({ event, data }),
      },
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      audienceEntityId: audience,
      goalId,
      openQuestionId,
    });

    expect(records).toEqual([]);
    expect(repository.add).not.toHaveBeenCalled();
    expect(events).toContainEqual({
      event: "action_persistence.dedup.skipped",
      data: expect.objectContaining({
        turnId: "turn_action_dedup",
        classification: "concrete_action",
        description_excerpt: candidateDescription,
        reason: "embedding_dedup",
        similarity: 1,
      }),
    });
    expect(events).toContainEqual({
      event: "extraction.actions.rejected",
      data: {
        turnId: "turn_action_dedup",
        classification: "concrete_action",
        description_excerpt: candidateDescription,
        reason: "embedding_dedup",
      },
    });
  });

  it("closes an existing active action when a terminal emission embedding-matches it", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const previousStreamEntryId = createStreamEntryId();
    const audience = createEntityId();
    const actor = createEntityId();
    const existingDescription = "deploy the fix";
    const terminalDescription = "deployed the fix";
    const existingAction = makeActionRecord({
      description: existingDescription,
      actor,
      audience_entity_id: audience,
      provenance_stream_entry_ids: [previousStreamEntryId],
      state: "committed_to_do",
      committed_at: 1_000,
    });
    const repository = makeActionRepository([existingAction]);
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [existingDescription, [1, 0]],
        [terminalDescription, [1, 0]],
      ]),
    );
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description: terminalDescription,
            actor,
            state: "completed",
            audience_entity_id: audience,
            evidence_stream_entry_ids: [currentUserStreamEntryId],
            confidence: 0.94,
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: repository,
      embeddingClient,
      clock: new FixedClock(2_000),
      turnId: "turn_action_terminal_close",
      tracer: {
        enabled: true,
        includePayloads: true,
        emit: (event, data) => events.push({ event, data }),
      },
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      userMessage: "X deployed the fix.",
      audienceEntityId: audience,
    });

    expect(records).toEqual([]);
    expect(repository.add).not.toHaveBeenCalled();
    expect(repository.update).toHaveBeenCalledOnce();
    expect(repository.records).toHaveLength(1);
    expect(repository.records[0]).toMatchObject({
      id: existingAction.id,
      description: existingDescription,
      actor,
      audience_entity_id: audience,
      state: "completed",
      confidence: 0.94,
      provenance_stream_entry_ids: [previousStreamEntryId, currentUserStreamEntryId],
      updated_at: 2_000,
      committed_at: 1_000,
      completed_at: 2_000,
    });
    expect(events).toContainEqual({
      event: "action_state.transitioned",
      data: expect.objectContaining({
        turnId: "turn_action_terminal_close",
        action_id: existingAction.id,
        candidate_index: 0,
        terminal_state: "completed",
        description_excerpt: terminalDescription,
        similarity: 1,
      }),
    });
    expect(events).toContainEqual({
      event: "extraction.actions.completed",
      data: expect.objectContaining({
        turnId: "turn_action_terminal_close",
        persisted_count: 0,
        skipped_count: 0,
        actions_closed_by_terminal_emission: 1,
      }),
    });
  });

  it("does not suppress a new candidate with an unknown-state existing action", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const audience = createEntityId();
    const description = "review the onboarding checklist";
    const repository = makeActionRepository([
      makeActionRecord({
        description,
        actor: "user",
        audience_entity_id: audience,
        state: "unknown",
        committed_at: null,
        unknown_at: 1_000,
      }),
    ]);
    const embeddingClient = new ScriptedEmbeddingClient(new Map([[description, [1, 0]]]));
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description,
            state: "committed_to_do",
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: repository,
      embeddingClient,
      clock: new FixedClock(2_000),
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      audienceEntityId: audience,
    });

    expect(records.map((record) => record.description)).toEqual([description]);
    expect(repository.add).toHaveBeenCalledOnce();
  });

  it("closes an existing unknown-state action when a terminal emission embedding-matches it", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const previousStreamEntryId = createStreamEntryId();
    const audience = createEntityId();
    const existingDescription = "confirm the release window";
    const terminalDescription = "confirmed the release window";
    const existingAction = makeActionRecord({
      description: existingDescription,
      actor: "user",
      audience_entity_id: audience,
      state: "unknown",
      committed_at: null,
      unknown_at: 1_000,
      provenance_stream_entry_ids: [previousStreamEntryId],
    });
    const repository = makeActionRepository([existingAction]);
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [existingDescription, [1, 0]],
        [terminalDescription, [1, 0]],
      ]),
    );
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description: terminalDescription,
            state: "completed",
            audience_entity_id: audience,
            evidence_stream_entry_ids: [currentUserStreamEntryId],
            confidence: 0.93,
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: repository,
      embeddingClient,
      clock: new FixedClock(2_000),
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      userMessage: "I confirmed the release window.",
      audienceEntityId: audience,
    });

    expect(records).toEqual([]);
    expect(repository.add).not.toHaveBeenCalled();
    expect(repository.update).toHaveBeenCalledOnce();
    expect(repository.records).toHaveLength(1);
    expect(repository.records[0]).toMatchObject({
      id: existingAction.id,
      description: existingDescription,
      state: "completed",
      confidence: 0.93,
      provenance_stream_entry_ids: [previousStreamEntryId, currentUserStreamEntryId],
      unknown_at: 1_000,
      completed_at: 2_000,
      updated_at: 2_000,
    });
  });

  it("does not dedup same-description actions with different goal ids", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const audience = createEntityId();
    const existingGoalId = createGoalId();
    const candidateGoalId = createGoalId();
    const description = "send the rollout update";
    const repository = makeActionRepository([
      makeActionRecord({
        description,
        actor: "user",
        audience_entity_id: audience,
        goal_id: existingGoalId,
      }),
    ]);
    const embeddingClient = new ScriptedEmbeddingClient(new Map([[description, [1, 0]]]));
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description,
            state: "committed_to_do",
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: repository,
      embeddingClient,
      clock: new FixedClock(2_000),
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      audienceEntityId: audience,
      goalId: candidateGoalId,
    });

    expect(records.map((record) => record.description)).toEqual([description]);
    expect(repository.add).toHaveBeenCalledOnce();
  });

  it("does not dedup same-description actions with different open question ids", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const audience = createEntityId();
    const existingOpenQuestionId = createOpenQuestionId();
    const candidateOpenQuestionId = createOpenQuestionId();
    const description = "send the rollout update";
    const repository = makeActionRepository([
      makeActionRecord({
        description,
        actor: "user",
        audience_entity_id: audience,
        open_question_id: existingOpenQuestionId,
      }),
    ]);
    const embeddingClient = new ScriptedEmbeddingClient(new Map([[description, [1, 0]]]));
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description,
            state: "committed_to_do",
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: repository,
      embeddingClient,
      clock: new FixedClock(2_000),
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      audienceEntityId: audience,
      openQuestionId: candidateOpenQuestionId,
    });

    expect(records.map((record) => record.description)).toEqual([description]);
    expect(repository.add).toHaveBeenCalledOnce();
  });

  it("dedups near-identical concrete actions within the same batch", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const firstDescription = "review the billing PR";
    const secondDescription = "review the billing pull request";
    const repository = makeActionRepository();
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [firstDescription, [1, 0]],
        [secondDescription, [1, 0]],
      ]),
    );
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description: firstDescription,
            state: "committed_to_do",
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
          {
            description: secondDescription,
            state: "committed_to_do",
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: repository,
      embeddingClient,
      clock: new FixedClock(2_000),
    });

    const records = await extractor.extract(makeExtractorInput(currentUserStreamEntryId));

    expect(records.map((record) => record.description)).toEqual([firstDescription]);
    expect(repository.add).toHaveBeenCalledOnce();
  });

  it.each([
    {
      label: "actor",
      existing: { actor: "borg" as const, audience_entity_id: null },
      inputAudience: null,
    },
    {
      label: "audience",
      existing: { actor: "user" as const, audience_entity_id: createEntityId() },
      inputAudience: createEntityId(),
    },
  ])("does not dedup against a different $label axis", async ({ existing, inputAudience }) => {
    const currentUserStreamEntryId = createStreamEntryId();
    const description = "send the project status note";
    const repository = makeActionRepository([
      makeActionRecord({
        description,
        actor: existing.actor,
        audience_entity_id: existing.audience_entity_id,
      }),
    ]);
    const embeddingClient = new ScriptedEmbeddingClient(new Map([[description, [1, 0]]]));
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description,
            actor: "user",
            state: "committed_to_do",
            audience_entity_id: inputAudience,
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: repository,
      embeddingClient,
      clock: new FixedClock(2_000),
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      audienceEntityId: inputAudience,
    });

    expect(records.map((record) => record.description)).toEqual([description]);
    expect(repository.add).toHaveBeenCalledOnce();
  });

  it("fails open when embedding dedup is unavailable", async () => {
    const currentUserStreamEntryId = createStreamEntryId();
    const audience = createEntityId();
    const description = "send the design review note";
    const repository = makeActionRepository([
      makeActionRecord({
        description,
        actor: "user",
        audience_entity_id: audience,
      }),
    ]);
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const llm = new FakeLLMClient({
      responses: [
        actionStateResponse([
          {
            description,
            state: "committed_to_do",
            evidence_stream_entry_ids: [currentUserStreamEntryId],
          },
        ]),
      ],
    });
    const extractor = new ActionStateExtractor({
      llmClient: llm,
      model: "haiku",
      actionRepository: repository,
      embeddingClient: new ScriptedEmbeddingClient(new Map(), true),
      clock: new FixedClock(2_000),
      turnId: "turn_action_dedup_degraded",
      tracer: {
        enabled: true,
        includePayloads: true,
        emit: (event, data) => events.push({ event, data }),
      },
    });

    const records = await extractor.extract({
      ...makeExtractorInput(currentUserStreamEntryId),
      audienceEntityId: audience,
    });

    expect(records.map((record) => record.description)).toEqual([description]);
    expect(repository.add).toHaveBeenCalledOnce();
    expect(events).toContainEqual({
      event: "action_persistence.dedup.degraded",
      data: expect.objectContaining({
        turnId: "turn_action_dedup_degraded",
        reason: "active_action_embedding_failed",
        error: "embedding offline",
      }),
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
