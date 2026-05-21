import { describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "../../../config/index.js";
import type {
  ActionRecord,
  ActionRecordListFilter,
  ActionRecordPatch,
} from "../../../memory/actions/index.js";
import { createWorkingMemory } from "../../../memory/working/index.js";
import { FakeLLMClient } from "../../../llm/test-support/fake-client.js";
import {
  createActionId,
  createEntityId,
  createSessionId,
  createStreamEntryId,
} from "../../../util/ids.js";
import { runPostGenerationPhase } from "./post-generation-phase.js";

function makeAction(overrides: Partial<ActionRecord> = {}): ActionRecord {
  const nowMs = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createActionId(),
    description: overrides.description ?? "Finish the pending action",
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
    expired_at: overrides.expired_at ?? null,
    archived_at: overrides.archived_at ?? null,
    unknown_at: overrides.unknown_at ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
    session_scope: overrides.session_scope ?? null,
    session_anchor_id: overrides.session_anchor_id ?? null,
    last_referenced_at_ms: overrides.last_referenced_at_ms ?? nowMs,
    last_referenced_turn_counter: overrides.last_referenced_turn_counter ?? null,
  };
}

function makeActionRepository(records: ActionRecord[]) {
  return {
    list: vi.fn((filter: ActionRecordListFilter = {}) =>
      records.filter((record) => {
        if (filter.states !== undefined && !filter.states.includes(record.state)) {
          return false;
        }

        return true;
      }),
    ),
    get: vi.fn((id: ActionRecord["id"]) => records.find((record) => record.id === id) ?? null),
    update: vi.fn((id: ActionRecord["id"], patch: ActionRecordPatch) => {
      const index = records.findIndex((record) => record.id === id);
      const current = records[index];

      if (current === undefined) {
        throw new Error(`Unknown action ${id}`);
      }

      records[index] = { ...current, ...patch };
    }),
    records,
  };
}

describe("runPostGenerationPhase", () => {
  it("archives only inactive participant actions during the post-generation scan", async () => {
    const sessionId = createSessionId();
    const audienceId = createEntityId();
    const participantId = createEntityId();
    const freshParticipantId = createEntityId();
    const staleParticipantAction = makeAction({
      actor: participantId,
      audience_entity_id: null,
      state: "committed_to_do",
      last_referenced_turn_counter: 25,
    });
    const borgAction = makeAction({
      actor: "borg",
      last_referenced_turn_counter: 5,
    });
    const groupAction = makeAction({
      actor: audienceId,
      audience_entity_id: audienceId,
      last_referenced_turn_counter: 5,
    });
    const freshParticipantAction = makeAction({
      actor: freshParticipantId,
      last_referenced_turn_counter: 49,
    });
    const unknownAction = makeAction({
      actor: participantId,
      state: "unknown",
      committed_at: null,
      unknown_at: 1_000,
      last_referenced_turn_counter: 5,
    });
    const terminalAction = makeAction({
      actor: participantId,
      state: "completed",
      completed_at: 1_000,
      committed_at: null,
      last_referenced_turn_counter: 5,
    });
    const actionRepository = makeActionRepository([
      staleParticipantAction,
      borgAction,
      groupAction,
      freshParticipantAction,
      unknownAction,
      terminalAction,
    ]);
    const workingMemory = {
      ...createWorkingMemory(sessionId, 1_000),
      turn_counter: 50,
    };
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const observedEntry = {
      id: createStreamEntryId(),
      kind: "event",
      turn_id: "turn_post_generation_archive",
      content: "",
    };

    await runPostGenerationPhase({
      options: {
        config: DEFAULT_CONFIG,
        clock: { now: () => 10_000 },
        actionRepository,
        tracer: {
          enabled: true,
          includePayloads: true,
          emit: (event: string, data: Record<string, unknown>) => events.push({ event, data }),
        },
        turnActionCoordinator: {
          run: vi.fn(async () => ({
            actionResult: {
              response: "",
              tool_calls: [],
              intents: [],
              workingMemory,
              pending_action_merge_count: 0,
            },
            actionEmission: {
              kind: "observed",
              reason: "no_response_needed",
            },
          })),
        },
        discourseStateService: {
          appendObservationMarker: vi.fn(async () => observedEntry),
        },
        turnReflectionCoordinator: {
          run: vi.fn(async () => undefined),
        },
        turnActionStateService: {
          closeBorgSelfPerformedActions: vi.fn(async () => undefined),
        },
        correctivePreferenceTurnService: {
          persistCommitment: vi.fn(async () => undefined),
        },
        streamIngestionCoordinator: undefined,
      } as never,
      appendHookFailureEvent: vi.fn(async () => undefined),
      llmClient: new FakeLLMClient({ responses: [] }),
      sessionId,
      turnId: "turn_post_generation_archive",
      turnInput: {
        userMessage: "Observation turn",
      },
      streamWriter: {} as never,
      lifecycleTracker: {
        trackPendingActionMerges: vi.fn(),
        trackReflectionEffects: vi.fn(),
      } as never,
      cognitionInput: "Observation turn",
      perception: {
        mode: "conversational",
        entities: [],
      } as never,
      workingMemory,
      workingMood: null as never,
      persistedPerceptionEntry: null as never,
      persistedUserEntryId: createStreamEntryId(),
      correctiveCommitment: null,
      correctiveCommitmentSupersession: null,
      deliberation: {
        path: "system1",
        thoughts: [],
        usage: null,
        retrievedEpisodes: [],
        referencedEpisodeIds: [],
      } as never,
      retrievalPhase: {
        applicableCommitments: [],
        retrievedEpisodes: [],
        selfSnapshot: null,
        retrieval: { confidence: 1 },
        executiveFocusWithStep: null,
        selectedSkill: null,
        proceduralContext: null,
        evidenceLedgerContext: { ledger: null },
      } as never,
      origin: "user",
      autonomyTrigger: undefined,
      closureLoopCurrentUserAct: null,
      audienceEntityId: audienceId,
      audienceIsGroup: true,
      senderEntityId: participantId,
      socialInteractionEntityId: null,
      pendingSocialAttribution: null,
      suppressionSet: null as never,
      isUserTurn: true,
      frameAnomalyClassification: null,
    });

    expect(actionRepository.records).toContainEqual(
      expect.objectContaining({
        id: staleParticipantAction.id,
        state: "archived",
        archived_at: 10_000,
      }),
    );
    expect(actionRepository.records).toContainEqual(
      expect.objectContaining({ id: borgAction.id, state: "committed_to_do" }),
    );
    expect(actionRepository.records).toContainEqual(
      expect.objectContaining({ id: groupAction.id, state: "committed_to_do" }),
    );
    expect(actionRepository.records).toContainEqual(
      expect.objectContaining({ id: freshParticipantAction.id, state: "committed_to_do" }),
    );
    expect(actionRepository.records).toContainEqual(
      expect.objectContaining({ id: unknownAction.id, state: "unknown" }),
    );
    expect(actionRepository.records).toContainEqual(
      expect.objectContaining({ id: terminalAction.id, state: "completed" }),
    );
    expect(events).toContainEqual({
      event: "action_archive.completed",
      data: expect.objectContaining({
        action_id: staleParticipantAction.id,
        archive_after_turns: 20,
        inactive_turns: 25,
      }),
    });
    expect(events).toContainEqual({
      event: "action_archive_scan.completed",
      data: expect.objectContaining({
        scanned_count: 4,
        eligible_count: 1,
        archived_count: 1,
        archive_after_turns: 20,
        skipped_by_reason: {
          below_inactive_threshold: 1,
          borg_owned: 1,
          group_owned: 1,
        },
      }),
    });
  });

  it("runs the action archive scan on suppressed no-output turns", async () => {
    const sessionId = createSessionId();
    const participantId = createEntityId();
    const staleParticipantAction = makeAction({
      actor: participantId,
      state: "committed_to_do",
      last_referenced_turn_counter: 1,
    });
    const actionRepository = makeActionRepository([staleParticipantAction]);
    const workingMemory = {
      ...createWorkingMemory(sessionId, 1_000),
      turn_counter: 70,
    };
    const events: Array<{ event: string; data: Record<string, unknown> }> = [];
    const suppressedEntry = {
      id: createStreamEntryId(),
      kind: "event",
      turn_id: "turn_post_generation_suppressed_archive",
      content: "",
    };

    await runPostGenerationPhase({
      options: {
        config: DEFAULT_CONFIG,
        clock: { now: () => 20_000 },
        actionRepository,
        tracer: {
          enabled: true,
          includePayloads: true,
          emit: (event: string, data: Record<string, unknown>) => events.push({ event, data }),
        },
        turnActionCoordinator: {
          run: vi.fn(async () => ({
            actionResult: {
              response: "",
              tool_calls: [],
              intents: [],
              workingMemory,
              pending_action_merge_count: 0,
            },
            actionEmission: {
              kind: "suppressed",
              reason: "finalizer_no_output",
            },
          })),
        },
        discourseStateService: {
          appendSuppressionMarker: vi.fn(async () => suppressedEntry),
          applySuppressedEmissionState: vi.fn(({ workingMemory: memory }) => memory),
        },
        workingMemoryStore: {
          save: vi.fn(),
        },
        correctivePreferenceTurnService: {
          persistCommitment: vi.fn(async () => undefined),
        },
        streamIngestionCoordinator: undefined,
      } as never,
      appendHookFailureEvent: vi.fn(async () => undefined),
      llmClient: new FakeLLMClient({ responses: [] }),
      sessionId,
      turnId: "turn_post_generation_suppressed_archive",
      turnInput: {
        userMessage: "No output needed",
      },
      streamWriter: {} as never,
      lifecycleTracker: {
        trackPendingActionMerges: vi.fn(),
        trackReflectionEffects: vi.fn(),
      } as never,
      cognitionInput: "No output needed",
      perception: {
        mode: "conversational",
        entities: [],
      } as never,
      workingMemory,
      workingMood: null as never,
      persistedPerceptionEntry: null as never,
      persistedUserEntryId: createStreamEntryId(),
      correctiveCommitment: null,
      correctiveCommitmentSupersession: null,
      deliberation: {
        path: "system1",
        thoughts: [],
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          stop_reason: "suppressed",
        },
        retrievedEpisodes: [],
        referencedEpisodeIds: [],
      } as never,
      retrievalPhase: {
        applicableCommitments: [],
        retrievedEpisodes: [],
        selfSnapshot: null,
        retrieval: { confidence: 1 },
        executiveFocusWithStep: null,
        selectedSkill: null,
        proceduralContext: null,
        evidenceLedgerContext: { ledger: null },
      } as never,
      origin: "user",
      autonomyTrigger: undefined,
      closureLoopCurrentUserAct: null,
      audienceEntityId: null,
      audienceIsGroup: false,
      senderEntityId: participantId,
      socialInteractionEntityId: null,
      pendingSocialAttribution: null,
      suppressionSet: null as never,
      isUserTurn: true,
      frameAnomalyClassification: null,
    });

    expect(actionRepository.records).toContainEqual(
      expect.objectContaining({
        id: staleParticipantAction.id,
        state: "archived",
        archived_at: 20_000,
      }),
    );
    expect(events).toContainEqual({
      event: "action_archive_scan.completed",
      data: expect.objectContaining({
        scanned_count: 1,
        eligible_count: 1,
        archived_count: 1,
        skipped_by_reason: {},
      }),
    });
  });
});
