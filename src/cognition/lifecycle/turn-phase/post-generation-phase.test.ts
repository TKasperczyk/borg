import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "../../../config/index.js";
import { sharedStateMigrations } from "../../../memory/shared-state/index.js";
import { SharedStateRepository } from "../../../memory/shared-state/repository.js";
import {
  TrainOfThoughtRepository,
  trainOfThoughtMigrations,
} from "../../../memory/train-of-thought/index.js";
import type {
  ActionRecord,
  ActionRecordListFilter,
  ActionRecordPatch,
} from "../../../memory/actions/index.js";
import { createWorkingMemory } from "../../../memory/working/index.js";
import { FakeLLMClient } from "../../../llm/test-support/fake-client.js";
import type { LLMCompleteOptions } from "../../../llm/index.js";
import { openDatabase } from "../../../storage/sqlite/index.js";
import { FixedClock } from "../../../util/clock.js";
import {
  createActionId,
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type EntityId,
  type SessionId,
} from "../../../util/ids.js";
import { StreamReader, StreamWriter, type StreamResponseTo } from "../../../stream/index.js";
import { SHARED_STATE_TOOL_NAME } from "../../shared-state/constants.js";
import type { PerceptionResult } from "../../types.js";
import { compileSharedStateArtifactForEvidenceLedger } from "./retrieval-phase.js";
import { runPostGenerationPhase } from "./post-generation-phase.js";
import type { TurnPhaseCoordinatorOptions } from "./types.js";

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
  const cleanup: Array<() => void> = [];

  afterEach(() => {
    while (cleanup.length > 0) {
      cleanup.pop()?.();
    }
  });

  it("upserts EmitContinueThought into the singleton and appends only a minimal marker", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-post-generation-continue-thought-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: trainOfThoughtMigrations,
    });
    cleanup.push(() => db.close());
    const sessionId = createSessionId();
    const turnId = "turn-continue-thought";
    const clock = new FixedClock(50_000);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId,
      clock,
    });
    cleanup.push(() => writer.close());
    const trainOfThoughtRepository = new TrainOfThoughtRepository({ db, clock });
    const selfEntityId = createEntityId();
    const carriedText = "I should resume with the unresolved continuity question.";
    const actionRepository = makeActionRepository([]);
    const options = {
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
      },
      clock,
      tracer: {
        enabled: false,
        includePayloads: false,
        emit: vi.fn(),
      },
      entityRepository: {
        resolve: () => selfEntityId,
        get: () => null,
      },
      relationalSlotRepository: {
        list: () => [],
      },
      actionRepository,
      goalsRepository: {
        list: () => [],
      },
      commitmentRepository: {
        list: () => [],
      },
      openQuestionsRepository: {
        list: () => [],
        findByHandles: () => [],
      },
      attachmentRepository: {
        isActiveForStreamEntry: () => true,
      },
      sharedStateRepository: {
        get: () => null,
      },
      trainOfThoughtRepository,
      createStreamReader: (readerSessionId: SessionId) =>
        new StreamReader({ dataDir: tempDir, sessionId: readerSessionId }),
      turnActionCoordinator: {
        run: vi.fn(async () => ({
          actionResult: {
            response: "",
            tool_calls: [],
            intents: [],
            workingMemory: createWorkingMemory(sessionId, 50_000),
            pending_action_merge_count: 0,
          },
          actionEmission: {
            kind: "continue_thought",
            text: carriedText,
          },
          deliberation: {
            path: "system_1",
            thoughts: [],
            usage: { input_tokens: 0, output_tokens: 0, stop_reason: null },
            retrievedEpisodes: [],
            referencedEpisodeIds: [],
          },
        })),
      },
      discourseStateService: {
        appendClosurePressureHistory: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
        setStopState: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
        markClosureLoopNamed: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
      },
      turnReflectionCoordinator: { run: vi.fn(async () => undefined) },
      turnActionStateService: { closeBorgSelfPerformedActions: vi.fn(async () => undefined) },
      correctivePreferenceTurnService: { persistCommitment: vi.fn(async () => undefined) },
      streamIngestionCoordinator: undefined,
    } as unknown as TurnPhaseCoordinatorOptions;
    const perception = {
      entities: [],
      mode: "idle",
      affectiveSignal: {
        valence: 0,
        arousal: 0,
        dominant_emotion: null,
      },
      temporalCue: null,
    } satisfies PerceptionResult;

    const result = await runPostGenerationPhase({
      options,
      appendHookFailureEvent: vi.fn(async () => undefined),
      llmClient: new FakeLLMClient({ responses: [] }),
      sessionId,
      turnId,
      turnInput: {
        userMessage: "",
        audience: "self",
        origin: "autonomous",
      },
      streamWriter: writer,
      lifecycleTracker: {
        trackPendingActionMerges: vi.fn(),
        trackReflectionEffects: vi.fn(),
      } as never,
      cognitionInput: carriedText,
      perception,
      workingMemory: createWorkingMemory(sessionId, 50_000),
      workingMood: null as never,
      persistedPerceptionEntry: null as never,
      persistedUserEntry: undefined,
      persistedUserEntryId: undefined,
      correctiveCommitment: null,
      correctiveCommitmentSupersession: null,
      correctiveCommitmentRetirement: null,
      deliberation: {
        path: "system_1",
        thoughts: [],
        usage: null,
        retrievedEpisodes: [],
        referencedEpisodeIds: [],
      } as never,
      retrievalPhase: {
        applicableCommitments: [],
        actionApplicableCommitments: [],
        retrievedEpisodes: [],
        selfSnapshot: null,
        retrieval: { confidence: 1 },
        executiveFocusWithStep: null,
        selectedSkill: null,
        proceduralContext: null,
        evidenceLedgerContext: { ledger: null },
      } as never,
      origin: "autonomous",
      autonomyTrigger: undefined,
      closureLoopCurrentUserAct: null,
      audienceEntityId: null,
      audienceIsGroup: false,
      senderEntityId: null,
      socialInteractionEntityId: null,
      pendingSocialAttribution: null,
      suppressionSet: null as never,
      isUserTurn: false,
      currentTurnFrameAnomaly: null,
      closureLoopAssessment: null,
      activeParticipants: [],
    });

    expect(result.response).toBe("");
    expect(result.emitted).toBe(false);
    expect(result.emission).toMatchObject({
      kind: "continue_thought",
      markerEntryId: expect.any(String),
    });
    expect(trainOfThoughtRepository.get()).toMatchObject({
      text: carriedText,
      self_entity_id: selfEntityId,
      disclosure_class: "self_private",
    });
    const marker = new StreamReader({ dataDir: tempDir, sessionId })
      .tail(5)
      .find(
        (entry) =>
          entry.kind === "internal_event" &&
          typeof entry.content === "object" &&
          entry.content !== null &&
          "kind" in entry.content &&
          entry.content.kind === "train_of_thought_continued",
      );

    expect(marker?.content).toMatchObject({
      kind: "train_of_thought_continued",
      self_entity_id: selfEntityId,
      text_length: carriedText.length,
    });
    expect(JSON.stringify(marker?.content)).not.toContain(carriedText);
  });

  it("compiles shared state from the persisted assistant response before reflection", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-post-generation-shared-state-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const sessionId = createSessionId();
    const turnId = "turn-post-response-shared-state";
    const clock = new FixedClock(50_000);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId,
      clock,
    });
    cleanup.push(() => writer.close());
    const priorSourceEntry = await writer.append({
      kind: "user_msg",
      turn_id: "turn-merge-discussion",
      content: "We are discussing the merge order.",
    });
    const currentUserEntry = await writer.append({
      kind: "user_msg",
      turn_id: turnId,
      content: "Please choose the merge order.",
    });
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const speakerEntityId = createEntityId();
    const preAnswerLlm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 12,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_shared_state_pre",
              name: SHARED_STATE_TOOL_NAME,
              input: { operations: [] },
            },
          ],
        },
      ],
    });

    sharedStateRepository.upsert(
      audienceEntityId,
      [
        {
          type: "add",
          state_key: "decision.merge_order",
          kind: "live",
          text: "The merge order is under discussion.",
          provenance_stream_entry_ids: [priorSourceEntry.id],
          last_updated_stream_entry_ids: [priorSourceEntry.id],
          created_at: 40_000,
          last_updated_at: 40_000,
        },
      ],
      {
        now: 40_000,
        lastCompiledStreamEntryId: priorSourceEntry.id,
      },
    );

    const makeBaseOptions = (llmClient: FakeLLMClient) =>
      ({
        config: {
          ...DEFAULT_CONFIG,
          dataDir: tempDir,
          generation: {
            ...DEFAULT_CONFIG.generation,
            evidenceLedger: {
              ...DEFAULT_CONFIG.generation.evidenceLedger,
              enabled: true,
              decisionArtifact: {
                ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
                compilerPrefilter: {
                  enabled: false,
                },
              },
            },
          },
        },
        sharedStateRepository,
        llmFactory: () => llmClient,
        clock,
        tracer: {
          enabled: false,
          includePayloads: false,
          emit: vi.fn(),
        },
        entityRepository: {
          resolve: () => selfEntityId,
          get: (entityId: EntityId) =>
            entityId === audienceEntityId
              ? { id: audienceEntityId, canonical_name: "Project Crew", kind: "group" }
              : null,
        },
        relationalSlotRepository: {
          list: () => [],
        },
        actionRepository: makeActionRepository([]),
        goalsRepository: {
          list: () => [],
        },
        commitmentRepository: {
          list: () => [],
        },
        openQuestionsRepository: {
          list: () => [],
          findByHandles: () => [],
        },
        attachmentRepository: {
          isActiveForStreamEntry: () => true,
        },
        createStreamReader: (readerSessionId: SessionId) =>
          new StreamReader({ dataDir: tempDir, sessionId: readerSessionId }),
      }) as unknown as TurnPhaseCoordinatorOptions;

    const perception = {
      entities: [],
      mode: "problem_solving",
      affectiveSignal: {
        valence: 0,
        arousal: 0,
        dominant_emotion: null,
      },
      temporalCue: null,
    } satisfies PerceptionResult;

    await compileSharedStateArtifactForEvidenceLedger({
      options: makeBaseOptions(preAnswerLlm),
      input: {
        sessionId,
        turnId,
        audienceEntityId,
        currentUserMessage: String(currentUserEntry.content),
        currentUserEntry,
        globalTurnCounter: 7,
        workingMemory: {
          turn_counter: 7,
        } as never,
        applicableCommitments: [],
        retrievedEvidence: [],
        retrievedEpisodes: [],
        openQuestions: [],
        pendingCorrections: [],
        activeParticipants: [
          { entityId: speakerEntityId, displayName: "Operator", role: "speaker" },
        ],
        participantRoster: null,
        isUserTurn: true,
        perception,
        closureLoopAssessment: null,
      },
      ledger: {
        sections: [
          {
            id: "current_user_message",
            label: "1. Current User Message",
            entries: [
              {
                id: `current_user_message:${currentUserEntry.id}`,
                source_type: "current_user_message",
                session_scope: "current_session",
                actor: "user",
                trust_rank: 1,
                text: String(currentUserEntry.content),
                citations: [currentUserEntry.id],
                stream_index: currentUserEntry.entry_index,
              },
            ],
          },
        ],
        transcriptIncluded: true,
        transcriptCompacted: false,
        originalTranscriptTokenEstimate: 0,
        compactedTranscriptEntryCount: 1,
        rawPreservedUserTranscriptEntryCount: 1,
        estimatedTokens: 0,
      },
      promptVisibleLedger: "",
    });

    expect(preAnswerLlm.requests).toHaveLength(1);
    expect(sharedStateRepository.get(audienceEntityId)?.last_compiled_stream_entry_id).toBe(
      currentUserEntry.id,
    );

    const postResponseLlm = new FakeLLMClient({
      responses: [
        (request: LLMCompleteOptions) => {
          const payload = JSON.parse(String(request.messages[0]?.content ?? "{}")) as {
            compile_pass?: string;
            assistant_response?: { stream_entry_id?: string; text?: string } | null;
          };
          const assistantStreamEntryId = payload.assistant_response?.stream_entry_id;

          return {
            text: "",
            input_tokens: 20,
            output_tokens: 10,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_shared_state_post",
                name: SHARED_STATE_TOOL_NAME,
                input: {
                  operations: [
                    {
                      type: "supersede",
                      id: sharedStateRepository
                        .get(audienceEntityId)
                        ?.entries.find((entry) => entry.state_key === "decision.merge_order")?.id,
                      replacement: {
                        state_key: "decision.merge_order",
                        kind: "locked",
                        text: "Backend merges first, then frontend.",
                        owner_entity_id: audienceEntityId,
                        source_stream_entry_ids: [assistantStreamEntryId],
                      },
                    },
                  ],
                },
              },
            ],
          };
        },
      ],
    });
    const postOptions = {
      ...makeBaseOptions(postResponseLlm),
      turnActionCoordinator: {
        run: vi.fn(async () => ({
          actionResult: {
            response: "Agreed: backend merges first, then frontend.",
            tool_calls: [],
            intents: [],
            workingMemory: createWorkingMemory(sessionId, 50_000),
            pending_action_merge_count: 0,
          },
          actionEmission: { kind: "message" },
          deliberation: {
            path: "system_1",
            thoughts: [],
            usage: { input_tokens: 0, output_tokens: 0, stop_reason: null },
            retrievedEpisodes: [],
            referencedEpisodeIds: [],
          },
        })),
      },
      discourseStateService: {
        appendClosurePressureHistory: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
        setStopState: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
        markClosureLoopNamed: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
      },
      turnReflectionCoordinator: { run: vi.fn(async () => undefined) },
      turnActionStateService: { closeBorgSelfPerformedActions: vi.fn(async () => undefined) },
      correctivePreferenceTurnService: { persistCommitment: vi.fn(async () => undefined) },
      streamIngestionCoordinator: undefined,
    } as unknown as TurnPhaseCoordinatorOptions;

    const result = await runPostGenerationPhase({
      options: postOptions,
      appendHookFailureEvent: vi.fn(async () => undefined),
      llmClient: new FakeLLMClient({ responses: [] }),
      sessionId,
      turnId,
      turnInput: {
        userMessage: String(currentUserEntry.content),
        globalTurnCounter: 7,
      },
      streamWriter: writer,
      lifecycleTracker: {
        trackPendingActionMerges: vi.fn(),
        trackReflectionEffects: vi.fn(),
      } as never,
      cognitionInput: String(currentUserEntry.content),
      perception,
      workingMemory: createWorkingMemory(sessionId, 50_000),
      workingMood: null as never,
      persistedPerceptionEntry: null as never,
      persistedUserEntry: currentUserEntry,
      persistedUserEntryId: currentUserEntry.id,
      correctiveCommitment: null,
      correctiveCommitmentSupersession: null,
      correctiveCommitmentRetirement: null,
      deliberation: {
        path: "system_1",
        thoughts: [],
        usage: null,
        retrievedEpisodes: [],
        referencedEpisodeIds: [],
      } as never,
      retrievalPhase: {
        applicableCommitments: [],
        actionApplicableCommitments: [],
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
      audienceEntityId,
      audienceIsGroup: true,
      senderEntityId: speakerEntityId,
      socialInteractionEntityId: null,
      pendingSocialAttribution: null,
      suppressionSet: null as never,
      isUserTurn: true,
      currentTurnFrameAnomaly: null,
      closureLoopAssessment: null,
      activeParticipants: [{ entityId: speakerEntityId, displayName: "Operator", role: "speaker" }],
    });

    const artifact = sharedStateRepository.get(audienceEntityId);
    const lockedDecision = artifact?.entries.find(
      (entry) => entry.state_key === "decision.merge_order" && entry.kind === "locked",
    );
    const supersededLiveEntry = artifact?.entries.find(
      (entry) => entry.state_key === "decision.merge_order" && entry.kind === "live",
    );

    expect(result.agentMessageId).toBeDefined();
    expect(postResponseLlm.requests).toHaveLength(1);
    expect(lockedDecision).toMatchObject({
      kind: "locked",
      text: "Backend merges first, then frontend.",
      provenance_stream_entry_ids: [result.agentMessageId],
      last_updated_stream_entry_ids: [result.agentMessageId],
    });
    expect(supersededLiveEntry?.superseded_by_id).toBe(lockedDecision?.id);
    expect(artifact?.last_compiled_stream_entry_id).toBe(result.agentMessageId);

    const postResponsePayload = JSON.parse(
      String(postResponseLlm.requests[0]?.messages[0]?.content ?? "{}"),
    ) as {
      compile_pass?: string;
      assistant_response?: { stream_entry_id?: string; text?: string } | null;
      source_trust?: { off_limits_source_stream_entry_ids?: string[] };
    };

    expect(postResponsePayload.compile_pass).toBe("post_response");
    expect(postResponsePayload.assistant_response).toEqual({
      stream_entry_id: result.agentMessageId,
      text: "Agreed: backend merges first, then frontend.",
    });
    expect(postResponsePayload.source_trust?.off_limits_source_stream_entry_ids).toContain(
      currentUserEntry.id,
    );
  });

  it("does not compile shared state from a directed outbound message that was not transported", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-post-generation-outbound-shared-state-"));
    cleanup.push(() => rmSync(tempDir, { recursive: true, force: true }));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: sharedStateMigrations,
    });
    cleanup.push(() => db.close());
    const sessionId = createSessionId();
    const turnId = "turn-post-response-outbound-not-transported";
    const clock = new FixedClock(60_000);
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId,
      clock,
    });
    cleanup.push(() => writer.close());
    const sharedStateRepository = new SharedStateRepository({ db, clock });
    const audienceEntityId = createEntityId();
    const selfEntityId = createEntityId();
    const postResponseLlm = new FakeLLMClient({
      responses: [
        (request: LLMCompleteOptions) => {
          const payload = JSON.parse(String(request.messages[0]?.content ?? "{}")) as {
            assistant_response?: { stream_entry_id?: string } | null;
          };

          return {
            text: "",
            input_tokens: 20,
            output_tokens: 10,
            stop_reason: "tool_use",
            tool_calls: [
              {
                id: "toolu_shared_state_outbound",
                name: SHARED_STATE_TOOL_NAME,
                input: {
                  operations: [
                    {
                      type: "add",
                      state_key: "decision.outbound_only",
                      new_key_reason: "The outbound-only response would establish a new decision.",
                      kind: "locked",
                      text: "The outbound-only decision is locked.",
                      owner_entity_id: audienceEntityId,
                      source_stream_entry_ids: [payload.assistant_response?.stream_entry_id],
                    },
                  ],
                },
              },
            ],
          };
        },
      ],
    });
    const workingMemory = createWorkingMemory(sessionId, 60_000);
    const deliver = vi.fn(
      async ({
        streamWriter,
        message,
      }: {
        streamWriter: StreamWriter;
        message: {
          content: string;
          streamInput: Omit<Parameters<StreamWriter["append"]>[0], "kind" | "content">;
        };
      }) => ({
        status: "transport_failed" as const,
        streamEntry: await streamWriter.append({
          kind: "agent_msg",
          content: message.content,
          ...message.streamInput,
        }),
        sourceType: "demo" as const,
        error: "transport failed",
      }),
    );

    await runPostGenerationPhase({
      options: {
        config: {
          ...DEFAULT_CONFIG,
          dataDir: tempDir,
          generation: {
            ...DEFAULT_CONFIG.generation,
            evidenceLedger: {
              ...DEFAULT_CONFIG.generation.evidenceLedger,
              enabled: true,
              decisionArtifact: {
                ...DEFAULT_CONFIG.generation.evidenceLedger.decisionArtifact,
                compilerPrefilter: {
                  enabled: false,
                },
              },
            },
          },
        },
        sharedStateRepository,
        llmFactory: () => postResponseLlm,
        clock,
        tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
        entityRepository: {
          resolve: () => selfEntityId,
          get: (entityId: EntityId) =>
            entityId === audienceEntityId
              ? { id: audienceEntityId, canonical_name: "Outbound Crew", kind: "group" }
              : null,
        },
        relationalSlotRepository: { list: () => [] },
        actionRepository: makeActionRepository([]),
        goalsRepository: { list: () => [] },
        commitmentRepository: { list: () => [] },
        openQuestionsRepository: {
          list: () => [],
          findByHandles: () => [],
        },
        attachmentRepository: {
          get: () => null,
          isActiveForStreamEntry: () => true,
        },
        createStreamReader: (readerSessionId: SessionId) =>
          new StreamReader({ dataDir: tempDir, sessionId: readerSessionId }),
        sessionsRepository: {
          get: () => ({
            session_id: sessionId,
            source_type: "demo",
            audience_label: "Outbound Crew",
            audience_entity_id: audienceEntityId,
          }),
        },
        outboundDelivery: { deliver },
        turnActionCoordinator: {
          run: vi.fn(async () => ({
            actionResult: {
              response: "Outbound-only decision: locked.",
              tool_calls: [],
              intents: [],
              workingMemory,
              pending_action_merge_count: 0,
            },
            actionEmission: { kind: "message" },
            deliberation: {
              path: "system_1",
              thoughts: [],
              usage: { input_tokens: 0, output_tokens: 0, stop_reason: null },
              retrievedEpisodes: [],
              referencedEpisodeIds: [],
            },
          })),
        },
        discourseStateService: {
          appendClosurePressureHistory: vi.fn(
            (arg: { workingMemory: unknown }) => arg.workingMemory,
          ),
          setStopState: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
          markClosureLoopNamed: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
        },
        turnReflectionCoordinator: { run: vi.fn(async () => undefined) },
        turnActionStateService: { closeBorgSelfPerformedActions: vi.fn(async () => undefined) },
        correctivePreferenceTurnService: { persistCommitment: vi.fn(async () => undefined) },
        streamIngestionCoordinator: undefined,
      } as unknown as TurnPhaseCoordinatorOptions,
      appendHookFailureEvent: vi.fn(async () => undefined),
      llmClient: new FakeLLMClient({ responses: [] }),
      sessionId,
      turnId,
      turnInput: {
        userMessage: "Directed outbound instruction",
        origin: "directed_outbound",
      },
      streamWriter: writer,
      lifecycleTracker: {
        trackPendingActionMerges: vi.fn(),
        trackReflectionEffects: vi.fn(),
      } as never,
      cognitionInput: "Directed outbound instruction",
      perception: {
        mode: "problem_solving",
        entities: [],
      } as never,
      workingMemory,
      workingMood: null as never,
      persistedPerceptionEntry: null as never,
      correctiveCommitment: null,
      correctiveCommitmentSupersession: null,
      correctiveCommitmentRetirement: null,
      deliberation: {
        path: "system_1",
        thoughts: [],
        usage: null,
        retrievedEpisodes: [],
        referencedEpisodeIds: [],
      } as never,
      retrievalPhase: {
        applicableCommitments: [],
        actionApplicableCommitments: [],
        retrievedEpisodes: [],
        selfSnapshot: null,
        retrieval: { confidence: 1, evidence: [], open_questions: [] },
        executiveFocusWithStep: null,
        selectedSkill: null,
        proceduralContext: null,
        evidenceLedgerContext: { ledger: null },
        pendingCorrections: [],
        retrievedSemantic: null,
        participantRoster: null,
      } as never,
      origin: "directed_outbound",
      autonomyTrigger: undefined,
      closureLoopCurrentUserAct: null,
      audienceEntityId,
      audienceIsGroup: true,
      senderEntityId: null,
      socialInteractionEntityId: null,
      pendingSocialAttribution: null,
      suppressionSet: null as never,
      isUserTurn: false,
      currentTurnFrameAnomaly: null,
      closureLoopAssessment: null,
      activeParticipants: [],
    });

    expect(deliver).toHaveBeenCalledTimes(1);
    expect(postResponseLlm.requests).toHaveLength(0);
    expect(sharedStateRepository.get(audienceEntityId)).toBeNull();
  });

  it("starts live ingestion with an explicit answered window for catch-up terminal messages", async () => {
    const sessionId = createSessionId();
    const turnId = "turn_post_generation_answered_window";
    const sourceEntryIds = [createStreamEntryId(), createStreamEntryId()];
    const responseTo: StreamResponseTo = {
      kind: "stream_backlog",
      from_cursor_exclusive: null,
      through_cursor_inclusive: {
        ts: 2_000,
        entryId: sourceEntryIds[1]!,
      },
      source_entry_ids: sourceEntryIds,
      count: sourceEntryIds.length,
    };
    const workingMemory = createWorkingMemory(sessionId, 1_000);
    const agentEntry = {
      id: createStreamEntryId(),
      kind: "agent_msg",
      turn_id: turnId,
      turn_status: "active",
      content: "Caught up.",
      session_id: sessionId,
      timestamp: 3_000,
      reply_target_entity_id: null,
    };
    const ingest = vi.fn(async () => ({ ran: true, processedEntries: 3 }));
    const advanceThrough = vi.fn();

    await runPostGenerationPhase({
      options: {
        config: DEFAULT_CONFIG,
        clock: { now: () => 10_000 },
        actionRepository: makeActionRepository([]),
        tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
        chatResponseWatermarkCoordinator: { advanceThrough },
        streamIngestionCoordinator: { ingest },
        turnActionCoordinator: {
          run: vi.fn(async () => ({
            actionResult: {
              response: "Caught up.",
              tool_calls: [],
              intents: [],
              workingMemory,
              pending_action_merge_count: 0,
            },
            actionEmission: { kind: "message" },
            deliberation: {
              path: "system_1",
              thoughts: [],
              usage: { input_tokens: 0, output_tokens: 0, stop_reason: null },
              retrievedEpisodes: [],
              referencedEpisodeIds: [],
            },
          })),
        },
        discourseStateService: {
          appendClosurePressureHistory: vi.fn(
            (arg: { workingMemory: unknown }) => arg.workingMemory,
          ),
          setStopState: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
          markClosureLoopNamed: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
        },
        turnReflectionCoordinator: { run: vi.fn(async () => undefined) },
        turnActionStateService: { closeBorgSelfPerformedActions: vi.fn(async () => undefined) },
        correctivePreferenceTurnService: { persistCommitment: vi.fn(async () => undefined) },
      } as never,
      appendHookFailureEvent: vi.fn(async () => undefined),
      llmClient: new FakeLLMClient({ responses: [] }),
      sessionId,
      turnId,
      turnInput: { userMessage: "Caught-up batch" },
      streamWriter: { append: vi.fn(async () => agentEntry) } as never,
      lifecycleTracker: {
        trackPendingActionMerges: vi.fn(),
        trackReflectionEffects: vi.fn(),
      } as never,
      cognitionInput: "Caught-up batch",
      perception: { mode: "conversational", entities: [] } as never,
      workingMemory,
      workingMood: null as never,
      persistedPerceptionEntry: null as never,
      persistedUserEntryId: sourceEntryIds[0],
      sourceUserEntryIds: sourceEntryIds,
      correctiveCommitment: null,
      correctiveCommitmentSupersession: null,
      correctiveCommitmentRetirement: null,
      deliberation: {
        path: "system_1",
        thoughts: [],
        usage: null,
        retrievedEpisodes: [],
        referencedEpisodeIds: [],
      } as never,
      retrievalPhase: {
        applicableCommitments: [],
        actionApplicableCommitments: [],
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
      senderEntityId: null,
      socialInteractionEntityId: null,
      pendingSocialAttribution: null,
      suppressionSet: null as never,
      isUserTurn: true,
      currentTurnFrameAnomaly: null,
      responseTo,
    });

    expect(advanceThrough).toHaveBeenCalledWith(sessionId, responseTo.through_cursor_inclusive);
    expect(ingest).toHaveBeenCalledWith(sessionId, {
      answeredWindow: {
        responseTo,
        terminalCursor: {
          ts: agentEntry.timestamp,
          entryId: agentEntry.id,
        },
      },
    });
  });

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
            deliberation: {
              path: "system_1",
              thoughts: [],
              usage: {
                input_tokens: 0,
                output_tokens: 0,
                stop_reason: null,
              },
              retrievedEpisodes: [],
              referencedEpisodeIds: [],
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
      correctiveCommitmentRetirement: null,
      deliberation: {
        path: "system1",
        thoughts: [],
        usage: null,
        retrievedEpisodes: [],
        referencedEpisodeIds: [],
      } as never,
      retrievalPhase: {
        applicableCommitments: [],
        actionApplicableCommitments: [],
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
      currentTurnFrameAnomaly: null,
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
              no_output_categories: ["closure", "with_open_question"],
              primary_no_output_reason: "closure",
              structural_no_output_flags: ["with_open_question", "open_question_rendered"],
            },
            deliberation: {
              path: "system_1",
              thoughts: [],
              usage: {
                input_tokens: 0,
                output_tokens: 0,
                stop_reason: "suppressed",
              },
              retrievedEpisodes: [],
              referencedEpisodeIds: [],
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
      correctiveCommitmentRetirement: null,
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
        actionApplicableCommitments: [],
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
      currentTurnFrameAnomaly: null,
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
    expect(events).toContainEqual({
      event: "post_generation.rejected",
      data: expect.objectContaining({
        reason: "finalizer_no_output",
        no_output_categories: ["closure", "with_open_question"],
        primary_no_output_reason: "closure",
        structural_no_output_flags: ["with_open_question", "open_question_rendered"],
      }),
    });
  });
});

describe("runPostGenerationPhase outbound activity gate", () => {
  // A directed-outbound message turn records cross-session activity
  // (borg_replied / turn_completed) ONLY when the message actually transported.
  // OutboundDelivery always stream-appends the message to the target first, so a
  // composed-but-not-transported message exists in the stream -- but recording a
  // "reply" the operator's projection would render as "Borg replied to X" when
  // nothing reached X would inject a falsehood into Borg's continuity context.
  // Gate: post-generation-phase.ts shouldRecordActivity = outboundDelivery
  // undefined || status === "transported".
  type DeliveryStatus = "transported" | "composed_not_transported" | "transport_failed";

  async function runDirectedOutboundMessageTurn(deliveryStatus: DeliveryStatus) {
    const sessionId = createSessionId();
    const audienceId = createEntityId();
    const senderId = createEntityId();
    const turnId = "turn_outbound_activity_gate";
    const workingMemory = createWorkingMemory(sessionId, 2_000);
    const record = vi.fn();
    const agentEntry = {
      id: createStreamEntryId(),
      kind: "agent_msg",
      turn_id: turnId,
      turn_status: "active",
      content: "Note to the crew.",
      session_id: sessionId,
      timestamp: 10_000,
      reply_target_entity_id: null,
    };
    const targetSession = {
      session_id: sessionId,
      source_type: "demo",
      audience_label: "Project Crew",
      audience_entity_id: audienceId,
    };
    const deliver = vi.fn(async () => ({
      status: deliveryStatus,
      streamEntry: agentEntry,
      sourceType: "demo",
    }));

    await runPostGenerationPhase({
      options: {
        config: {
          ...DEFAULT_CONFIG,
          generation: {
            ...DEFAULT_CONFIG.generation,
            evidenceLedger: {
              ...DEFAULT_CONFIG.generation.evidenceLedger,
              enabled: false,
            },
          },
        },
        clock: { now: () => 10_000 },
        actionRepository: makeActionRepository([]),
        tracer: { enabled: false, includePayloads: false, emit: vi.fn() },
        activityRepository: { record },
        sessionsRepository: { get: () => targetSession },
        outboundDelivery: { deliver },
        turnActionCoordinator: {
          run: vi.fn(async () => ({
            actionResult: {
              response: "Note to the crew.",
              tool_calls: [],
              intents: [],
              workingMemory,
              pending_action_merge_count: 0,
            },
            actionEmission: { kind: "message" },
            deliberation: {
              path: "system_1",
              thoughts: [],
              usage: { input_tokens: 0, output_tokens: 0, stop_reason: null },
              retrievedEpisodes: [],
              referencedEpisodeIds: [],
            },
          })),
        },
        discourseStateService: {
          appendObservationMarker: vi.fn(),
          appendClosurePressureHistory: vi.fn(
            (arg: { workingMemory: unknown }) => arg.workingMemory,
          ),
          setStopState: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
          markClosureLoopNamed: vi.fn((arg: { workingMemory: unknown }) => arg.workingMemory),
        },
        turnReflectionCoordinator: { run: vi.fn(async () => undefined) },
        turnActionStateService: { closeBorgSelfPerformedActions: vi.fn(async () => undefined) },
        correctivePreferenceTurnService: { persistCommitment: vi.fn(async () => undefined) },
        streamIngestionCoordinator: undefined,
      } as never,
      appendHookFailureEvent: vi.fn(async () => undefined),
      llmClient: new FakeLLMClient({ responses: [] }),
      sessionId,
      turnId,
      turnInput: { userMessage: "Directed outbound instruction", origin: "directed_outbound" },
      streamWriter: { append: vi.fn(async () => agentEntry) } as never,
      lifecycleTracker: {
        trackPendingActionMerges: vi.fn(),
        trackReflectionEffects: vi.fn(),
      } as never,
      cognitionInput: "Directed outbound instruction",
      perception: { mode: "relational", entities: [] } as never,
      workingMemory,
      workingMood: null as never,
      persistedPerceptionEntry: null as never,
      persistedUserEntryId: createStreamEntryId(),
      correctiveCommitment: null,
      correctiveCommitmentSupersession: null,
      correctiveCommitmentRetirement: null,
      deliberation: {
        path: "system_1",
        thoughts: [],
        usage: null,
        retrievedEpisodes: [],
        referencedEpisodeIds: [],
      } as never,
      retrievalPhase: {
        applicableCommitments: [],
        actionApplicableCommitments: [],
        retrievedEpisodes: [],
        selfSnapshot: null,
        retrieval: { confidence: 1 },
        executiveFocusWithStep: null,
        selectedSkill: null,
        proceduralContext: null,
        evidenceLedgerContext: { ledger: null },
      } as never,
      origin: "directed_outbound",
      autonomyTrigger: undefined,
      closureLoopCurrentUserAct: null,
      audienceEntityId: audienceId,
      audienceIsGroup: true,
      senderEntityId: senderId,
      socialInteractionEntityId: null,
      pendingSocialAttribution: null,
      suppressionSet: null as never,
      isUserTurn: false,
      currentTurnFrameAnomaly: null,
    });

    return { record, deliver };
  }

  it.each(["composed_not_transported", "transport_failed"] as const)(
    "records zero activity events when a directed-outbound delivery is %s",
    async (status) => {
      const { record, deliver } = await runDirectedOutboundMessageTurn(status);

      expect(deliver).toHaveBeenCalledTimes(1);
      expect(record).not.toHaveBeenCalled();
    },
  );

  it("records borg_replied and turn_completed when the directed-outbound delivery transports", async () => {
    const { record } = await runDirectedOutboundMessageTurn("transported");

    expect(record).toHaveBeenCalledTimes(2);
    expect(record).toHaveBeenCalledWith(expect.objectContaining({ kind: "borg_replied" }));
    expect(record).toHaveBeenCalledWith(expect.objectContaining({ kind: "turn_completed" }));
  });
});
