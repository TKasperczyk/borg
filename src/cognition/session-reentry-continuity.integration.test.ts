import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "../config/index.js";
import { FakeEmbeddingClient } from "../embeddings/index.js";
import type { LLMCompleteResult } from "../llm/index.js";
import { FakeLLMClient, createFakeEmitAnswerResponse } from "../llm/test-support/fake-client.js";
import type { SharedStateArtifact } from "../memory/decision-artifacts/index.js";
import { createWorkingMemory, type WorkingMemory } from "../memory/working/index.js";
import type { RetrievedContext, RetrievalConfidence } from "../retrieval/index.js";
import { StreamWriter, type StreamEntry, type StreamReader } from "../stream/index.js";
import {
  makeLiveSharedStateEntry,
  makeLockedSharedStateEntry,
  makeSharedStateArtifact,
  makeTestTurnTraceRecorder,
  makeToolUseCompleteResult,
} from "../test-support/factories/index.js";
import { ToolDispatcher } from "../tools/index.js";
import { FixedClock } from "../util/clock.js";
import {
  DEFAULT_SESSION_ID,
  createEntityId,
  createSharedStateEntryId,
  createStreamEntryId,
  type EntityId,
} from "../util/ids.js";
import { ContradictionRoutingCooldown } from "./deliberation/contradiction-routing-cooldown.js";
import { runDeliberationPhase } from "./lifecycle/turn-phase/deliberation-phase.js";
import { runRetrievalPhase } from "./lifecycle/turn-phase/retrieval-phase.js";
import type { TurnPhaseCoordinatorOptions } from "./lifecycle/turn-phase/types.js";
import { SESSION_REENTRY_CONTINUITY_TAG } from "./session-reentry-continuity.js";
import type { PerceptionResult } from "./types.js";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    const tempDir = tempDirs.pop();

    if (tempDir !== undefined) {
      rmSync(tempDir, { recursive: true, force: true });
    }
  }
});

function makePerception(): PerceptionResult {
  return {
    entities: [],
    mode: "problem_solving",
    affectiveSignal: {
      valence: 0,
      arousal: 0,
      dominant_emotion: null,
    },
    temporalCue: null,
  };
}

function makeStreamEntry(input: {
  id?: StreamEntry["id"];
  kind?: StreamEntry["kind"];
  content?: unknown;
  timestamp?: number;
}): StreamEntry {
  return {
    id: input.id ?? createStreamEntryId(),
    kind: input.kind ?? "user_msg",
    content: input.content ?? "Start the incident handoff log.",
    timestamp: input.timestamp ?? 10_000,
    session_id: DEFAULT_SESSION_ID,
    compressed: false,
    sender_entity_id: null,
    reply_target_entity_id: null,
  };
}

function makeStreamReader(entries: readonly StreamEntry[]): StreamReader {
  return {
    async *iterate() {
      for (const entry of entries) {
        yield entry;
      }
    },
  } as StreamReader;
}

function makeRetrievalConfidence(
  overall = 0.9,
  overrides: Partial<RetrievalConfidence> = {},
): RetrievalConfidence {
  return {
    overall,
    evidenceStrength: overrides.evidenceStrength ?? overall,
    coverage: overrides.coverage ?? 1,
    sourceDiversity: overrides.sourceDiversity ?? 1,
    contradictionPresent: overrides.contradictionPresent ?? false,
    sampleSize: overrides.sampleSize ?? 1,
  };
}

function makeRetrievedContext(): RetrievedContext {
  return {
    episodes: [],
    semantic: {
      supports: [],
      contradicts: [],
      categories: [],
      matched_node_ids: [],
      matched_nodes: [],
      support_hits: [],
      causal_hits: [],
      contradiction_hits: [],
      category_hits: [],
    },
    open_questions: [],
    evidence: [],
    recall_intents: [],
    contradiction_present: false,
    contradictionRouting: {
      contradictions: [],
    },
    confidence: makeRetrievalConfidence(0.9),
  };
}

function makeRetrievalOptions(input: {
  audienceEntityId: EntityId;
  artifact: SharedStateArtifact;
  streamEntries: readonly StreamEntry[];
  tracer: ReturnType<typeof makeTestTurnTraceRecorder>;
}): TurnPhaseCoordinatorOptions {
  const retrieval = makeRetrievedContext();

  return {
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
    embeddingClient: new FakeEmbeddingClient(8),
    sharedStateRepository: {
      get: vi.fn(() => input.artifact),
      upsert: vi.fn(),
    },
    selfContextBuilder: {
      build: vi.fn(async () => ({
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        activeScoringValues: [],
        selfScoringFeatures: {
          goalVectors: [],
          valueVectors: [],
        },
        retrievalScoringFeatures: {
          goalVectors: [],
          valueVectors: [],
        },
        executiveFocus: {
          selected_goal: null,
          selected_score: null,
          candidates: [],
          threshold: 0,
        },
      })),
    },
    turnRetrievalCoordinator: {
      coordinate: vi.fn(async () => ({
        applicableCommitments: [],
        pendingCorrections: [],
        affectiveTrajectory: [],
        retrieval,
        retrievedEpisodes: [],
        retrievedSemantic: null,
        proceduralContext: null,
        selectedSkill: null,
        retrievalOptions: {},
        reRetrieve: vi.fn(async () => retrieval),
      })),
    },
    relationalSlotRepository: {
      list: () => [],
      listConstrained: () => [],
    },
    openQuestionsRepository: {
      findByHandles: () => [],
      get: () => null,
      list: () => [],
      resolve: vi.fn(),
    },
    createStreamReader: () => makeStreamReader(input.streamEntries),
    clock: new FixedClock(10_000),
    tracer: input.tracer,
    entityRepository: {
      get: () => null,
      findByName: () => null,
      resolve: () => input.audienceEntityId,
    },
    socialRepository: {
      getProfile: () => null,
    },
    actionRepository: {
      list: () => [],
      get: () => null,
      update: vi.fn(),
    },
    goalsRepository: {
      list: () => [],
    },
    commitmentRepository: {
      list: () => [],
    },
    workingMemoryStore: {},
    toolDispatcher: createToolDispatcher(),
    llmFactory: () => new FakeLLMClient({ responses: [] }),
    perceptionGateway: {},
    turnOpeningPersistence: {},
    attributionLifecycleService: {},
    correctivePreferenceTurnService: {},
    turnActionStateService: {},
    turnGoalPromotionService: {},
    discourseStateService: {},
    postGenerationGuardRunner: {
      listRecentCompletedActions: () => [],
    },
    turnActionCoordinator: {},
    turnReflectionCoordinator: {},
  } as unknown as TurnPhaseCoordinatorOptions;
}

async function runRetrievalFixture(input: {
  artifact: SharedStateArtifact;
  audienceEntityId: EntityId;
  priorEntries?: readonly StreamEntry[];
  userMessage?: string;
}) {
  const tracer = makeTestTurnTraceRecorder();
  const userMessage = input.userMessage ?? "Let's start a fresh incident handoff log.";
  const currentUserEntry = makeStreamEntry({
    content: userMessage,
    timestamp: 10_000,
  });
  const options = makeRetrievalOptions({
    audienceEntityId: input.audienceEntityId,
    artifact: input.artifact,
    streamEntries: [...(input.priorEntries ?? []), currentUserEntry],
    tracer,
  });
  const workingMemory: WorkingMemory = {
    ...createWorkingMemory(DEFAULT_SESSION_ID, 10_000),
    turn_counter: 1,
    mode: "problem_solving",
  };

  const result = await runRetrievalPhase({
    options,
    sessionId: DEFAULT_SESSION_ID,
    turnId: "turn-session-reentry-continuity",
    turnInput: {
      userMessage,
      audience: "incident-team",
      origin: "user",
    },
    isSelfAudience: false,
    isUserTurn: true,
    cognitionInput: userMessage,
    llmClient: new FakeLLMClient({ responses: [] }),
    recencyMessages: [],
    audienceEntityId: input.audienceEntityId,
    audienceEntity: null,
    audienceProfile: null,
    perception: makePerception(),
    workingMemory,
    suppressionSet: {} as never,
    actionLinkSelfContext: null,
    persistedPromotions: {
      goalIds: [],
      executiveStepIds: [],
    },
    correctiveCommitment: null,
    activeParticipants: [],
    participantRoster: null,
    participantProfiles: [],
    persistedUserEntry: currentUserEntry,
    currentTurnFrameAnomaly: null,
    closureLoopAssessment: null,
  });

  return {
    result,
    options,
    tracer,
    currentUserEntry,
    workingMemory,
  };
}

function plannerResponse(): LLMCompleteResult {
  return makeToolUseCompleteResult({
    toolName: "EmitTurnPlan",
    toolInput: {
      uncertainty: "",
      verification_steps: [],
      tensions: [],
      voice_note: "Continue from the surfaced incident state.",
      emission_recommendation: "emit",
      intents: [],
    },
  });
}

function requestSystemText(system: unknown): string {
  if (typeof system === "string") {
    return system;
  }

  if (!Array.isArray(system)) {
    return "";
  }

  return system
    .map((block) =>
      block !== null &&
      typeof block === "object" &&
      "text" in block &&
      typeof block.text === "string"
        ? block.text
        : "",
    )
    .join("\n\n");
}

function eventsNamed(tracer: ReturnType<typeof makeTestTurnTraceRecorder>, name: string) {
  return tracer.events.filter((entry) => entry.event === name);
}

function createToolDispatcher(): ToolDispatcher {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-session-reentry-"));
  tempDirs.push(tempDir);
  const clock = new FixedClock(10_000);

  return new ToolDispatcher({
    clock,
    createStreamWriter: (sessionId) =>
      new StreamWriter({
        dataDir: tempDir,
        sessionId,
        clock,
      }),
  });
}

function createStreamWriter(): StreamWriter {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-session-reentry-stream-"));
  tempDirs.push(tempDir);

  return new StreamWriter({
    dataDir: tempDir,
    sessionId: DEFAULT_SESSION_ID,
    clock: new FixedClock(10_000),
  });
}

async function runDeliberationFixture(input: {
  options: TurnPhaseCoordinatorOptions;
  llmClient: FakeLLMClient;
  audienceEntityId: EntityId;
  currentUserEntry: StreamEntry;
  workingMemory: WorkingMemory;
  retrievalPhase: Awaited<ReturnType<typeof runRetrievalPhase>>;
}) {
  return runDeliberationPhase({
    options: input.options,
    llmClient: input.llmClient,
    sessionId: DEFAULT_SESSION_ID,
    turnId: "turn-session-reentry-continuity",
    turnInput: {
      userMessage: String(input.currentUserEntry.content),
      audience: "incident-team",
      origin: "user",
      stakes: "high",
    },
    streamWriter: createStreamWriter(),
    audienceEntityId: input.audienceEntityId,
    participationPolicy: "active",
    creatorContext: null,
    operatorSessionSnapshot: null,
    persistedUserEntryId: input.currentUserEntry.id,
    perception: makePerception(),
    activeParticipants: [],
    participantProfiles: [],
    workingMemory: input.workingMemory,
    audienceProfile: null,
    recencyMessages: [],
    currentTurnFrameAnomaly: null,
    retrievalPhase: input.retrievalPhase,
    contradictionRoutingCooldown: new ContradictionRoutingCooldown(),
    participantRoster: null,
  });
}

function makeArtifactWithActiveAndSupersededEntries(audienceEntityId: EntityId) {
  const rollback = makeLockedSharedStateEntry({
    audience_entity_id: audienceEntityId,
    state_key: "incident.rollback",
    text: "Rollback checkpoint is approved if error budget breach continues.",
    last_updated_at: 1_000,
    last_updated_stream_entry_ids: [createStreamEntryId()],
  });
  const customerNote = makeLiveSharedStateEntry({
    audience_entity_id: audienceEntityId,
    state_key: "incident.customer-note",
    text: "Customer-facing note must avoid internal service names.",
    last_updated_at: 2_000,
    last_updated_stream_entry_ids: [createStreamEntryId()],
  });
  const legacy = {
    ...makeLockedSharedStateEntry({
      audience_entity_id: audienceEntityId,
      text: "Legacy note: escalation owner is the release lead.",
      last_updated_at: 3_000,
      last_updated_stream_entry_ids: [createStreamEntryId()],
    }),
    state_key: null,
  };
  const superseded = makeLockedSharedStateEntry({
    audience_entity_id: audienceEntityId,
    state_key: "incident.superseded",
    text: "Superseded incident note.",
    last_updated_at: 4_000,
    superseded_by_id: createSharedStateEntryId(),
  });

  return makeSharedStateArtifact([rollback, customerNote, legacy, superseded], {
    audience_entity_id: audienceEntityId,
  });
}

function makeLegacyOnlyArtifact(audienceEntityId: EntityId) {
  const first = {
    ...makeLockedSharedStateEntry({
      audience_entity_id: audienceEntityId,
      text: "Legacy note: deployment owner is the release lead.",
      last_updated_at: 1_000,
      last_updated_stream_entry_ids: [createStreamEntryId()],
    }),
    state_key: null,
  };
  const second = {
    ...makeLiveSharedStateEntry({
      audience_entity_id: audienceEntityId,
      text: "Legacy live note: vendor update should wait for rollback confirmation.",
      last_updated_at: 2_000,
      last_updated_stream_entry_ids: [createStreamEntryId()],
    }),
    state_key: null,
  };

  return makeSharedStateArtifact([first, second], {
    audience_entity_id: audienceEntityId,
  });
}

describe("session re-entry continuity integration", () => {
  it("carries a real artifact continuity card to both planner and finalizer prompts", async () => {
    const audienceEntityId = createEntityId();
    const artifact = makeArtifactWithActiveAndSupersededEntries(audienceEntityId);
    const { result, options, tracer, currentUserEntry, workingMemory } = await runRetrievalFixture({
      audienceEntityId,
      artifact,
    });
    const promptSection = result.evidenceLedgerContext.sessionReentryContinuityPromptSection;

    expect(promptSection).toContain(`<${SESSION_REENTRY_CONTINUITY_TAG}>`);
    expect(promptSection).toContain("active_entry_count=3");
    expect(promptSection).toContain("active_keyed_entry_count=2");
    expect(promptSection).toContain("active_legacy_unkeyed_entry_count=1");
    expect(promptSection).toContain("state_key_bucket=incident.rollback");
    expect(promptSection).toContain("state_key_bucket=incident.customer-note");
    expect(promptSection).toContain("state_key_bucket=legacy bucket_source=unkeyed_legacy_state");
    expect(promptSection).not.toContain("incident.superseded");
    const evaluatedEvents = eventsNamed(tracer, "session_reentry.continuity.evaluated");
    const renderedEvents = eventsNamed(tracer, "session_reentry.continuity.rendered");

    expect(evaluatedEvents).toHaveLength(1);
    expect(evaluatedEvents[0]?.data).toMatchObject({
      status: "rendered",
      active_entry_count: 3,
      active_keyed_entry_count: 2,
      active_legacy_entry_count: 1,
      active_state_key_count: 3,
      active_entries_by_key: {
        "incident.customer-note": 1,
        "incident.rollback": 1,
        legacy: 1,
      },
    });
    expect(renderedEvents).toHaveLength(1);
    expect(renderedEvents[0]?.data).toMatchObject({
      status: "rendered",
    });

    const llmClient = new FakeLLMClient({
      responses: [
        plannerResponse(),
        createFakeEmitAnswerResponse("I see the existing incident state."),
      ],
    });

    await runDeliberationFixture({
      options,
      llmClient,
      audienceEntityId,
      currentUserEntry,
      workingMemory,
      retrievalPhase: result,
    });

    const plannerSystem = requestSystemText(llmClient.requests[0]?.system);
    const finalizerSystem = requestSystemText(llmClient.requests[1]?.system);

    expect(plannerSystem).toContain(`<${SESSION_REENTRY_CONTINUITY_TAG}>`);
    expect(finalizerSystem).toContain(`<${SESSION_REENTRY_CONTINUITY_TAG}>`);
    expect(plannerSystem).toContain("state_key_bucket=incident.rollback");
    expect(finalizerSystem).toContain("state_key_bucket=incident.rollback");
    expect(plannerSystem).toContain("state_key_bucket=legacy bucket_source=unkeyed_legacy_state");
    expect(finalizerSystem).toContain("state_key_bucket=legacy bucket_source=unkeyed_legacy_state");
  });

  it("renders carryover as possible prior context for fresh-start framing", async () => {
    const audienceEntityId = createEntityId();
    const artifact = makeArtifactWithActiveAndSupersededEntries(audienceEntityId);
    const { result } = await runRetrievalFixture({
      audienceEntityId,
      artifact,
      userMessage: "I haven't told them yet; let's start a thread for the rollout decision.",
    });
    const promptSection = result.evidenceLedgerContext.sessionReentryContinuityPromptSection;

    expect(promptSection).toContain(`<${SESSION_REENTRY_CONTINUITY_TAG}>`);
    expect(promptSection).toContain(
      "Surface the carryover as possible prior context and ask whether to continue that thread, reset it, or start a new one.",
    );
  });

  it("renders for audiences with only legacy null-key active entries", async () => {
    const audienceEntityId = createEntityId();
    const artifact = makeLegacyOnlyArtifact(audienceEntityId);
    const { result, tracer } = await runRetrievalFixture({
      audienceEntityId,
      artifact,
    });
    const promptSection = result.evidenceLedgerContext.sessionReentryContinuityPromptSection;

    expect(promptSection).toContain(`<${SESSION_REENTRY_CONTINUITY_TAG}>`);
    expect(promptSection).toContain("active_entry_count=2");
    expect(promptSection).toContain("active_keyed_entry_count=0");
    expect(promptSection).toContain("active_legacy_unkeyed_entry_count=2");
    expect(promptSection).toContain(
      "state_key_bucket=legacy bucket_source=unkeyed_legacy_state entries=2",
    );
    expect(promptSection).not.toContain("bucket_source=keyed_thread");

    const evaluatedEvents = eventsNamed(tracer, "session_reentry.continuity.evaluated");
    const renderedEvents = eventsNamed(tracer, "session_reentry.continuity.rendered");

    expect(evaluatedEvents).toHaveLength(1);
    expect(evaluatedEvents[0]?.data).toMatchObject({
      status: "rendered",
      active_entry_count: 2,
      active_keyed_entry_count: 0,
      active_legacy_entry_count: 2,
      active_state_key_count: 1,
      active_entries_by_key: {
        legacy: 2,
      },
    });
    expect(renderedEvents).toHaveLength(1);
    expect(renderedEvents[0]?.data).toMatchObject({
      status: "rendered",
    });
  });

  it("omits the card for a blank audience and emits the blank-audience metric source trace", async () => {
    const audienceEntityId = createEntityId();
    const artifact = makeSharedStateArtifact([], {
      audience_entity_id: audienceEntityId,
      entries: [],
    });
    const { result, tracer } = await runRetrievalFixture({
      audienceEntityId,
      artifact,
    });

    expect(result.evidenceLedgerContext.sessionReentryContinuityPromptSection).toBeNull();
    expect(eventsNamed(tracer, "session_reentry.continuity.rendered")).toHaveLength(0);
    expect(eventsNamed(tracer, "session_reentry.continuity.evaluated")).toHaveLength(1);
    expect(eventsNamed(tracer, "session_reentry.continuity.evaluated")[0]?.data).toMatchObject({
      status: "blank_audience",
      active_entry_count: 0,
      active_keyed_entry_count: 0,
      active_legacy_entry_count: 0,
      active_state_key_count: 0,
      active_entries_by_key: {},
    });
  });

  it("omits the card on mid-session turns with prior user messages", async () => {
    const audienceEntityId = createEntityId();
    const artifact = makeArtifactWithActiveAndSupersededEntries(audienceEntityId);
    const priorUserEntry = makeStreamEntry({
      content: "Prior incident handoff request.",
      timestamp: 9_000,
    });
    const { result, tracer } = await runRetrievalFixture({
      audienceEntityId,
      artifact,
      priorEntries: [priorUserEntry],
    });

    expect(result.evidenceLedgerContext.sessionReentryContinuityPromptSection).toBeNull();
    expect(eventsNamed(tracer, "session_reentry.continuity.rendered")).toHaveLength(0);
    expect(eventsNamed(tracer, "session_reentry.continuity.evaluated")).toHaveLength(1);
    expect(eventsNamed(tracer, "session_reentry.continuity.evaluated")[0]?.data).toMatchObject({
      status: "not_first_user_turn",
      active_entry_count: 3,
      active_keyed_entry_count: 2,
      active_legacy_entry_count: 1,
      active_state_key_count: 3,
      active_entries_by_key: {
        "incident.customer-note": 1,
        "incident.rollback": 1,
        legacy: 1,
      },
    });
  });
});
