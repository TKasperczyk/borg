import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { SuppressionSet } from "../attention/index.js";
import { Reflector, type ReflectorOptions } from "./reflector.js";
import { FakeLLMClient } from "../../llm/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { selfMigrations } from "../../memory/self/migrations.js";
import { episodicMigrations } from "../../memory/episodic/migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "../../memory/episodic/repository.js";
import { executiveMigrations, ExecutiveStepsRepository } from "../../executive/index.js";
import {
  GoalsRepository,
  OpenQuestionsRepository,
  TraitsRepository,
  ValuesRepository,
} from "../../memory/self/index.js";
import { retrievalMigrations } from "../../retrieval/migrations.js";
import { StreamReader, StreamWriter } from "../../stream/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, createExecutiveStepId, createStreamEntryId } from "../../util/ids.js";
import type { RetrievalConfidence, RetrievedEpisode } from "../../retrieval/index.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createRetrievalScoreFixture,
  createWorkingMemoryFixture,
} from "../../offline/test-support.js";

function createReflectionResponse(
  advancedGoals: Array<{ goal_id: string; evidence: string }> = [],
  proceduralOutcomes: Array<{
    classification: "success" | "failure" | "unclear";
    evidence: string;
    grounded?: boolean;
    attempt_turn_counter?: number;
    skill_actually_applied?: boolean;
  }> = [],
  intentUpdates: Array<{
    description: string;
    next_action: string | null;
    actor: "user" | "borg";
    status: "completed" | "abandoned";
    evidence: string;
  }> = [],
  traitDemonstrations: Array<{
    trait_label: string;
    evidence: string;
    strength_delta: number;
  }> = [],
  stepOutcomes: Array<{
    step_id: string;
    new_status: "doing" | "done" | "blocked" | "abandoned";
    evidence: string;
  }> = [],
  proposedSteps: Array<{
    goal_id: string;
    description: string;
    kind: "think" | "ask_user" | "research" | "act" | "wait";
    due_at?: number | null;
    rationale: string;
  }> = [],
  openQuestions: Array<{
    question: string;
    urgency: number;
    related_episode_ids: string[];
  }> = [],
  resolvedOpenQuestions: Array<{
    question_id: string;
    resolution_note: string;
    evidence_episode_ids: string[];
    evidence_stream_entry_ids: string[];
  }> = [],
) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: advancedGoals,
          procedural_outcomes: proceduralOutcomes.map((outcome) => ({
            grounded: true,
            // Default to grading the turn-1 attempt unless the test specifies.
            attempt_turn_counter: 1,
            // Sprint 56: default to crediting the selected skill when the
            // test does not explicitly stub otherwise.
            skill_actually_applied: true,
            ...outcome,
          })),
          trait_demonstrations: traitDemonstrations,
          intent_updates: intentUpdates,
          step_outcomes: stepOutcomes,
          proposed_steps: proposedSteps,
          open_questions: openQuestions,
          resolved_open_questions: resolvedOpenQuestions,
        },
      },
    ],
  };
}

class CaptureTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = true;
  readonly events: Array<{ event: TurnTraceEventName; data: TurnTraceData }> = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.events.push({ event, data });
  }
}

function createRawReflectionResponse(input: Record<string, unknown>) {
  return {
    text: "",
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input,
      },
    ],
  };
}

function createRetrievedEpisode(
  episode: RetrievedEpisode["episode"],
  score = 0.8,
): RetrievedEpisode {
  return {
    episode,
    score,
    scoreBreakdown: createRetrievalScoreFixture({
      similarity: score,
      decayedSalience: 0.4,
      heat: 0.3,
    }),
    citationChain: [],
  };
}

function createRetrievalConfidence(
  overrides: Partial<RetrievalConfidence> = {},
): RetrievalConfidence {
  return {
    overall: overrides.overall ?? 0.8,
    evidenceStrength: overrides.evidenceStrength ?? 0.8,
    coverage: overrides.coverage ?? 1,
    sourceDiversity: overrides.sourceDiversity ?? 1,
    contradictionPresent: overrides.contradictionPresent ?? false,
    sampleSize: overrides.sampleSize ?? 3,
  };
}

type ReflectionHarness = Awaited<ReturnType<typeof createOfflineTestHarness>>;

function createHarnessReflector(
  harness: ReflectionHarness,
  overrides: Partial<ReflectorOptions> = {},
): Reflector {
  return new Reflector({
    episodicRepository: harness.episodicRepository,
    goalsRepository: harness.goalsRepository,
    traitsRepository: harness.traitsRepository,
    actionRepository: harness.actionRepository,
    ...overrides,
  });
}

function createOpenQuestionReflectionContext(
  options: {
    retrievedEpisodes?: RetrievedEpisode[];
    referencedEpisodeIds?: RetrievedEpisode["episode"]["id"][] | null;
    retrievalConfidence?: RetrievalConfidence;
    response?: string;
  } = {},
) {
  const workingMemory = createWorkingMemoryFixture({
    hot_entities: ["Atlas"],
    mode: "reflective",
  });
  const response = options.response ?? "I still need to compare more evidence.";

  return {
    userMessage: "Why is Atlas still failing?",
    workingMemory,
    selfSnapshot: {
      values: [],
      goals: [],
      traits: [],
    },
    deliberationResult: {
      path: "system_2" as const,
      response,
      thoughts: [],
      tool_calls: [],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "end_turn" as const,
      },
      decision_reason: "low confidence" as const,
      retrievedEpisodes: options.retrievedEpisodes ?? [],
      referencedEpisodeIds: options.referencedEpisodeIds ?? null,
      intents: [],
      thoughtsPersisted: false,
    },
    actionResult: {
      response,
      tool_calls: [],
      intents: [],
      workingMemory,
    },
    retrievedEpisodes: options.retrievedEpisodes ?? [],
    retrievalConfidence:
      options.retrievalConfidence ??
      createRetrievalConfidence({
        overall: 0.2,
        evidenceStrength: 0.2,
        coverage: 0.2,
        sampleSize: 1,
      }),
    suppressionSet: new SuppressionSet(1),
  } satisfies Parameters<Reflector["reflect"]>[0];
}

function createPendingProceduralReflectionContext() {
  const pendingAttempt = {
    problem_text: "I hit a Rust lifetime issue again.",
    approach_summary: "Shrink borrow scopes and use intermediate bindings.",
    selected_skill_id: null,
    source_stream_ids: [createStreamEntryId(), createStreamEntryId()],
    turn_counter: 1,
    audience_entity_id: null,
  };
  const workingMemory = {
    session_id: DEFAULT_SESSION_ID,
    turn_counter: 2,
    hot_entities: ["Rust"],
    pending_actions: [],
    pending_social_attribution: null,
    pending_trait_attribution: null,
    suppressed: [],
    mood: null,
    pending_procedural_attempts: [pendingAttempt],
    discourse_state: {
      stop_until_substantive_content: null,
    },
    mode: "problem_solving" as const,
    updated_at: 0,
  };

  return {
    userMessage: "No clear result yet.",
    perception: {
      entities: ["Rust"],
      mode: "problem_solving" as const,
      affectiveSignal: {
        valence: 0,
        arousal: 0,
        dominant_emotion: null,
      },
      temporalCue: null,
    },
    workingMemory,
    selfSnapshot: {
      values: [],
      goals: [],
      traits: [],
    },
    deliberationResult: {
      path: "system_1" as const,
      response: "Next response.",
      thoughts: [],
      tool_calls: [],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "end_turn" as const,
      },
      decision_reason: "confidence" as const,
      retrievedEpisodes: [],
      referencedEpisodeIds: null,
      intents: [],
      thoughtsPersisted: false,
    },
    actionResult: {
      response: "Next response.",
      tool_calls: [],
      intents: [],
      workingMemory,
    },
    retrievedEpisodes: [],
    retrievalConfidence: createRetrievalConfidence(),
    selectedSkillId: null,
    suppressionSet: new SuppressionSet(2),
  } satisfies Parameters<Reflector["reflect"]>[0];
}

async function createExecutiveReflectionHarness(clock = new FixedClock(1_000)) {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
  const store = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: composeMigrations(
      episodicMigrations,
      selfMigrations,
      executiveMigrations,
      retrievalMigrations,
    ),
  });
  const table = await store.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(4),
  });
  const episodicRepository = new EpisodicRepository({
    table,
    db,
    clock,
  });
  const goalsRepository = new GoalsRepository({
    db,
    clock,
  });
  const traitsRepository = new TraitsRepository({
    db,
    clock,
  });
  const executiveStepsRepository = new ExecutiveStepsRepository({
    db,
    clock,
  });
  const writer = new StreamWriter({
    dataDir: tempDir,
    sessionId: DEFAULT_SESSION_ID,
    clock,
  });

  return {
    tempDir,
    clock,
    store,
    db,
    episodicRepository,
    goalsRepository,
    traitsRepository,
    executiveStepsRepository,
    writer,
    cleanup: async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    },
  };
}

function createExecutiveReflectionContext(input: {
  origin?: "user" | "autonomous";
  goal: ReturnType<GoalsRepository["add"]>;
  nextStep?: ReturnType<ExecutiveStepsRepository["add"]> | null;
  pendingIntents?: Array<{ description: string; next_action: string | null }>;
}) {
  const pendingIntents = input.pendingIntents ?? [];
  const workingMemory = {
    session_id: DEFAULT_SESSION_ID,
    turn_counter: 1,
    hot_entities: ["Apollo"],
    pending_actions: pendingIntents,
    pending_social_attribution: null,
    pending_trait_attribution: null,
    suppressed: [],
    mood: null,
    pending_procedural_attempts: [],
    discourse_state: {
      stop_until_substantive_content: null,
    },
    mode: "problem_solving" as const,
    updated_at: 0,
  };

  return {
    origin: input.origin ?? "user",
    userMessage: "Let's move the Apollo launch plan forward.",
    perception: {
      entities: ["Apollo"],
      mode: "problem_solving" as const,
      affectiveSignal: {
        valence: 0,
        arousal: 0,
        dominant_emotion: null,
      },
      temporalCue: null,
    },
    workingMemory,
    selfSnapshot: {
      values: [],
      goals: [input.goal],
      traits: [],
    },
    deliberationResult: {
      path: "system_1" as const,
      response: "I worked on Apollo.",
      thoughts: [],
      tool_calls: [],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "end_turn" as const,
      },
      decision_reason: "confidence" as const,
      retrievedEpisodes: [],
      referencedEpisodeIds: null,
      intents: [],
      thoughtsPersisted: false,
    },
    actionResult: {
      response: "I worked on Apollo.",
      tool_calls: [],
      intents: pendingIntents,
      workingMemory,
    },
    retrievedEpisodes: [],
    retrievalConfidence: createRetrievalConfidence(),
    executiveFocus: {
      selected_goal: input.goal,
      selected_score: null,
      next_step: input.nextStep ?? null,
      candidates: [],
      threshold: 0.45,
    },
    selectedSkillId: null,
    suppressionSet: new SuppressionSet(1),
  } satisfies Parameters<Reflector["reflect"]>[0];
}

describe("reflector", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("applies valid executive step outcomes and drops invalid outcomes", async () => {
    const harness = await createExecutiveReflectionHarness();
    cleanup.push(harness.cleanup);
    const goal = harness.goalsRepository.add({
      description: "Apollo launch plan",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const legal = harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Start launch readiness review",
      kind: "act",
      provenance: { kind: "manual" },
    });
    const illegal = harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Skip straight to done",
      kind: "act",
      provenance: { kind: "manual" },
    });
    const emptyEvidence = harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Needs actual evidence",
      kind: "think",
      provenance: { kind: "manual" },
    });
    const missingStepId = createExecutiveStepId();
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse(
          [],
          [],
          [],
          [],
          [
            {
              step_id: legal.id,
              new_status: "doing",
              evidence: "The assistant started the readiness review.",
            },
            {
              step_id: illegal.id,
              new_status: "done",
              evidence: "Queued cannot jump directly to done.",
            },
            {
              step_id: missingStepId,
              new_status: "doing",
              evidence: "The model referenced a stale step id.",
            },
            {
              step_id: emptyEvidence.id,
              new_status: "doing",
              evidence: "   ",
            },
          ],
        ),
      ],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      executiveStepsRepository: harness.executiveStepsRepository,
    });

    await reflector.reflect(
      createExecutiveReflectionContext({
        goal,
        nextStep: legal,
      }),
      harness.writer,
    );

    expect(harness.executiveStepsRepository.get(legal.id)).toMatchObject({
      status: "doing",
      last_attempt_ts: harness.clock.now(),
    });
    expect(harness.executiveStepsRepository.get(illegal.id)?.status).toBe("queued");
    expect(harness.executiveStepsRepository.get(emptyEvidence.id)?.status).toBe("queued");

    const droppedReasons = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    })
      .tail(10)
      .filter((entry) => entry.kind === "internal_event")
      .map((entry) => (entry.content as { reason?: string }).reason);

    expect(droppedReasons).toEqual(
      expect.arrayContaining(["repository_update_failed", "missing_step", "empty_evidence"]),
    );
  });

  it("drops executive step outcomes for goals outside the audience-scoped self snapshot", async () => {
    const harness = await createExecutiveReflectionHarness();
    cleanup.push(harness.cleanup);
    const visibleGoal = harness.goalsRepository.add({
      description: "Visible Apollo launch plan",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const hiddenGoal = harness.goalsRepository.add({
      description: "Private background plan",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const hiddenStep = harness.executiveStepsRepository.add({
      goalId: hiddenGoal.id,
      description: "Mutate hidden step",
      kind: "act",
      provenance: { kind: "manual" },
    });
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse(
          [],
          [],
          [],
          [],
          [
            {
              step_id: hiddenStep.id,
              new_status: "doing",
              evidence: "The reflected tool call reached across goal scope.",
            },
          ],
        ),
      ],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      executiveStepsRepository: harness.executiveStepsRepository,
    });

    await reflector.reflect(
      createExecutiveReflectionContext({
        goal: visibleGoal,
        nextStep: null,
      }),
      harness.writer,
    );

    expect(harness.executiveStepsRepository.get(hiddenStep.id)?.status).toBe("queued");
    expect(
      new StreamReader({
        dataDir: harness.tempDir,
        sessionId: DEFAULT_SESSION_ID,
      })
        .tail(5)
        .some(
          (entry) =>
            entry.kind === "internal_event" &&
            (entry.content as { reason?: string }).reason === "step_goal_not_visible",
        ),
    ).toBe(true);
  });

  it("drops autonomous done step outcomes but applies the same outcome on user turns", async () => {
    const harness = await createExecutiveReflectionHarness();
    cleanup.push(harness.cleanup);
    const goal = harness.goalsRepository.add({
      description: "Apollo launch plan",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const step = harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Finish readiness review",
      kind: "act",
      status: "doing",
      provenance: { kind: "manual" },
    });
    const stepDoneOutcome = [
      {
        step_id: step.id,
        new_status: "done" as const,
        evidence: "The turn claims the readiness review is complete.",
      },
    ];
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse([], [], [], [], stepDoneOutcome),
        createReflectionResponse([], [], [], [], stepDoneOutcome),
      ],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      executiveStepsRepository: harness.executiveStepsRepository,
    });

    await reflector.reflect(
      createExecutiveReflectionContext({
        origin: "autonomous",
        goal,
        nextStep: step,
      }),
      harness.writer,
    );

    expect(harness.executiveStepsRepository.get(step.id)?.status).toBe("doing");

    await reflector.reflect(
      createExecutiveReflectionContext({
        origin: "user",
        goal,
        nextStep: step,
      }),
      harness.writer,
    );

    expect(harness.executiveStepsRepository.get(step.id)?.status).toBe("done");
  });

  it("does not apply intent updates from autonomous reflection", async () => {
    const harness = await createExecutiveReflectionHarness();
    cleanup.push(harness.cleanup);
    const goal = harness.goalsRepository.add({
      description: "Apollo launch plan",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const pendingIntent = {
      description: "Ask the user to review Apollo readiness",
      next_action: "Ask for review",
    };
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse(
          [],
          [],
          [
            {
              ...pendingIntent,
              actor: "borg",
              status: "completed",
              evidence: "Autonomous self-talk claimed the intent was done.",
            },
          ],
          [],
          [
            {
              step_id: createExecutiveStepId(),
              new_status: "doing",
              evidence: "Keeps executive reflection work present.",
            },
          ],
        ),
      ],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      executiveStepsRepository: harness.executiveStepsRepository,
    });

    const reflected = await reflector.reflect(
      createExecutiveReflectionContext({
        origin: "autonomous",
        goal,
        nextStep: null,
        pendingIntents: [pendingIntent],
      }),
      harness.writer,
    );

    expect(reflected.pending_actions).toEqual([pendingIntent]);
    expect(
      new StreamReader({
        dataDir: harness.tempDir,
        sessionId: DEFAULT_SESSION_ID,
      })
        .tail(10)
        .some(
          (entry) =>
            entry.kind === "internal_event" &&
            (entry.content as { hook?: string }).hook === "reflector_intent_update_dropped",
        ),
    ).toBe(true);
  });

  it.each([
    {
      executiveOutput: {
        step_outcomes: [
          {
            new_status: "doing",
            evidence: "Missing step_id rejects the whole reflection.",
          },
        ],
        proposed_steps: [],
      },
    },
    {
      executiveOutput: {
        step_outcomes: [],
        proposed_steps: [
          {
            description: "Review launch notes",
            kind: "think",
            rationale: "Missing goal_id rejects the whole reflection.",
          },
        ],
      },
    },
  ])(
    "rejects the whole reflection when executive output is malformed",
    async ({ executiveOutput }) => {
      const harness = await createOfflineTestHarness({
        llmClient: new FakeLLMClient({
          responses: [
            createRawReflectionResponse({
              advanced_goals: [],
              procedural_outcomes: [
                {
                  attempt_turn_counter: 1,
                  classification: "success",
                  evidence: "The user's follow-up confirmed the approach worked.",
                  grounded: true,
                  skill_actually_applied: true,
                },
              ],
              trait_demonstrations: [],
              intent_updates: [],
              ...executiveOutput,
            }),
          ],
        }),
      });
      cleanup.push(harness.cleanup);
      const reflector = createHarnessReflector(harness, {
        clock: harness.clock,
        llmClient: harness.llmClient,
        model: "haiku",
        proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
      });

      const reflected = await reflector.reflect(
        createPendingProceduralReflectionContext(),
        harness.streamWriter,
      );
      const events = new StreamReader({
        dataDir: harness.tempDir,
        sessionId: DEFAULT_SESSION_ID,
      }).tail(10);

      expect(reflected.pending_procedural_attempts).toHaveLength(1);
      expect(harness.proceduralEvidenceRepository.list()).toEqual([]);
      expect(
        events.some(
          (entry) =>
            entry.kind === "internal_event" &&
            (entry.content as { hook?: string; error?: string }).hook === "reflection_judgment" &&
            (entry.content as { error?: string }).error?.startsWith(
              "LLMError: Reflector returned invalid reflection payload",
            ),
        ),
      ).toBe(true);
      expect(
        events.some(
          (entry) =>
            entry.kind === "internal_event" &&
            (entry.content as { hook?: string }).hook === "reflector_executive_item_dropped",
        ),
      ).toBe(false);
    },
  );

  it("creates proposed executive steps and drops proposals over the open-step cap", async () => {
    const harness = await createExecutiveReflectionHarness();
    cleanup.push(harness.cleanup);
    const goal = harness.goalsRepository.add({
      description: "Apollo launch plan",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const proposedSteps = [
      "Review launch notes",
      "Ask user for risk tolerance",
      "Wait for date",
      "Extra",
    ].map((description, index) => ({
      goal_id: goal.id,
      description,
      kind: index === 1 ? ("ask_user" as const) : ("think" as const),
      due_at: index === 2 ? 2_000 : null,
      rationale: "No open step exists for the selected goal.",
    }));
    const llm = new FakeLLMClient({
      responses: [createReflectionResponse([], [], [], [], [], proposedSteps)],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      executiveStepsRepository: harness.executiveStepsRepository,
    });

    await reflector.reflect(
      createExecutiveReflectionContext({
        goal,
        nextStep: null,
      }),
      harness.writer,
    );

    const openStepDescriptions = harness.executiveStepsRepository
      .listOpen(goal.id)
      .map((step) => step.description);

    expect(openStepDescriptions[0]).toBe("Wait for date");
    expect(openStepDescriptions).toEqual(
      expect.arrayContaining(["Review launch notes", "Ask user for risk tolerance"]),
    );
    expect(openStepDescriptions).toHaveLength(3);
    expect(
      new StreamReader({
        dataDir: harness.tempDir,
        sessionId: DEFAULT_SESSION_ID,
      })
        .tail(10)
        .some(
          (entry) =>
            entry.kind === "internal_event" &&
            (entry.content as { reason?: string }).reason === "open_step_cap",
        ),
    ).toBe(true);
  });

  it("drops autonomous close-and-replace proposals for the same goal", async () => {
    const harness = await createExecutiveReflectionHarness();
    cleanup.push(harness.cleanup);
    const goal = harness.goalsRepository.add({
      description: "Apollo launch plan",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const step = harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Investigate the launch blocker",
      kind: "act",
      status: "doing",
      provenance: { kind: "manual" },
    });
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse(
          [],
          [],
          [],
          [],
          [
            {
              step_id: step.id,
              new_status: "blocked",
              evidence: "The autonomous turn hit a missing dependency.",
            },
          ],
          [
            {
              goal_id: goal.id,
              description: "Try a replacement step immediately",
              kind: "act",
              due_at: harness.clock.now(),
              rationale: "The previous step was blocked.",
            },
          ],
        ),
      ],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      executiveStepsRepository: harness.executiveStepsRepository,
    });

    await reflector.reflect(
      createExecutiveReflectionContext({
        origin: "autonomous",
        goal,
        nextStep: step,
      }),
      harness.writer,
    );

    expect(harness.executiveStepsRepository.get(step.id)?.status).toBe("blocked");
    expect(harness.executiveStepsRepository.listOpen(goal.id)).toEqual([]);
    expect(
      new StreamReader({
        dataDir: harness.tempDir,
        sessionId: DEFAULT_SESSION_ID,
      })
        .tail(10)
        .some(
          (entry) =>
            entry.kind === "internal_event" &&
            (entry.content as { hook?: string }).hook ===
              "reflector_proposal_dropped_close_and_replace",
        ),
    ).toBe(true);
  });

  it("drops proposed wait steps without due_at before repository write", async () => {
    const harness = await createExecutiveReflectionHarness();
    cleanup.push(harness.cleanup);
    const goal = harness.goalsRepository.add({
      description: "Apollo launch plan",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse(
          [],
          [],
          [],
          [],
          [],
          [
            {
              goal_id: goal.id,
              description: "Wait without a concrete time",
              kind: "wait",
              due_at: null,
              rationale: "The model proposed a wait but no deadline.",
            },
          ],
        ),
      ],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      executiveStepsRepository: harness.executiveStepsRepository,
    });

    await reflector.reflect(
      createExecutiveReflectionContext({
        goal,
        nextStep: null,
      }),
      harness.writer,
    );

    expect(harness.executiveStepsRepository.listOpen(goal.id)).toEqual([]);
    expect(
      new StreamReader({
        dataDir: harness.tempDir,
        sessionId: DEFAULT_SESSION_ID,
      })
        .tail(10)
        .some(
          (entry) =>
            entry.kind === "internal_event" &&
            (entry.content as { reason?: string }).reason === "wait_without_due_at",
        ),
    ).toBe(true);
  });

  it("bumps LLM-marked goal progress, skips unreferenced episode use, and ticks suppression", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new FixedClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(episodicMigrations, selfMigrations, retrievalMigrations),
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const episodicRepository = new EpisodicRepository({
      table,
      db,
      clock,
    });
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const traitsRepository = new TraitsRepository({
      db,
      clock,
    });
    const valuesRepository = new ValuesRepository({
      db,
      clock,
    });
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const goal = goalsRepository.add({
      description: "stabilize atlas release",
      priority: 5,
      provenance: { kind: "manual" },
    });
    const episode = await episodicRepository.insert({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas incident",
      narrative: "Atlas deployment failed.",
      participants: ["team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as never],
      significance: 0.8,
      tags: ["atlas"],
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    const suppressionSet = SuppressionSet.fromEntries(
      [
        {
          id: "ep_stale",
          reason: "temporary",
          until_turn: 1,
        },
      ],
      1,
    );

    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse([
          {
            goal_id: goal.id,
            evidence: "Updated the Atlas release stabilization plan.",
          },
        ]),
      ],
    });
    const reflector = new Reflector({
      clock,
      llmClient: llm,
      model: "haiku",
      episodicRepository,
      goalsRepository,
      traitsRepository,
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.9,
      scoreBreakdown: createRetrievalScoreFixture({
        similarity: 0.9,
        decayedSalience: 0.3,
        heat: 1,
        goalRelevance: 0.2,
        timeRelevance: 0,
        suppressionPenalty: 0,
      }),
      citationChain: [],
    };
    const reflected = await reflector.reflect(
      {
        userMessage: "We need to stabilize the Atlas release",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: ["Atlas"],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: valuesRepository.list(),
          goals: [goal],
          traits: traitsRepository.list(),
        },
        deliberationResult: {
          path: "system_1",
          response: "To stabilize the Atlas release, check the atlas deployment.",
          thoughts: ["brief thought"],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: true,
        },
        actionResult: {
          response: "To stabilize the Atlas release, check the atlas deployment.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: ["Atlas"],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet,
      },
      writer,
    );

    expect(goalsRepository.list({ status: "active" })[0]?.progress_notes).toContain(
      "Updated the Atlas release stabilization plan.",
    );
    expect(episodicRepository.getStats(episode.id)?.use_count).toBe(0);
    expect(traitsRepository.list()).toEqual([]);
    expect(reflected.pending_trait_attribution).toBeNull();
    expect(suppressionSet.isSuppressed(episode.id)).toBe(false);
    expect(suppressionSet.isSuppressed("ep_stale")).toBe(false);
    // Phase E removed scratchpad/recent_thoughts from working memory. The
    // reflector no longer clears scratchpad or pushes thoughts into the
    // cache; thoughts live in the stream (persisted by the deliberator),
    // and working memory holds derived live-turn state only.
    expect(reflected.turn_counter).toBe(1);
    expect(reflected.hot_entities).toEqual(["Atlas"]);
  });

  it("applies user-turn reflector progress updates without review", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(4_000),
    });
    cleanup.push(harness.cleanup);

    const goal = harness.goalsRepository.add({
      description: "stabilize atlas release",
      priority: 5,
      provenance: {
        kind: "episodes",
        episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse([
          {
            goal_id: goal.id,
            evidence: "Updated the deployment checklist.",
          },
        ]),
      ],
    });
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
      identityService: harness.identityService,
      reviewQueueRepository: harness.reviewQueueRepository,
    });

    await reflector.reflect(
      {
        userMessage: "We need to stabilize the Atlas release",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: ["Atlas"],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: harness.valuesRepository.list(),
          goals: [goal],
          traits: harness.traitsRepository.list(),
        },
        deliberationResult: {
          path: "system_1",
          response: "To stabilize the Atlas release, update the deployment checklist.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: true,
        },
        actionResult: {
          response: "To stabilize the Atlas release, update the deployment checklist.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: ["Atlas"],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(),
      },
      harness.streamWriter,
    );

    const updatedGoal = harness.goalsRepository.get(goal.id);
    expect(updatedGoal?.progress_notes).toContain("Updated the deployment checklist.");
    expect(updatedGoal?.last_progress_ts).toBe(4_000);
    expect(harness.reviewQueueRepository.getOpen()).toEqual([]);
    expect(
      harness.identityEventRepository.list({ recordType: "goal", recordId: goal.id })[0],
    ).toEqual(
      expect.objectContaining({
        action: "update",
        provenance: {
          kind: "online",
          process: "reflector",
        },
        new_value: expect.objectContaining({
          progress_notes: expect.stringContaining("Updated the deployment checklist."),
          last_progress_ts: 4_000,
        }),
        review_item_id: null,
        overwrite_without_review: false,
      }),
    );
    expect(llm.requests[0]?.system).toContain(
      "Apply common-sense task linkage: when a turn describes the user completing a recognizable sub-task of an active goal, mark advanced_goals for that goal even if the user doesn't name the goal explicitly.",
    );
  });

  it("ignores autonomous-turn advanced goal output", async () => {
    const harness = await createExecutiveReflectionHarness(new FixedClock(4_500));
    cleanup.push(harness.cleanup);
    const goal = harness.goalsRepository.add({
      description: "Apollo launch plan",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const step = harness.executiveStepsRepository.add({
      goalId: goal.id,
      description: "Start launch readiness review",
      kind: "act",
      provenance: { kind: "manual" },
    });
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse(
          [
            {
              goal_id: goal.id,
              evidence: "The autonomous turn claimed launch readiness moved forward.",
            },
          ],
          [],
          [],
          [],
          [
            {
              step_id: step.id,
              new_status: "doing",
              evidence: "Keeps executive reflection work present.",
            },
          ],
        ),
      ],
    });
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      executiveStepsRepository: harness.executiveStepsRepository,
    });

    await reflector.reflect(
      createExecutiveReflectionContext({
        origin: "autonomous",
        goal,
        nextStep: step,
      }),
      harness.writer,
    );

    expect(llm.requests).toHaveLength(1);
    expect(harness.goalsRepository.get(goal.id)?.progress_notes).toBeNull();
  });

  it("does not update goal progress when reflection output is empty even if text overlaps", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(5_000),
    });
    cleanup.push(harness.cleanup);

    const goal = harness.goalsRepository.add({
      description: "stabilize atlas release",
      priority: 5,
      provenance: { kind: "manual" },
    });
    const llm = new FakeLLMClient({
      responses: [createReflectionResponse()],
    });
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
    });

    await reflector.reflect(
      {
        userMessage: "We need to stabilize the Atlas release.",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: ["Atlas"],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: harness.valuesRepository.list(),
          goals: [goal],
          traits: harness.traitsRepository.list(),
        },
        deliberationResult: {
          path: "system_1",
          response: "To stabilize the Atlas release, we should discuss the risk.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: true,
        },
        actionResult: {
          response: "To stabilize the Atlas release, we should discuss the risk.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: ["Atlas"],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(),
      },
      harness.streamWriter,
    );

    expect(harness.goalsRepository.get(goal.id)?.progress_notes).toBeNull();
  });

  it("clears completed and abandoned pending actions and persists action records", async () => {
    const pendingIntents: Array<{ description: string; next_action: string | null }> = [
      {
        description: "Check the Atlas rollout after tests finish",
        next_action: "review deploy status",
      },
      {
        description: "Open a new incident if rollout fails",
        next_action: "create incident",
      },
    ];
    const newIntent = {
      description: "Write a rollback note",
      next_action: "draft rollback note",
    };
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse(
          [],
          [],
          [
            {
              ...pendingIntents[0]!,
              actor: "borg",
              status: "completed",
              evidence: "The response reviewed the deploy status.",
            },
            {
              ...pendingIntents[1]!,
              actor: "user",
              status: "abandoned",
              evidence: "The user said not to create an incident.",
            },
            {
              description: "Hallucinated intent",
              next_action: null,
              actor: "borg",
              status: "completed",
              evidence: "Not present in prior pending actions.",
            },
          ],
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "haiku",
      proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
    });
    const workingMemory = createWorkingMemoryFixture({
      turn_counter: 2,
      hot_entities: ["Atlas"],
      pending_actions: pendingIntents,
      mode: "problem_solving" as const,
    });
    const currentTurnStreamEntryIds = [createStreamEntryId(), createStreamEntryId()];

    const reflected = await reflector.reflect(
      {
        userMessage: "The rollout passed and don't open a new incident.",
        workingMemory,
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Rollout status is clean. I will write a rollback note.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [newIntent],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Rollout status is clean. I will write a rollback note.",
          tool_calls: [],
          intents: [newIntent],
          workingMemory: {
            ...workingMemory,
            pending_actions: [...pendingIntents, newIntent],
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(),
        currentTurnStreamEntryIds,
      },
      harness.streamWriter,
    );

    expect(reflected.pending_actions).toEqual([newIntent]);
    expect(harness.actionRepository.list({ state: "completed" })).toEqual([
      expect.objectContaining({
        description: "Check the Atlas rollout after tests finish",
        actor: "borg",
        state: "completed",
        completed_at: 1_000_000,
        provenance_stream_entry_ids: currentTurnStreamEntryIds,
      }),
    ]);
    expect(harness.actionRepository.list({ state: "not_done" })).toEqual([
      expect.objectContaining({
        description: "Open a new incident if rollout fails",
        actor: "user",
        state: "not_done",
        not_done_at: 1_000_000,
        provenance_stream_entry_ids: currentTurnStreamEntryIds,
      }),
    ]);
    expect(llm.requests[0]?.messages[0]?.content).toContain("pending_actions");
  });

  it("preserves a pending procedural attempt when reflection returns no procedural outcome for it", async () => {
    // Sprint 53: omitted outcomes leave the attempt pending so a later
    // turn can grade it. The orchestrator expires attempts via TTL.
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [createReflectionResponse([], [])],
      }),
    });
    cleanup.push(harness.cleanup);
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "haiku",
      proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
    });

    const reflected = await reflector.reflect(
      createPendingProceduralReflectionContext(),
      harness.streamWriter,
    );

    expect(reflected.pending_procedural_attempts).toHaveLength(1);
    expect(harness.proceduralEvidenceRepository.list()).toEqual([]);
  });

  it("preserves a pending procedural attempt when reflection judgment fails", async () => {
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          () => {
            throw new Error("reflection unavailable");
          },
        ],
      }),
    });
    cleanup.push(harness.cleanup);
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "haiku",
      proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
    });

    const reflected = await reflector.reflect(
      createPendingProceduralReflectionContext(),
      harness.streamWriter,
    );
    const events = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(1);

    expect(reflected.pending_procedural_attempts).toHaveLength(1);
    expect(harness.proceduralEvidenceRepository.list()).toEqual([]);
    expect(events[0]).toMatchObject({
      kind: "internal_event",
      content: {
        hook: "reflection_judgment",
      },
    });
  });

  it("counts an episode as used when the planner referenced it", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new FixedClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(episodicMigrations, selfMigrations, retrievalMigrations),
    });
    const table = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const episodicRepository = new EpisodicRepository({
      table,
      db,
      clock,
    });
    const goalsRepository = new GoalsRepository({
      db,
      clock,
    });
    const traitsRepository = new TraitsRepository({
      db,
      clock,
    });
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      clock,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episode = await episodicRepository.insert({
      id: "ep_bbbbbbbbbbbbbbbb" as never,
      title: "Unexpected rollback plan",
      narrative: "Database migration blocked the release and required a rollback.",
      participants: ["ops-team"],
      location: null,
      start_time: 0,
      end_time: 1,
      source_stream_ids: ["strm_bbbbbbbbbbbbbbbb" as never],
      significance: 0.8,
      tags: ["ops"],
      confidence: 0.8,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 0,
      updated_at: 0,
    });
    const suppressionSet = new SuppressionSet(1);
    const reflector = new Reflector({
      clock,
      episodicRepository,
      goalsRepository,
      traitsRepository,
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.9,
      scoreBreakdown: createRetrievalScoreFixture({
        similarity: 0.9,
        decayedSalience: 0.3,
        heat: 1,
        goalRelevance: 0.2,
        timeRelevance: 0,
        suppressionPenalty: 0,
      }),
      citationChain: [],
    };

    await reflector.reflect(
      {
        userMessage: "How should we recover the release?",
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: [],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Use the safer recovery path.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Use the safer recovery path.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: [],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet,
      },
      writer,
    );

    expect(episodicRepository.getStats(episode.id)?.use_count).toBe(1);
  });

  it("persists LLM-emitted open questions with structured urgency and provenance", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Atlas uncertainty",
        narrative: "The logs were incomplete and the root cause stayed unclear.",
        tags: ["atlas"],
      }),
    );
    const retrieved = createRetrievedEpisode(episode, 0.92);
    llm.pushResponse(
      createReflectionResponse(
        [],
        [],
        [],
        [],
        [],
        [],
        [
          {
            question: "¿Qué evidencia falta sobre Atlas?",
            urgency: 0.73,
            related_episode_ids: [episode.id],
          },
        ],
      ),
    );
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "claude-opus-4-7",
      identityService: harness.identityService,
    });
    const workingMemory = createWorkingMemoryFixture({
      hot_entities: ["Atlas"],
      mode: "reflective",
    });

    await reflector.reflect(
      {
        userMessage: "Why is Atlas still failing?",
        workingMemory,
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "I still need to compare more evidence.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "low confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I still need to compare more evidence.",
          tool_calls: [],
          intents: [],
          workingMemory,
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence({
          overall: 0.2,
          evidenceStrength: 0.1,
          coverage: 0.2,
          sampleSize: 1,
        }),
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    const promptContext = JSON.parse(llm.requests[0]?.messages[0]?.content ?? "{}") as {
      retrieval_confidence?: RetrievalConfidence;
    };

    expect(promptContext.retrieval_confidence?.overall).toBe(0.2);
    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([
      expect.objectContaining({
        question: "¿Qué evidencia falta sobre Atlas?",
        urgency: 0.73,
        source: "reflection",
        related_episode_ids: [episode.id],
        provenance: {
          kind: "episodes",
          episode_ids: [episode.id],
        },
      }),
    ]);
  });

  it("logs and skips LLM-emitted open questions when identity service is unavailable", async () => {
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [],
            [],
            [],
            [],
            [],
            [
              {
                question: "What uncertainty remains about Atlas?",
                urgency: 0.6,
                related_episode_ids: [],
              },
            ],
          ),
        ],
      }),
    });
    cleanup.push(harness.cleanup);

    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "claude-opus-4-7",
    });
    const workingMemory = createWorkingMemoryFixture({
      hot_entities: ["Atlas"],
      mode: "reflective",
    });

    await reflector.reflect(
      {
        userMessage: "Why is Atlas still failing?",
        workingMemory,
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "I still need to compare more evidence.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "low confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I still need to compare more evidence.",
          tool_calls: [],
          intents: [],
          workingMemory,
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence({
          overall: 0.2,
          evidenceStrength: 0.1,
          coverage: 0.2,
          sampleSize: 1,
        }),
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    const entries = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(1);

    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
    expect(entries[0]).toMatchObject({
      kind: "internal_event",
      content: {
        hook: "reflection_open_question",
        error: "Error: identity_service_unavailable",
      },
    });
  });

  it("routes reflector-created open questions through identity events", async () => {
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [],
            [],
            [],
            [],
            [],
            [
              {
                question: "What uncertainty remains about Atlas?",
                urgency: 0.6,
                related_episode_ids: [],
              },
            ],
          ),
        ],
      }),
    });
    cleanup.push(harness.cleanup);

    const audienceEntityId = harness.entityRepository.resolve("Bob");
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "claude-opus-4-7",
      identityService: harness.identityService,
    });
    const workingMemory = createWorkingMemoryFixture({
      hot_entities: ["Atlas"],
      mode: "reflective",
    });

    await reflector.reflect(
      {
        userMessage: "Why is Atlas still failing?",
        workingMemory,
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "I still need to compare more evidence.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "low confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I still need to compare more evidence.",
          tool_calls: [],
          intents: [],
          workingMemory,
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence({
          overall: 0.2,
          evidenceStrength: 0.1,
          coverage: 0.2,
          sampleSize: 1,
        }),
        audienceEntityId,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    const openQuestion = harness.openQuestionsRepository.list({ status: "open" })[0];

    expect(openQuestion).toEqual(
      expect.objectContaining({
        source: "reflection",
        audience_entity_id: audienceEntityId,
      }),
    );
    expect(
      harness.identityEventRepository.list({
        recordType: "open_question",
        recordId: openQuestion?.id,
      }),
    ).toEqual([
      expect.objectContaining({
        action: "create",
        record_type: "open_question",
        provenance: {
          kind: "online",
          process: "reflector",
        },
      }),
    ]);
  });

  it("resolves active open questions with retrieved episode evidence", async () => {
    const episode = createEpisodeFixture({
      id: "ep_aaaaaaaaaaaaaaaa" as never,
      title: "Atlas rollback answer",
      narrative: "Atlas stabilized after rollback rehearsal.",
    });
    const retrieved = createRetrievedEpisode(episode);
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const q1 = harness.openQuestionsRepository.add({
      question: "Why did Atlas recover?",
      urgency: 0.8,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    const q2 = harness.openQuestionsRepository.add({
      question: "What remains unclear about Atlas?",
      urgency: 0.7,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    llm.pushResponse(
      createReflectionResponse(
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [
          {
            question_id: q1.id,
            resolution_note: "Atlas recovered after rollback rehearsal.",
            evidence_episode_ids: [episode.id],
            evidence_stream_entry_ids: [],
          },
        ],
      ),
    );
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: llm,
      model: "claude-opus-4-7",
      identityService: harness.identityService,
      openQuestionsRepository: harness.openQuestionsRepository,
    });

    await reflector.reflect(
      {
        ...createOpenQuestionReflectionContext({
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          response: "Atlas recovered after rollback rehearsal.",
        }),
        activeOpenQuestions: [q1, q2],
      },
      harness.streamWriter,
    );

    expect(harness.openQuestionsRepository.get(q1.id)).toMatchObject({
      status: "resolved",
      resolution_evidence_episode_ids: [episode.id],
      resolution_evidence_stream_entry_ids: [],
      resolution_note: "Atlas recovered after rollback rehearsal.",
    });
    expect(harness.openQuestionsRepository.get(q2.id)?.status).toBe("open");
    expect(
      harness.identityEventRepository.list({
        recordType: "open_question",
        recordId: q1.id,
      }),
    ).toEqual([
      expect.objectContaining({
        action: "resolve",
        provenance: {
          kind: "online_reflector",
          evidence_episode_ids: [episode.id],
          evidence_stream_entry_ids: [],
        },
      }),
    ]);
  });

  it("resolves active open questions with current-turn stream evidence only", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const streamEntryId = "strm_aaaaaaaaaaaaaaaa" as never;
    const question = harness.openQuestionsRepository.add({
      question: "What did this turn clarify?",
      urgency: 0.8,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    llm.pushResponse(
      createReflectionResponse(
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [
          {
            question_id: question.id,
            resolution_note: "The current turn clarified the answer.",
            evidence_episode_ids: [],
            evidence_stream_entry_ids: [streamEntryId],
          },
        ],
      ),
    );
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: llm,
      model: "claude-opus-4-7",
      identityService: harness.identityService,
      openQuestionsRepository: harness.openQuestionsRepository,
    });

    await reflector.reflect(
      {
        ...createOpenQuestionReflectionContext({
          response: "The current turn clarified the answer.",
        }),
        activeOpenQuestions: [question],
        currentTurnStreamEntryIds: [streamEntryId],
      },
      harness.streamWriter,
    );

    expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
      status: "resolved",
      resolution_evidence_episode_ids: [],
      resolution_evidence_stream_entry_ids: [streamEntryId],
    });
  });

  it.each([
    {
      name: "source-less",
      reason: "no_evidence",
      active: true,
      evidence_episode_ids: [],
      evidence_stream_entry_ids: [],
    },
    {
      name: "unknown question",
      reason: "unknown_question",
      active: false,
      evidence_episode_ids: [],
      evidence_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa"],
    },
    {
      name: "hallucinated episode id",
      reason: "unknown_episode",
      active: true,
      evidence_episode_ids: ["ep_bbbbbbbbbbbbbbbb"],
      evidence_stream_entry_ids: [],
    },
    {
      name: "hallucinated stream id",
      reason: "unknown_stream",
      active: true,
      evidence_episode_ids: [],
      evidence_stream_entry_ids: ["strm_bbbbbbbbbbbbbbbb"],
    },
  ])(
    "rejects $name open-question resolutions with a degraded trace event",
    async ({ reason, active, evidence_episode_ids, evidence_stream_entry_ids }) => {
      const tracer = new CaptureTracer();
      const llm = new FakeLLMClient();
      const harness = await createOfflineTestHarness({
        llmClient: llm,
      });
      cleanup.push(harness.cleanup);
      const question = harness.openQuestionsRepository.add({
        question: "What resolution should be validated?",
        urgency: 0.8,
        source: "reflection",
        provenance: { kind: "manual" },
      });
      llm.pushResponse(
        createReflectionResponse(
          [],
          [],
          [],
          [],
          [],
          [],
          [],
          [
            {
              question_id: question.id,
              resolution_note: "Validation should decide whether this closes.",
              evidence_episode_ids,
              evidence_stream_entry_ids,
            },
          ],
        ),
      );
      const reflector = createHarnessReflector(harness, {
        clock: harness.clock,
        llmClient: llm,
        model: "claude-opus-4-7",
        identityService: harness.identityService,
        openQuestionsRepository: harness.openQuestionsRepository,
        tracer,
      });

      await reflector.reflect(
        {
          ...createOpenQuestionReflectionContext(),
          turnId: "turn_resolution_validation",
          activeOpenQuestions: active ? [question] : [],
          currentTurnStreamEntryIds: ["strm_aaaaaaaaaaaaaaaa" as never],
        },
        harness.streamWriter,
      );

      expect(harness.openQuestionsRepository.get(question.id)?.status).toBe("open");
      expect(tracer.events).toEqual(
        expect.arrayContaining([
          {
            event: "open_question_resolution_degraded",
            data: expect.objectContaining({
              turnId: "turn_resolution_validation",
              reason,
              question_id: question.id,
            }),
          },
        ]),
      );
    },
  );

  it("rejects audience-isolated open-question resolutions outside the active list", async () => {
    const tracer = new CaptureTracer();
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const alice = harness.entityRepository.resolve("Alice");
    const bob = harness.entityRepository.resolve("Bob");
    const streamEntryId = "strm_aaaaaaaaaaaaaaaa" as never;
    const privateQuestion = harness.openQuestionsRepository.add({
      question: "What should Alice know privately?",
      urgency: 0.8,
      audience_entity_id: alice,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    const activeForBob = harness.openQuestionsRepository.list({
      status: "open",
      visibleToAudienceEntityId: bob,
      limit: 20,
    });

    expect(activeForBob.map((question) => question.id)).not.toContain(privateQuestion.id);
    llm.pushResponse(
      createReflectionResponse(
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [
          {
            question_id: privateQuestion.id,
            resolution_note: "This should not cross audience scope.",
            evidence_episode_ids: [],
            evidence_stream_entry_ids: [streamEntryId],
          },
        ],
      ),
    );
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: llm,
      model: "claude-opus-4-7",
      identityService: harness.identityService,
      openQuestionsRepository: harness.openQuestionsRepository,
      tracer,
    });

    await reflector.reflect(
      {
        ...createOpenQuestionReflectionContext(),
        turnId: "turn_audience_isolation",
        audienceEntityId: bob,
        activeOpenQuestions: activeForBob,
        currentTurnStreamEntryIds: [streamEntryId],
      },
      harness.streamWriter,
    );

    expect(harness.openQuestionsRepository.get(privateQuestion.id)?.status).toBe("open");
    expect(tracer.events).toEqual(
      expect.arrayContaining([
        {
          event: "open_question_resolution_degraded",
          data: expect.objectContaining({
            reason: "unknown_question",
            question_id: privateQuestion.id,
          }),
        },
      ]),
    );
  });

  it("skips already resolved open questions when a stale active list is replayed", async () => {
    const tracer = new CaptureTracer();
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);
    const streamEntryId = "strm_aaaaaaaaaaaaaaaa" as never;
    const question = harness.openQuestionsRepository.add({
      question: "What was already resolved?",
      urgency: 0.8,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    harness.openQuestionsRepository.resolve(question.id, {
      resolution_evidence_episode_ids: [],
      resolution_evidence_stream_entry_ids: [streamEntryId],
      resolution_note: "Resolved before reflection replay.",
    });
    llm.pushResponse(
      createReflectionResponse(
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [
          {
            question_id: question.id,
            resolution_note: "A second resolution should be ignored.",
            evidence_episode_ids: [],
            evidence_stream_entry_ids: [streamEntryId],
          },
        ],
      ),
    );
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: llm,
      model: "claude-opus-4-7",
      identityService: harness.identityService,
      openQuestionsRepository: harness.openQuestionsRepository,
      tracer,
    });

    await reflector.reflect(
      {
        ...createOpenQuestionReflectionContext(),
        turnId: "turn_resolution_idempotency",
        activeOpenQuestions: [question],
        currentTurnStreamEntryIds: [streamEntryId],
      },
      harness.streamWriter,
    );

    expect(harness.openQuestionsRepository.get(question.id)).toMatchObject({
      status: "resolved",
      resolution_note: "Resolved before reflection replay.",
    });
    expect(
      harness.identityEventRepository.list({
        recordType: "open_question",
        recordId: question.id,
      }),
    ).toEqual([]);
    expect(tracer.events).toEqual(
      expect.arrayContaining([
        {
          event: "open_question_resolution_degraded",
          data: expect.objectContaining({
            reason: "not_open",
            question_id: question.id,
          }),
        },
      ]),
    );
  });

  it("does not add an open question when reflection output has an empty open_questions array", async () => {
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [createReflectionResponse()],
      }),
    });
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Atlas settled cause",
        narrative: "Atlas failures were traced to a known rollback gap.",
        tags: ["atlas"],
      }),
    );
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "claude-opus-4-7",
      identityService: harness.identityService,
    });
    const retrieved = createRetrievedEpisode(episode, 0.1);
    const workingMemory = createWorkingMemoryFixture({
      hot_entities: ["Atlas"],
      mode: "reflective",
    });

    await reflector.reflect(
      {
        userMessage: "Why is Atlas still failing?",
        workingMemory,
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "The rollback gap explains the failure.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "low score",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "The rollback gap explains the failure.",
          tool_calls: [],
          intents: [],
          workingMemory,
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence({
          overall: 0.2,
          evidenceStrength: 0.2,
          coverage: 0.2,
          sampleSize: 1,
        }),
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
  });

  it("does not add an open question when the reflection tool input omits open_questions", async () => {
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          createRawReflectionResponse({
            advanced_goals: [],
            procedural_outcomes: [],
            trait_demonstrations: [],
            intent_updates: [],
          }),
        ],
      }),
    });
    cleanup.push(harness.cleanup);
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "claude-opus-4-7",
      identityService: harness.identityService,
    });

    await reflector.reflect(createOpenQuestionReflectionContext(), harness.streamWriter);

    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
  });

  it.each([
    {
      name: "six proposals",
      openQuestions: Array.from({ length: 6 }, (_, index) => ({
        question: `Question ${index + 1}?`,
        urgency: 0.5,
        related_episode_ids: [],
      })),
    },
    {
      name: "negative urgency",
      openQuestions: [
        {
          question: "What uncertainty remains about Atlas?",
          urgency: -0.1,
          related_episode_ids: [],
        },
      ],
    },
    {
      name: "too-large urgency",
      openQuestions: [
        {
          question: "What uncertainty remains about Atlas?",
          urgency: 1.1,
          related_episode_ids: [],
        },
      ],
    },
    {
      name: "invalid episode id format",
      openQuestions: [
        {
          question: "What uncertainty remains about Atlas?",
          urgency: 0.6,
          related_episode_ids: ["not-a-real-id"],
        },
      ],
    },
  ])(
    "rejects malformed open_questions from reflection output: $name",
    async ({ openQuestions }) => {
      const harness = await createOfflineTestHarness({
        llmClient: new FakeLLMClient({
          responses: [
            createRawReflectionResponse({
              open_questions: openQuestions,
            }),
          ],
        }),
      });
      cleanup.push(harness.cleanup);
      const reflector = createHarnessReflector(harness, {
        clock: harness.clock,
        llmClient: harness.llmClient,
        model: "claude-opus-4-7",
        identityService: harness.identityService,
      });

      await reflector.reflect(createOpenQuestionReflectionContext(), harness.streamWriter);

      expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([]);
      expect(
        new StreamReader({
          dataDir: harness.tempDir,
          sessionId: DEFAULT_SESSION_ID,
        })
          .tail(5)
          .some(
            (entry) =>
              entry.kind === "internal_event" &&
              (entry.content as { hook?: string }).hook === "reflection_judgment",
          ),
      ).toBe(true);
    },
  );

  it("drops format-valid episode ids that are not referenced by the reflection context", async () => {
    const hallucinatedEpisodeId = "ep_aaaaaaaaaaaaaaaa";
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [],
            [],
            [],
            [],
            [],
            [
              {
                question: "What uncertainty remains about Atlas?",
                urgency: 0.6,
                related_episode_ids: [hallucinatedEpisodeId],
              },
            ],
          ),
        ],
      }),
    });
    cleanup.push(harness.cleanup);
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "claude-opus-4-7",
      identityService: harness.identityService,
    });

    await reflector.reflect(createOpenQuestionReflectionContext(), harness.streamWriter);

    expect(harness.openQuestionsRepository.list({ status: "open" })).toEqual([
      expect.objectContaining({
        question: "What uncertainty remains about Atlas?",
        related_episode_ids: [],
        provenance: {
          kind: "online",
          process: "reflector",
        },
      }),
    ]);
    expect(
      new StreamReader({
        dataDir: harness.tempDir,
        sessionId: DEFAULT_SESSION_ID,
      })
        .tail(5)
        .some(
          (entry) =>
            entry.kind === "internal_event" &&
            (entry.content as { hook?: string; dropped_episode_ids?: string[] }).hook ===
              "reflection_open_question_filtered_episode_ids" &&
            (
              entry.content as {
                dropped_episode_ids?: string[];
              }
            ).dropped_episode_ids?.includes(hallucinatedEpisodeId) === true,
        ),
    ).toBe(true);
  });

  it("logs and continues when the reflection open-question hook fails", async () => {
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [],
            [],
            [],
            [],
            [],
            [
              {
                question: "What uncertainty remains about Atlas?",
                urgency: 0.6,
                related_episode_ids: [],
              },
            ],
          ),
        ],
      }),
    });
    cleanup.push(harness.cleanup);

    const brokenIdentityService = {
      addOpenQuestion() {
        throw new Error("hook exploded");
      },
      updateGoal() {
        throw new Error("unexpected goal update");
      },
      updateGoalProgressFromReflection() {
        throw new Error("unexpected goal progress update");
      },
      resolveOpenQuestion() {
        throw new Error("unexpected open question resolution");
      },
    };
    const reflector = new Reflector({
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "claude-opus-4-7",
      episodicRepository: harness.episodicRepository,
      goalsRepository: harness.goalsRepository,
      traitsRepository: harness.traitsRepository,
      identityService: brokenIdentityService,
    });
    const workingMemory = createWorkingMemoryFixture({
      hot_entities: ["Atlas"],
      mode: "reflective",
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "Why is Atlas still failing?",
        workingMemory,
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "I still need to compare more evidence.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "low confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I still need to compare more evidence.",
          tool_calls: [],
          intents: [],
          workingMemory,
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence({
          overall: 0.2,
          evidenceStrength: 0.2,
          coverage: 0.2,
          sampleSize: 1,
        }),
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    const entries = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    }).tail(1);

    expect(reflected.turn_counter).toBe(1);
    expect(entries[0]).toMatchObject({
      kind: "internal_event",
      content: {
        hook: "reflection_open_question",
      },
    });
  });

  it.each(["success", "failure", "unclear"] as const)(
    "stages structured procedural %s outcomes as evidence",
    async (classification) => {
      const harness = await createOfflineTestHarness({
        llmClient: new FakeLLMClient({
          responses: [
            createReflectionResponse(
              [],
              [
                {
                  classification,
                  evidence:
                    classification === "success"
                      ? "User confirmed the fix worked."
                      : classification === "failure"
                        ? "User reported the same error is still happening."
                        : "User replied without saying whether the attempt worked.",
                },
              ],
            ),
          ],
        }),
      });
      cleanup.push(harness.cleanup);

      const episode = await harness.episodicRepository.insert(
        createEpisodeFixture({
          title: "Rust lifetime attempt",
          source_stream_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"] as never,
        }),
      );
      const skill = await harness.skillRepository.add({
        applies_when: "Rust lifetime debugging",
        approach: "Shrink borrow scopes and use intermediate bindings.",
        sourceEpisodes: [episode.id],
      });
      const reflector = createHarnessReflector(harness, {
        clock: harness.clock,
        llmClient: harness.llmClient,
        model: "haiku",
        skillRepository: harness.skillRepository,
        proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
      });

      const reflected = await reflector.reflect(
        {
          userMessage:
            classification === "success"
              ? "That worked, thanks."
              : classification === "failure"
                ? "That didn't work; the same error remains."
                : "I need to look at it again.",
          perception: {
            entities: ["Rust"],
            mode: "problem_solving",
            affectiveSignal: {
              valence: 0,
              arousal: 0,
              dominant_emotion: null,
            },
            temporalCue: null,
          },
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 2,
            hot_entities: ["Rust"],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            pending_procedural_attempts: [
              {
                problem_text: "I hit a Rust lifetime issue again.",
                approach_summary: "Shrink borrow scopes and use intermediate bindings.",
                selected_skill_id: skill.id,
                source_stream_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"] as never,
                turn_counter: 1,
                audience_entity_id: null,
              },
            ],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            mode: "problem_solving",
            updated_at: 0,
          },
          selfSnapshot: {
            values: [],
            goals: [],
            traits: [],
          },
          deliberationResult: {
            path: "system_1",
            response: "Next response.",
            thoughts: [],
            tool_calls: [],
            usage: {
              input_tokens: 1,
              output_tokens: 1,
              stop_reason: "end_turn",
            },
            decision_reason: "confidence",
            retrievedEpisodes: [],
            referencedEpisodeIds: null,
            intents: [],
            thoughtsPersisted: false,
          },
          actionResult: {
            response: "Next response.",
            tool_calls: [],
            intents: [],
            workingMemory: {
              session_id: DEFAULT_SESSION_ID,
              turn_counter: 2,
              hot_entities: ["Rust"],
              pending_actions: [],
              pending_social_attribution: null,
              pending_trait_attribution: null,
              suppressed: [],
              mood: null,
              pending_procedural_attempts: [
                {
                  problem_text: "I hit a Rust lifetime issue again.",
                  approach_summary: "Shrink borrow scopes and use intermediate bindings.",
                  selected_skill_id: skill.id,
                  source_stream_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"] as never,
                  turn_counter: 1,
                  audience_entity_id: null,
                },
              ],
              discourse_state: {
                stop_until_substantive_content: null,
              },
              mode: "problem_solving",
              updated_at: 0,
            },
          },
          retrievedEpisodes: [],
          retrievalConfidence: createRetrievalConfidence(),
          selectedSkillId: null,
          suppressionSet: new SuppressionSet(2),
        },
        harness.streamWriter,
      );

      // Sprint 53: only actionable (success/failure) outcomes retire the
      // attempt. Grounded "unclear" still records evidence but keeps the
      // attempt pending so a later turn can grade it.
      if (classification === "unclear") {
        expect(reflected.pending_procedural_attempts).toHaveLength(1);
      } else {
        expect(reflected.pending_procedural_attempts).toEqual([]);
      }
      expect(harness.proceduralEvidenceRepository.list()).toEqual([
        expect.objectContaining({
          classification,
          resolved_episode_ids: [episode.id],
          audience_entity_id: null,
        }),
      ]);

      if (classification === "success") {
        expect(harness.skillRepository.get(skill.id)).toMatchObject({
          attempts: 1,
          successes: 1,
          alpha: 2,
        });
      } else if (classification === "failure") {
        expect(harness.skillRepository.get(skill.id)).toMatchObject({
          attempts: 1,
          failures: 1,
          beta: 2,
        });
      } else {
        expect(harness.skillRepository.get(skill.id)?.attempts).toBe(0);
      }
    },
  );

  it("stores non-applied skill evidence without crediting the skill posterior", async () => {
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [
              {
                classification: "success",
                evidence:
                  "User confirmed the workaround helped, but the selected approach was not used.",
                skill_actually_applied: false,
              },
            ],
          ),
        ],
      }),
    });
    cleanup.push(harness.cleanup);

    const sourceStreamIds = ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"] as never;
    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Rust lifetime workaround",
        source_stream_ids: sourceStreamIds,
      }),
    );
    const skill = await harness.skillRepository.add({
      applies_when: "Rust lifetime debugging",
      approach: "Shrink borrow scopes and use intermediate bindings.",
      sourceEpisodes: [episode.id],
    });
    const pendingAttempt = {
      problem_text: "I hit a Rust lifetime issue again.",
      approach_summary: "Shrink borrow scopes and use intermediate bindings.",
      selected_skill_id: skill.id,
      source_stream_ids: sourceStreamIds,
      turn_counter: 1,
      audience_entity_id: null,
    };
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "haiku",
      skillRepository: harness.skillRepository,
      proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "That worked after I changed the whole shape of the code.",
        perception: {
          entities: ["Rust"],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 2,
          hot_entities: ["Rust"],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          suppressed: [],
          mood: null,
          pending_procedural_attempts: [pendingAttempt],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Next response.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Next response.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 2,
            hot_entities: ["Rust"],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            pending_procedural_attempts: [pendingAttempt],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        selectedSkillId: null,
        suppressionSet: new SuppressionSet(2),
      },
      harness.streamWriter,
    );

    expect(reflected.pending_procedural_attempts).toEqual([]);
    expect(harness.proceduralEvidenceRepository.list()).toEqual([
      expect.objectContaining({
        classification: "success",
        skill_actually_applied: false,
        resolved_episode_ids: [episode.id],
      }),
    ]);
    expect(harness.skillRepository.get(skill.id)?.attempts).toBe(0);
  });

  it("does not infer procedural success from assistant wording alone", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
    });

    await reflector.reflect(
      {
        userMessage: "I'm frustrated and tired of this.",
        perception: {
          entities: ["Rust"],
          mode: "problem_solving",
          affectiveSignal: {
            valence: -0.8,
            arousal: 0.3,
            dominant_emotion: "sadness",
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: ["Rust"],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          suppressed: [],
          mood: null,
          pending_procedural_attempts: [
            {
              problem_text: "I hit a Rust lifetime issue again.",
              approach_summary: "This works when you shrink the borrow scope.",
              selected_skill_id: null,
              source_stream_ids: ["strm_aaaaaaaaaaaaaaaa"] as never,
              turn_counter: 1,
              audience_entity_id: null,
            },
          ],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "This works when you shrink the borrow scope.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "This works when you shrink the borrow scope.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: ["Rust"],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            pending_procedural_attempts: [
              {
                problem_text: "I hit a Rust lifetime issue again.",
                approach_summary: "This works when you shrink the borrow scope.",
                selected_skill_id: null,
                source_stream_ids: ["strm_aaaaaaaaaaaaaaaa"] as never,
                turn_counter: 1,
                audience_entity_id: null,
              },
            ],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        selectedSkillId: null,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.proceduralEvidenceRepository.list()).toEqual([]);
  });

  it("rejects self-validating success evidence grounded only in assistant wording", async () => {
    const harness = await createOfflineTestHarness({
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [
              {
                classification: "success",
                evidence: "The assistant response said this works.",
                grounded: false,
              },
            ],
          ),
        ],
      }),
    });
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture({
      title: "Rust lifetime frustration",
      narrative: "Rust lifetime errors kept blocking progress.",
      tags: ["rust", "lifetimes"],
    });
    await harness.episodicRepository.insert(episode);

    const skill = await harness.skillRepository.add({
      applies_when: "Rust lifetime debugging",
      approach: "Shrink borrow scopes and use intermediate bindings.",
      sourceEpisodes: [episode.id],
    });
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: harness.llmClient,
      model: "haiku",
      skillRepository: harness.skillRepository,
      proceduralEvidenceRepository: harness.proceduralEvidenceRepository,
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "I'm frustrated and tired of this.",
        perception: {
          entities: ["Rust"],
          mode: "problem_solving",
          affectiveSignal: {
            valence: -0.5,
            arousal: 0.5,
            dominant_emotion: "anger",
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: ["Rust"],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          suppressed: [],
          mood: null,
          pending_procedural_attempts: [
            {
              problem_text: "I hit a Rust lifetime issue again.",
              approach_summary: "Shrink borrow scopes and use intermediate bindings.",
              selected_skill_id: skill.id,
              source_stream_ids: episode.source_stream_ids,
              turn_counter: 1,
              audience_entity_id: null,
            },
          ],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Let's narrow the borrow lifetime a bit more.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Let's narrow the borrow lifetime a bit more.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: ["Rust"],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            suppressed: [],
            mood: null,
            pending_procedural_attempts: [
              {
                problem_text: "I hit a Rust lifetime issue again.",
                approach_summary: "Shrink borrow scopes and use intermediate bindings.",
                selected_skill_id: skill.id,
                source_stream_ids: episode.source_stream_ids,
                turn_counter: 1,
                audience_entity_id: null,
              },
            ],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        selectedSkillId: null,
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    // Sprint 53: ungrounded outcomes leave the attempt pending instead of
    // unconditionally clearing it. The attempt may get graded on a later
    // turn or expire via TTL.
    expect(reflected.pending_procedural_attempts).toHaveLength(1);
    expect(harness.proceduralEvidenceRepository.list()).toEqual([]);
    expect(harness.skillRepository.get(skill.id)?.attempts).toBe(0);
  });

  it("runs trait judgment on ordinary user turns with no goals, intents, attempts, or referenced episodes", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse(
          [],
          [],
          [],
          [
            {
              trait_label: "careful",
              evidence: "The response checked the user's constraint before answering.",
              strength_delta: 0.05,
            },
          ],
        ),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    cleanup.push(harness.cleanup);

    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "Please keep the answer short.",
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: [],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Short answer with the requested constraint respected.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Short answer with the requested constraint respected.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: [],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(1),
        currentTurnStreamEntryIds: ["strm_aaaaaaaaaaaaaaaa" as never],
      },
      harness.streamWriter,
    );

    expect(llm.requests).toHaveLength(1);
    expect(reflected.pending_trait_attribution).toMatchObject({
      trait_label: "careful",
      strength_delta: 0.05,
      source_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa"],
    });
  });

  it("anchors S2 trait demonstrations to the current turn's stream entries", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episodeA = await harness.episodicRepository.insert(
      createEpisodeFixture({
        id: "ep_aaaaaaaaaaaaaaaa" as never,
        title: "Planning sync A",
      }),
    );
    const episodeB = await harness.episodicRepository.insert(
      createEpisodeFixture({
        id: "ep_bbbbbbbbbbbbbbbb" as never,
        title: "Planning sync B",
      }),
    );
    const retrievedA = createRetrievedEpisode(episodeA);
    const retrievedB = createRetrievedEpisode(episodeB);
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [],
            [],
            [
              {
                trait_label: "engaged",
                evidence: "The response used both planning syncs as concrete evidence.",
                strength_delta: 0.07,
              },
            ],
          ),
        ],
      }),
      model: "haiku",
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "Let's work through the plan.",
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: [],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "Use both planning syncs as evidence.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "reflective",
          retrievedEpisodes: [retrievedA, retrievedB],
          referencedEpisodeIds: [episodeA.id, episodeB.id],
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Use both planning syncs as evidence.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: [],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrievedA, retrievedB],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(1),
        currentTurnStreamEntryIds: [
          "strm_aaaaaaaaaaaaaaaa" as never,
          "strm_bbbbbbbbbbbbbbbb" as never,
        ],
      },
      harness.streamWriter,
    );

    // Sprint 56: trait demonstration evidence is the current turn that
    // displayed the trait, not arbitrary memories the planner referenced.
    expect(reflected.pending_trait_attribution).toMatchObject({
      trait_label: "engaged",
      strength_delta: 0.07,
      source_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"],
    });
  });

  it("does not attach S2 trait evidence when no current-turn stream entries are provided", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(createEpisodeFixture());
    const retrieved = createRetrievedEpisode(episode);
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [],
            [],
            [
              {
                trait_label: "focused",
                evidence: "The response narrowed the evidence to the retrieved episode.",
                strength_delta: 0.04,
              },
            ],
          ),
        ],
      }),
      model: "haiku",
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "Let's work through the plan.",
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: [],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "No episode evidence was needed.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "reflective",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [],
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "No episode evidence was needed.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: [],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(reflected.pending_trait_attribution).toBeNull();
  });

  it("anchors S2 trait evidence to current-turn stream entries even when planner referenced episodes", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        id: "ep_aaaaaaaaaaaaaaaa" as never,
      }),
    );
    const retrieved = createRetrievedEpisode(episode);
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [],
            [],
            [
              {
                trait_label: "focused",
                evidence: "The response narrowed the evidence to the retrieved episode.",
                strength_delta: 0.04,
              },
            ],
          ),
        ],
      }),
      model: "haiku",
    });

    const reflected = await reflector.reflect(
      {
        userMessage: "Let's work through the plan.",
        perception: {
          entities: [],
          mode: "problem_solving",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: [],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "problem_solving",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "Only the retrieved episode should count.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "reflective",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id, "ep_bbbbbbbbbbbbbbbb"],
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Only the retrieved episode should count.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: [],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "problem_solving",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(1),
        currentTurnStreamEntryIds: ["strm_aaaaaaaaaaaaaaaa" as never],
      },
      harness.streamWriter,
    );

    // Sprint 56: planner-referenced episodes are no longer the source of
    // trait evidence; the current turn that demonstrated the trait is.
    expect(reflected.pending_trait_attribution).toMatchObject({
      trait_label: "focused",
      strength_delta: 0.04,
      source_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa"],
    });
  });

  it("does not reinforce traits when no current-turn stream entries are available", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
    });

    await reflector.reflect(
      {
        userMessage: "Thinking out loud.",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: [],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "reflective",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_1",
          response: "Staying with it.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Staying with it.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: [],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.traitsRepository.list()).toEqual([]);
  });

  it("uses LLM-judged trait demonstrations for pending attribution", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture({
      title: "Reflective walk",
      narrative: "A slow reflective walk helped untangle a hard feeling.",
      tags: ["reflection"],
    });
    await harness.episodicRepository.insert(episode);

    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: new FakeLLMClient({
        responses: [
          createReflectionResponse(
            [],
            [],
            [],
            [
              {
                trait_label: "patient",
                evidence: "The response stayed with the feeling and traced it carefully.",
                strength_delta: 0.06,
              },
            ],
          ),
        ],
      }),
      model: "haiku",
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.7,
      scoreBreakdown: createRetrievalScoreFixture({
        similarity: 0.7,
        decayedSalience: 0.4,
        heat: 0.2,
        goalRelevance: 0,
        valueAlignment: 0,
        timeRelevance: 0,
        moodBoost: 0,
        socialRelevance: 0,
        suppressionPenalty: 0,
      }),
      citationChain: [],
    };

    const reflected = await reflector.reflect(
      {
        userMessage: "I want to sit with this feeling for a minute.",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: -0.2,
            arousal: 0.1,
            dominant_emotion: "sadness",
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: [],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "reflective",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "Let me stay with it and trace what keeps resurfacing.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "Let me stay with it and trace what keeps resurfacing.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: [],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(1),
        currentTurnStreamEntryIds: [
          "strm_aaaaaaaaaaaaaaaa" as never,
          "strm_bbbbbbbbbbbbbbbb" as never,
        ],
      },
      harness.streamWriter,
    );

    expect(harness.traitsRepository.list()).toEqual([]);
    expect(reflected.pending_trait_attribution).toMatchObject({
      trait_label: "patient",
      strength_delta: 0.06,
      source_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa", "strm_bbbbbbbbbbbbbbbb"],
      audience_entity_id: null,
    });
  });

  it("does not queue pending trait attribution for autonomous turns", async () => {
    const harness = await createOfflineTestHarness();
    cleanup.push(harness.cleanup);

    const episode = await harness.episodicRepository.insert(
      createEpisodeFixture({
        title: "Internal reflective note",
        narrative: "A private autonomous reflection about an earlier feeling.",
      }),
    );
    const llm = new FakeLLMClient({
      responses: [
        createReflectionResponse(
          [],
          [],
          [],
          [
            {
              trait_label: "introspective",
              evidence: "The autonomous response traced the private reflection.",
              strength_delta: 0.05,
            },
          ],
        ),
      ],
    });
    const reflector = createHarnessReflector(harness, {
      clock: harness.clock,
      llmClient: llm,
      model: "haiku",
    });
    const retrieved: RetrievedEpisode = {
      episode,
      score: 0.7,
      scoreBreakdown: createRetrievalScoreFixture({
        similarity: 0.7,
        decayedSalience: 0.4,
        heat: 0.2,
        goalRelevance: 0,
        valueAlignment: 0,
        timeRelevance: 0,
        moodBoost: 0,
        socialRelevance: 0,
        entityRelevance: 0,
        suppressionPenalty: 0,
      }),
      citationChain: [],
    };

    const reflected = await reflector.reflect(
      {
        origin: "autonomous",
        userMessage: "Let me sit with this for a moment.",
        perception: {
          entities: [],
          mode: "reflective",
          affectiveSignal: {
            valence: 0,
            arousal: 0.1,
            dominant_emotion: null,
          },
          temporalCue: null,
        },
        workingMemory: {
          session_id: DEFAULT_SESSION_ID,
          turn_counter: 1,
          hot_entities: [],
          pending_actions: [],
          pending_social_attribution: null,
          pending_trait_attribution: null,
          mood: null,
          pending_procedural_attempts: [],
          discourse_state: {
            stop_until_substantive_content: null,
          },
          suppressed: [],
          mode: "reflective",
          updated_at: 0,
        },
        selfSnapshot: {
          values: [],
          goals: [],
          traits: [],
        },
        deliberationResult: {
          path: "system_2",
          response: "I'll keep tracing this privately.",
          thoughts: [],
          tool_calls: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          decision_reason: "confidence",
          retrievedEpisodes: [retrieved],
          referencedEpisodeIds: [episode.id],
          intents: [],
          thoughtsPersisted: false,
        },
        actionResult: {
          response: "I'll keep tracing this privately.",
          tool_calls: [],
          intents: [],
          workingMemory: {
            session_id: DEFAULT_SESSION_ID,
            turn_counter: 1,
            hot_entities: [],
            pending_actions: [],
            pending_social_attribution: null,
            pending_trait_attribution: null,
            mood: null,
            pending_procedural_attempts: [],
            discourse_state: {
              stop_until_substantive_content: null,
            },
            suppressed: [],
            mode: "reflective",
            updated_at: 0,
          },
        },
        retrievedEpisodes: [retrieved],
        retrievalConfidence: createRetrievalConfidence(),
        suppressionSet: new SuppressionSet(1),
      },
      harness.streamWriter,
    );

    expect(harness.traitsRepository.list()).toEqual([]);
    expect(llm.requests).toHaveLength(0);
    expect(reflected.pending_trait_attribution).toBeNull();
  });
});
