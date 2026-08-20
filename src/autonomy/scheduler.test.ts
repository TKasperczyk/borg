import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { StreamReader, StreamWatermarkRepository, StreamWriter } from "../stream/index.js";
import {
  ToolDispatcher,
  createCommitmentsListTool,
  createIdentityEventsListForCognitionTool,
} from "../tools/index.js";
import { ManualClock } from "../util/clock.js";
import { DEFAULT_SESSION_ID } from "../util/ids.js";
import { createOfflineTestHarness } from "../offline/test-support.js";
import { openDatabase, type SqliteDatabase } from "../storage/sqlite/index.js";
import { LLMError, SessionBusyError } from "../util/errors.js";
import { SelfDecisionRepository } from "../memory/self-decisions/index.js";
import { selectSelfDecisionIntrospection } from "../memory/self-decisions/projection.js";
import { TrainOfThoughtRepository } from "../memory/train-of-thought/index.js";
import type { TurnResult } from "../cognition/index.js";

import {
  createCommitmentExpiringTrigger,
  createGoalFollowupDueTrigger,
  createScheduledReflectionTrigger,
} from "./index.js";
import {
  AutonomyScheduler,
  goalConcernPayload,
  turnEmittedHeadway,
  type AutonomySchedulerOptions,
} from "./scheduler.js";
import type { AutonomyWakeSource } from "./types.js";
import { AutonomyWakesRepository } from "./wakes-repository.js";
import { getExecutiveFocusGoalStaleBackoffProcessName } from "./executive-focus-stale-backoff.js";
import {
  DEFAULT_FLEET_BRAKE_OPTIONS,
  FLEET_BRAKE_PROCESS_NAME,
  readFleetBrakeMetadata,
  type FleetBrakeOptions,
} from "./fleet-brake.js";

function createScheduler(
  options: Omit<AutonomySchedulerOptions, "budgetWindowMs" | "wakeRepository"> & {
    db: SqliteDatabase;
    budgetWindowMs?: number;
    wakeRepository?: AutonomyWakesRepository;
  },
): AutonomyScheduler {
  const { db, budgetWindowMs = 3_600_000, wakeRepository, ...schedulerOptions } = options;

  return new AutonomyScheduler({
    ...schedulerOptions,
    fleetBrake: schedulerOptions.fleetBrake ?? {
      ...DEFAULT_FLEET_BRAKE_OPTIONS,
      enabled: false,
    },
    budgetWindowMs,
    wakeRepository:
      wakeRepository ??
      new AutonomyWakesRepository({
        db,
        clock: schedulerOptions.clock,
      }),
  });
}

function createTestDueSource(
  eventId = "goal_aaaaaaaaaaaaaaaa:no-target:1000",
  sourceName: AutonomyWakeSource["name"] = "goal_followup_due",
  sourceCategory: AutonomyWakeSource["sourceCategory"] = "operational",
): AutonomyWakeSource {
  return {
    name: sourceName,
    type: "trigger",
    sourceCategory,
    async scan() {
      return [
        {
          id: eventId,
          sourceName,
          sourceType: "trigger",
          watermarkProcessName: `autonomy:test:${eventId}`,
          sortTs: 1_000,
          payload: {
            goal_id: "goal_aaaaaaaaaaaaaaaa",
          },
        },
      ];
    },
    buildTurn() {
      return {
        audience: "self",
        stakes: "low",
        userMessage: "Follow up on a goal.",
      };
    },
  };
}

function createExecutiveFocusGoalStaleSource(goalId: string): AutonomyWakeSource {
  return {
    name: "executive_focus_due",
    type: "trigger",
    sourceCategory: "operational",
    async scan() {
      return [
        {
          id: `${goalId}:stale:1000`,
          sourceName: "executive_focus_due",
          sourceType: "trigger",
          watermarkProcessName: `autonomy:test:${goalId}:stale`,
          sortTs: 1_000,
          payload: {
            reason: "goal_stale",
            selected_goal_id: goalId,
            selected_goal: {
              last_progress_ts: 500,
            },
          },
        },
      ];
    },
    buildTurn() {
      return {
        audience: "self",
        stakes: "low",
        userMessage: "Revisit the stale executive-focus goal.",
      };
    },
  };
}

const TEST_FLEET_BRAKE: FleetBrakeOptions = {
  ...DEFAULT_FLEET_BRAKE_OPTIONS,
  emptyStreakThreshold: 5,
  baseCooldownMs: 100,
  cooldownMultiplier: 2,
  maxCooldownMs: 600,
  errorStreakThreshold: 3,
  errorBasePauseMs: 50,
  errorMaxPauseMs: 300,
  freshnessBypassCap: 3,
};

function createStructuralTurnResult(input: {
  emissionKind: "message" | "continue_thought" | "suppressed";
  deliveredOutbound?: boolean;
}): TurnResult {
  const emission: TurnResult["emission"] =
    input.emissionKind === "message"
      ? {
          kind: "message",
          content: "",
          agentMessageId: "strm_test_message" as never,
        }
      : input.emissionKind === "continue_thought"
        ? { kind: "continue_thought" }
        : { kind: "suppressed", reason: "finalizer_no_output" };

  return {
    turn_id: "turn_autonomy_test",
    mode: "idle",
    path: input.emissionKind === "suppressed" ? "suppressed" : "system_1",
    response: "",
    emitted: input.emissionKind === "message",
    emission,
    thoughts: [],
    usage: {
      input_tokens: 1,
      output_tokens: 1,
      stop_reason: "end_turn",
    },
    retrievedEpisodeIds: [],
    referencedEpisodeIds: [],
    intents: [],
    toolCalls:
      input.deliveredOutbound === true
        ? [
            {
              callId: "toolu_outbound",
              name: "tool.outbound.post",
              input: {},
              output: {
                outbound: {
                  emitted: true,
                  delivery_outcome: {
                    state: "delivered",
                    agent_message_id: "strm_delivered",
                  },
                },
              },
              ok: true,
              durationMs: 1,
            },
          ]
        : [],
    agentMessageId: "strm_autonomy_test" as never,
  };
}

function createPersistentDueSource(input: {
  watermarkRepository: StreamWatermarkRepository;
  eventIds: readonly string[];
  sourceName?: AutonomyWakeSource["name"];
  sourceCategory?: AutonomyWakeSource["sourceCategory"];
  sortTs?: (eventId: string, index: number) => number;
  stateTs?: (eventId: string, index: number) => number | undefined;
  payload?: (eventId: string, index: number) => Record<string, unknown>;
  goalStaleBackoffActionAvailabilityKey?: string;
}): AutonomyWakeSource {
  const sourceName = input.sourceName ?? "goal_followup_due";
  const sourceCategory = input.sourceCategory ?? "operational";

  return {
    name: sourceName,
    type: "trigger",
    sourceCategory,
    async scan() {
      return input.eventIds.flatMap((eventId, index) => {
        const watermarkProcessName = `autonomy:test:persistent:${eventId}`;

        if (input.watermarkRepository.get(watermarkProcessName, DEFAULT_SESSION_ID) !== null) {
          return [];
        }

        const stateTs = input.stateTs?.(eventId, index);

        return [
          {
            id: eventId,
            sourceName,
            sourceType: "trigger" as const,
            watermarkProcessName,
            sortTs: input.sortTs?.(eventId, index) ?? index + 1,
            ...(stateTs === undefined ? {} : { stateTs }),
            ...(input.goalStaleBackoffActionAvailabilityKey === undefined
              ? {}
              : {
                  goalStaleBackoffActionAvailabilityKey:
                    input.goalStaleBackoffActionAvailabilityKey,
                }),
            payload: input.payload?.(eventId, index) ?? {},
          },
        ];
      });
    },
    buildTurn() {
      return {
        audience: "self",
        stakes: "low",
        userMessage: "",
      };
    },
  };
}

function setFleetBrakeState(
  watermarkRepository: StreamWatermarkRepository,
  clock: ManualClock,
  metadata: Partial<ReturnType<typeof readFleetBrakeMetadata>>,
): void {
  watermarkRepository.set(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID, {
    lastTs: clock.now(),
    lastEntryId: "fleet-state",
    metadata: {
      empty_streak: 0,
      streak_anchor_ts: 0,
      last_wake_ts: 0,
      error_streak: 0,
      last_error_ts: 0,
      bypass_count: 0,
      ...metadata,
    },
  });
}

describe("AutonomyScheduler", () => {
  let cleanup: (() => Promise<void>) | undefined;

  afterEach(async () => {
    vi.restoreAllMocks();
    await cleanup?.();
    cleanup = undefined;
  });

  it("recognizes only the two structural per-goal empty-wake payload shapes", () => {
    const baseEvent = {
      id: "goal-event",
      sourceType: "trigger" as const,
      watermarkProcessName: "autonomy:test:goal-event",
      sortTs: 1_000,
    };

    expect(
      goalConcernPayload({
        ...baseEvent,
        sourceName: "goal_followup_due",
        payload: {
          goal_id: "goal_aaaaaaaaaaaaaaaa",
          last_progress_ts: null,
        },
      }),
    ).toEqual({
      goalId: "goal_aaaaaaaaaaaaaaaa",
      lastProgressTs: null,
    });
    expect(
      goalConcernPayload({
        ...baseEvent,
        sourceName: "executive_focus_due",
        payload: {
          reason: "goal_stale",
          selected_goal_id: "goal_bbbbbbbbbbbbbbbb",
          selected_goal: { last_progress_ts: 900 },
        },
      }),
    ).toEqual({
      goalId: "goal_bbbbbbbbbbbbbbbb",
      lastProgressTs: 900,
    });
    expect(
      goalConcernPayload({
        ...baseEvent,
        sourceName: "executive_focus_due",
        payload: {
          reason: "step_due",
          selected_goal_id: "goal_bbbbbbbbbbbbbbbb",
          selected_goal: { last_progress_ts: 900 },
        },
      }),
    ).toBeNull();
    expect(
      goalConcernPayload({
        ...baseEvent,
        sourceName: "goal_followup_due",
        payload: { goal_id: "goal_aaaaaaaaaaaaaaaa" },
      }),
    ).toBeNull();
  });

  it("uses one structural headway predicate for messages, private carry, and delivery", () => {
    expect(turnEmittedHeadway(createStructuralTurnResult({ emissionKind: "message" }))).toBe(true);
    expect(
      turnEmittedHeadway(createStructuralTurnResult({ emissionKind: "continue_thought" })),
    ).toBe(true);
    expect(
      turnEmittedHeadway(
        createStructuralTurnResult({ emissionKind: "suppressed", deliveredOutbound: true }),
      ),
    ).toBe(true);
    expect(turnEmittedHeadway(createStructuralTurnResult({ emissionKind: "suppressed" }))).toBe(
      false,
    );
  });

  it("fires due events once and respects trigger watermarks", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );

    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "Reflected on recent changes.",
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_result",
      }),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const firstTick = await scheduler.tick();
    expect(firstTick.firedEvents).toBe(1);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);
    expect(
      watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID),
    ).toMatchObject({
      lastTs: 1_000_000,
      lastEntryId: expect.any(String),
    });

    const secondTick = await scheduler.tick();
    expect(secondTick.firedEvents).toBe(0);
    expect(secondTick.dueEvents).toBe(0);

    const kinds = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    })
      .tail(4)
      .map((entry) => entry.kind);
    expect(kinds).toEqual(["internal_event", "tool_call", "tool_result", "internal_event"]);
  });

  it("merges prior self thought into scheduled reflection payload with self-private labels", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const trainOfThoughtRepository = new TrainOfThoughtRepository({
      db: harness.db,
      clock,
    });
    const selfEntityId = harness.entityRepository.resolve("self", {
      kind: "self",
      provenance: "assistant_seeded",
    });
    trainOfThoughtRepository.upsert({
      text: "I am still circling the open question.",
      selfEntityId,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "Reflected.",
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_prior_thought",
      }),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      trainOfThoughtRepository,
      sources: [trigger],
    });

    await scheduler.tick();

    const turnInput = turnRunner.run.mock.calls[0]?.[0];
    expect(turnInput?.autonomyTrigger?.payload).toMatchObject({
      prior_self_thought: {
        text: "I am still circling the open question.",
        self_entity_id: selfEntityId,
        disclosure_label: {
          disclosure_class: "self_private",
          private_to_entity_ids: [],
          public_to_entity_ids: [],
        },
      },
    });
  });

  it("merges prior self thought into every autonomous wake payload, not only reflection", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const trainOfThoughtRepository = new TrainOfThoughtRepository({
      db: harness.db,
      clock,
    });
    const selfEntityId = harness.entityRepository.resolve("self", {
      kind: "self",
      provenance: "assistant_seeded",
    });
    trainOfThoughtRepository.upsert({
      text: "Third retire tonight, and it worked again.",
      selfEntityId,
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue(
        createStructuralTurnResult({
          emissionKind: "continue_thought",
        }),
      ),
    };
    const baseSource = createTestDueSource();
    const passthroughSource: AutonomyWakeSource = {
      ...baseSource,
      buildTurn(event) {
        return {
          audience: "self",
          stakes: "low",
          userMessage: "",
          autonomyTrigger: {
            source_name: event.sourceName,
            source_type: event.sourceType,
            event_id: event.id,
            sort_ts: event.sortTs,
            payload: event.payload,
          },
        };
      },
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      trainOfThoughtRepository,
      sources: [passthroughSource],
    });

    await scheduler.tick();

    const turnInput = turnRunner.run.mock.calls[0]?.[0];
    expect(turnInput?.autonomyTrigger?.source_name).toBe("goal_followup_due");
    expect(turnInput?.autonomyTrigger?.payload).toMatchObject({
      goal_id: "goal_aaaaaaaaaaaaaaaa",
      prior_self_thought: {
        text: "Third retire tonight, and it worked again.",
        self_entity_id: selfEntityId,
        disclosure_label: {
          disclosure_class: "self_private",
        },
      },
    });
  });

  it("records a self decision only after a successful autonomous turn", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const selfDecisionRepository = {
      record: vi.fn(),
    };
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "  Decidí revisar los objetivos pendientes.  ",
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_decision",
      }),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      selfDecisionRepository,
      sources: [createTestDueSource()],
    });

    const result = await scheduler.tick();

    expect(result.firedEvents).toBe(1);
    expect(selfDecisionRepository.record).toHaveBeenCalledTimes(1);
    expect(selfDecisionRepository.record).toHaveBeenCalledWith(
      expect.objectContaining({
        occurredAt: 1_000_000,
        sessionId: DEFAULT_SESSION_ID,
        triggerName: "goal_followup_due",
        triggerType: "trigger",
        sourceEventId: "goal_aaaaaaaaaaaaaaaa:no-target:1000",
        fireEventId: expect.any(String),
        decisionSummary: "Decidí revisar los objetivos pendientes.",
        turnResultId: "strm_agent_decision",
        sourceStreamEntryIds: [expect.any(String), expect.any(String)],
      }),
    );
  });

  it("records separate decisions for recurring committed fires with the same due event id", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const selfDecisionRepository = new SelfDecisionRepository({
      db: harness.db,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "Rechecked the same recurring condition.",
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_same_due_event",
      }),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      selfDecisionRepository,
      sources: [createTestDueSource("recurring-source-event")],
    });

    const firstResult = await scheduler.tick();
    clock.advance(3_600_000);
    const secondResult = await scheduler.tick();

    expect(firstResult.firedEvents).toBe(1);
    expect(secondResult.firedEvents).toBe(1);
    expect(
      selfDecisionRepository.listRecentAutonomousSelfPrivate({
        sinceMs: 0,
        limit: 10,
      }),
    ).toHaveLength(2);
  });

  it("records and recalls the finalizer rationale for a no-output autonomous wake", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const selfDecisionRepository = new SelfDecisionRepository({ db: harness.db, clock });
    const decisionRationale = "Nie pojawiło się nic nowego, więc odpowiedź byłaby tylko echem.";
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "suppressed",
        response: "",
        emitted: false,
        emission: {
          kind: "suppressed",
          reason: "finalizer_no_output",
          primary_no_output_reason: "low_value_echo",
          decision_rationale: decisionRationale,
        },
        thoughts: [],
        usage: { input_tokens: 1, output_tokens: 1, stop_reason: "end_turn" },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_suppressed",
      }),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository: new StreamWatermarkRepository({ db: harness.db, clock }),
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      selfDecisionRepository,
      sources: [createTestDueSource("suppressed-source-event")],
    });

    const result = await scheduler.tick();
    expect(result.firedEvents).toBe(1);

    const rows = selfDecisionRepository.listRecentAutonomousSelfPrivate({
      sinceMs: 0,
      limit: 10,
    });
    expect(rows).toHaveLength(1);
    expect(rows[0]?.decisionSummary).toContain("Stayed silent");
    expect(rows[0]?.decisionSummary).toContain("low value echo");
    expect(rows[0]?.decisionRationale).toBe(decisionRationale);

    const recallRows = selectSelfDecisionIntrospection({
      repository: selfDecisionRepository,
      nowMs: clock.now(),
    });
    expect(recallRows).toHaveLength(1);
    expect(recallRows[0]?.text).toContain(rows[0]?.decisionSummary);
    expect(recallRows[0]?.text).toContain(`because ${decisionRationale}`);
  });

  it("resets stale-goal wake backoff only for delivered outbound outcomes", async () => {
    const runCase = async (input: {
      deliveryOutcome: Record<string, unknown>;
      emitted: boolean;
    }) => {
      const clock = new ManualClock(1_000_000);
      const harness = await createOfflineTestHarness({
        clock,
      });
      const watermarkRepository = new StreamWatermarkRepository({
        db: harness.db,
        clock,
      });
      const goalId = "goal_aaaaaaaaaaaaaaaa";
      const otherGoalId = "goal_bbbbbbbbbbbbbbbb";
      const backoffProcessName = getExecutiveFocusGoalStaleBackoffProcessName(goalId);
      const otherBackoffProcessName = getExecutiveFocusGoalStaleBackoffProcessName(otherGoalId);
      watermarkRepository.set(backoffProcessName, DEFAULT_SESSION_ID, {
        lastTs: 750,
        lastEntryId: "previous-stale-wake",
        metadata: {
          empty_count: 2,
        },
      });
      watermarkRepository.set(otherBackoffProcessName, DEFAULT_SESSION_ID, {
        lastTs: 700,
        lastEntryId: "other-goal-stale-wake",
        metadata: {
          empty_count: 3,
        },
      });
      const scheduler = createScheduler({
        db: harness.db,
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerWindow: 6,
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        watermarkRepository,
        turnOrchestrator: {
          run: vi.fn().mockResolvedValue({
            turn_id: "turn-stale-goal",
            mode: "idle",
            path: "suppressed",
            response: "",
            emitted: false,
            emission: {
              kind: "suppressed",
              reason: "finalizer_no_output",
            },
            thoughts: [],
            usage: {
              input_tokens: 1,
              output_tokens: 1,
              stop_reason: "end_turn",
            },
            retrievedEpisodeIds: [],
            referencedEpisodeIds: [],
            intents: [],
            toolCalls: [
              {
                callId: "toolu_outbound",
                name: "tool.outbound.post",
                input: {},
                output: {
                  outbound: {
                    emitted: input.emitted,
                    delivery_outcome: input.deliveryOutcome,
                  },
                },
                ok: true,
                durationMs: 1,
              },
            ],
            agentMessageId: "strm_stale_goal",
          }),
        },
        toolDispatcher: new ToolDispatcher({
          createStreamWriter: (sessionId) =>
            new StreamWriter({
              dataDir: harness.tempDir,
              sessionId,
              clock,
            }),
          clock,
        }),
        sources: [createExecutiveFocusGoalStaleSource(goalId)],
      });

      try {
        const result = await scheduler.tick();

        expect(result.firedEvents).toBe(1);
        return {
          progressedGoal: watermarkRepository.get(backoffProcessName, DEFAULT_SESSION_ID),
          otherGoal: watermarkRepository.get(otherBackoffProcessName, DEFAULT_SESSION_ID),
        };
      } finally {
        await harness.cleanup();
      }
    };

    await expect(
      runCase({
        deliveryOutcome: {
          state: "delivered",
          agent_message_id: "strm_delivered",
        },
        emitted: true,
      }),
    ).resolves.toEqual({
      progressedGoal: null,
      otherGoal: expect.objectContaining({
        lastEntryId: "other-goal-stale-wake",
        metadata: { empty_count: 3 },
      }),
    });

    await expect(
      runCase({
        deliveryOutcome: {
          state: "suppressed",
          reason: "finalizer_no_output",
        },
        emitted: false,
      }),
    ).resolves.toMatchObject({
      progressedGoal: {
        lastTs: 1_000,
        lastEntryId: "goal_aaaaaaaaaaaaaaaa:stale:1000",
        metadata: {
          empty_count: 3,
        },
      },
      otherGoal: {
        lastEntryId: "other-goal-stale-wake",
        metadata: { empty_count: 3 },
      },
    });

    await expect(
      runCase({
        deliveryOutcome: {
          state: "transport_failed",
          reason: "transport_failed",
          agent_message_id: "strm_transport_failed",
          stream_entry_id: "strm_transport_failed",
          source_type: "demo",
          error: "connector unavailable",
        },
        emitted: true,
      }),
    ).resolves.toMatchObject({
      progressedGoal: {
        lastTs: 1_000,
        lastEntryId: "goal_aaaaaaaaaaaaaaaa:stale:1000",
        metadata: {
          empty_count: 3,
        },
      },
      otherGoal: {
        lastEntryId: "other-goal-stale-wake",
        metadata: { empty_count: 3 },
      },
    });
  });

  it("stamps the shared per-goal brake for empty followup wakes and resets only its headway goal", async () => {
    const clock = new ManualClock(1_100_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const emptyGoalId = "goal_aaaaaaaaaaaaaaaa";
    const headwayGoalId = "goal_bbbbbbbbbbbbbbbb";
    const untouchedGoalId = "goal_cccccccccccccccc";

    for (const goalId of [emptyGoalId, headwayGoalId, untouchedGoalId]) {
      watermarkRepository.set(
        getExecutiveFocusGoalStaleBackoffProcessName(goalId),
        DEFAULT_SESSION_ID,
        {
          lastTs: 500,
          lastEntryId: `prior-${goalId}`,
          metadata: { empty_count: 2 },
        },
      );
    }

    const source = createPersistentDueSource({
      watermarkRepository,
      eventIds: ["followup-empty", "followup-headway"],
      goalStaleBackoffActionAvailabilityKey: "outbound_action_surface_v1:test",
      payload: (_eventId, index) => ({
        goal_id: index === 0 ? emptyGoalId : headwayGoalId,
        last_progress_ts: 500,
      }),
    });
    const turnRunner = {
      run: vi
        .fn()
        .mockResolvedValueOnce(createStructuralTurnResult({ emissionKind: "suppressed" }))
        .mockResolvedValueOnce(
          createStructuralTurnResult({
            emissionKind: "suppressed",
            deliveredOutbound: true,
          }),
        ),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [source],
    });

    const result = await scheduler.tick();

    expect(result.firedEvents).toBe(2);
    expect(
      watermarkRepository.get(
        getExecutiveFocusGoalStaleBackoffProcessName(emptyGoalId),
        DEFAULT_SESSION_ID,
      ),
    ).toMatchObject({
      metadata: {
        empty_count: 3,
        action_availability_key: "outbound_action_surface_v1:test",
      },
    });
    expect(
      watermarkRepository.get(
        getExecutiveFocusGoalStaleBackoffProcessName(headwayGoalId),
        DEFAULT_SESSION_ID,
      ),
    ).toBeNull();
    expect(
      watermarkRepository.get(
        getExecutiveFocusGoalStaleBackoffProcessName(untouchedGoalId),
        DEFAULT_SESSION_ID,
      ),
    ).toMatchObject({
      lastEntryId: `prior-${untouchedGoalId}`,
      metadata: { empty_count: 2 },
    });
  });

  it("preserves the stamped action topology across an empty wake with no available action", async () => {
    const clock = new ManualClock(1_125_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goalId = "goal_aaaaaaaaaaaaaaaa";
    const processName = getExecutiveFocusGoalStaleBackoffProcessName(goalId);
    const actionAvailabilityKey = "outbound_action_surface_v1:stable-topology";
    watermarkRepository.set(processName, DEFAULT_SESSION_ID, {
      lastTs: 500,
      lastEntryId: "prior-empty-wake",
      metadata: {
        empty_count: 2,
        action_availability_key: actionAvailabilityKey,
      },
    });
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [
        createPersistentDueSource({
          watermarkRepository,
          eventIds: ["followup-empty-without-action"],
          payload: () => ({
            goal_id: goalId,
            last_progress_ts: 500,
          }),
        }),
      ],
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(watermarkRepository.get(processName, DEFAULT_SESSION_ID)).toMatchObject({
      metadata: {
        empty_count: 3,
        action_availability_key: actionAvailabilityKey,
      },
    });
  });

  it("leaves the shared goal watermark untouched when followup stale-backoff respect is disabled", async () => {
    const clock = new ManualClock(1_150_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const goalId = "goal_aaaaaaaaaaaaaaaa";
    const processName = getExecutiveFocusGoalStaleBackoffProcessName(goalId);
    watermarkRepository.set(processName, DEFAULT_SESSION_ID, {
      lastTs: 500,
      lastEntryId: "rollback-shared-watermark",
      metadata: { empty_count: 3 },
    });
    const before = watermarkRepository.get(processName, DEFAULT_SESSION_ID);
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      respectGoalFollowupStaleBackoff: false,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [
        createPersistentDueSource({
          watermarkRepository,
          eventIds: ["rollback-followup"],
          payload: () => ({
            goal_id: goalId,
            last_progress_ts: 500,
          }),
        }),
      ],
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(watermarkRepository.get(processName, DEFAULT_SESSION_ID)).toEqual(before);
  });

  it("classifies post-turn watermark failures as bookkeeping without recording a self decision", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const selfDecisionRepository = new SelfDecisionRepository({
      db: harness.db,
      clock,
    });
    const throwingWatermarkRepository = {
      set: vi.fn(() => {
        throw new Error("watermark unavailable");
      }),
    } as unknown as StreamWatermarkRepository;
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: throwingWatermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue({
          mode: "idle",
          path: "system_1",
          response: "The turn succeeded but the watermark will fail.",
          thoughts: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          retrievedEpisodeIds: [],
          referencedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_agent_watermark_failure",
        }),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      selfDecisionRepository,
      sources: [createTestDueSource("watermark-failure-event")],
    });

    const result = await scheduler.tick();

    expect(result).toMatchObject({
      firedEvents: 1,
      errorCount: 0,
      bookkeepingErrorCount: 1,
      events: [expect.objectContaining({ status: "bookkeeping_error" })],
    });
    expect(
      selfDecisionRepository.listRecentAutonomousSelfPrivate({
        sinceMs: 0,
        limit: 10,
      }),
    ).toEqual([]);
  });

  it("does not record self decisions for budget skips, preparation errors, busy skips, or turn errors", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;

    const runCase = async (input: {
      source: AutonomyWakeSource;
      maxWakesPerWindow?: number;
      seedBudgetWake?: boolean;
      turnResult: unknown;
    }) => {
      const selfDecisionRepository = {
        record: vi.fn(),
      };
      const wakeRepository = new AutonomyWakesRepository({
        db: harness.db,
        clock,
      });

      if (input.seedBudgetWake === true) {
        wakeRepository.record({
          trigger_name: "scheduled_reflection",
          condition_name: null,
          session_id: DEFAULT_SESSION_ID,
          wake_source_type: "trigger",
        });
      }

      const scheduler = createScheduler({
        db: harness.db,
        wakeRepository,
        budgetWindowMs: 60_000,
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerWindow: input.maxWakesPerWindow ?? 6,
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        watermarkRepository: new StreamWatermarkRepository({
          db: harness.db,
          clock,
        }),
        turnOrchestrator: {
          run:
            input.turnResult instanceof Error
              ? vi.fn().mockRejectedValue(input.turnResult)
              : vi.fn().mockResolvedValue(input.turnResult),
        },
        toolDispatcher: new ToolDispatcher({
          createStreamWriter: (sessionId) =>
            new StreamWriter({
              dataDir: harness.tempDir,
              sessionId,
              clock,
            }),
          clock,
        }),
        selfDecisionRepository,
        sources: [input.source],
      });

      const result = await scheduler.tick();

      expect(selfDecisionRepository.record).not.toHaveBeenCalled();
      return result;
    };

    await expect(
      runCase({
        source: createTestDueSource("budget-skipped-event"),
        maxWakesPerWindow: 1,
        seedBudgetWake: true,
        turnResult: {
          mode: "idle",
          path: "system_1",
          response: "Should not run.",
          thoughts: [],
          usage: { input_tokens: 1, output_tokens: 1, stop_reason: "end_turn" },
          retrievedEpisodeIds: [],
          referencedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_agent_budget_skip",
        },
      }),
    ).resolves.toMatchObject({ budgetSkipped: 1 });

    await expect(
      runCase({
        source: createTestDueSource("preparation-error-event", "scheduled_reflection"),
        turnResult: {
          mode: "idle",
          path: "system_1",
          response: "Should not run.",
          thoughts: [],
          usage: { input_tokens: 1, output_tokens: 1, stop_reason: "end_turn" },
          retrievedEpisodeIds: [],
          referencedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_agent_preparation_error",
        },
      }),
    ).resolves.toMatchObject({ errorCount: 1 });

    await expect(
      runCase({
        source: createTestDueSource("busy-skipped-event"),
        turnResult: new SessionBusyError("busy"),
      }),
    ).resolves.toMatchObject({ busySkipped: 1 });

    await expect(
      runCase({
        source: createTestDueSource("turn-error-event"),
        turnResult: new Error("turn failed"),
      }),
    ).resolves.toMatchObject({ errorCount: 1 });
  });

  it("respects maxWakesPerWindow", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    harness.commitmentRepository.add({
      type: "promise",
      directiveFamily: "first_expiring_commitment",
      directive: "First expiring commitment",
      priority: 5,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 5_000,
    });
    harness.commitmentRepository.add({
      type: "promise",
      directiveFamily: "second_expiring_commitment",
      directive: "Second expiring commitment",
      priority: 4,
      provenance: { kind: "manual" },
      expiresAt: clock.now() + 6_000,
    });

    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createCommitmentsListTool({
        listCommitments: () =>
          harness.commitmentRepository.list({
            activeOnly: true,
          }),
      }),
    );
    const trigger = createCommitmentExpiringTrigger({
      commitmentRepository: harness.commitmentRepository,
      watermarkRepository,
      lookaheadMs: 20_000,
      clock,
    });
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 1,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue({
          mode: "idle",
          path: "system_1",
          response: "Processed one commitment.",
          thoughts: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          retrievedEpisodeIds: [],
          referencedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_agent_budget",
        }),
      },
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const result = await scheduler.tick();
    expect(result.firedEvents).toBe(1);
    expect(result.budgetSkipped).toBe(1);
  });

  it("reserves configured budget slots for contemplative wake sources", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "Handled wake.",
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_reserved_budget",
      }),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 2,
      reservedContemplativeWakesPerWindow: 1,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [
        createTestDueSource("a-operational", "goal_followup_due", "operational"),
        createTestDueSource("b-operational", "goal_followup_due", "operational"),
        createTestDueSource("c-contemplative", "scheduled_wake", "contemplative"),
      ],
    });

    const result = await scheduler.tick();

    expect(result.firedEvents).toBe(2);
    expect(result.budgetSkipped).toBe(1);
    expect(result.events.map((event) => [event.id, event.status, event.sourceCategory])).toEqual([
      ["a-operational", "fired", "operational"],
      ["b-operational", "budget_skipped", "operational"],
      ["c-contemplative", "fired", "contemplative"],
    ]);
    expect(turnRunner.run).toHaveBeenCalledTimes(2);
  });

  it("checks persisted wake history when a fresh scheduler enforces budget", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const wakeRepository = new AutonomyWakesRepository({
      db: harness.db,
      clock,
    });
    wakeRepository.record({
      trigger_name: "scheduled_reflection",
      condition_name: null,
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "Should not run.",
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_should_not_run",
      }),
    };
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository: new AutonomyWakesRepository({
        db: harness.db,
        clock,
      }),
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 1,
      budgetWindowMs: 60_000,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [
        {
          name: "scheduled_reflection",
          type: "trigger",
          sourceCategory: "contemplative",
          async scan() {
            return [
              {
                id: "persisted-budget-event",
                sourceName: "scheduled_reflection",
                sourceType: "trigger",
                watermarkProcessName: "autonomy:test:persisted-budget",
                sortTs: clock.now(),
                payload: {},
              },
            ];
          },
          buildTurn() {
            return {
              audience: "self",
              stakes: "low",
              userMessage: "Reflect",
            };
          },
        },
      ],
    });

    const result = await scheduler.tick();
    expect(result.firedEvents).toBe(0);
    expect(result.budgetSkipped).toBe(1);
    expect(turnRunner.run).not.toHaveBeenCalled();
  });

  it("shares persisted budget across SQLite connections", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const secondDb = openDatabase(join(harness.tempDir, "borg.db"), {
      // The harness already migrated this file. A second connection only needs
      // to attach to the existing schema; re-running a divergent migration
      // subset would collide on tables the harness already created.
      migrations: [],
    });

    try {
      const firstTurnRunner = {
        run: vi.fn().mockResolvedValue({
          mode: "idle",
          path: "system_1",
          response: "Process A wake.",
          thoughts: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          retrievedEpisodeIds: [],
          referencedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_process_a",
        }),
      };
      const firstScheduler = createScheduler({
        db: harness.db,
        wakeRepository: new AutonomyWakesRepository({
          db: harness.db,
          clock,
        }),
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerWindow: 1,
        budgetWindowMs: 60_000,
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        watermarkRepository: new StreamWatermarkRepository({
          db: harness.db,
          clock,
        }),
        turnOrchestrator: firstTurnRunner,
        toolDispatcher: new ToolDispatcher({
          createStreamWriter: (sessionId) =>
            new StreamWriter({
              dataDir: harness.tempDir,
              sessionId,
              clock,
            }),
          clock,
        }),
        sources: [
          {
            name: "goal_followup_due",
            type: "trigger",
            sourceCategory: "operational",
            async scan() {
              return [
                {
                  id: "process-a-event",
                  sourceName: "goal_followup_due",
                  sourceType: "trigger",
                  watermarkProcessName: "autonomy:test:process-a",
                  sortTs: 1,
                  payload: {
                    goal_id: "goal_aaaaaaaaaaaaaaaa",
                  },
                },
              ];
            },
            buildTurn() {
              return {
                audience: "self",
                stakes: "low",
                userMessage: "Process A",
              };
            },
          },
        ],
      });
      const firstResult = await firstScheduler.tick();
      expect(firstResult.firedEvents).toBe(1);

      const secondTurnRunner = {
        run: vi.fn(),
      };
      const secondScheduler = createScheduler({
        db: secondDb,
        wakeRepository: new AutonomyWakesRepository({
          db: secondDb,
          clock,
        }),
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerWindow: 1,
        budgetWindowMs: 60_000,
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        watermarkRepository: new StreamWatermarkRepository({
          db: secondDb,
          clock,
        }),
        turnOrchestrator: secondTurnRunner,
        toolDispatcher: new ToolDispatcher({
          createStreamWriter: (sessionId) =>
            new StreamWriter({
              dataDir: harness.tempDir,
              sessionId,
              clock,
            }),
          clock,
        }),
        sources: [
          {
            name: "goal_followup_due",
            type: "trigger",
            sourceCategory: "operational",
            async scan() {
              return [
                {
                  id: "process-b-event",
                  sourceName: "goal_followup_due",
                  sourceType: "trigger",
                  watermarkProcessName: "autonomy:test:process-b",
                  sortTs: 2,
                  payload: {
                    goal_id: "goal_bbbbbbbbbbbbbbbb",
                  },
                },
              ];
            },
            buildTurn() {
              return {
                audience: "self",
                stakes: "low",
                userMessage: "Process B",
              };
            },
          },
        ],
      });

      const secondResult = await secondScheduler.tick();
      expect(secondResult.firedEvents).toBe(0);
      expect(secondResult.budgetSkipped).toBe(1);
      expect(secondTurnRunner.run).not.toHaveBeenCalled();
    } finally {
      secondDb.close();
    }
  });

  it("prunes wake records after each enabled tick", async () => {
    const clock = new ManualClock(10_000_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const budgetWindowMs = 60_000;
    const safetyBufferMs = 7 * 24 * 60 * 60 * 1_000;
    const pruneCutoff = clock.now() - budgetWindowMs - safetyBufferMs;
    const wakeRepository = new AutonomyWakesRepository({
      db: harness.db,
      clock,
    });

    clock.set(pruneCutoff - 1);
    const oldWake = wakeRepository.record({
      trigger_name: "scheduled_reflection",
      condition_name: null,
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
    });
    clock.set(pruneCutoff);
    const retainedWake = wakeRepository.record({
      trigger_name: "scheduled_reflection",
      condition_name: null,
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
    });
    clock.set(10_000_000_000);

    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      budgetWindowMs,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: {
        run: vi.fn(),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [],
    });

    await scheduler.tick();
    const wakeIds = wakeRepository.listSince(0, 10).map((wake) => wake.id);
    expect(wakeIds).not.toContain(oldWake.id);
    expect(wakeIds).toContain(retainedWake.id);
  });

  it("skips busy autonomous turns", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: 4,
      streak_anchor_ts: clock.now() - 1_000,
      last_wake_ts: clock.now() - 1_000,
      error_streak: 2,
      last_error_ts: clock.now() - 1_000,
      bypass_count: 1,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockRejectedValue(new SessionBusyError("busy")),
    };
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const result = await scheduler.tick();
    expect(result.busySkipped).toBe(1);
    expect(result.events[0]?.status).toBe("busy_skipped");
    expect(wakeRepository.countSince(0, { outcome: "busy" })).toBe(1);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({ empty_streak: 4, error_streak: 2, bypass_count: 1 });
    expect(watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID)).toBeNull();

    const secondResult = await scheduler.tick();
    expect(secondResult.busySkipped).toBe(0);
    expect(secondResult.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(30_000);
    const thirdResult = await scheduler.tick();
    expect(thirdResult.busySkipped).toBe(1);
    expect(thirdResult.events[0]?.status).toBe("busy_skipped");
    expect(turnRunner.run).toHaveBeenCalledTimes(2);
    expect(wakeRepository.countSince(0, { outcome: "busy" })).toBe(2);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({ empty_streak: 4, error_streak: 2, bypass_count: 1 });
  });

  it("reuses scheduled reflection backoff within a due window and refreshes it in the next window", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 60_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockRejectedValue(new SessionBusyError("busy")),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const firstTick = await scheduler.tick();
    expect(firstTick.events[0]?.id).toBe("scheduled-reflection:1000000");
    expect(firstTick.busySkipped).toBe(1);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(10_000);
    const secondTick = await scheduler.tick();
    expect(secondTick.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(19_999);
    const thirdTick = await scheduler.tick();
    expect(thirdTick.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(30_001);
    const fourthTick = await scheduler.tick();
    expect(fourthTick.events[0]?.id).toBe("scheduled-reflection:1060000");
    expect(fourthTick.busySkipped).toBe(1);
    expect(turnRunner.run).toHaveBeenCalledTimes(2);
  });

  it("leaves trigger watermarks untouched when an autonomous turn throws", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockRejectedValue(new Error("turn failed")),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const result = await scheduler.tick();
    expect(result.errorCount).toBe(1);
    expect(result.events[0]?.status).toBe("error");
    expect(watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID)).toBeNull();

    const secondResult = await scheduler.tick();
    expect(secondResult.errorCount).toBe(0);
    expect(secondResult.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(30_000);
    const thirdResult = await scheduler.tick();
    expect(thirdResult.errorCount).toBe(1);
    expect(thirdResult.events[0]?.status).toBe("error");
    expect(turnRunner.run).toHaveBeenCalledTimes(2);
  });

  it("engages the fleet cooldown after five operational silences and escalates durably", async () => {
    const clock = new ManualClock(2_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    const source = createPersistentDueSource({
      watermarkRepository,
      eventIds: ["empty-1", "empty-2", "empty-3", "empty-4", "empty-5", "empty-6", "empty-7"],
      stateTs: () => clock.now() - 10_000,
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
    };
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      budgetWindowMs: 60_000,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [source],
    });

    const first = await scheduler.tick();
    expect(first.firedEvents).toBe(5);
    expect(first.fleetCooldownSkipped).toBe(2);
    expect(wakeRepository.countSince(0, { outcome: "silent" })).toBe(5);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({
      empty_streak: 5,
      streak_anchor_ts: 2_000_000,
      last_wake_ts: 2_000_000,
    });

    clock.advance(100);
    const second = await scheduler.tick();
    expect(second.firedEvents).toBe(1);
    expect(second.fleetCooldownSkipped).toBe(1);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID))
        .empty_streak,
    ).toBe(6);

    const restartedScheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      budgetWindowMs: 60_000,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [source],
    });
    const beforeEscalatedEnd = await restartedScheduler.tick();
    expect(beforeEscalatedEnd.fleetCooldownSkipped).toBe(1);

    clock.advance(200);
    const third = await restartedScheduler.tick();
    expect(third.firedEvents).toBe(1);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID))
        .empty_streak,
    ).toBe(7);
    await expect(restartedScheduler.describe()).resolves.toMatchObject({
      fleet_brake: {
        enabled: true,
        empty_streak: 7,
        cooldown_until: clock.now() + 400,
        window_outcomes: {
          headway: 0,
          silent: 7,
          error: 0,
          busy: 0,
        },
      },
    });
  });

  it("enables the default fleet brake for direct scheduler construction", async () => {
    const clock = new ManualClock(2_050_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: DEFAULT_FLEET_BRAKE_OPTIONS.emptyStreakThreshold,
      streak_anchor_ts: clock.now() - 1_000,
      last_wake_ts: clock.now(),
    });
    const scheduler = new AutonomyScheduler({
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      budgetWindowMs: 60_000,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      wakeRepository: new AutonomyWakesRepository({ db: harness.db, clock }),
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [
        createPersistentDueSource({
          watermarkRepository,
          eventIds: ["default-brake-direct-construction"],
          stateTs: () => clock.now() - 2_000,
        }),
      ],
    });

    const result = await scheduler.tick();
    expect(result.events[0]?.status).toBe("fleet_cooldown_skipped");
    await expect(scheduler.describe()).resolves.toMatchObject({
      fleet_brake: { enabled: true },
    });
  });

  it("resets operational silence on headway while contemplative thought does not reset it", async () => {
    const clock = new ManualClock(2_100_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: 5,
      streak_anchor_ts: clock.now() - 1_000,
      last_wake_ts: clock.now() - TEST_FLEET_BRAKE.baseCooldownMs,
      bypass_count: 2,
    });
    const operationalScheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "message" })),
      },
      toolDispatcher: dispatcher,
      sources: [
        createPersistentDueSource({
          watermarkRepository,
          eventIds: ["operational-headway"],
          stateTs: () => clock.now() - 2_000,
        }),
      ],
    });

    expect((await operationalScheduler.tick()).firedEvents).toBe(1);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({ empty_streak: 0, bypass_count: 0 });

    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: 20,
      streak_anchor_ts: clock.now() - 1_000,
      last_wake_ts: clock.now(),
      bypass_count: 2,
    });
    const reflectionScheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi
          .fn()
          .mockResolvedValueOnce(createStructuralTurnResult({ emissionKind: "suppressed" }))
          .mockResolvedValueOnce(createStructuralTurnResult({ emissionKind: "continue_thought" }))
          .mockResolvedValueOnce(createStructuralTurnResult({ emissionKind: "message" })),
      },
      toolDispatcher: dispatcher,
      sources: [
        createPersistentDueSource({
          watermarkRepository,
          eventIds: ["reflection-silent", "reflection-thought", "reflection-message"],
          sourceName: "scheduled_reflection",
          sourceCategory: "contemplative",
        }),
      ],
    });

    const reflection = await reflectionScheduler.tick();
    expect(reflection.firedEvents).toBe(3);
    expect(reflection.fleetCooldownSkipped).toBe(0);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({ empty_streak: 20, bypass_count: 2 });

    const deliveredScheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(
          createStructuralTurnResult({
            emissionKind: "suppressed",
            deliveredOutbound: true,
          }),
        ),
      },
      toolDispatcher: dispatcher,
      sources: [
        createPersistentDueSource({
          watermarkRepository,
          eventIds: ["reflection-delivered"],
          sourceName: "scheduled_reflection",
          sourceCategory: "contemplative",
        }),
      ],
    });

    expect((await deliveredScheduler.tick()).firedEvents).toBe(1);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({ empty_streak: 0, bypass_count: 0 });
  });

  it("persists delivered-outbound fleet reset before later bookkeeping fails", async () => {
    const clock = new ManualClock(2_150_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: 8,
      streak_anchor_ts: clock.now() - 10_000,
      last_wake_ts: clock.now(),
      error_streak: 2,
      last_error_ts: clock.now() - 1_000,
      bypass_count: 3,
    });
    const dueWatermark = "autonomy:test:persistent:delivered-bookkeeping";
    const originalSet = watermarkRepository.set.bind(watermarkRepository);
    vi.spyOn(watermarkRepository, "set").mockImplementation((processName, sessionId, input) => {
      if (processName === dueWatermark) {
        throw new Error("due-event watermark unavailable");
      }

      return originalSet(processName, sessionId, input);
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(
          createStructuralTurnResult({
            emissionKind: "suppressed",
            deliveredOutbound: true,
          }),
        ),
      },
      toolDispatcher: dispatcher,
      sources: [
        createPersistentDueSource({
          watermarkRepository,
          eventIds: ["delivered-bookkeeping"],
          sourceName: "scheduled_reflection",
          sourceCategory: "contemplative",
        }),
      ],
    });

    const result = await scheduler.tick();
    expect(result).toMatchObject({
      firedEvents: 1,
      errorCount: 0,
      bookkeepingErrorCount: 1,
      events: [expect.objectContaining({ status: "bookkeeping_error" })],
    });
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({
      empty_streak: 0,
      error_streak: 0,
      bypass_count: 0,
    });
    expect(wakeRepository.countSince(0, { outcome: "headway" })).toBe(1);
  });

  it("pauses on three infrastructure errors and fairly probes contemplation after production retry spacing", async () => {
    const clock = new ManualClock(2_200_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const turnRunner = {
      run: vi
        .fn()
        .mockRejectedValueOnce(new LLMError("outage-1"))
        .mockRejectedValueOnce(new LLMError("outage-2"))
        .mockRejectedValueOnce(new LLMError("outage-3"))
        .mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
    };
    const operationalSource = createPersistentDueSource({
      watermarkRepository,
      eventIds: ["error-1", "error-2", "error-3"],
      sortTs: (_eventId, index) => index + 1,
    });
    const reflectionSource = createPersistentDueSource({
      watermarkRepository,
      eventIds: ["reflection-after-errors"],
      sourceName: "scheduled_reflection",
      sourceCategory: "contemplative",
      sortTs: () => 10,
    });
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 4,
      fleetBrake: {
        ...TEST_FLEET_BRAKE,
        errorBasePauseMs: DEFAULT_FLEET_BRAKE_OPTIONS.errorBasePauseMs,
        errorMaxPauseMs: DEFAULT_FLEET_BRAKE_OPTIONS.errorMaxPauseMs,
      },
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [operationalSource, reflectionSource],
    });

    const first = await scheduler.tick();
    expect(first.errorCount).toBe(3);
    expect(first.errorCircuitSkipped).toBe(1);
    expect(wakeRepository.countSince(0, { outcome: "error" })).toBe(3);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({ error_streak: 3, empty_streak: 0 });

    // The 5-minute production pause is much longer than the 30-second
    // per-event retry, so all three failing events are eligible again.
    clock.advance(DEFAULT_FLEET_BRAKE_OPTIONS.errorBasePauseMs);
    const second = await scheduler.tick();
    expect(second.firedEvents).toBe(1);
    expect(second.events[0]?.sourceName).toBe("scheduled_reflection");
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID))
        .error_streak,
    ).toBe(0);
  });

  it("isolates persistent source-preparation failures from healthy reflection", async () => {
    const clock = new ManualClock(2_250_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );
    const failingSource = (
      name: "goal_followup_due" | "commitment_expiring",
    ): AutonomyWakeSource => ({
      name,
      type: "trigger",
      sourceCategory: "operational",
      scan: vi.fn().mockRejectedValue(new Error(`${name} repository unavailable`)),
      buildTurn: vi.fn(),
    });
    const reflection = createPersistentDueSource({
      watermarkRepository,
      eventIds: ["healthy-reflection"],
      sourceName: "scheduled_reflection",
      sourceCategory: "contemplative",
    });
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
      },
      toolDispatcher: dispatcher,
      sources: [
        failingSource("goal_followup_due"),
        failingSource("commitment_expiring"),
        reflection,
      ],
    });

    const first = await scheduler.tick();
    expect(first).toMatchObject({
      firedEvents: 1,
      errorCount: 2,
      sourceErrorCount: 2,
      errorCircuitSkipped: 0,
    });
    expect(first.events[0]).toMatchObject({
      sourceName: "scheduled_reflection",
      status: "fired",
    });
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID))
        .error_streak,
    ).toBe(0);

    const second = await scheduler.tick();
    expect(second.sourceErrorCount).toBe(0);
    expect(second.errorCircuitSkipped).toBe(0);
  });

  it("caps freshness bypasses and fails closed when stateTs is missing", async () => {
    const clock = new ManualClock(2_300_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: 5,
      streak_anchor_ts: clock.now() - 1_000,
      last_wake_ts: clock.now(),
    });
    const source = createPersistentDueSource({
      watermarkRepository,
      eventIds: ["fresh-1", "fresh-2", "fresh-3", "fresh-over-cap", "missing-state"],
      stateTs: (eventId) => (eventId === "missing-state" ? undefined : clock.now()),
    });
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [source],
    });

    const result = await scheduler.tick();
    expect(result.firedEvents).toBe(3);
    expect(result.fleetCooldownSkipped).toBe(2);
    expect(result.events.map((event) => [event.id, event.status])).toEqual([
      ["fresh-1", "fired"],
      ["fresh-2", "fired"],
      ["fresh-3", "fired"],
      ["fresh-over-cap", "fleet_cooldown_skipped"],
      ["missing-state", "fleet_cooldown_skipped"],
    ]);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({ empty_streak: 8, bypass_count: 3 });
    expect(wakeRepository.countSince(0)).toBe(3);
  });

  it("admits a deadline concern within the current cooldown window without consuming freshness", async () => {
    const clock = new ManualClock(2_400_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: 5,
      streak_anchor_ts: clock.now() - 1_000,
      last_wake_ts: clock.now(),
    });
    const source = createPersistentDueSource({
      watermarkRepository,
      eventIds: ["older-silent-1", "older-silent-2", "older-silent-3", "deadline-concern"],
      stateTs: () => clock.now() - 2_000,
      payload: (eventId) => (eventId === "deadline-concern" ? { target_at: clock.now() + 50 } : {}),
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
    };
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [source],
    });

    const result = await scheduler.tick();
    expect(result.events.map((event) => [event.id, event.status])).toEqual([
      ["older-silent-1", "fleet_cooldown_skipped"],
      ["older-silent-2", "fleet_cooldown_skipped"],
      ["older-silent-3", "fleet_cooldown_skipped"],
      ["deadline-concern", "fired"],
    ]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({ empty_streak: 6, bypass_count: 0 });
    expect(wakeRepository.countSince(0)).toBe(1);
  });

  it("keeps fleet admission skips non-consuming and re-presents the event", async () => {
    const clock = new ManualClock(2_450_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    const recordSpy = vi.spyOn(wakeRepository, "record");
    const appendSpies: Array<ReturnType<typeof vi.spyOn>> = [];
    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: 5,
      streak_anchor_ts: clock.now() - 1_000,
      last_wake_ts: clock.now(),
    });
    const source = createPersistentDueSource({
      watermarkRepository,
      eventIds: ["non-consuming-skip"],
      stateTs: () => clock.now() - 2_000,
    });
    const createWriter = (sessionId: typeof DEFAULT_SESSION_ID) => {
      const writer = new StreamWriter({ dataDir: harness.tempDir, sessionId, clock });
      appendSpies.push(vi.spyOn(writer, "append"));
      return writer;
    };
    const turnRunner = {
      run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
    };
    const schedulerOptions = {
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: createWriter,
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({ createStreamWriter: createWriter, clock }),
      sources: [source],
    };
    const scheduler = createScheduler(schedulerOptions);

    expect((await scheduler.tick()).events[0]?.status).toBe("fleet_cooldown_skipped");
    expect(recordSpy).not.toHaveBeenCalled();
    expect(appendSpies[0]).not.toHaveBeenCalled();
    expect(turnRunner.run).not.toHaveBeenCalled();

    const restartedScheduler = createScheduler(schedulerOptions);
    const sameDecision = await restartedScheduler.tick();
    expect(sameDecision.events[0]?.status).toBe("fleet_cooldown_skipped");
    expect(recordSpy).not.toHaveBeenCalled();
    expect(appendSpies[1]).not.toHaveBeenCalled();

    clock.advance(TEST_FLEET_BRAKE.baseCooldownMs);
    const admitted = await restartedScheduler.tick();
    expect(admitted.firedEvents).toBe(1);
    expect(recordSpy).toHaveBeenCalledTimes(1);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);
    expect(wakeRepository.countSince(0, { outcome: "silent" })).toBe(1);
  });

  it("restores pre-governor admission when the fleet-brake flag is disabled", async () => {
    const clock = new ManualClock(2_500_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: 20,
      streak_anchor_ts: clock.now(),
      last_wake_ts: clock.now(),
      error_streak: 10,
      last_error_ts: clock.now(),
      bypass_count: 3,
    });
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: {
        ...TEST_FLEET_BRAKE,
        enabled: false,
      },
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [
        createPersistentDueSource({
          watermarkRepository,
          eventIds: ["rollback-admission"],
        }),
      ],
    });

    const result = await scheduler.tick();

    expect(result.firedEvents).toBe(1);
    expect(result.fleetCooldownSkipped).toBe(0);
    expect(result.errorCircuitSkipped).toBe(0);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({
      empty_streak: 20,
      error_streak: 10,
      bypass_count: 3,
    });
  });

  it("reports malformed fleet metadata once and self-heals on bookkeeping", async () => {
    const clock = new ManualClock(2_600_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    watermarkRepository.set(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID, {
      lastTs: clock.now(),
      lastEntryId: "malformed-fleet-state",
      metadata: {
        empty_streak: "invalid",
      },
    });
    const onError = vi.fn();
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 20,
      fleetBrake: TEST_FLEET_BRAKE,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [
        createPersistentDueSource({
          watermarkRepository,
          eventIds: ["self-heal-fleet"],
        }),
      ],
    });
    scheduler.setObserver({ onError });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(
      readFleetBrakeMetadata(watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, DEFAULT_SESSION_ID)),
    ).toMatchObject({
      empty_streak: 1,
      error_streak: 0,
      bypass_count: 0,
    });
  });

  it("is inert when autonomy is disabled", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const setIntervalFn = vi.fn<typeof setInterval>();
    const scheduler = createScheduler({
      db: harness.db,
      enabled: false,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn(),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [],
      setIntervalFn: setIntervalFn as unknown as typeof setInterval,
      clearIntervalFn: vi.fn(),
    });

    scheduler.start();
    expect(setIntervalFn).not.toHaveBeenCalled();
    await expect(scheduler.tick()).resolves.toMatchObject({
      status: "disabled",
      firedEvents: 0,
    });
  });

  it("describes sources and budget while counting a null-outcome wake only in wake_count", async () => {
    const clock = new ManualClock(950_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const wakeRepository = new AutonomyWakesRepository({
      db: harness.db,
      clock,
    });
    const headwayWake = wakeRepository.record({
      trigger_name: "scheduled_wake",
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
      source_category: "contemplative",
    });
    wakeRepository.recordOutcome(headwayWake.id, "headway");
    clock.set(980_000);
    const silentWake = wakeRepository.record({
      trigger_name: "scheduled_wake",
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
      source_category: "contemplative",
    });
    wakeRepository.recordOutcome(silentWake.id, "silent");
    const errorWake = wakeRepository.record({
      trigger_name: "commitment_revoked",
      condition_name: "commitment_revoked",
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "condition",
      source_category: "operational",
    });
    wakeRepository.recordOutcome(errorWake.id, "error");
    const busyWake = wakeRepository.record({
      trigger_name: "commitment_revoked",
      condition_name: "commitment_revoked",
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "condition",
      source_category: "operational",
    });
    wakeRepository.recordOutcome(busyWake.id, "busy");
    const nullOutcomeWake = wakeRepository.record({
      trigger_name: "commitment_revoked",
      condition_name: "commitment_revoked",
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "condition",
      source_category: "operational",
    });
    expect(nullOutcomeWake.outcome).toBeNull();
    clock.set(1_000_000);
    const setIntervalFn = vi.fn((callback: () => void) => {
      void callback;
      return 1 as unknown as ReturnType<typeof setInterval>;
    });
    const clearIntervalFn = vi.fn<typeof clearInterval>();
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 5_000,
      maxWakesPerWindow: 6,
      budgetWindowMs: 60_000,
      reservedContemplativeWakesPerWindow: 2,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: {
        run: vi.fn(),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [
        {
          name: "scheduled_wake",
          type: "trigger",
          sourceCategory: "contemplative",
          scan: vi.fn(async () => []),
          nextDueAt: vi.fn(async () => clock.now() + 30_000),
          buildTurn() {
            return {
              audience: "self",
              stakes: "low",
              userMessage: "Wake",
            };
          },
        },
        {
          name: "commitment_revoked",
          type: "condition",
          sourceCategory: "operational",
          scan: vi.fn(async () => []),
          buildTurn() {
            return {
              audience: "self",
              stakes: "low",
              userMessage: "Condition",
            };
          },
        },
      ],
      setIntervalFn: setIntervalFn as unknown as typeof setInterval,
      clearIntervalFn,
    });

    await expect(scheduler.describe()).resolves.toMatchObject({
      enabled: true,
      interval_ms: 5_000,
      next_tick_at: null,
      budget: {
        max_wakes_per_window: 6,
        window_ms: 60_000,
        used_in_current_window: 5,
        reserved_contemplative_wakes_per_window: 2,
        contemplative_used_in_current_window: 2,
        wakes_in_current_window_by_trigger: [
          {
            trigger_name: "scheduled_wake",
            wake_count: 2,
            outcome_counts: {
              headway: 1,
              silent: 1,
              error: 0,
              busy: 0,
            },
          },
          {
            trigger_name: "commitment_revoked",
            wake_count: 3,
            outcome_counts: {
              headway: 0,
              silent: 0,
              error: 1,
              busy: 1,
            },
          },
        ],
        next_budget_slot_frees_at: 1_010_001,
      },
      sources: expect.arrayContaining([
        {
          name: "scheduled_wake",
          type: "trigger",
          category: "contemplative",
          enabled: true,
          next_due_at: 1_030_000,
        },
        {
          name: "scheduled_reflection",
          type: "trigger",
          category: "contemplative",
          enabled: false,
          next_due_at: null,
        },
        {
          name: "commitment_revoked",
          type: "condition",
          category: "operational",
          enabled: true,
        },
      ]),
    });

    const condition = (await scheduler.describe()).sources.find(
      (source) => source.name === "commitment_revoked",
    );
    expect(condition).not.toHaveProperty("next_due_at");

    scheduler.start();
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: 1_005_000,
    });
    await scheduler.stop();
  });

  it("reports the first millisecond after the inclusive budget window as the next slot", async () => {
    const clock = new ManualClock(1_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const wakeRepository = new AutonomyWakesRepository({
      db: harness.db,
      clock,
    });
    wakeRepository.record({
      trigger_name: "scheduled_wake",
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
      source_category: "contemplative",
    });
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 5_000,
      maxWakesPerWindow: 6,
      budgetWindowMs: 60_000,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: {
        run: vi.fn(),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [],
      setIntervalFn: vi.fn() as unknown as typeof setInterval,
      clearIntervalFn: vi.fn(),
    });

    clock.set(61_000);
    await expect(scheduler.describe()).resolves.toMatchObject({
      budget: {
        used_in_current_window: 1,
        next_budget_slot_frees_at: 61_001,
      },
    });

    clock.set(61_001);
    await expect(scheduler.describe()).resolves.toMatchObject({
      budget: {
        used_in_current_window: 0,
        wakes_in_current_window_by_trigger: [],
        next_budget_slot_frees_at: null,
      },
    });
  });

  it("keeps next_tick_at null while stopped and resets stale tick anchors on start", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const setIntervalFn = vi.fn((callback: () => void) => {
      void callback;
      return 1 as unknown as ReturnType<typeof setInterval>;
    });
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 5_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: {
        run: vi.fn(),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [],
      setIntervalFn: setIntervalFn as unknown as typeof setInterval,
      clearIntervalFn: vi.fn(),
    });

    await scheduler.tick();
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: null,
    });

    clock.advance(20_000);
    scheduler.start();
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: 1_025_000,
    });
  });

  it("clears running tick anchors across stop/start", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const setIntervalFn = vi.fn((callback: () => void) => {
      void callback;
      return 1 as unknown as ReturnType<typeof setInterval>;
    });
    const clearIntervalFn = vi.fn<typeof clearInterval>();
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 5_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: {
        run: vi.fn(),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [],
      setIntervalFn: setIntervalFn as unknown as typeof setInterval,
      clearIntervalFn,
    });

    scheduler.start();
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: 1_005_000,
    });
    await scheduler.tick();
    clock.advance(20_000);
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: 1_020_000,
    });
    await scheduler.stop();
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: null,
    });

    clock.advance(10_000);
    scheduler.start();
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: 1_035_000,
    });
  });

  it("waits for an active tick to finish during graceful stop", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );

    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });

    let intervalCallback: (() => void) | undefined;
    const setIntervalFn = vi.fn((callback: () => void) => {
      intervalCallback = callback;
      return 1 as unknown as ReturnType<typeof setInterval>;
    });
    const clearIntervalFn = vi.fn<typeof clearInterval>();
    let resolveTurn:
      | ((value: {
          mode: "idle";
          path: "system_1";
          response: string;
          thoughts: [];
          usage: {
            input_tokens: number;
            output_tokens: number;
            stop_reason: "end_turn";
          };
          retrievedEpisodeIds: [];
          referencedEpisodeIds: [];
          intents: [];
          toolCalls: [];
          agentMessageId: string;
        }) => void)
      | undefined;
    const turnCompletion = new Promise<{
      mode: "idle";
      path: "system_1";
      response: string;
      thoughts: [];
      usage: {
        input_tokens: number;
        output_tokens: number;
        stop_reason: "end_turn";
      };
      retrievedEpisodeIds: [];
      referencedEpisodeIds: [];
      intents: [];
      toolCalls: [];
      agentMessageId: string;
    }>((resolve) => {
      resolveTurn = resolve;
    });
    const turnRunner = {
      run: vi.fn().mockReturnValue(turnCompletion),
    };

    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
      setIntervalFn: setIntervalFn as unknown as typeof setInterval,
      clearIntervalFn,
    });

    scheduler.start();
    intervalCallback?.();
    await vi.waitFor(() => {
      expect(turnRunner.run).toHaveBeenCalledTimes(1);
    });

    let stopped = false;
    const stopPromise = scheduler.stop();
    void stopPromise.then(() => {
      stopped = true;
    });

    await Promise.resolve();
    expect(stopped).toBe(false);

    resolveTurn?.({
      mode: "idle",
      path: "system_1",
      response: "Finished reflective work.",
      thoughts: [],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "end_turn",
      },
      retrievedEpisodeIds: [],
      referencedEpisodeIds: [],
      intents: [],
      toolCalls: [],
      agentMessageId: "strm_stop_wait",
    });

    await stopPromise;
    expect(stopped).toBe(true);
    expect(clearIntervalFn).toHaveBeenCalledTimes(1);
  });

  it("waits for a direct tick to finish during graceful stop", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );

    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });

    let resolveTurn:
      | ((value: {
          mode: "idle";
          path: "system_1";
          response: string;
          thoughts: [];
          usage: {
            input_tokens: number;
            output_tokens: number;
            stop_reason: "end_turn";
          };
          retrievedEpisodeIds: [];
          referencedEpisodeIds: [];
          intents: [];
          toolCalls: [];
          agentMessageId: string;
        }) => void)
      | undefined;
    const turnCompletion = new Promise<{
      mode: "idle";
      path: "system_1";
      response: string;
      thoughts: [];
      usage: {
        input_tokens: number;
        output_tokens: number;
        stop_reason: "end_turn";
      };
      retrievedEpisodeIds: [];
      referencedEpisodeIds: [];
      intents: [];
      toolCalls: [];
      agentMessageId: string;
    }>((resolve) => {
      resolveTurn = resolve;
    });
    const turnRunner = {
      run: vi.fn().mockReturnValue(turnCompletion),
    };

    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const tickPromise = scheduler.tick();
    await vi.waitFor(() => {
      expect(turnRunner.run).toHaveBeenCalledTimes(1);
    });

    let stopped = false;
    const stopPromise = scheduler.stop();
    void stopPromise.then(() => {
      stopped = true;
    });

    await Promise.resolve();
    expect(stopped).toBe(false);

    resolveTurn?.({
      mode: "idle",
      path: "system_1",
      response: "Finished reflective work.",
      thoughts: [],
      usage: {
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "end_turn",
      },
      retrievedEpisodeIds: [],
      referencedEpisodeIds: [],
      intents: [],
      toolCalls: [],
      agentMessageId: "strm_direct_stop_wait",
    });

    await Promise.all([tickPromise, stopPromise]);
    expect(stopped).toBe(true);
  });

  it("reports watermark commit failures as bookkeeping errors and retries the source", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      clock,
    });
    dispatcher.register(
      createIdentityEventsListForCognitionTool({
        listEvents: (options) => harness.identityService.listEvents(options),
      }),
    );

    const trigger = createScheduledReflectionTrigger({
      watermarkRepository,
      intervalMs: 10_000,
      clock,
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue({
        mode: "idle",
        path: "system_1",
        response: "Reflected on recent changes.",
        thoughts: [],
        usage: {
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        },
        retrievedEpisodeIds: [],
        referencedEpisodeIds: [],
        intents: [],
        toolCalls: [],
        agentMessageId: "strm_agent_result",
      }),
    };
    vi.spyOn(watermarkRepository, "set").mockImplementationOnce(() => {
      throw new Error("watermark commit failed");
    });

    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: dispatcher,
      sources: [trigger],
    });

    const firstTick = await scheduler.tick();
    expect(firstTick.firedEvents).toBe(1);
    expect(firstTick.errorCount).toBe(0);
    expect(firstTick.bookkeepingErrorCount).toBe(1);
    expect(firstTick.events[0]).toMatchObject({
      status: "bookkeeping_error",
      turnResultId: "strm_agent_result",
    });
    expect(firstTick.events[0]?.outcomeSummary).toContain("watermark commit failed");
    expect(watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID)).toBeNull();

    const secondTick = await scheduler.tick();
    expect(secondTick.firedEvents).toBe(0);
    expect(secondTick.events).toEqual([]);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);

    clock.advance(30_000);
    const thirdTick = await scheduler.tick();
    expect(thirdTick.firedEvents).toBe(1);
    expect(thirdTick.events[0]?.status).toBe("fired");
    expect(turnRunner.run).toHaveBeenCalledTimes(2);
    expect(
      watermarkRepository.get("autonomy:scheduled-reflection", DEFAULT_SESSION_ID),
    ).toMatchObject({
      lastTs: 1_030_000,
      lastEntryId: expect.any(String),
    });
  });

  it("commits shared source watermarks to the processed event cursor", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const events = [
      {
        id: "event-a",
        sourceName: "goal_followup_due" as const,
        sourceType: "trigger" as const,
        watermarkProcessName: "autonomy:test:shared-cursor",
        sortTs: 100,
        payload: {
          goal_id: "goal_aaaaaaaaaaaaaaaa",
        },
      },
      {
        id: "event-b",
        sourceName: "goal_followup_due" as const,
        sourceType: "trigger" as const,
        watermarkProcessName: "autonomy:test:shared-cursor",
        sortTs: 100,
        payload: {
          goal_id: "goal_bbbbbbbbbbbbbbbb",
        },
      },
    ];
    const turnRunner = {
      run: vi
        .fn()
        .mockResolvedValueOnce({
          mode: "idle",
          path: "system_1",
          response: "Handled event A.",
          thoughts: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          retrievedEpisodeIds: [],
          referencedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_event_a",
        })
        .mockRejectedValueOnce(new Error("event B failed"))
        .mockResolvedValueOnce({
          mode: "idle",
          path: "system_1",
          response: "Handled event B.",
          thoughts: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          retrievedEpisodeIds: [],
          referencedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_event_b",
        }),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [
        {
          name: "goal_followup_due",
          type: "trigger",
          sourceCategory: "operational",
          async scan() {
            const watermark = watermarkRepository.get(
              "autonomy:test:shared-cursor",
              DEFAULT_SESSION_ID,
            );

            return events.filter(
              (event) =>
                watermark === null ||
                event.sortTs > watermark.lastTs ||
                (event.sortTs === watermark.lastTs && event.id > watermark.lastEntryId),
            );
          },
          buildTurn(event) {
            return {
              audience: "self",
              stakes: "low",
              userMessage: `Handle ${event.id}`,
            };
          },
        },
      ],
    });

    const firstTick = await scheduler.tick();
    expect(firstTick.firedEvents).toBe(1);
    expect(firstTick.errorCount).toBe(1);
    expect(
      watermarkRepository.get("autonomy:test:shared-cursor", DEFAULT_SESSION_ID),
    ).toMatchObject({
      lastTs: 100,
      lastEntryId: "event-a",
    });

    clock.advance(30_000);
    const secondTick = await scheduler.tick();
    expect(secondTick.firedEvents).toBe(1);
    expect(secondTick.events[0]).toMatchObject({
      id: "event-b",
      status: "fired",
    });
    expect(
      watermarkRepository.get("autonomy:test:shared-cursor", DEFAULT_SESSION_ID),
    ).toMatchObject({
      lastTs: 100,
      lastEntryId: "event-b",
    });
  });

  it("dispatches mixed trigger and condition sources and records wake metadata", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository,
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue({
          mode: "idle",
          path: "system_1",
          response: "Handled the wake.",
          thoughts: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: "end_turn",
          },
          retrievedEpisodeIds: [],
          referencedEpisodeIds: [],
          intents: [],
          toolCalls: [],
          agentMessageId: "strm_mixed_sources",
        }),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [
        {
          name: "goal_followup_due",
          type: "trigger",
          sourceCategory: "operational",
          async scan() {
            return [
              {
                id: "goal-1",
                sourceName: "goal_followup_due",
                sourceType: "trigger",
                watermarkProcessName: "autonomy:test:goal",
                sortTs: 1,
                payload: {
                  goal_id: "goal_aaaaaaaaaaaaaaaa",
                },
              },
            ];
          },
          buildTurn() {
            return {
              audience: "self",
              stakes: "low",
              userMessage: "Goal follow-up",
            };
          },
        },
        {
          name: "commitment_revoked",
          type: "condition",
          sourceCategory: "operational",
          async scan() {
            return [
              {
                id: "condition-1",
                sourceName: "commitment_revoked",
                sourceType: "condition",
                watermarkProcessName: "autonomy:test:condition",
                sortTs: 2,
                payload: {
                  commitment_id: "cmt_aaaaaaaaaaaaaaaa",
                },
              },
            ];
          },
          buildTurn() {
            return {
              audience: "self",
              stakes: "low",
              userMessage: "Commitment reflection",
            };
          },
        },
      ],
    });

    const result = await scheduler.tick();
    expect(result.firedEvents).toBe(2);
    expect(result.events.map((event) => event.sourceName)).toEqual([
      "goal_followup_due",
      "commitment_revoked",
    ]);

    const wakeEntries = new StreamReader({
      dataDir: harness.tempDir,
      sessionId: DEFAULT_SESSION_ID,
    })
      .tail(8)
      .filter((entry) => entry.kind === "internal_event" && typeof entry.content === "object");

    expect(wakeEntries[0]?.content).toMatchObject({
      kind: "autonomous_wake",
      trigger_type: "trigger",
      source_name: "goal_followup_due",
    });
    expect(wakeEntries[2]?.content).toMatchObject({
      kind: "autonomous_wake",
      trigger_type: "condition",
      source_name: "commitment_revoked",
    });
  });

  it("reports source scan errors and retries only after bounded source backoff", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup = harness.cleanup;

    let intervalCallback: (() => void) | undefined;
    const setIntervalFn = vi.fn((callback: () => void) => {
      intervalCallback = callback;
      return 1 as unknown as ReturnType<typeof setInterval>;
    });
    const onTick = vi.fn();
    const onError = vi.fn();
    let scanCount = 0;

    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({
          dataDir: harness.tempDir,
          sessionId,
          clock,
        }),
      watermarkRepository: new StreamWatermarkRepository({
        db: harness.db,
        clock,
      }),
      turnOrchestrator: {
        run: vi.fn(),
      },
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({
            dataDir: harness.tempDir,
            sessionId,
            clock,
          }),
        clock,
      }),
      sources: [
        {
          name: "scheduled_reflection",
          type: "trigger",
          sourceCategory: "contemplative",
          scan: vi.fn().mockImplementation(async () => {
            scanCount += 1;

            if (scanCount === 1) {
              throw new Error("scan failed");
            }

            return [];
          }),
          buildTurn: vi.fn(),
        },
      ],
      setIntervalFn: setIntervalFn as unknown as typeof setInterval,
      clearIntervalFn: vi.fn(),
    });
    scheduler.setObserver({
      onTick,
      onError,
    });

    scheduler.start();
    intervalCallback?.();
    await vi.waitFor(() => {
      expect(onError).toHaveBeenCalledTimes(1);
      expect(onTick).toHaveBeenCalledTimes(1);
    });

    intervalCallback?.();
    await vi.waitFor(() => {
      expect(onTick).toHaveBeenCalledTimes(2);
    });
    expect(scanCount).toBe(1);

    clock.advance(30_000);
    intervalCallback?.();
    await vi.waitFor(() => {
      expect(onTick).toHaveBeenCalledTimes(3);
    });
    expect(scanCount).toBe(2);

    await scheduler.stop();
  });
});
