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
  silentWakeOutcomeDetail,
  turnEmittedHeadway,
  type AutonomySchedulerOptions,
} from "./scheduler.js";
import type { AutonomyWakeSource } from "./types.js";
import { AutonomyWakesRepository } from "./wakes-repository.js";
import {
  getExecutiveFocusGoalStaleBackoffProcessName,
  goalStaleBackoffState,
} from "./executive-focus-stale-backoff.js";
import {
  DEFAULT_FLEET_BRAKE_OPTIONS,
  FLEET_BRAKE_PROCESS_NAME,
  readFleetBrakeMetadata,
  type FleetBrakeMetadata,
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
              status: "active",
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
  undeliveredOutboundState?: string;
  outboundCallOk?: boolean;
  retiredGoalToolId?: string;
  reflectionRetiredGoalIds?: TurnResult["reflectionRetiredGoalIds"];
  suppressionReason?: Extract<TurnResult["emission"], { kind: "suppressed" }>["reason"];
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
        : { kind: "suppressed", reason: input.suppressionReason ?? "finalizer_no_output" };

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
    toolCalls: [
      ...(input.deliveredOutbound === true || input.undeliveredOutboundState !== undefined
        ? [
            {
              callId: "toolu_outbound",
              name: "tool.outbound.post",
              input: {},
              output: {
                outbound: {
                  emitted: input.deliveredOutbound === true,
                  delivery_outcome:
                    input.undeliveredOutboundState === undefined
                      ? {
                          state: "delivered",
                          agent_message_id: "strm_delivered",
                        }
                      : { state: input.undeliveredOutboundState },
                },
              },
              ok: input.outboundCallOk ?? true,
              durationMs: 1,
            },
          ]
        : []),
      ...(input.retiredGoalToolId === undefined
        ? []
        : [
            {
              callId: "toolu_goal_retire",
              name: "tool.goals.retire",
              input: { goal_id: input.retiredGoalToolId, reason: "Superseded." },
              output: {
                status: "applied",
                goal: { id: input.retiredGoalToolId, status: "abandoned" },
              },
              ok: true,
              durationMs: 1,
            },
          ]),
    ],
    ...(input.reflectionRetiredGoalIds === undefined
      ? {}
      : { reflectionRetiredGoalIds: input.reflectionRetiredGoalIds }),
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
  executiveGoalRank?: (eventId: string, index: number) => number;
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
            ...(input.executiveGoalRank === undefined
              ? {}
              : { executiveGoalRank: input.executiveGoalRank(eventId, index) }),
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

const TEST_SELF_PRIVATE_DISCLOSURE = {
  disclosure:
    "disclosure_class=self_private private-to=unknown; I can use this internally; I do not disclose it to the current audience unless authorized",
  disclosure_label: {
    disclosure_class: "self_private",
    origin_audience_entity_ids: [],
    private_to_entity_ids: [],
    public_to_entity_ids: [],
  },
} as const;

type BatchedGoalDueSpec = {
  eventId: string;
  goalId: string;
  description: string;
  priority: number;
  targetAt: number | null;
  lastProgressTs: number | null;
  reason: "deadline" | "stale" | "both" | "goal_stale";
  rank: number;
  sortTs?: number;
};

function createBatchedGoalDueSource(input: {
  watermarkRepository: StreamWatermarkRepository;
  sourceName: "goal_followup_due" | "executive_focus_due";
  goals: readonly BatchedGoalDueSpec[];
  actionAvailabilityKey?: string;
}): AutonomyWakeSource {
  return {
    name: input.sourceName,
    type: "trigger",
    sourceCategory: "operational",
    async scan() {
      return input.goals.flatMap((goal, index) => {
        const watermarkProcessName = `autonomy:test:batch:${input.sourceName}:${goal.eventId}`;

        if (input.watermarkRepository.get(watermarkProcessName, DEFAULT_SESSION_ID) !== null) {
          return [];
        }

        const selectedGoal = {
          goal_id: goal.goalId,
          description: goal.description,
          priority: goal.priority,
          status: "active" as const,
          target_at: goal.targetAt,
          last_progress_ts: goal.lastProgressTs,
          ...TEST_SELF_PRIVATE_DISCLOSURE,
        };
        const payload =
          input.sourceName === "goal_followup_due"
            ? {
                goal_id: goal.goalId,
                selected_goal_id: goal.goalId,
                description: goal.description,
                priority: goal.priority,
                target_at: goal.targetAt,
                last_progress_ts: goal.lastProgressTs,
                selected_goal: selectedGoal,
                days_stale: 10,
                reason: goal.reason,
                ...TEST_SELF_PRIVATE_DISCLOSURE,
              }
            : {
                reason: "goal_stale",
                selected_goal_id: goal.goalId,
                selected_goal: selectedGoal,
              };

        return [
          {
            id: goal.eventId,
            sourceName: input.sourceName,
            sourceType: "trigger" as const,
            watermarkProcessName,
            sortTs: goal.sortTs ?? index + 1,
            stateTs: goal.lastProgressTs ?? undefined,
            executiveGoalRank: goal.rank,
            ...(input.actionAvailabilityKey === undefined
              ? {}
              : { goalStaleBackoffActionAvailabilityKey: input.actionAvailabilityKey }),
            payload,
          },
        ];
      });
    },
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
      status: null,
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
      status: null,
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

  it("labels the complement of headway with the class that produced it", () => {
    // The predicate above answers "did anything come out"; every no lands in
    // `silent`, but a guard block now holds the streak while the other endings
    // advance it. The same class keeps the diagnostic and policy aligned.
    expect(
      silentWakeOutcomeDetail(createStructuralTurnResult({ emissionKind: "suppressed" })),
    ).toBe("deliberate-silence: finalizer_no_output");
    expect(
      silentWakeOutcomeDetail(
        createStructuralTurnResult({
          emissionKind: "suppressed",
          suppressionReason: "commitment_violation_after_regenerate",
        }),
      ),
    ).toBe("guard-blocked: commitment_violation_after_regenerate");
    expect(
      silentWakeOutcomeDetail(
        createStructuralTurnResult({
          emissionKind: "suppressed",
          suppressionReason: "closure_pressure_only",
        }),
      ),
    ).toBe("guard-blocked: closure_pressure_only");
    expect(
      silentWakeOutcomeDetail(
        createStructuralTurnResult({
          emissionKind: "suppressed",
          suppressionReason: "empty_finalizer",
        }),
      ),
    ).toBe("emission-failed: empty_finalizer");
    // Never throws on a shape it did not expect: this runs inside the block that
    // persists the wake's outcome, so a throw would recast a completed wake as a
    // bookkeeping failure.
    expect(
      silentWakeOutcomeDetail({
        ...createStructuralTurnResult({ emissionKind: "suppressed" }),
        emission: undefined as never,
      }),
    ).toBe("no emission recorded");
  });

  it("names a reach that did not land rather than the closure its emission looks like", () => {
    // A wake whose only act was an outbound post closes with
    // `finalizer_no_output` whatever the post did, so the emission label reads
    // as a silence the entity chose. `turnEmittedHeadway` already reads the
    // delivery state to decide headway; these keep it as far as the reason.
    for (const state of [
      "target_busy",
      "transport_failed",
      "not_transportable",
      "not_emitted",
      "suppressed",
    ]) {
      expect(
        silentWakeOutcomeDetail(
          createStructuralTurnResult({
            emissionKind: "suppressed",
            undeliveredOutboundState: state,
          }),
        ),
      ).toBe(`outbound-undelivered: ${state}`);
    }

    // An unrecognised state widens the tally by one bucket, not one per value,
    // so a new state cannot fragment the counter it is summarised into.
    expect(
      silentWakeOutcomeDetail(
        createStructuralTurnResult({
          emissionKind: "suppressed",
          undeliveredOutboundState: "some_future_state",
        }),
      ),
    ).toBe("outbound-undelivered: unknown_state");

    // A call the dispatcher failed is still a reach that did not land.
    expect(
      silentWakeOutcomeDetail(
        createStructuralTurnResult({
          emissionKind: "suppressed",
          undeliveredOutboundState: "transport_failed",
          outboundCallOk: false,
        }),
      ),
    ).toBe("outbound-undelivered: call_failed");

    // A delivered post is headway, so it never reaches this label, and a turn
    // that posted nothing keeps the emission-derived reason unchanged.
    expect(
      silentWakeOutcomeDetail(
        createStructuralTurnResult({ emissionKind: "suppressed", deliveredOutbound: true }),
      ),
    ).toBe("deliberate-silence: finalizer_no_output");
  });

  it("batches due goals into one budgeted wake while firing every per-goal event", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    const goals = [
      harness.goalsRepository.add({
        description: "Highest scored due goal",
        priority: 9,
        provenance: { kind: "manual" },
      }),
      harness.goalsRepository.add({
        description: "Second due goal",
        priority: 7,
        provenance: { kind: "manual" },
      }),
      harness.goalsRepository.add({
        description: "Third due goal",
        priority: 5,
        provenance: { kind: "manual" },
      }),
    ];
    const source = createBatchedGoalDueSource({
      watermarkRepository,
      sourceName: "goal_followup_due",
      goals: goals.map((goal, index) => ({
        eventId: `batch-event-${index}`,
        goalId: goal.id,
        description: goal.description,
        priority: goal.priority,
        targetAt: null,
        lastProgressTs: null,
        reason: "stale",
        rank: index,
      })),
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
    };
    const scheduler = createScheduler({
      db: harness.db,
      wakeRepository,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 1,
      goalWakeBatchMax: 5,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      goalsRepository: harness.goalsRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [source],
    });

    const result = await scheduler.tick();

    expect(turnRunner.run).toHaveBeenCalledTimes(1);
    expect(wakeRepository.countSince(0)).toBe(1);
    expect(result).toMatchObject({
      dueEvents: 3,
      firedEvents: 1,
      budgetSkipped: 0,
    });
    expect(result.events).toHaveLength(3);
    expect(result.events.every((event) => event.status === "fired")).toBe(true);
    expect(turnRunner.run.mock.calls[0]?.[0].autonomyTrigger?.payload).toMatchObject({
      goal_id: goals[0]!.id,
      secondary_due_goals: [
        {
          goal_id: goals[1]!.id,
          description: goals[1]!.description,
          disclosure_label: { disclosure_class: "self_private" },
        },
        {
          goal_id: goals[2]!.id,
          description: goals[2]!.description,
          disclosure_label: { disclosure_class: "self_private" },
        },
      ],
    });
    for (let index = 0; index < goals.length; index += 1) {
      expect(
        watermarkRepository.get(
          `autonomy:test:batch:goal_followup_due:batch-event-${index}`,
          DEFAULT_SESSION_ID,
        ),
      ).toMatchObject({ lastEntryId: `batch-event-${index}` });
    }
  });

  it.each([1, 2, 3, 4, 5, 6])(
    "atomically rolls back all goal accounting and source latches when watermark write %i fails",
    async (failurePosition) => {
      const clock = new ManualClock(1_050_000);
      const harness = await createOfflineTestHarness({ clock });
      cleanup = harness.cleanup;
      const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
      const actionAvailabilityKey = "outbound_action_surface_v1:atomic-batch";
      const goals = Array.from({ length: 3 }, (_, index) =>
        harness.goalsRepository.add({
          description: `Atomic batch goal ${index}`,
          priority: 10 - index,
          provenance: { kind: "manual" },
        }),
      );
      const source = createBatchedGoalDueSource({
        watermarkRepository,
        sourceName: "goal_followup_due",
        actionAvailabilityKey,
        goals: goals.map((goal, index) => ({
          eventId: `atomic-event-${index}`,
          goalId: goal.id,
          description: goal.description,
          priority: goal.priority,
          targetAt: null,
          lastProgressTs: null,
          reason: "stale",
          rank: index,
          sortTs: 800 + index,
        })),
      });
      const seedByProcess = new Map<
        string,
        NonNullable<ReturnType<typeof watermarkRepository.get>>
      >();

      for (const [index, goal] of goals.entries()) {
        const processName = getExecutiveFocusGoalStaleBackoffProcessName(goal.id);
        watermarkRepository.set(processName, DEFAULT_SESSION_ID, {
          lastTs: 400 + index,
          lastEntryId: `prior-empty-${index}`,
          metadata: {
            empty_count: 2,
            action_availability_key: actionAvailabilityKey,
          },
        });
        seedByProcess.set(processName, watermarkRepository.get(processName, DEFAULT_SESSION_ID)!);
      }

      const scheduler = createScheduler({
        db: harness.db,
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerWindow: 6,
        goalWakeBatchMax: 5,
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        watermarkRepository,
        goalsRepository: harness.goalsRepository,
        turnOrchestrator: {
          run: vi.fn(async () => {
            harness.goalsRepository.updateProgress(goals[1]!.id, "Atomic acted-goal progress", {
              kind: "manual",
            });
            return createStructuralTurnResult({ emissionKind: "suppressed" });
          }),
        },
        toolDispatcher: new ToolDispatcher({
          createStreamWriter: (sessionId) =>
            new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
          clock,
        }),
        sources: [source],
      });
      const originalSet = watermarkRepository.set.bind(watermarkRepository);
      const originalReset = watermarkRepository.reset.bind(watermarkRepository);
      let writePosition = 0;
      const failAfterWrite = <T>(result: T): T => {
        writePosition += 1;

        if (writePosition === failurePosition) {
          throw new Error(`Injected watermark failure at position ${failurePosition}`);
        }

        return result;
      };
      const setSpy = vi
        .spyOn(watermarkRepository, "set")
        .mockImplementation((...args: Parameters<StreamWatermarkRepository["set"]>) => {
          return failAfterWrite(originalSet(...args));
        });
      const resetSpy = vi
        .spyOn(watermarkRepository, "reset")
        .mockImplementation((...args: Parameters<StreamWatermarkRepository["reset"]>) => {
          originalReset(...args);
          failAfterWrite(undefined);
        });

      const failed = await scheduler.tick();

      expect(writePosition).toBe(failurePosition);
      expect(failed).toMatchObject({ firedEvents: 1, bookkeepingErrorCount: 1 });
      expect(failed.events.every((event) => event.status === "bookkeeping_error")).toBe(true);
      for (const [index, goal] of goals.entries()) {
        const processName = getExecutiveFocusGoalStaleBackoffProcessName(goal.id);
        expect(watermarkRepository.get(processName, DEFAULT_SESSION_ID)).toEqual(
          seedByProcess.get(processName),
        );
        expect(
          watermarkRepository.get(
            `autonomy:test:batch:goal_followup_due:atomic-event-${index}`,
            DEFAULT_SESSION_ID,
          ),
        ).toBeNull();
      }

      setSpy.mockRestore();
      resetSpy.mockRestore();
      clock.advance(30_000);
      const retried = await scheduler.tick();

      expect(retried).toMatchObject({ firedEvents: 1, bookkeepingErrorCount: 0 });
      expect(retried.events.every((event) => event.status === "fired")).toBe(true);
      for (const [index, goal] of goals.entries()) {
        expect(
          watermarkRepository.get(
            `autonomy:test:batch:goal_followup_due:atomic-event-${index}`,
            DEFAULT_SESSION_ID,
          ),
        ).not.toBeNull();
        const backoff = watermarkRepository.get(
          getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
          DEFAULT_SESSION_ID,
        );

        if (index === 1) {
          expect(backoff).toBeNull();
        } else {
          expect(backoff?.metadata).toEqual({
            empty_count: 3,
            action_availability_key: actionAvailabilityKey,
          });
        }
      }
    },
  );

  it("reserves capped batch admission for a deadline goal before stale demand", async () => {
    const clock = new ManualClock(1_100_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    const goals = [
      harness.goalsRepository.add({
        description: "Top stale candidate",
        priority: 10,
        provenance: { kind: "manual" },
      }),
      harness.goalsRepository.add({
        description: "Second stale candidate",
        priority: 9,
        provenance: { kind: "manual" },
      }),
      harness.goalsRepository.add({
        description: "Low-score deadline candidate",
        priority: 1,
        provenance: { kind: "manual" },
        targetAt: clock.now() + 1_000,
      }),
      harness.goalsRepository.add({
        description: "Third stale candidate",
        priority: 8,
        provenance: { kind: "manual" },
      }),
    ];
    const source = createBatchedGoalDueSource({
      watermarkRepository,
      sourceName: "goal_followup_due",
      goals: goals.map((goal, index) => ({
        eventId: `capped-event-${index}`,
        goalId: goal.id,
        description: goal.description,
        priority: goal.priority,
        targetAt: goal.target_at,
        lastProgressTs: null,
        reason: index === 2 ? "deadline" : "stale",
        rank: index === 2 ? 99 : index,
      })),
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 1,
      goalWakeBatchMax: 2,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      wakeRepository,
      goalsRepository: harness.goalsRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [source],
    });

    const result = await scheduler.tick();
    const payload = turnRunner.run.mock.calls[0]?.[0].autonomyTrigger?.payload;

    expect(turnRunner.run).toHaveBeenCalledTimes(1);
    expect(wakeRepository.countSince(0)).toBe(1);
    expect(result).toMatchObject({ firedEvents: 1, budgetSkipped: 2 });
    expect(payload).toMatchObject({
      goal_id: goals[0]!.id,
      secondary_due_goals: [{ goal_id: goals[2]!.id, reason: "deadline" }],
    });
    expect(JSON.stringify(payload)).not.toContain(goals[1]!.id);
    expect(JSON.stringify(payload)).not.toContain(goals[3]!.id);
  });

  it("reserves one stale slot across consecutive batches under sustained deadline demand", async () => {
    const clock = new ManualClock(1_150_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const actionAvailabilityKey = "outbound_action_surface_v1:stale-reserve";
    const deadlineGoals = Array.from({ length: 6 }, (_, index) =>
      harness.goalsRepository.add({
        description: `Deadline pressure goal ${index}`,
        priority: 10 - index,
        targetAt: clock.now() + 1_000 + index,
        provenance: { kind: "manual" },
      }),
    );
    const staleGoals = Array.from({ length: 3 }, (_, index) =>
      harness.goalsRepository.add({
        description: `Reserved stale goal ${index}`,
        priority: 3 - index,
        provenance: { kind: "manual" },
      }),
    );

    for (const [index, goal] of staleGoals.entries()) {
      watermarkRepository.set(
        getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
        DEFAULT_SESSION_ID,
        {
          lastTs: 500 + index,
          lastEntryId: `stale-prior-empty-${index}`,
          metadata: {
            empty_count: 2,
            action_availability_key: actionAvailabilityKey,
          },
        },
      );
    }

    const allGoals = [...deadlineGoals, ...staleGoals];
    const source = createBatchedGoalDueSource({
      watermarkRepository,
      sourceName: "goal_followup_due",
      actionAvailabilityKey,
      goals: allGoals.map((goal, index) => ({
        eventId: `reserve-event-${index}`,
        goalId: goal.id,
        description: goal.description,
        priority: goal.priority,
        targetAt: goal.target_at,
        lastProgressTs: null,
        reason: index < deadlineGoals.length ? "deadline" : "stale",
        rank: index,
        sortTs: 700 + index,
      })),
    });
    const turnRunner = {
      run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 3,
      goalWakeBatchMax: 3,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      goalsRepository: harness.goalsRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [source],
    });

    const result = await scheduler.tick();
    const deadlineIds = new Set<string>(deadlineGoals.map((goal) => goal.id));
    const staleIds = new Set<string>(staleGoals.map((goal) => goal.id));

    expect(result).toMatchObject({ firedEvents: 3, budgetSkipped: 0 });
    expect(result.events).toHaveLength(9);
    expect(turnRunner.run).toHaveBeenCalledTimes(3);
    for (const [turnInput] of turnRunner.run.mock.calls) {
      const payload = turnInput.autonomyTrigger?.payload;
      const presentedIds = [
        typeof payload?.goal_id === "string" ? payload.goal_id : "",
        ...((payload?.secondary_due_goals as Array<{ goal_id: string }> | undefined) ?? []).map(
          (goal) => goal.goal_id,
        ),
      ];

      expect(presentedIds.filter((goalId) => staleIds.has(goalId))).toHaveLength(1);
      expect(presentedIds.filter((goalId) => deadlineIds.has(goalId))).toHaveLength(2);
    }

    for (const goal of staleGoals) {
      const backoff = watermarkRepository.get(
        getExecutiveFocusGoalStaleBackoffProcessName(goal.id),
        DEFAULT_SESSION_ID,
      );
      expect(backoff?.metadata).toEqual({
        empty_count: 3,
        action_availability_key: actionAvailabilityKey,
      });
      expect(
        goalStaleBackoffState({
          watermark: backoff,
          lastProgressTs: null,
          baseCooldownMs: 100,
          multiplier: 2,
          maxCooldownMs: 1_000,
          dormancyCount: 3,
          actionAvailabilityKey,
        }).endMs,
      ).toBe(Number.POSITIVE_INFINITY);
    }
  });

  it("combines followup and executive goal-stale lanes into one scored wake", async () => {
    const clock = new ManualClock(1_200_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    const executiveGoal = harness.goalsRepository.add({
      description: "Executive stale primary",
      priority: 10,
      provenance: { kind: "manual" },
    });
    const followupGoal = harness.goalsRepository.add({
      description: "Followup stale secondary",
      priority: 7,
      provenance: { kind: "manual" },
    });
    const sources = [
      createBatchedGoalDueSource({
        watermarkRepository,
        sourceName: "goal_followup_due",
        goals: [
          {
            eventId: "combined-followup",
            goalId: followupGoal.id,
            description: followupGoal.description,
            priority: followupGoal.priority,
            targetAt: null,
            lastProgressTs: null,
            reason: "stale",
            rank: 1,
          },
        ],
      }),
      createBatchedGoalDueSource({
        watermarkRepository,
        sourceName: "executive_focus_due",
        goals: [
          {
            eventId: "combined-executive",
            goalId: executiveGoal.id,
            description: executiveGoal.description,
            priority: executiveGoal.priority,
            targetAt: null,
            lastProgressTs: null,
            reason: "goal_stale",
            rank: 0,
          },
        ],
      }),
    ];
    const turnRunner = {
      run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 1,
      goalWakeBatchMax: 5,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      wakeRepository,
      goalsRepository: harness.goalsRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources,
    });

    const result = await scheduler.tick();

    expect(wakeRepository.countSince(0)).toBe(1);
    expect(turnRunner.run).toHaveBeenCalledTimes(1);
    expect(result.firedEvents).toBe(1);
    expect(result.events.map((event) => event.sourceName)).toEqual([
      "executive_focus_due",
      "goal_followup_due",
    ]);
    expect(turnRunner.run.mock.calls[0]?.[0].autonomyTrigger).toMatchObject({
      source_name: "executive_focus_due",
      payload: {
        selected_goal_id: executiveGoal.id,
        secondary_due_goals: [
          {
            source_name: "goal_followup_due",
            goal_id: followupGoal.id,
          },
        ],
      },
    });
  });

  it("keeps a single due goal payload byte-identical when batching is enabled", async () => {
    const clock = new ManualClock(1_300_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    const goal = harness.goalsRepository.add({
      description: "Only due goal",
      priority: 8,
      provenance: { kind: "manual" },
    });
    const source = createBatchedGoalDueSource({
      watermarkRepository,
      sourceName: "goal_followup_due",
      goals: [
        {
          eventId: "single-batch-compatible",
          goalId: goal.id,
          description: goal.description,
          priority: goal.priority,
          targetAt: null,
          lastProgressTs: null,
          reason: "stale",
          rank: 0,
        },
      ],
    });
    const originalPayload = (await source.scan())[0]?.payload;
    const turnRunner = {
      run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
    };
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      maxWakesPerWindow: 1,
      goalWakeBatchMax: 5,
      wakeRepository,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository,
      goalsRepository: harness.goalsRepository,
      turnOrchestrator: turnRunner,
      toolDispatcher: new ToolDispatcher({
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        clock,
      }),
      sources: [source],
    });

    await scheduler.tick();

    expect(JSON.stringify(turnRunner.run.mock.calls[0]?.[0].autonomyTrigger?.payload)).toBe(
      JSON.stringify(originalPayload),
    );
    expect(wakeRepository.listSince(0, 1)[0]?.selected_goal_id).toBe(goal.id);
  });

  it("accounts for an untouched batched goal exactly like a silent single-goal wake", async () => {
    const harnesses: Array<Awaited<ReturnType<typeof createOfflineTestHarness>>> = [];

    const runCase = async (batched: boolean) => {
      const clock = new ManualClock(1_400_000);
      const harness = await createOfflineTestHarness({ clock });
      harnesses.push(harness);
      const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
      const untouchedGoal = harness.goalsRepository.add({
        description: "Untouched due goal",
        priority: 7,
        provenance: { kind: "manual" },
      });
      const actedGoal = batched
        ? harness.goalsRepository.add({
            description: "Acted secondary goal",
            priority: 10,
            provenance: { kind: "manual" },
          })
        : null;
      const untouchedBackoffName = getExecutiveFocusGoalStaleBackoffProcessName(untouchedGoal.id);
      watermarkRepository.set(untouchedBackoffName, DEFAULT_SESSION_ID, {
        lastTs: 500,
        lastEntryId: "prior-untouched-empty",
        metadata: { empty_count: 2 },
      });

      if (actedGoal !== null) {
        watermarkRepository.set(
          getExecutiveFocusGoalStaleBackoffProcessName(actedGoal.id),
          DEFAULT_SESSION_ID,
          {
            lastTs: 500,
            lastEntryId: "prior-acted-empty",
            metadata: { empty_count: 2 },
          },
        );
      }

      const specs: BatchedGoalDueSpec[] = [
        ...(actedGoal === null
          ? []
          : [
              {
                eventId: "acted-batch-event",
                goalId: actedGoal.id,
                description: actedGoal.description,
                priority: actedGoal.priority,
                targetAt: null,
                lastProgressTs: null,
                reason: "stale" as const,
                rank: 1,
                sortTs: 700,
              },
            ]),
        {
          eventId: "untouched-batch-event",
          goalId: untouchedGoal.id,
          description: untouchedGoal.description,
          priority: untouchedGoal.priority,
          targetAt: null,
          lastProgressTs: null,
          reason: "stale",
          rank: 0,
          sortTs: 700,
        },
      ];
      const source = createBatchedGoalDueSource({
        watermarkRepository,
        sourceName: "goal_followup_due",
        goals: specs,
        actionAvailabilityKey: "outbound_action_surface_v1:batch-test",
      });
      const turnRunner = {
        run: vi.fn(async () => {
          if (actedGoal !== null) {
            harness.goalsRepository.updateProgress(actedGoal.id, "Concrete autonomous headway", {
              kind: "manual",
            });
          }

          return createStructuralTurnResult({ emissionKind: "suppressed" });
        }),
      };
      const scheduler = createScheduler({
        db: harness.db,
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerWindow: 6,
        goalWakeBatchMax: batched ? 5 : 1,
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        watermarkRepository,
        goalsRepository: harness.goalsRepository,
        turnOrchestrator: turnRunner,
        toolDispatcher: new ToolDispatcher({
          createStreamWriter: (sessionId) =>
            new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
          clock,
        }),
        sources: [source],
      });

      await scheduler.tick();

      return {
        untouched: watermarkRepository.get(untouchedBackoffName, DEFAULT_SESSION_ID),
        acted:
          actedGoal === null
            ? undefined
            : watermarkRepository.get(
                getExecutiveFocusGoalStaleBackoffProcessName(actedGoal.id),
                DEFAULT_SESSION_ID,
              ),
      };
    };

    try {
      const batched = await runCase(true);
      const single = await runCase(false);

      expect(batched.untouched).toMatchObject({
        lastTs: 700,
        lastEntryId: "untouched-batch-event",
        metadata: {
          empty_count: 3,
          action_availability_key: "outbound_action_surface_v1:batch-test",
        },
      });
      expect(batched.untouched).toMatchObject({
        lastTs: single.untouched?.lastTs,
        lastEntryId: single.untouched?.lastEntryId,
        sessionId: single.untouched?.sessionId,
        updatedAt: single.untouched?.updatedAt,
        metadata: single.untouched?.metadata,
      });
      expect(batched.acted).toBeNull();
    } finally {
      await Promise.all(harnesses.map((harness) => harness.cleanup()));
    }
  });

  it("records ordered headway bases and credits only structurally attributable closure", async () => {
    const runCase = async (input: {
      mutation: "progress_and_tool_retire" | "reflection_retire" | "external_retire" | "none";
      emissionKind: "message" | "continue_thought" | "suppressed";
      deliveredOutbound?: boolean;
    }) => {
      const clock = new ManualClock(1_450_000);
      const harness = await createOfflineTestHarness({ clock });
      const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
      const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
      const goal = harness.goalsRepository.add({
        description: "Attribution target",
        priority: 8,
        provenance: { kind: "manual" },
      });
      const backoffProcessName = getExecutiveFocusGoalStaleBackoffProcessName(goal.id);
      watermarkRepository.set(backoffProcessName, DEFAULT_SESSION_ID, {
        lastTs: 500,
        lastEntryId: "prior-empty-wake",
        metadata: { empty_count: 2 },
      });
      const source = createBatchedGoalDueSource({
        watermarkRepository,
        sourceName: "goal_followup_due",
        goals: [
          {
            eventId: `attribution-${input.mutation}`,
            goalId: goal.id,
            description: goal.description,
            priority: goal.priority,
            targetAt: null,
            lastProgressTs: null,
            reason: "stale",
            rank: 0,
          },
        ],
      });
      const turnRunner = {
        run: vi.fn(async () => {
          if (input.mutation === "progress_and_tool_retire") {
            harness.goalsRepository.updateProgress(goal.id, "Concrete result produced", {
              kind: "manual",
            });
            harness.goalsRepository.retire(goal.id, "Goal superseded", { kind: "manual" });
          } else if (input.mutation === "reflection_retire") {
            harness.goalsRepository.updateStatus(goal.id, "abandoned", { kind: "manual" });
          } else if (input.mutation === "external_retire") {
            harness.goalsRepository.updateStatus(goal.id, "done", { kind: "manual" });
          }

          return createStructuralTurnResult({
            emissionKind: input.emissionKind,
            deliveredOutbound: input.deliveredOutbound,
            ...(input.mutation === "progress_and_tool_retire"
              ? { retiredGoalToolId: goal.id }
              : {}),
            ...(input.mutation === "reflection_retire"
              ? { reflectionRetiredGoalIds: [goal.id] }
              : {}),
          });
        }),
      };
      const scheduler = createScheduler({
        db: harness.db,
        wakeRepository,
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerWindow: 6,
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        watermarkRepository,
        goalsRepository: harness.goalsRepository,
        turnOrchestrator: turnRunner,
        toolDispatcher: new ToolDispatcher({
          createStreamWriter: (sessionId) =>
            new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
          clock,
        }),
        sources: [source],
      });

      try {
        await scheduler.tick();
        return {
          goalId: goal.id,
          wake: wakeRepository.listSince(0, 1)[0],
          backoff: watermarkRepository.get(backoffProcessName, DEFAULT_SESSION_ID),
        };
      } finally {
        await harness.cleanup();
      }
    };

    const toolClosure = await runCase({
      mutation: "progress_and_tool_retire",
      emissionKind: "message",
      deliveredOutbound: true,
    });
    expect(toolClosure.wake).toMatchObject({
      selected_goal_id: toolClosure.goalId,
      outcome: "headway",
      outcome_detail: `emitted message; delivered outbound post; progress recorded on ${toolClosure.goalId}; goal ${toolClosure.goalId} retired by this turn`,
    });
    expect(toolClosure.backoff).toBeNull();

    const reflectionClosure = await runCase({
      mutation: "reflection_retire",
      emissionKind: "suppressed",
    });
    expect(reflectionClosure.wake).toMatchObject({
      outcome: "headway",
      outcome_detail: `goal ${reflectionClosure.goalId} retired by this turn`,
    });
    expect(reflectionClosure.backoff).toBeNull();

    const externalClosure = await runCase({
      mutation: "external_retire",
      emissionKind: "suppressed",
    });
    expect(externalClosure.wake).toMatchObject({
      outcome: "silent",
      outcome_detail: "deliberate-silence: finalizer_no_output",
    });
    expect(externalClosure.backoff).toMatchObject({ metadata: { empty_count: 3 } });

    const continuedThought = await runCase({
      mutation: "none",
      emissionKind: "continue_thought",
    });
    expect(continuedThought.wake).toMatchObject({
      outcome: "headway",
      outcome_detail: "continue_thought",
    });
    expect(continuedThought.backoff).toMatchObject({ metadata: { empty_count: 3 } });
  });

  it("deduplicates the same goal across both lanes while preserving exact single-wake dormancy accounting", async () => {
    const harnesses: Array<Awaited<ReturnType<typeof createOfflineTestHarness>>> = [];
    const sharedGoalId = "goal_cccccccccccccccc" as never;
    const actionAvailabilityKey = "outbound_action_surface_v1:duplicate-lanes";
    const backoffProcessName = getExecutiveFocusGoalStaleBackoffProcessName(sharedGoalId);

    const runCase = async (includeExecutiveLane: boolean) => {
      const clock = new ManualClock(1_600_000);
      const harness = await createOfflineTestHarness({ clock });
      harnesses.push(harness);
      const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
      const goal = harness.goalsRepository.add({
        id: sharedGoalId,
        description: "One goal surfaced through two due lanes",
        priority: 8,
        provenance: { kind: "manual" },
      });
      watermarkRepository.set(backoffProcessName, DEFAULT_SESSION_ID, {
        lastTs: 400,
        lastEntryId: "prior-duplicate-empty",
        metadata: {
          empty_count: 2,
          action_availability_key: actionAvailabilityKey,
        },
      });
      const followupSource = createBatchedGoalDueSource({
        watermarkRepository,
        sourceName: "goal_followup_due",
        actionAvailabilityKey,
        goals: [
          {
            eventId: "shared-accounting-event",
            goalId: goal.id,
            description: goal.description,
            priority: goal.priority,
            targetAt: null,
            lastProgressTs: null,
            reason: "stale",
            rank: 0,
            sortTs: 700,
          },
        ],
      });
      const executiveSource = createBatchedGoalDueSource({
        watermarkRepository,
        sourceName: "executive_focus_due",
        actionAvailabilityKey,
        goals: [
          {
            eventId: "zz-duplicate-executive-event",
            goalId: goal.id,
            description: goal.description,
            priority: goal.priority,
            targetAt: null,
            lastProgressTs: null,
            reason: "goal_stale",
            rank: 0,
            sortTs: 700,
          },
        ],
      });
      const turnRunner = {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
      };
      const scheduler = createScheduler({
        db: harness.db,
        enabled: true,
        intervalMs: 1_000,
        maxWakesPerWindow: 6,
        goalWakeBatchMax: includeExecutiveLane ? 5 : 1,
        clock,
        createStreamWriter: (sessionId) =>
          new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
        watermarkRepository,
        goalsRepository: harness.goalsRepository,
        turnOrchestrator: turnRunner,
        toolDispatcher: new ToolDispatcher({
          createStreamWriter: (sessionId) =>
            new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
          clock,
        }),
        sources: includeExecutiveLane ? [followupSource, executiveSource] : [followupSource],
      });

      const result = await scheduler.tick();

      return {
        backoff: watermarkRepository.get(backoffProcessName, DEFAULT_SESSION_ID),
        followupLatch: watermarkRepository.get(
          "autonomy:test:batch:goal_followup_due:shared-accounting-event",
          DEFAULT_SESSION_ID,
        ),
        executiveLatch: watermarkRepository.get(
          "autonomy:test:batch:executive_focus_due:zz-duplicate-executive-event",
          DEFAULT_SESSION_ID,
        ),
        result,
        turnRunner,
      };
    };

    try {
      const batched = await runCase(true);
      const control = await runCase(false);

      expect(batched.turnRunner.run).toHaveBeenCalledTimes(1);
      expect(batched.result).toMatchObject({ firedEvents: 1 });
      expect(batched.result.events).toHaveLength(2);
      expect(batched.turnRunner.run.mock.calls[0]?.[0].autonomyTrigger?.payload).not.toHaveProperty(
        "secondary_due_goals",
      );
      expect(batched.followupLatch).toMatchObject({ lastEntryId: "shared-accounting-event" });
      expect(batched.executiveLatch).toMatchObject({
        lastEntryId: "zz-duplicate-executive-event",
      });
      expect(batched.backoff).toEqual(control.backoff);
      expect(batched.backoff).toEqual({
        processName: backoffProcessName,
        sessionId: DEFAULT_SESSION_ID,
        lastTs: 700,
        lastEntryId: "shared-accounting-event",
        updatedAt: 1_600_000,
        metadata: {
          empty_count: 3,
          action_availability_key: actionAvailabilityKey,
        },
      });
      expect(
        goalStaleBackoffState({
          watermark: batched.backoff,
          lastProgressTs: null,
          baseCooldownMs: 100,
          multiplier: 2,
          maxCooldownMs: 1_000,
          dormancyCount: 3,
          actionAvailabilityKey,
        }).endMs,
      ).toBe(Number.POSITIVE_INFINITY);
    } finally {
      await Promise.all(harnesses.map((harness) => harness.cleanup()));
    }
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

  it("does not reset stale-goal wake backoff for outbound emission alone", async () => {
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
    ).resolves.toMatchObject({
      progressedGoal: {
        lastTs: 1_000,
        lastEntryId: "goal_aaaaaaaaaaaaaaaa:stale:1000",
        metadata: {
          empty_count: 3,
        },
      },
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

  it("stamps the per-goal brake when a followup wake only emits outbound", async () => {
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
    ).toMatchObject({
      metadata: {
        empty_count: 3,
        action_availability_key: "outbound_action_surface_v1:test",
      },
    });
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

  it("passes the configured preparation-tool timeout to the dispatcher", async () => {
    const clock = new ManualClock(1_000_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const dispatcher = new ToolDispatcher({
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      clock,
    });
    dispatcher.register(
      createCommitmentsListTool({
        listCommitments: () => [],
      }),
    );
    const dispatch = vi.spyOn(dispatcher, "dispatch");
    const scheduler = createScheduler({
      db: harness.db,
      enabled: true,
      intervalMs: 1_000,
      prepToolTimeoutMs: 45_000,
      maxWakesPerWindow: 6,
      clock,
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir: harness.tempDir, sessionId, clock }),
      watermarkRepository: new StreamWatermarkRepository({ db: harness.db, clock }),
      turnOrchestrator: {
        run: vi.fn().mockResolvedValue(createStructuralTurnResult({ emissionKind: "suppressed" })),
      },
      toolDispatcher: dispatcher,
      sources: [createTestDueSource("prep-timeout-event", "commitment_expiring")],
    });

    expect((await scheduler.tick()).firedEvents).toBe(1);
    expect(dispatch).toHaveBeenCalledWith(expect.objectContaining({ timeoutMs: 45_000 }));
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
    // The rows driving the streak carry what ended them, so the count above can
    // be read as a composition rather than as one behaviour repeated five times.
    expect(wakeRepository.summarizeOutcomeDetailsSince(0, "silent")).toEqual({
      total: 5,
      without_detail: 0,
      reasons: [{ detail: "deliberate-silence: finalizer_no_output", count: 5 }],
    });
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

  it("holds guard-blocked wakes out of the empty streak while preserving its other outcomes", async () => {
    const clock = new ManualClock(2_025_000);
    const harness = await createOfflineTestHarness({ clock });
    cleanup = harness.cleanup;
    const watermarkRepository = new StreamWatermarkRepository({ db: harness.db, clock });
    const wakeRepository = new AutonomyWakesRepository({ db: harness.db, clock });
    setFleetBrakeState(watermarkRepository, clock, {
      empty_streak: 2,
      streak_anchor_ts: clock.now() - 1_000,
      last_wake_ts: clock.now() - 100,
    });
    const fleetUpdates: FleetBrakeMetadata[] = [];
    const originalSet = watermarkRepository.set.bind(watermarkRepository);
    vi.spyOn(watermarkRepository, "set").mockImplementation((processName, sessionId, input) => {
      const result = originalSet(processName, sessionId, input);

      if (processName === FLEET_BRAKE_PROCESS_NAME) {
        fleetUpdates.push(readFleetBrakeMetadata(watermarkRepository.get(processName, sessionId)));
      }

      return result;
    });
    const source = createPersistentDueSource({
      watermarkRepository,
      eventIds: ["guard-blocked", "chosen-silence", "transport-failure", "headway"],
    });
    const turnRunner = {
      run: vi
        .fn()
        .mockResolvedValueOnce(
          createStructuralTurnResult({
            emissionKind: "suppressed",
            suppressionReason: "commitment_violation_after_regenerate",
          }),
        )
        .mockResolvedValueOnce(createStructuralTurnResult({ emissionKind: "suppressed" }))
        .mockResolvedValueOnce(
          createStructuralTurnResult({
            emissionKind: "suppressed",
            undeliveredOutboundState: "transport_failed",
          }),
        )
        .mockResolvedValueOnce(createStructuralTurnResult({ emissionKind: "message" })),
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

    expect((await scheduler.tick()).firedEvents).toBe(4);
    expect(fleetUpdates.map((metadata) => metadata.empty_streak)).toEqual([2, 3, 4, 0]);
    await expect(scheduler.describe()).resolves.toMatchObject({
      fleet_brake: {
        empty_streak: 0,
        window_silent_reasons: {
          total: 3,
          without_detail: 0,
          reasons: [
            { detail: "deliberate-silence: finalizer_no_output", count: 1 },
            { detail: "guard-blocked: commitment_violation_after_regenerate", count: 1 },
            { detail: "outbound-undelivered: transport_failed", count: 1 },
          ],
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

  it("describes sources and budget with null-outcome wakes stamped in flight", async () => {
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
    wakeRepository.recordOutcome(headwayWake.id, "headway", "emitted message");
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
            in_flight: 0,
            in_flight_started_at: [],
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
            in_flight: 1,
            // The fire stamp of the row counted by in_flight. Nothing ever
            // writes an outcome for a row whose bookkeeping threw, so the count
            // alone cannot tell a permanent orphan from a healthy transient --
            // only a stamp that repeats across two reads can.
            in_flight_started_at: [980_000],
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
      fleet_brake: {
        window_headway_reasons: {
          total: 1,
          without_detail: 0,
          reasons: [{ detail: "emitted message", count: 1 }],
        },
      },
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
      scheduled_tick_at: 1_005_000,
    });
    await scheduler.tick();
    clock.advance(20_000);
    // The tick came due at 1_005_000 and the read is at 1_020_000: next_tick_at floors to the read
    // and loses the 15s the loop is behind by, scheduled_tick_at keeps it.
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: 1_020_000,
      scheduled_tick_at: 1_005_000,
    });
    await scheduler.stop();
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: null,
      scheduled_tick_at: null,
    });

    clock.advance(10_000);
    scheduler.start();
    await expect(scheduler.describe()).resolves.toMatchObject({
      next_tick_at: 1_035_000,
      scheduled_tick_at: 1_035_000,
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

  // The early return that refuses a fire while a tick runs is the one path in
  // the scheduler that writes nothing at all: no wake row, no outcome, no error.
  // So a stretch in which every fire was refused and a stretch in which nothing
  // was due are the same (empty) evidence in the budget and outcome tallies --
  // which is what this asserts alongside the counter, because the counter is
  // only worth having if the surfaces beside it genuinely cannot supply it.
  it("counts interval fires refused while a tick runs, which leave no other trace", async () => {
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
    let resolveTurn: ((value: unknown) => void) | undefined;
    const turnCompletion = new Promise((resolve) => {
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
      clearIntervalFn: vi.fn(),
    });

    scheduler.start();
    await expect(scheduler.describe()).resolves.toMatchObject({
      interval_ms: 1_000,
      dropped_interval_fires: { since_interval_armed: 0, current_tick: null },
    });

    intervalCallback?.();
    await vi.waitFor(() => {
      expect(turnRunner.run).toHaveBeenCalledTimes(1);
    });

    for (let fire = 0; fire < 3; fire += 1) {
      clock.advance(1_000);
      intervalCallback?.();
    }

    const duringFreeze = await scheduler.describe();
    expect(duringFreeze.tick_in_flight).toBe(true);
    expect(duringFreeze.dropped_interval_fires).toEqual({
      since_interval_armed: 3,
      current_tick: 3,
    });
    // Four fires, one wake: the other three are absent from every count on the
    // page rather than represented in any of them.
    expect(turnRunner.run).toHaveBeenCalledTimes(1);
    expect(duringFreeze.budget.used_in_current_window).toBe(1);
    expect(duringFreeze.fleet_brake.window_outcomes).toEqual({
      headway: 0,
      silent: 0,
      error: 0,
      busy: 0,
    });

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
      agentMessageId: "strm_dropped_fires",
    });
    await vi.waitFor(async () => {
      expect((await scheduler.describe()).tick_in_flight).toBe(false);
    });

    // The running total survives the tick that earned it; the per-tick count
    // goes null rather than holding a figure that would read as a live cost.
    await expect(scheduler.describe()).resolves.toMatchObject({
      dropped_interval_fires: { since_interval_armed: 3, current_tick: null },
    });

    await scheduler.stop();
    await expect(scheduler.describe()).resolves.toMatchObject({
      dropped_interval_fires: { since_interval_armed: 0, current_tick: null },
    });
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
