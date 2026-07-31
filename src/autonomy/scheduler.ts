import { StreamWatermarkRepository, StreamWriter } from "../stream/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import {
  AuthError,
  AutonomyError,
  LLMError,
  SessionBusyError,
  findInErrorCauseChain,
} from "../util/errors.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../util/ids.js";
import type { ToolDispatcher } from "../tools/dispatcher.js";
import { classifySuppressionReason } from "../cognition/generation/suppression-outcome.js";
import type { TurnOrchestrator, TurnResult } from "../cognition/index.js";
import { memoryDisclosurePayloadFields } from "../memory/common/disclosure-serializers.js";
import type { SelfDecisionRepository } from "../memory/self-decisions/index.js";
import type { TrainOfThoughtRepository } from "../memory/train-of-thought/index.js";
import type { GoalsRepository } from "../memory/self/index.js";
import { selfPrivateMemoryDisclosureLabel } from "../memory/common/disclosure-label.js";
import { OUTBOUND_POST_TOOL_NAME } from "../tools/internal/outbound-post-name.js";

import type {
  AutonomyConditionName,
  AutonomySchedulerDescription,
  AutonomySchedulerSourceDescription,
  AutonomyTickEventResult,
  AutonomyTriggerName,
  AutonomyWakeSource,
  AutonomyWakeSourceCategory,
  TickResult,
  DueEvent,
} from "./types.js";
import { AUTONOMY_WAKE_SOURCE_METADATA, AUTONOMY_WAKE_SOURCE_NAMES } from "./types.js";
import type { AutonomyWakesRepository } from "./wakes-repository.js";
import {
  getExecutiveFocusGoalStaleBackoffProcessName,
  readExecutiveFocusGoalStaleBackoffMetadata,
} from "./executive-focus-stale-backoff.js";
import {
  DEFAULT_FLEET_BRAKE_OPTIONS,
  FLEET_BRAKE_PROCESS_NAME,
  emptyFleetBrakeMetadata,
  fleetBrakeCooldownUntilMs,
  fleetBrakeErrorPausedUntilMs,
  readFleetBrakeMetadata,
  type FleetBrakeMetadata,
  type FleetBrakeOptions,
} from "./fleet-brake.js";

type IntervalHandle = ReturnType<typeof setInterval>;
type RetryBackoffState = {
  delayMs: number;
  nextEligibleTs: number;
};
type FleetAdmissionDecision = {
  skipStatus: "fleet_cooldown_skipped" | "error_circuit_skipped" | null;
  bypassKind: "deadline" | "freshness" | null;
  metadata: FleetBrakeMetadata | null;
};
type GoalWakeOutcome = {
  headway: boolean;
  concern: ReturnType<typeof goalConcernPayload>;
};
type ScannedDueEvents = {
  events: Array<{ source: AutonomyWakeSource; event: DueEvent }>;
  sourceErrorCount: number;
};

const INITIAL_RETRY_BACKOFF_MS = 30_000;
const MAX_RETRY_BACKOFF_MS = 3_600_000;
const WAKE_PRUNE_SAFETY_BUFFER_MS = 7 * 24 * 60 * 60 * 1_000;
// Autonomy wake PREPARATION runs a context-gathering tool (e.g. episodic
// search = embed + LanceDB vector search) before a background wake. It is not
// latency-sensitive, and the dispatcher's 5s default is too tight for that
// search under load, which made prep fail and the trigger retry-loop. Live/
// reactive tool calls keep the 5s default; only prep gets this longer bound.
const AUTONOMY_PREP_TOOL_TIMEOUT_MS = 30_000;

export type AutonomySchedulerObserver = {
  onTick?(result: TickResult): void | Promise<void>;
  onError?(error: unknown): void | Promise<void>;
};

export type AutonomySchedulerStopOptions = {
  graceful?: boolean;
};

export type AutonomySchedulerOptions = {
  enabled: boolean;
  intervalMs: number;
  maxWakesPerWindow: number;
  budgetWindowMs: number;
  reservedContemplativeWakesPerWindow?: number;
  fleetBrake?: FleetBrakeOptions;
  respectGoalFollowupStaleBackoff?: boolean;
  sessionId?: SessionId;
  clock?: Clock;
  createStreamWriter: (sessionId: SessionId) => StreamWriter;
  watermarkRepository: StreamWatermarkRepository;
  wakeRepository: AutonomyWakesRepository;
  selfDecisionRepository?: Pick<SelfDecisionRepository, "record">;
  trainOfThoughtRepository?: Pick<TrainOfThoughtRepository, "get">;
  goalsRepository?: Pick<GoalsRepository, "get">;
  turnOrchestrator: Pick<TurnOrchestrator, "run">;
  toolDispatcher: ToolDispatcher;
  sources: readonly AutonomyWakeSource[];
  setIntervalFn?: typeof setInterval;
  clearIntervalFn?: typeof clearInterval;
};

function summarizeOutcome(text: string): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  return collapsed.length <= 240 ? collapsed : `${collapsed.slice(0, 239)}…`;
}

// Phase 1.1 (B): when an autonomous wake produces no user-facing output, the
// decision WAS to stay silent. Record the structural suppression reason (a
// finalizer/suppression enum -- never user-content words, so multilingual-safe)
// so the operator-introspection lane isn't empty for no-output reflections.
// We store structure only; the model phrases it naturally at read time.
function summarizeAutonomousDecision(turnResult: TurnResult): string {
  const emitted = summarizeOutcome(turnResult.response);

  if (emitted.length > 0) {
    return emitted;
  }

  if (turnResult.emission.kind === "suppressed") {
    const outcomeClass = classifySuppressionReason(turnResult.emission.reason).replaceAll("-", " ");
    const detail = (
      turnResult.emission.primary_no_output_reason ?? turnResult.emission.reason
    ).replaceAll("_", " ");

    return summarizeOutcome(`Stayed silent (${outcomeClass}): ${detail}`);
  }

  if (turnResult.emission.kind === "continue_thought") {
    return "Continued private train of thought.";
  }

  return "";
}

function autonomousDecisionRationale(turnResult: TurnResult): string | null {
  return turnResult.emission?.kind === "suppressed"
    ? (turnResult.emission.decision_rationale ?? null)
    : null;
}

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
}

export class AutonomySourcePreparationError extends AutonomyError {
  constructor(sourceName: AutonomyWakeSource["name"], error: unknown) {
    super(`Autonomy source ${sourceName} could not prepare due events`, {
      code: "AUTONOMY_SOURCE_PREPARATION_FAILED",
      cause: error,
    });
  }
}

export class AutonomyBookkeepingError extends AutonomyError {
  constructor(event: DueEvent, error: unknown) {
    super(`Autonomy bookkeeping failed for ${event.sourceName}:${event.id}`, {
      code: "AUTONOMY_BOOKKEEPING_FAILED",
      cause: error,
    });
  }
}

function backoffKey(event: DueEvent): string {
  return `${event.sourceType}:${event.sourceName}:${event.id}`;
}

function isGlobalCircuitFailure(error: unknown): boolean {
  return (
    findInErrorCauseChain(
      error,
      (candidate): candidate is LLMError | AuthError =>
        candidate instanceof LLMError || candidate instanceof AuthError,
    ) !== undefined
  );
}

function structuralDeadlineTimestamp(event: DueEvent): number | null {
  const candidates = [event.payload.target_at, event.payload.expires_at].filter(
    (value): value is number => typeof value === "number" && Number.isFinite(value),
  );

  return candidates.length === 0 ? null : Math.min(...candidates);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function goalConcernPayload(event: DueEvent): {
  goalId: string;
  lastProgressTs: number | null;
} | null {
  if (event.sourceType !== "trigger") {
    return null;
  }

  const payload = event.payload;

  if (event.sourceName === "goal_followup_due") {
    const lastProgressTs = payload.last_progress_ts;

    if (
      typeof payload.goal_id !== "string" ||
      (lastProgressTs !== null && typeof lastProgressTs !== "number")
    ) {
      return null;
    }

    return {
      goalId: payload.goal_id,
      lastProgressTs,
    };
  }

  if (
    event.sourceName !== "executive_focus_due" ||
    payload.reason !== "goal_stale" ||
    typeof payload.selected_goal_id !== "string" ||
    !isRecord(payload.selected_goal)
  ) {
    return null;
  }

  const lastProgressTs = payload.selected_goal.last_progress_ts;

  if (lastProgressTs !== null && typeof lastProgressTs !== "number") {
    return null;
  }

  return {
    goalId: payload.selected_goal_id,
    lastProgressTs,
  };
}

function outboundPostEmitted(call: TurnResult["toolCalls"][number]): boolean {
  if (call.name !== OUTBOUND_POST_TOOL_NAME || !call.ok || !isRecord(call.output)) {
    return false;
  }

  const outbound = call.output.outbound;

  if (!isRecord(outbound)) {
    return false;
  }

  const deliveryOutcome = outbound.delivery_outcome;

  if (isRecord(deliveryOutcome)) {
    return deliveryOutcome.state === "delivered";
  }

  return outbound.emitted === true;
}

function deliveredOutboundPost(turnResult: TurnResult): boolean {
  return turnResult.toolCalls.some(outboundPostEmitted);
}

export function turnEmittedHeadway(turnResult: TurnResult): boolean {
  const emissionKind = turnResult.emission?.kind;

  return (
    emissionKind === "message" ||
    emissionKind === "continue_thought" ||
    deliveredOutboundPost(turnResult)
  );
}

function goalProgressAdvanced(input: { before: number | null; after: number | null }): boolean {
  return input.after !== null && (input.before === null || input.after > input.before);
}

export class AutonomyScheduler {
  private readonly clock: Clock;
  private readonly sessionId: SessionId;
  private readonly setIntervalFn: typeof setInterval;
  private readonly clearIntervalFn: typeof clearInterval;
  private readonly fleetBrakeOptions: FleetBrakeOptions;
  private readonly respectGoalFollowupStaleBackoff: boolean;
  private readonly retryBackoff = new Map<string, RetryBackoffState>();
  private readonly sourceRetryBackoff = new Map<AutonomyWakeSource["name"], RetryBackoffState>();
  private intervalHandle: IntervalHandle | null = null;
  private activeTick: Promise<TickResult> | null = null;
  private observer: AutonomySchedulerObserver | null = null;
  private intervalStartedTs: number | null = null;
  private lastTickTs: number | null = null;

  constructor(private readonly options: AutonomySchedulerOptions) {
    this.clock = options.clock ?? new SystemClock();
    this.sessionId = options.sessionId ?? DEFAULT_SESSION_ID;
    this.setIntervalFn = options.setIntervalFn ?? setInterval;
    this.clearIntervalFn = options.clearIntervalFn ?? clearInterval;
    this.fleetBrakeOptions = options.fleetBrake ?? { ...DEFAULT_FLEET_BRAKE_OPTIONS };
    this.respectGoalFollowupStaleBackoff = options.respectGoalFollowupStaleBackoff ?? true;
  }

  setObserver(observer: AutonomySchedulerObserver | null): void {
    this.observer = observer;
  }

  isEnabled(): boolean {
    return this.options.enabled;
  }

  start(): void {
    if (!this.options.enabled || this.intervalHandle !== null) {
      return;
    }

    this.intervalStartedTs = this.clock.now();
    this.lastTickTs = null;
    this.intervalHandle = this.setIntervalFn(() => {
      if (this.activeTick !== null) {
        return;
      }

      void this.runScheduledTick();
    }, this.options.intervalMs);
  }

  async stop(options: AutonomySchedulerStopOptions = {}): Promise<void> {
    if (this.intervalHandle !== null) {
      this.clearIntervalFn(this.intervalHandle);
      this.intervalHandle = null;
    }
    this.intervalStartedTs = null;
    this.lastTickTs = null;

    if (options.graceful === false) {
      return;
    }

    const activeTick = this.activeTick;

    if (activeTick !== null) {
      await activeTick;
    }
  }

  async tick(): Promise<TickResult> {
    return this.runTrackedTick();
  }

  async describe(): Promise<AutonomySchedulerDescription> {
    const nowMs = this.clock.now();
    const budgetCutoff = nowMs - this.options.budgetWindowMs;
    const registeredSources = new Map(this.options.sources.map((source) => [source.name, source]));
    const sources: AutonomySchedulerSourceDescription[] = [];

    for (const name of AUTONOMY_WAKE_SOURCE_NAMES) {
      const metadata = AUTONOMY_WAKE_SOURCE_METADATA[name];
      const source = registeredSources.get(name);

      if (metadata.type === "condition") {
        sources.push({
          name: name as AutonomyConditionName,
          type: "condition",
          category: metadata.category,
          enabled: source !== undefined,
        });
        continue;
      }

      sources.push({
        name: name as AutonomyTriggerName,
        type: "trigger",
        category: metadata.category,
        enabled: source !== undefined,
        next_due_at: source?.nextDueAt === undefined ? null : await source.nextDueAt(),
      });
    }

    const fleetBrakeMetadata = this.readFleetBrakeState();

    return {
      enabled: this.options.enabled,
      interval_ms: this.options.intervalMs,
      next_tick_at: this.describeNextTickAt(nowMs),
      budget: {
        max_wakes_per_window: this.options.maxWakesPerWindow,
        window_ms: this.options.budgetWindowMs,
        used_in_current_window: this.options.wakeRepository.countSince(budgetCutoff),
        reserved_contemplative_wakes_per_window: Math.min(
          this.options.maxWakesPerWindow,
          Math.max(0, Math.floor(this.options.reservedContemplativeWakesPerWindow ?? 0)),
        ),
        contemplative_used_in_current_window: this.options.wakeRepository.countSince(budgetCutoff, {
          sourceCategory: "contemplative",
        }),
      },
      fleet_brake: {
        enabled: this.fleetBrakeOptions.enabled,
        empty_streak: fleetBrakeMetadata.empty_streak,
        streak_anchor_ts:
          fleetBrakeMetadata.streak_anchor_ts === 0 ? null : fleetBrakeMetadata.streak_anchor_ts,
        cooldown_until: fleetBrakeCooldownUntilMs(fleetBrakeMetadata, this.fleetBrakeOptions),
        error_streak: fleetBrakeMetadata.error_streak,
        error_paused_until: fleetBrakeErrorPausedUntilMs(
          fleetBrakeMetadata,
          this.fleetBrakeOptions,
        ),
        bypass_count: fleetBrakeMetadata.bypass_count,
        window_outcomes: {
          headway: this.options.wakeRepository.countSince(budgetCutoff, { outcome: "headway" }),
          silent: this.options.wakeRepository.countSince(budgetCutoff, { outcome: "silent" }),
          error: this.options.wakeRepository.countSince(budgetCutoff, { outcome: "error" }),
          busy: this.options.wakeRepository.countSince(budgetCutoff, { outcome: "busy" }),
        },
      },
      sources,
    };
  }

  private describeNextTickAt(nowMs: number): number | null {
    if (this.intervalHandle === null) {
      return null;
    }

    const tickAnchor =
      this.lastTickTs === null || this.intervalStartedTs === null
        ? (this.lastTickTs ?? this.intervalStartedTs ?? nowMs)
        : Math.max(this.lastTickTs, this.intervalStartedTs);

    return Math.max(tickAnchor + this.options.intervalMs, nowMs);
  }

  private async tickOnce(): Promise<TickResult> {
    const nowMs = this.clock.now();
    this.lastTickTs = nowMs;
    const scannedSources = this.options.sources.map((source) => source.name);

    if (!this.options.enabled) {
      return {
        status: "disabled",
        ts: nowMs,
        scannedSources,
        dueEvents: 0,
        firedEvents: 0,
        budgetSkipped: 0,
        fleetCooldownSkipped: 0,
        errorCircuitSkipped: 0,
        busySkipped: 0,
        errorCount: 0,
        sourceErrorCount: 0,
        bookkeepingErrorCount: 0,
        events: [],
      };
    }

    try {
      const scanResult = await this.scanDueEvents();
      const scannedDueEvents = this.orderDueEventsForRecoveryProbe(scanResult.events, nowMs);
      const dueEventKeys = new Set(scannedDueEvents.map(({ event }) => backoffKey(event)));

      for (const key of this.retryBackoff.keys()) {
        if (!dueEventKeys.has(key)) {
          this.retryBackoff.delete(key);
        }
      }

      const dueEvents = scannedDueEvents.filter(({ event }) => {
        const backoff = this.retryBackoff.get(backoffKey(event));
        return backoff === undefined || backoff.nextEligibleTs <= nowMs;
      });
      const writer = this.options.createStreamWriter(this.sessionId);
      const eventResults: AutonomyTickEventResult[] = [];
      let firedEvents = 0;
      let budgetSkipped = 0;
      let fleetCooldownSkipped = 0;
      let errorCircuitSkipped = 0;
      let busySkipped = 0;
      let errorCount = scanResult.sourceErrorCount;
      let sourceErrorCount = scanResult.sourceErrorCount;
      let bookkeepingErrorCount = 0;

      try {
        for (const scannedEvent of dueEvents) {
          const dueEvent = scannedEvent.event;
          const sourceCategory = scannedEvent.source.sourceCategory;

          if (this.sourceRetryIsActive(dueEvent.sourceName, this.clock.now())) {
            continue;
          }

          const fleetAdmission = this.fleetAdmissionDecision(
            dueEvent,
            sourceCategory,
            this.clock.now(),
          );

          if (fleetAdmission.skipStatus !== null) {
            if (fleetAdmission.skipStatus === "error_circuit_skipped") {
              errorCircuitSkipped += 1;
            } else {
              fleetCooldownSkipped += 1;
            }

            eventResults.push({
              id: dueEvent.id,
              sourceName: dueEvent.sourceName,
              sourceType: dueEvent.sourceType,
              sourceCategory,
              status: fleetAdmission.skipStatus,
              payload: dueEvent.payload,
              outcomeSummary:
                fleetAdmission.skipStatus === "error_circuit_skipped"
                  ? "Skipped while the durable autonomy error circuit was paused."
                  : "Skipped while the durable operational fleet cooldown was active.",
            });
            continue;
          }

          const budgetCutoff = this.clock.now() - this.options.budgetWindowMs;
          const totalWakesInWindow = this.options.wakeRepository.countSince(budgetCutoff);
          const reservedContemplativeWakes = Math.min(
            this.options.maxWakesPerWindow,
            Math.max(0, Math.floor(this.options.reservedContemplativeWakesPerWindow ?? 0)),
          );
          const contemplativeWakesInWindow = this.options.wakeRepository.countSince(budgetCutoff, {
            sourceCategory: "contemplative",
          });
          const reservedContemplativeSlotsRemaining = Math.max(
            0,
            reservedContemplativeWakes - contemplativeWakesInWindow,
          );
          const operationalWakeLimit =
            this.options.maxWakesPerWindow - reservedContemplativeSlotsRemaining;

          if (
            totalWakesInWindow >= this.options.maxWakesPerWindow ||
            (sourceCategory !== "contemplative" && totalWakesInWindow >= operationalWakeLimit)
          ) {
            budgetSkipped += 1;
            eventResults.push({
              id: dueEvent.id,
              sourceName: dueEvent.sourceName,
              sourceType: dueEvent.sourceType,
              sourceCategory,
              status: "budget_skipped",
              payload: dueEvent.payload,
              outcomeSummary: "Skipped because autonomy wake budget was exhausted.",
            });
            continue;
          }

          const autonomousWakeEntry = await writer.append({
            kind: "internal_event",
            content: {
              kind: "autonomous_wake",
              trigger_type: dueEvent.sourceType,
              source_name: dueEvent.sourceName,
              source_category: sourceCategory,
              payload: dueEvent.payload,
              ts: this.clock.now(),
            },
          });
          const wakeRecord = this.options.wakeRepository.record({
            trigger_name: dueEvent.sourceName,
            condition_name:
              dueEvent.sourceType === "condition"
                ? (dueEvent.sourceName as AutonomyConditionName)
                : null,
            session_id: this.sessionId,
            wake_source_type: dueEvent.sourceType,
            source_category: sourceCategory,
          });

          const preparedEvent = await this.prepareEvent(dueEvent);

          if ("toolError" in preparedEvent) {
            errorCount += 1;
            sourceErrorCount += 1;
            const outcomeSummary = `Autonomous preparation failed: ${preparedEvent.toolError}`;
            this.options.wakeRepository.recordOutcome(wakeRecord.id, "error");
            this.consumeFleetFreshnessBypass(
              dueEvent,
              fleetAdmission.bypassKind,
              fleetAdmission.metadata,
            );
            this.scheduleSourceRetryBackoff(dueEvent.sourceName);
            await writer.append({
              kind: "internal_event",
              content: {
                kind: "autonomous_action",
                trigger: dueEvent.sourceName,
                outcome_summary: outcomeSummary,
                turn_result_id: null,
                ts: this.clock.now(),
              },
            });
            eventResults.push({
              id: dueEvent.id,
              sourceName: dueEvent.sourceName,
              sourceType: dueEvent.sourceType,
              sourceCategory,
              status: "error",
              payload: dueEvent.payload,
              error: preparedEvent.toolError,
              outcomeSummary,
              turnResultId: null,
            });
            await this.notifyError(
              new AutonomySourcePreparationError(dueEvent.sourceName, preparedEvent.toolError),
            );
            continue;
          }

          let turnInput;

          try {
            turnInput = preparedEvent.source.buildTurn(preparedEvent.event);
          } catch (error) {
            errorCount += 1;
            sourceErrorCount += 1;
            const preparationError = new AutonomySourcePreparationError(dueEvent.sourceName, error);
            const outcomeSummary = `Autonomous source preparation failed: ${formatError(error)}`;
            this.options.wakeRepository.recordOutcome(wakeRecord.id, "error");
            this.consumeFleetFreshnessBypass(
              dueEvent,
              fleetAdmission.bypassKind,
              fleetAdmission.metadata,
            );
            this.scheduleSourceRetryBackoff(dueEvent.sourceName);
            await writer.append({
              kind: "internal_event",
              content: {
                kind: "autonomous_action",
                trigger: dueEvent.sourceName,
                outcome_summary: outcomeSummary,
                turn_result_id: null,
                ts: this.clock.now(),
              },
            });
            eventResults.push({
              id: dueEvent.id,
              sourceName: dueEvent.sourceName,
              sourceType: dueEvent.sourceType,
              sourceCategory,
              status: "error",
              payload: preparedEvent.event.payload,
              outcomeSummary,
              turnResultId: null,
              error: formatError(preparationError),
            });
            await this.notifyError(preparationError);
            continue;
          }

          this.sourceRetryBackoff.delete(dueEvent.sourceName);

          let turnResult: TurnResult;

          try {
            turnResult = await this.options.turnOrchestrator.run({
              ...turnInput,
              sessionId: this.sessionId,
              audience: "self",
              stakes: "low",
              origin: "autonomous",
            });
          } catch (error) {
            const busy = error instanceof SessionBusyError;
            const outcomeSummary = busy
              ? "Skipped autonomous turn because the session was busy."
              : `Autonomous turn failed: ${formatError(error)}`;

            await writer.append({
              kind: "internal_event",
              content: {
                kind: "autonomous_action",
                trigger: dueEvent.sourceName,
                outcome_summary: outcomeSummary,
                turn_result_id: null,
                ts: this.clock.now(),
              },
            });

            if (busy) {
              busySkipped += 1;
              this.options.wakeRepository.recordOutcome(wakeRecord.id, "busy");
            } else {
              errorCount += 1;
              this.options.wakeRepository.recordOutcome(wakeRecord.id, "error");
              if (isGlobalCircuitFailure(error)) {
                this.updateFleetBrakeAfterGlobalError(
                  dueEvent,
                  fleetAdmission.bypassKind,
                  fleetAdmission.metadata,
                );
              } else {
                this.consumeFleetFreshnessBypass(
                  dueEvent,
                  fleetAdmission.bypassKind,
                  fleetAdmission.metadata,
                );
              }
            }
            this.scheduleRetryBackoff(dueEvent);

            eventResults.push({
              id: dueEvent.id,
              sourceName: dueEvent.sourceName,
              sourceType: dueEvent.sourceType,
              sourceCategory,
              status: busy ? "busy_skipped" : "error",
              payload: preparedEvent.event.payload,
              outcomeSummary,
              turnResultId: null,
              ...(busy ? {} : { error: formatError(error) }),
            });
            continue;
          }

          const outcomeSummary = summarizeOutcome(turnResult.response);
          const decisionSummary = summarizeAutonomousDecision(turnResult);
          let wakeOutcome: GoalWakeOutcome;

          try {
            wakeOutcome = this.evaluateGoalWakeOutcome(preparedEvent.event, turnResult);
            // A completed turn's structural outcome is authoritative. Persist
            // fleet headway before any stream/latch/introspection bookkeeping
            // can fail and accidentally preserve an empty streak.
            this.updateFleetBrakeAfterSuccess({
              event: dueEvent,
              sourceCategory,
              turnResult,
              headway: wakeOutcome.headway,
              bypassKind: fleetAdmission.bypassKind,
              admissionMetadata: fleetAdmission.metadata,
            });
            this.options.wakeRepository.recordOutcome(
              wakeRecord.id,
              wakeOutcome.headway ? "headway" : "silent",
            );
          } catch (error) {
            firedEvents += 1;
            bookkeepingErrorCount += 1;
            this.scheduleRetryBackoff(dueEvent);
            const bookkeepingError = new AutonomyBookkeepingError(dueEvent, error);
            eventResults.push({
              id: dueEvent.id,
              sourceName: dueEvent.sourceName,
              sourceType: dueEvent.sourceType,
              sourceCategory,
              status: "bookkeeping_error",
              payload: preparedEvent.event.payload,
              outcomeSummary: `Autonomous turn completed; bookkeeping failed: ${formatError(error)}`,
              turnResultId: turnResult.agentMessageId ?? null,
              error: formatError(bookkeepingError),
            });
            await this.notifyError(bookkeepingError);
            continue;
          }

          firedEvents += 1;

          try {
            const autonomousActionEntry = await writer.append({
              kind: "internal_event",
              content: {
                kind: "autonomous_action",
                trigger: dueEvent.sourceName,
                outcome_summary: outcomeSummary,
                turn_result_id: turnResult.agentMessageId ?? null,
                ts: this.clock.now(),
              },
            });
            this.options.watermarkRepository.set(dueEvent.watermarkProcessName, this.sessionId, {
              lastTs: dueEvent.sortTs,
              lastEntryId: dueEvent.id,
            });
            this.applyPerGoalEmptyWakeBackoff(preparedEvent.event, wakeOutcome);
            this.options.selfDecisionRepository?.record({
              occurredAt: autonomousActionEntry.timestamp,
              sessionId: this.sessionId,
              triggerName: dueEvent.sourceName,
              triggerType: dueEvent.sourceType,
              sourceEventId: dueEvent.id,
              fireEventId: autonomousActionEntry.id,
              decisionSummary,
              decisionRationale: autonomousDecisionRationale(turnResult),
              turnResultId: turnResult.agentMessageId ?? null,
              sourceStreamEntryIds: [autonomousWakeEntry.id, autonomousActionEntry.id],
            });
            try {
              await preparedEvent.source.onFired?.(preparedEvent.event);
            } catch {
              // Best-effort: the watermark already enforces one-time semantics
              // and scan() reconciles row state as a backstop, so a failed
              // onFired must not demote a successful fire to an error.
            }
            this.retryBackoff.delete(backoffKey(dueEvent));
            eventResults.push({
              id: dueEvent.id,
              sourceName: dueEvent.sourceName,
              sourceType: dueEvent.sourceType,
              sourceCategory,
              status: "fired",
              payload: preparedEvent.event.payload,
              outcomeSummary,
              turnResultId: turnResult.agentMessageId ?? null,
            });
          } catch (error) {
            bookkeepingErrorCount += 1;
            this.scheduleRetryBackoff(dueEvent);
            const bookkeepingError = new AutonomyBookkeepingError(dueEvent, error);
            eventResults.push({
              id: dueEvent.id,
              sourceName: dueEvent.sourceName,
              sourceType: dueEvent.sourceType,
              sourceCategory,
              status: "bookkeeping_error",
              payload: preparedEvent.event.payload,
              outcomeSummary: `Autonomous turn completed; bookkeeping failed: ${formatError(error)}`,
              turnResultId: turnResult.agentMessageId ?? null,
              error: formatError(bookkeepingError),
            });
            await this.notifyError(bookkeepingError);
          }
        }

        return {
          status: "ok",
          ts: nowMs,
          scannedSources,
          dueEvents: dueEvents.length,
          firedEvents,
          budgetSkipped,
          fleetCooldownSkipped,
          errorCircuitSkipped,
          busySkipped,
          errorCount,
          sourceErrorCount,
          bookkeepingErrorCount,
          events: eventResults,
        };
      } finally {
        writer.close();
      }
    } finally {
      this.pruneWakeRecords();
    }
  }

  private evaluateGoalWakeOutcome(event: DueEvent, turnResult: TurnResult): GoalWakeOutcome {
    const parsedConcern = goalConcernPayload(event);
    const concern =
      event.sourceName === "goal_followup_due" && !this.respectGoalFollowupStaleBackoff
        ? null
        : parsedConcern;
    const emittedHeadway = turnEmittedHeadway(turnResult);

    if (concern === null || emittedHeadway) {
      return {
        headway: emittedHeadway,
        concern,
      };
    }

    const currentGoal =
      this.options.goalsRepository?.get(concern.goalId as Parameters<GoalsRepository["get"]>[0]) ??
      null;
    const currentLastProgressTs = currentGoal?.last_progress_ts ?? concern.lastProgressTs;
    const progressedDuringTurn = goalProgressAdvanced({
      before: concern.lastProgressTs,
      after: currentLastProgressTs,
    });

    return {
      headway: progressedDuringTurn,
      concern,
    };
  }

  private applyPerGoalEmptyWakeBackoff(event: DueEvent, outcome: GoalWakeOutcome): void {
    const concern = outcome.concern;

    if (
      concern === null ||
      (event.sourceName === "goal_followup_due" && !this.respectGoalFollowupStaleBackoff)
    ) {
      return;
    }

    const processName = getExecutiveFocusGoalStaleBackoffProcessName(concern.goalId);

    if (outcome.headway) {
      this.options.watermarkRepository.reset(processName, this.sessionId);
      return;
    }

    const previousBackoff = this.options.watermarkRepository.get(processName, this.sessionId);
    const progressSincePreviousBackoff =
      previousBackoff !== null &&
      concern.lastProgressTs !== null &&
      concern.lastProgressTs >= previousBackoff.updatedAt;
    const previousMetadata = readExecutiveFocusGoalStaleBackoffMetadata(previousBackoff);
    const previousEmptyCount = progressSincePreviousBackoff ? 0 : previousMetadata.empty_count;

    this.options.watermarkRepository.set(processName, this.sessionId, {
      lastTs: event.sortTs,
      lastEntryId: event.id,
      metadata: {
        empty_count: previousEmptyCount + 1,
      },
    });
  }

  private fleetAdmissionDecision(
    event: DueEvent,
    sourceCategory: AutonomyWakeSourceCategory,
    nowMs: number,
  ): FleetAdmissionDecision {
    if (!this.fleetBrakeOptions.enabled) {
      return {
        skipStatus: null,
        bypassKind: null,
        metadata: null,
      };
    }

    const metadata = this.readFleetBrakeState({ notifyError: true });
    const errorPausedUntil = fleetBrakeErrorPausedUntilMs(metadata, this.fleetBrakeOptions);

    if (errorPausedUntil !== null && nowMs < errorPausedUntil) {
      return {
        skipStatus: "error_circuit_skipped",
        bypassKind: null,
        metadata,
      };
    }

    if (sourceCategory !== "operational") {
      return {
        skipStatus: null,
        bypassKind: null,
        metadata,
      };
    }

    const cooldownUntil = fleetBrakeCooldownUntilMs(metadata, this.fleetBrakeOptions);

    if (cooldownUntil === null || nowMs >= cooldownUntil) {
      return {
        skipStatus: null,
        bypassKind: null,
        metadata,
      };
    }

    const deadlineTs = structuralDeadlineTimestamp(event);

    if (deadlineTs !== null && deadlineTs <= cooldownUntil) {
      return {
        skipStatus: null,
        bypassKind: "deadline",
        metadata,
      };
    }

    const stateTs = event.stateTs;
    const freshConcern =
      typeof stateTs === "number" &&
      Number.isFinite(stateTs) &&
      stateTs > metadata.streak_anchor_ts;

    if (freshConcern && metadata.bypass_count < this.fleetBrakeOptions.freshnessBypassCap) {
      return {
        skipStatus: null,
        bypassKind: "freshness",
        metadata,
      };
    }

    return {
      skipStatus: "fleet_cooldown_skipped",
      bypassKind: null,
      metadata,
    };
  }

  private updateFleetBrakeAfterSuccess(input: {
    event: DueEvent;
    sourceCategory: AutonomyWakeSourceCategory;
    turnResult: TurnResult;
    headway: boolean;
    bypassKind: FleetAdmissionDecision["bypassKind"];
    admissionMetadata: FleetBrakeMetadata | null;
  }): void {
    if (!this.fleetBrakeOptions.enabled) {
      return;
    }

    const nowMs = this.clock.now();
    const current = input.admissionMetadata ?? emptyFleetBrakeMetadata();
    const next: FleetBrakeMetadata = {
      ...current,
      error_streak: 0,
      last_error_ts: 0,
    };

    if (input.sourceCategory === "operational") {
      if (input.headway) {
        next.empty_streak = 0;
        next.streak_anchor_ts = 0;
        next.last_wake_ts = nowMs;
        next.bypass_count = 0;
      } else {
        next.empty_streak = current.empty_streak + 1;
        next.streak_anchor_ts = current.empty_streak === 0 ? nowMs : current.streak_anchor_ts;
        next.last_wake_ts = nowMs;
        next.bypass_count = current.bypass_count + (input.bypassKind === "freshness" ? 1 : 0);
      }
    } else if (deliveredOutboundPost(input.turnResult)) {
      next.empty_streak = 0;
      next.streak_anchor_ts = 0;
      next.bypass_count = 0;
    }

    this.writeFleetBrakeState(input.event, next, nowMs);
  }

  private updateFleetBrakeAfterGlobalError(
    event: DueEvent,
    bypassKind: FleetAdmissionDecision["bypassKind"],
    admissionMetadata: FleetBrakeMetadata | null | undefined,
  ): void {
    if (!this.fleetBrakeOptions.enabled) {
      return;
    }

    const nowMs = this.clock.now();
    const current = admissionMetadata ?? this.readFleetBrakeState({ notifyError: true });

    this.writeFleetBrakeState(
      event,
      {
        ...current,
        error_streak: current.error_streak + 1,
        last_error_ts: nowMs,
        bypass_count: current.bypass_count + (bypassKind === "freshness" ? 1 : 0),
      },
      nowMs,
    );
  }

  private consumeFleetFreshnessBypass(
    event: DueEvent,
    bypassKind: FleetAdmissionDecision["bypassKind"],
    admissionMetadata: FleetBrakeMetadata | null | undefined,
  ): void {
    if (!this.fleetBrakeOptions.enabled || bypassKind !== "freshness") {
      return;
    }

    const nowMs = this.clock.now();
    const current = admissionMetadata ?? this.readFleetBrakeState({ notifyError: true });

    this.writeFleetBrakeState(
      event,
      {
        ...current,
        bypass_count: current.bypass_count + 1,
      },
      nowMs,
    );
  }

  private readFleetBrakeState(options: { notifyError?: boolean } = {}): FleetBrakeMetadata {
    const notify =
      options.notifyError === true ? (error: unknown) => void this.notifyError(error) : undefined;

    try {
      return readFleetBrakeMetadata(
        this.options.watermarkRepository.get(FLEET_BRAKE_PROCESS_NAME, this.sessionId),
        notify,
      );
    } catch (error) {
      notify?.(error);
      return emptyFleetBrakeMetadata();
    }
  }

  private writeFleetBrakeState(event: DueEvent, metadata: FleetBrakeMetadata, nowMs: number): void {
    this.options.watermarkRepository.set(FLEET_BRAKE_PROCESS_NAME, this.sessionId, {
      lastTs: nowMs,
      lastEntryId: event.id,
      metadata,
    });
  }

  private runTrackedTick(
    options: {
      notifyObserver?: boolean;
    } = {},
  ): Promise<TickResult> {
    const existing = this.activeTick;

    if (existing !== null) {
      return existing;
    }

    const notifyObserver = options.notifyObserver ?? false;
    const promise = (async () => {
      try {
        const result = await this.tickOnce();

        if (notifyObserver) {
          await this.notifyTick(result);
        }

        return result;
      } catch (error) {
        if (notifyObserver) {
          await this.notifyError(error);
        }

        throw error;
      }
    })().finally(() => {
      if (this.activeTick === promise) {
        this.activeTick = null;
      }
    });

    this.activeTick = promise;
    return promise;
  }

  private orderDueEventsForRecoveryProbe(
    events: ScannedDueEvents["events"],
    nowMs: number,
  ): ScannedDueEvents["events"] {
    if (!this.fleetBrakeOptions.enabled) {
      return events;
    }

    const metadata = this.readFleetBrakeState();
    const pausedUntil = fleetBrakeErrorPausedUntilMs(metadata, this.fleetBrakeOptions);

    if (pausedUntil === null || nowMs < pausedUntil) {
      return events;
    }

    return [...events].sort((left, right) => {
      const recoveryRank = (candidate: (typeof events)[number]): number => {
        if (candidate.source.sourceCategory === "contemplative") {
          return 0;
        }

        return this.retryBackoff.has(backoffKey(candidate.event)) ? 2 : 1;
      };
      const leftRank = recoveryRank(left);
      const rightRank = recoveryRank(right);

      return (
        leftRank - rightRank ||
        left.event.sortTs - right.event.sortTs ||
        left.event.id.localeCompare(right.event.id)
      );
    });
  }

  private async scanDueEvents(): Promise<ScannedDueEvents> {
    const dueEvents: Array<{ source: AutonomyWakeSource; event: DueEvent }> = [];
    let sourceErrorCount = 0;

    for (const source of this.options.sources) {
      if (this.sourceRetryIsActive(source.name, this.clock.now())) {
        continue;
      }

      let events: DueEvent[];

      try {
        events = await source.scan();
      } catch (error) {
        sourceErrorCount += 1;
        this.scheduleSourceRetryBackoff(source.name);
        await this.notifyError(new AutonomySourcePreparationError(source.name, error));
        continue;
      }

      if (events.length === 0) {
        this.sourceRetryBackoff.delete(source.name);
      }

      for (const event of events) {
        dueEvents.push({
          source,
          event,
        });
      }
    }

    return {
      events: dueEvents.sort(
        (left, right) =>
          left.event.sortTs - right.event.sortTs || left.event.id.localeCompare(right.event.id),
      ),
      sourceErrorCount,
    };
  }

  private pruneWakeRecords(): void {
    this.options.wakeRepository.prune(
      this.clock.now() - this.options.budgetWindowMs - WAKE_PRUNE_SAFETY_BUFFER_MS,
    );
  }

  private async prepareEvent(dueEvent: DueEvent): Promise<
    | {
        source: AutonomyWakeSource;
        event: DueEvent;
      }
    | {
        toolError: string;
      }
  > {
    const source = this.options.sources.find((entry) => entry.name === dueEvent.sourceName);

    if (source === undefined) {
      return {
        toolError: `Unknown autonomy source: ${dueEvent.sourceName}`,
      };
    }

    const provenance = {
      source_name: dueEvent.sourceName,
      event_id: dueEvent.id,
    };

    switch (dueEvent.sourceName) {
      case "commitment_expiring": {
        const result = await this.options.toolDispatcher.dispatch({
          toolName: "tool.commitments.list",
          input: {},
          origin: "autonomous",
          sessionId: this.sessionId,
          provenance,
          timeoutMs: AUTONOMY_PREP_TOOL_TIMEOUT_MS,
        });

        if (!result.ok) {
          return {
            toolError: result.error,
          };
        }

        const output = result.output as {
          commitments: unknown[];
        };

        return {
          source,
          event: {
            ...dueEvent,
            payload: {
              ...dueEvent.payload,
              active_commitments: output.commitments,
            },
          },
        };
      }

      case "open_question_dormant": {
        const payload = dueEvent.payload as {
          question: string;
        };
        const result = await this.options.toolDispatcher.dispatch({
          toolName: "tool.episodic.search",
          input: {
            query: payload.question,
            limit: 5,
          },
          origin: "autonomous",
          sessionId: this.sessionId,
          provenance,
          timeoutMs: AUTONOMY_PREP_TOOL_TIMEOUT_MS,
        });

        if (!result.ok) {
          return {
            toolError: result.error,
          };
        }

        const output = result.output as {
          episodes: unknown[];
        };

        return {
          source,
          event: {
            ...dueEvent,
            payload: {
              ...dueEvent.payload,
              related_episodes: output.episodes,
            },
          },
        };
      }

      case "scheduled_reflection": {
        const result = await this.options.toolDispatcher.dispatch({
          toolName: "tool.identityEvents.listForCognition",
          input: {
            limit: 10,
          },
          origin: "autonomous",
          sessionId: this.sessionId,
          provenance,
          timeoutMs: AUTONOMY_PREP_TOOL_TIMEOUT_MS,
        });

        if (!result.ok) {
          return {
            toolError: result.error,
          };
        }

        const output = result.output as {
          events: unknown[];
        };
        const priorSelfThought = this.options.trainOfThoughtRepository?.get() ?? null;

        return {
          source,
          event: {
            ...dueEvent,
            payload: {
              ...dueEvent.payload,
              recent_identity_events: output.events,
              ...(priorSelfThought === null
                ? {}
                : {
                    prior_self_thought: {
                      text: priorSelfThought.text,
                      updated_at: priorSelfThought.updated_at,
                      self_entity_id: priorSelfThought.self_entity_id,
                      ...memoryDisclosurePayloadFields(selfPrivateMemoryDisclosureLabel()),
                    },
                  }),
            },
          },
        };
      }
      default:
        return {
          source,
          event: dueEvent,
        };
    }
  }

  private async runScheduledTick(): Promise<void> {
    try {
      await this.runTrackedTick({
        notifyObserver: true,
      });
    } catch {
      // Scheduled ticks report failures through notifyError; the interval loop
      // should not surface an unhandled rejection.
    }
  }

  private async notifyTick(result: TickResult): Promise<void> {
    try {
      await this.observer?.onTick?.(result);
    } catch (error) {
      await this.notifyError(error);
    }
  }

  private async notifyError(error: unknown): Promise<void> {
    try {
      await this.observer?.onError?.(error);
    } catch {
      // Observer failures must not stop the scheduler loop.
    }
  }

  private scheduleRetryBackoff(dueEvent: DueEvent): void {
    const key = backoffKey(dueEvent);
    const previousBackoff = this.retryBackoff.get(key);
    const delayMs =
      previousBackoff === undefined
        ? INITIAL_RETRY_BACKOFF_MS
        : Math.min(previousBackoff.delayMs * 2, MAX_RETRY_BACKOFF_MS);

    this.retryBackoff.set(key, {
      delayMs,
      nextEligibleTs: this.clock.now() + delayMs,
    });
  }

  private sourceRetryIsActive(sourceName: AutonomyWakeSource["name"], nowMs: number): boolean {
    const backoff = this.sourceRetryBackoff.get(sourceName);

    return backoff !== undefined && backoff.nextEligibleTs > nowMs;
  }

  private scheduleSourceRetryBackoff(sourceName: AutonomyWakeSource["name"]): void {
    const previousBackoff = this.sourceRetryBackoff.get(sourceName);
    const delayMs =
      previousBackoff === undefined
        ? INITIAL_RETRY_BACKOFF_MS
        : Math.min(previousBackoff.delayMs * 2, MAX_RETRY_BACKOFF_MS);

    this.sourceRetryBackoff.set(sourceName, {
      delayMs,
      nextEligibleTs: this.clock.now() + delayMs,
    });
  }
}
