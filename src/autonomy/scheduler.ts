import { StreamWatermarkRepository, StreamWriter } from "../stream/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { SessionBusyError } from "../util/errors.js";
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
  TickResult,
  DueEvent,
} from "./types.js";
import { AUTONOMY_WAKE_SOURCE_METADATA, AUTONOMY_WAKE_SOURCE_NAMES } from "./types.js";
import type { AutonomyWakesRepository } from "./wakes-repository.js";
import {
  getExecutiveFocusGoalStaleBackoffProcessName,
  readExecutiveFocusGoalStaleBackoffMetadata,
} from "./executive-focus-stale-backoff.js";

type IntervalHandle = ReturnType<typeof setInterval>;
type RetryBackoffState = {
  delayMs: number;
  nextEligibleTs: number;
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

function backoffKey(event: DueEvent): string {
  return `${event.sourceType}:${event.sourceName}:${event.id}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function executiveFocusGoalStalePayload(event: DueEvent): {
  goalId: string;
  lastProgressTs: number | null;
} | null {
  if (event.sourceName !== "executive_focus_due" || event.sourceType !== "trigger") {
    return null;
  }

  const payload = event.payload;

  if (
    !isRecord(payload) ||
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

  return isRecord(outbound) && outbound.emitted === true;
}

function goalProgressAdvanced(input: { before: number | null; after: number | null }): boolean {
  return input.after !== null && (input.before === null || input.after > input.before);
}

export class AutonomyScheduler {
  private readonly clock: Clock;
  private readonly sessionId: SessionId;
  private readonly setIntervalFn: typeof setInterval;
  private readonly clearIntervalFn: typeof clearInterval;
  private readonly retryBackoff = new Map<string, RetryBackoffState>();
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
        busySkipped: 0,
        errorCount: 0,
        events: [],
      };
    }

    try {
      const scannedDueEvents = await this.scanDueEvents();
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
      let busySkipped = 0;
      let errorCount = 0;

      try {
        for (const scannedEvent of dueEvents) {
          const dueEvent = scannedEvent.event;
          const sourceCategory = scannedEvent.source.sourceCategory;
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
          this.options.wakeRepository.record({
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
            const outcomeSummary = `Autonomous preparation failed: ${preparedEvent.toolError}`;
            this.scheduleRetryBackoff(dueEvent);
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
            continue;
          }

          try {
            const turnInput = preparedEvent.source.buildTurn(preparedEvent.event);
            const turnResult = await this.options.turnOrchestrator.run({
              ...turnInput,
              sessionId: this.sessionId,
              audience: "self",
              stakes: "low",
              origin: "autonomous",
            });
            const outcomeSummary = summarizeOutcome(turnResult.response);
            const decisionSummary = summarizeAutonomousDecision(turnResult);

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

            try {
              this.options.watermarkRepository.set(dueEvent.watermarkProcessName, this.sessionId, {
                lastTs: dueEvent.sortTs,
                lastEntryId: dueEvent.id,
              });
              this.updateExecutiveFocusGoalStaleBackoff(preparedEvent.event, turnResult);
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
              firedEvents += 1;
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
              errorCount += 1;
              this.scheduleRetryBackoff(dueEvent);
              eventResults.push({
                id: dueEvent.id,
                sourceName: dueEvent.sourceName,
                sourceType: dueEvent.sourceType,
                sourceCategory,
                status: "error",
                payload: preparedEvent.event.payload,
                outcomeSummary: `Autonomous turn succeeded but watermark commit failed: ${formatError(error)}`,
                turnResultId: turnResult.agentMessageId ?? null,
                error: formatError(error),
              });
              await this.notifyError(error);
            }
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
            } else {
              errorCount += 1;
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
          }
        }

        return {
          status: "ok",
          ts: nowMs,
          scannedSources,
          dueEvents: dueEvents.length,
          firedEvents,
          budgetSkipped,
          busySkipped,
          errorCount,
          events: eventResults,
        };
      } finally {
        writer.close();
      }
    } finally {
      this.pruneWakeRecords();
    }
  }

  private updateExecutiveFocusGoalStaleBackoff(event: DueEvent, turnResult: TurnResult): void {
    const payload = executiveFocusGoalStalePayload(event);

    if (payload === null) {
      return;
    }

    const processName = getExecutiveFocusGoalStaleBackoffProcessName(payload.goalId);
    const previousBackoff = this.options.watermarkRepository.get(processName, this.sessionId);
    const currentGoal =
      this.options.goalsRepository?.get(payload.goalId as Parameters<GoalsRepository["get"]>[0]) ??
      null;
    const currentLastProgressTs = currentGoal?.last_progress_ts ?? payload.lastProgressTs;
    const progressedDuringTurn = goalProgressAdvanced({
      before: payload.lastProgressTs,
      after: currentLastProgressTs,
    });

    // Structural headway mapping only: progress timestamp advance, outward message,
    // private thought carryover, or a successful outbound post reset the stale loop.
    // Passive observed markers do not carry goal work forward, so they count empty.
    const emissionKind = turnResult.emission?.kind;
    const emittedHeadway =
      emissionKind === "message" ||
      emissionKind === "continue_thought" ||
      turnResult.toolCalls.some(outboundPostEmitted);

    if (progressedDuringTurn || emittedHeadway) {
      this.options.watermarkRepository.reset(processName, this.sessionId);
      return;
    }

    const progressSincePreviousBackoff =
      previousBackoff !== null &&
      payload.lastProgressTs !== null &&
      payload.lastProgressTs >= previousBackoff.updatedAt;
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

  private async scanDueEvents(): Promise<Array<{ source: AutonomyWakeSource; event: DueEvent }>> {
    const dueEvents: Array<{ source: AutonomyWakeSource; event: DueEvent }> = [];

    for (const source of this.options.sources) {
      const events = await source.scan();

      for (const event of events) {
        dueEvents.push({
          source,
          event,
        });
      }
    }

    return dueEvents.sort(
      (left, right) =>
        left.event.sortTs - right.event.sortTs || left.event.id.localeCompare(right.event.id),
    );
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
}
