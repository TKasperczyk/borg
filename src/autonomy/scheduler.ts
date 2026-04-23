import { StreamWatermarkRepository, StreamWriter } from "../stream/index.js";
import { SystemClock, type Clock } from "../util/clock.js";
import { SessionBusyError } from "../util/errors.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../util/ids.js";
import type { ToolDispatcher } from "../tools/dispatcher.js";
import type { TurnInput, TurnOrchestrator } from "../cognition/index.js";

import type {
  AutonomyTickEventResult,
  AutonomyWakeSource,
  TickResult,
  DueEvent,
} from "./types.js";

type IntervalHandle = ReturnType<typeof setInterval>;

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
  maxWakesPerHour: number;
  sessionId?: SessionId;
  clock?: Clock;
  createStreamWriter: (sessionId: SessionId) => StreamWriter;
  watermarkRepository: StreamWatermarkRepository;
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

function formatError(error: unknown): string {
  if (error instanceof Error) {
    return `${error.name}: ${error.message}`;
  }

  return String(error);
}

export class AutonomyScheduler {
  private readonly clock: Clock;
  private readonly sessionId: SessionId;
  private readonly setIntervalFn: typeof setInterval;
  private readonly clearIntervalFn: typeof clearInterval;
  private readonly wakeHistory: number[] = [];
  private intervalHandle: IntervalHandle | null = null;
  private activeTick: Promise<void> | null = null;
  private observer: AutonomySchedulerObserver | null = null;

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

    this.intervalHandle = this.setIntervalFn(() => {
      if (this.activeTick !== null) {
        return;
      }

      this.activeTick = this.runScheduledTick();
    }, this.options.intervalMs);
  }

  async stop(options: AutonomySchedulerStopOptions = {}): Promise<void> {
    if (this.intervalHandle !== null) {
      this.clearIntervalFn(this.intervalHandle);
      this.intervalHandle = null;
    }

    if (options.graceful === false) {
      return;
    }

    const activeTick = this.activeTick;

    if (activeTick !== null) {
      await activeTick;
    }
  }

  async tick(): Promise<TickResult> {
    const nowMs = this.clock.now();
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

    const dueEvents = await this.scanDueEvents();
    const writer = this.options.createStreamWriter(this.sessionId);
    const eventResults: AutonomyTickEventResult[] = [];
    let firedEvents = 0;
    let budgetSkipped = 0;
    let busySkipped = 0;
    let errorCount = 0;

    try {
      for (const scannedEvent of dueEvents) {
        const dueEvent = scannedEvent.event;
        this.pruneWakeHistory();

        if (this.wakeHistory.length >= this.options.maxWakesPerHour) {
          budgetSkipped += 1;
          eventResults.push({
            id: dueEvent.id,
            sourceName: dueEvent.sourceName,
            sourceType: dueEvent.sourceType,
            status: "budget_skipped",
            payload: dueEvent.payload,
            outcomeSummary: "Skipped because maxWakesPerHour budget was exhausted.",
          });
          continue;
        }

        const wakeEntry = await writer.append({
          kind: "internal_event",
          content: {
            kind: "autonomous_wake",
            trigger_type: dueEvent.sourceType,
            source_name: dueEvent.sourceName,
            payload: dueEvent.payload,
            ts: this.clock.now(),
          },
        });
        this.options.watermarkRepository.set(dueEvent.watermarkProcessName, this.sessionId, {
          lastTs: wakeEntry.timestamp,
          lastEntryId: wakeEntry.id,
        });
        this.wakeHistory.push(wakeEntry.timestamp);

        const preparedEvent = await this.prepareEvent(dueEvent);

        if ("toolError" in preparedEvent) {
          errorCount += 1;
          const outcomeSummary = `Autonomous preparation failed: ${preparedEvent.toolError}`;
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

          await writer.append({
            kind: "internal_event",
            content: {
              kind: "autonomous_action",
              trigger: dueEvent.sourceName,
              outcome_summary: outcomeSummary,
              turn_result_id: turnResult.agentMessageId ?? null,
              ts: this.clock.now(),
            },
          });

          firedEvents += 1;
          eventResults.push({
            id: dueEvent.id,
            sourceName: dueEvent.sourceName,
            sourceType: dueEvent.sourceType,
            status: "fired",
            payload: preparedEvent.event.payload,
            outcomeSummary,
            turnResultId: turnResult.agentMessageId ?? null,
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
          } else {
            errorCount += 1;
          }

          eventResults.push({
            id: dueEvent.id,
            sourceName: dueEvent.sourceName,
            sourceType: dueEvent.sourceType,
            status: busy ? "busy_skipped" : "error",
            payload: preparedEvent.event.payload,
            outcomeSummary,
            turnResultId: null,
            ...(busy ? {} : { error: formatError(error) }),
          });
        }
      }
    } finally {
      writer.close();
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
      (left, right) => left.event.sortTs - right.event.sortTs || left.event.id.localeCompare(right.event.id),
    );
  }

  private pruneWakeHistory(): void {
    const cutoff = this.clock.now() - 3_600_000;

    while (this.wakeHistory.length > 0 && this.wakeHistory[0] !== undefined && this.wakeHistory[0] < cutoff) {
      this.wakeHistory.shift();
    }
  }

  private async prepareEvent(
    dueEvent: DueEvent,
  ): Promise<
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
          toolName: "tool.identityEvents.list",
          input: {
            limit: 10,
          },
          origin: "autonomous",
          sessionId: this.sessionId,
          provenance,
        });

        if (!result.ok) {
          return {
            toolError: result.error,
          };
        }

        const output = result.output as {
          events: unknown[];
        };

        return {
          source,
          event: {
            ...dueEvent,
            payload: {
              ...dueEvent.payload,
              recent_identity_events: output.events,
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
      const result = await this.tick();
      await this.notifyTick(result);
    } catch (error) {
      await this.notifyError(error);
    } finally {
      this.activeTick = null;
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
}
