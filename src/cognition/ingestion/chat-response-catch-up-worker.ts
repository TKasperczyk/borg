/*
 * v1 delivery contract: Inbound is durable (committed before ack). Reply generation is replay-safe (cursor-stamped response_to + reconcile-before-generate; at-least-once). External delivery is NOT auto-retried by borg (no durable outbox in v1). Accepted D1 loss window: a crash after the stamped terminal append but before the transport delivers leaves the reply recorded-but-possibly-undelivered, not retried. Dedup is per source_message_key via the single-writer daemon's in-process serialization; cross-process concurrent writers are out of v1 scope.
 */
import type { StreamCursor, StreamEntry, StreamEntryIndexRepository } from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import { CognitionError, describeError, SessionBusyError } from "../../util/errors.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
import type { TurnOrchestrator } from "../turn-orchestrator.js";

import type { ChatResponseBacklogPrefixBuilder } from "./backlog-prefix.js";
import type { ChatResponseWatermarkCoordinator } from "./chat-response-watermark.js";
import type { TaskEventService, TaskEventCatchUpRunner } from "./task-events.js";

type TimeoutHandle = ReturnType<typeof setTimeout>;
type SetTimeoutFn = (callback: () => void, delayMs: number) => TimeoutHandle;
type ClearTimeoutFn = (handle: TimeoutHandle) => void;

type ResponseLane = "user" | "task_event";
type LaneState = {
  oldestPendingAt: number | null;
  timer: TimeoutHandle | null;
  backoffMs: number | null;
  repairOnly: boolean;
  retryAt: number | null;
};
type SessionState = LaneState & {
  task: LaneState;
  lease: ChatResponseCatchUpLease | null;
};

function newLaneState(): LaneState {
  return { oldestPendingAt: null, timer: null, backoffMs: null, repairOnly: false, retryAt: null };
}

type ScheduleReason = "append" | "startup" | "in_flight_settled" | "has_more" | "retry";

export type ChatResponseCatchUpWorkerConfig = {
  quietWindowMs: number;
  maxWaitMs: number;
  backoffBaseMs: number;
  maxBackoffMs: number;
};

export type DrainResult = {
  sessionId: SessionId;
  status: "empty" | "drained" | "busy" | "error";
  drained: number;
  hasMore: boolean;
  error?: string;
};

export type ChatResponseCatchUpRunInput = {
  sessionId: SessionId;
  inboundBatch: {
    kind: "stream_backlog";
    entryIds: readonly StreamEntryId[];
    throughCursorInclusive: StreamCursor;
  };
};

export type ChatResponseCatchUpRunner = {
  run(input: ChatResponseCatchUpRunInput): Promise<void>;
};

export type ChatResponseCatchUpLease = {
  release(): void;
};

export type ChatResponseReconcileAdvance = {
  sessionId: SessionId;
  advancedThrough: StreamCursor;
};

export class TurnOrchestratorChatResponseCatchUpRunner implements ChatResponseCatchUpRunner {
  constructor(private readonly turnOrchestrator: Pick<TurnOrchestrator, "run">) {}

  async run(input: ChatResponseCatchUpRunInput): Promise<void> {
    await this.turnOrchestrator.run({
      sessionId: input.sessionId,
      origin: "user",
      lockMode: "try",
      inboundBatch: input.inboundBatch,
    });
  }
}

export type ChatResponseCatchUpWorkerOptions = {
  coordinator: Pick<ChatResponseWatermarkCoordinator, "reconcile" | "compareCursors">;
  prefixBuilder: Pick<ChatResponseBacklogPrefixBuilder, "build">;
  entryIndex: Pick<StreamEntryIndexRepository, "listSessionIdsWithPendingResponseBacklog">;
  repairSessionStreamEntryIndex: (sessionId: SessionId) => Promise<unknown>;
  runner?: ChatResponseCatchUpRunner;
  taskEvents?: {
    service: Pick<TaskEventService, "listSessionIds" | "listUnanswered" | "findTerminal">;
    runner: TaskEventCatchUpRunner;
  };
  turnOrchestrator?: Pick<TurnOrchestrator, "run">;
  sessionPredicate?: (sessionId: SessionId) => boolean;
  acquireLease?: () => ChatResponseCatchUpLease;
  onReconcileAdvance?: (event: ChatResponseReconcileAdvance) => void;
  clock: Clock;
  setTimeoutFn?: SetTimeoutFn;
  clearTimeoutFn?: ClearTimeoutFn;
  config: ChatResponseCatchUpWorkerConfig;
};

function isQueuedUserMessage(entry: StreamEntry): boolean {
  return entry.kind === "user_msg" && entry.turn_id === undefined;
}

function isSessionTurnBusy(error: unknown): boolean {
  return error instanceof SessionBusyError && error.code === "SESSION_TURN_BUSY";
}

function isPoisonedStreamIndex(error: unknown): boolean {
  return (
    error !== null &&
    typeof error === "object" &&
    "code" in error &&
    (error as { code?: unknown }).code === "STREAM_INDEX_POISONED"
  );
}

function clampNonnegativeDelay(delayMs: number): number {
  return Math.max(0, delayMs);
}

export class ChatResponseCatchUpWorker {
  private readonly clock: Clock;
  private readonly setTimeoutFn: SetTimeoutFn;
  private readonly clearTimeoutFn: ClearTimeoutFn;
  private readonly sessionStates = new Map<SessionId, SessionState>();
  private readonly inFlight = new Map<SessionId, Promise<DrainResult>>();
  private started = false;
  private stopping = false;
  private startupScan: Promise<void> | null = null;
  private startupLease: ChatResponseCatchUpLease | null = null;
  private readonly runner: ChatResponseCatchUpRunner;

  constructor(private readonly options: ChatResponseCatchUpWorkerOptions) {
    this.clock = options.clock;
    this.setTimeoutFn =
      options.setTimeoutFn ?? ((callback, delayMs) => setTimeout(callback, delayMs));
    this.clearTimeoutFn = options.clearTimeoutFn ?? ((handle) => clearTimeout(handle));
    if (options.runner !== undefined) {
      this.runner = options.runner;
    } else if (options.turnOrchestrator !== undefined) {
      this.runner = new TurnOrchestratorChatResponseCatchUpRunner(options.turnOrchestrator);
    } else {
      throw new CognitionError("Chat response catch-up worker requires a runner", {
        code: "CHAT_RESPONSE_RUNNER_REQUIRED",
      });
    }
  }

  start(): void {
    if (this.started) {
      return;
    }

    this.stopping = false;
    this.started = true;
    const startupLease = this.acquireBackgroundLease();
    this.startupLease = startupLease;
    this.startupScan = this.runStartupScan()
      .catch((error) => {
        this.logError("startup scan failed", error);
      })
      .finally(() => {
        this.startupScan = null;
        startupLease?.release();
        if (this.startupLease === startupLease) {
          this.startupLease = null;
        }
      });
  }

  async stop(options: { graceful?: boolean } = {}): Promise<void> {
    this.started = false;
    this.stopping = true;

    for (const [sessionId, state] of this.sessionStates) {
      for (const lane of [state, state.task]) {
        if (lane.timer !== null) this.clearTimeoutFn(lane.timer);
        lane.timer = null;
      }
      this.releaseSessionLeaseIfIdle(sessionId);
    }

    if (options.graceful === false) {
      return;
    }

    const startupScan = this.startupScan;

    if (startupScan !== null) {
      await startupScan;
    }

    const inFlight = [...this.inFlight.values()];

    if (inFlight.length > 0) {
      await Promise.allSettled(inFlight);
    }
  }

  isEnabled(): boolean {
    return true;
  }

  isTaskEventEnabled(): boolean {
    return this.options.taskEvents !== undefined;
  }

  onAppend(entries: readonly StreamEntry[]): void {
    if (!this.started || this.stopping) {
      return;
    }

    for (const entry of entries) {
      if (
        this.options.taskEvents !== undefined &&
        entry.kind === "internal_event" &&
        entry.metadata?.task_event !== undefined
      ) {
        this.onPendingTaskSession(entry.session_id, entry.timestamp);
        continue;
      }
      if (!isQueuedUserMessage(entry)) {
        continue;
      }
      if (!this.acceptsSession(entry.session_id)) {
        continue;
      }

      this.markPending(entry.session_id, entry.timestamp);
      this.schedule(entry.session_id, "append");
    }
  }

  onPendingSession(sessionId: SessionId, pendingAt: number): void {
    if (!this.started || this.stopping) {
      return;
    }
    if (!this.acceptsSession(sessionId)) {
      return;
    }

    this.markPending(sessionId, pendingAt);
    this.schedule(sessionId, "append");
  }

  onPendingTaskSession(sessionId: SessionId, pendingAt: number): void {
    if (
      !this.started ||
      this.stopping ||
      this.options.taskEvents === undefined ||
      !this.acceptsSession(sessionId)
    )
      return;
    this.markPending(sessionId, pendingAt, "task_event");
    if (this.ensureSessionState(sessionId).task.timer === null) {
      this.schedule(sessionId, "append", "task_event");
    }
  }

  tick(sessionId: SessionId): Promise<DrainResult> {
    return this.runTrackedDrain(sessionId);
  }

  private async runStartupScan(): Promise<void> {
    const sessionIds = this.options.entryIndex.listSessionIdsWithPendingResponseBacklog();

    for (const sessionId of sessionIds) {
      if (!this.started || this.stopping) {
        return;
      }
      if (!this.acceptsSession(sessionId)) {
        continue;
      }

      try {
        const { watermark } = this.reconcile(sessionId);
        const prefix = await this.options.prefixBuilder.build({
          sessionId,
          fromCursorExclusive: watermark,
        });

        if (prefix.includedCount > 0) {
          this.markPending(sessionId, this.clock.now());
          this.schedule(sessionId, "startup");
        }
      } catch (error) {
        this.markPending(sessionId, this.clock.now());
        this.applyBackoff(sessionId);
        this.logError(`startup scan failed for session ${sessionId}`, error);
        this.schedule(sessionId, "retry");
      }
    }
    for (const sessionId of this.options.taskEvents?.service.listSessionIds() ?? []) {
      if (!this.started || this.stopping) return;
      if (!this.acceptsSession(sessionId)) continue;
      this.markPending(sessionId, this.clock.now(), "task_event");
      this.schedule(sessionId, "startup", "task_event");
    }
  }

  private schedule(
    sessionId: SessionId,
    reason: ScheduleReason,
    lane: ResponseLane = "user",
  ): void {
    if (!this.started || this.stopping) {
      return;
    }

    const session = this.ensureSessionState(sessionId);
    const state = this.laneState(sessionId, lane);

    if (state.oldestPendingAt === null) {
      return;
    }

    // User settling owns its deadline. Task continuation/retry must neither
    // trigger that batch early nor reset its timer or maximum wait age.
    if (lane === "task_event" && session.oldestPendingAt !== null) return;
    if (lane === "user" && session.task.timer !== null) {
      this.clearTimeoutFn(session.task.timer);
      session.task.timer = null;
    }

    if (this.inFlight.has(sessionId)) {
      return;
    }

    if (state.timer !== null) {
      this.clearTimeoutFn(state.timer);
      state.timer = null;
    }

    this.ensureSessionLease(session);
    const delayMs =
      lane === "task_event" && state.retryAt !== null
        ? clampNonnegativeDelay(state.retryAt - this.clock.now())
        : this.delayFor(state, reason);
    state.timer = this.setTimeoutFn(() => {
      state.timer = null;

      if (!this.started || this.stopping) {
        this.releaseSessionLeaseIfIdle(sessionId);
        return;
      }

      void this.runScheduledDrain(sessionId);
    }, delayMs);
  }

  private async drain(sessionId: SessionId, selectTaskLane: () => void): Promise<DrainResult> {
    const state = this.ensureSessionState(sessionId);
    const claimedPendingSince = state.oldestPendingAt ?? this.clock.now();
    const repairOnly = state.repairOnly;
    state.oldestPendingAt = null;
    state.repairOnly = false;

    try {
      if (!this.acceptsSession(sessionId)) {
        state.backoffMs = null;
        return {
          sessionId,
          status: "empty",
          drained: 0,
          hasMore: false,
        };
      }

      if (repairOnly) {
        await this.options.repairSessionStreamEntryIndex(sessionId);
        state.backoffMs = null;
        this.markPending(sessionId, this.clock.now());

        return {
          sessionId,
          status: "drained",
          drained: 0,
          hasMore: true,
        };
      }

      const { watermark } = this.reconcile(sessionId);
      const prefix = await this.options.prefixBuilder.build({
        sessionId,
        fromCursorExclusive: watermark,
      });

      if (prefix.includedCount === 0) {
        state.backoffMs = null;
        if (this.options.taskEvents !== undefined && state.oldestPendingAt === null) {
          selectTaskLane();
          return this.drainTaskEvent(sessionId);
        }

        return {
          sessionId,
          status: "empty",
          drained: 0,
          hasMore: false,
        };
      }

      if (prefix.throughCursorInclusive === null) {
        throw new Error("Chat response catch-up prefix included entries without a through cursor");
      }

      await this.runner.run({
        sessionId,
        inboundBatch: {
          kind: "stream_backlog",
          entryIds: prefix.entryIds,
          throughCursorInclusive: prefix.throughCursorInclusive,
        },
      });

      const reconciled = this.reconcile(sessionId).watermark;
      const coversSuppliedPrefix =
        reconciled !== null &&
        this.options.coordinator.compareCursors(
          sessionId,
          reconciled,
          prefix.throughCursorInclusive,
        ) >= 0;
      if (!coversSuppliedPrefix && !this.watermarkAdvanced(sessionId, watermark, reconciled)) {
        throw new CognitionError("Chat response runner did not durably cover its supplied prefix", {
          code: "CHAT_RESPONSE_RUNNER_PREFIX_NOT_COVERED",
        });
      }

      state.backoffMs = null;

      if (!coversSuppliedPrefix) {
        this.markPending(sessionId, this.clock.now());
        const coveredEntryIndex =
          reconciled === null ? -1 : prefix.entryIds.indexOf(reconciled.entryId);
        return {
          sessionId,
          status: "drained",
          drained: coveredEntryIndex + 1,
          hasMore: true,
        };
      }

      const hasMore = prefix.hasMore;
      if (hasMore) {
        this.markPending(sessionId, this.clock.now());
      }

      return {
        sessionId,
        status: "drained",
        drained: prefix.includedCount,
        hasMore,
      };
    } catch (error) {
      if (repairOnly) {
        return this.handleRepairOnlyError(sessionId, error, claimedPendingSince);
      }

      return this.handleDrainError(sessionId, error, claimedPendingSince);
    }
  }

  private async drainTaskEvent(sessionId: SessionId): Promise<DrainResult> {
    const lane = this.options.taskEvents!;
    const state = this.laneState(sessionId, "task_event");
    const claimedPendingSince = state.oldestPendingAt ?? this.clock.now();
    state.oldestPendingAt = null;
    try {
      if (state.repairOnly) {
        await this.options.repairSessionStreamEntryIndex(sessionId);
        state.repairOnly = false;
      }
      await lane.runner.reconcile(sessionId);
      const taskEvent = lane.service.listUnanswered(sessionId)[0];
      if (taskEvent !== undefined) {
        await lane.runner.run({ sessionId, taskEvent });
        if (lane.service.findTerminal(sessionId, taskEvent) === null) {
          throw new CognitionError("Task event runner did not durably answer its event", {
            code: "TASK_EVENT_RUNNER_EVENT_NOT_COVERED",
          });
        }
      }
      const hasMore = lane.service.listUnanswered(sessionId).length > 0;
      state.backoffMs = null;
      state.retryAt = null;
      if (hasMore) this.markPending(sessionId, this.clock.now(), "task_event");
      return {
        sessionId,
        status: taskEvent === undefined ? "empty" : "drained",
        drained: taskEvent === undefined ? 0 : 1,
        hasMore,
      };
    } catch (error) {
      if (isPoisonedStreamIndex(error)) state.repairOnly = true;
      this.restoreClaimedPending(sessionId, claimedPendingSince, "task_event");
      this.applyBackoff(sessionId, "task_event");
      state.retryAt = this.clock.now() + state.backoffMs!;
      this.logError(`task drain failed for session ${sessionId}`, error);
      return { sessionId, status: "error", drained: 0, hasMore: true, error: describeError(error) };
    }
  }

  private runTrackedDrain(sessionId: SessionId): Promise<DrainResult> {
    const existing = this.inFlight.get(sessionId);

    if (existing !== undefined) {
      return existing;
    }

    let result: DrainResult | undefined;
    let taskDrain = false;
    let promise: Promise<DrainResult>;

    this.ensureSessionLease(this.ensureSessionState(sessionId));
    promise = this.drain(sessionId, () => {
      taskDrain = true;
    })
      .then((drainResult) => {
        result = drainResult;
        return drainResult;
      })
      .finally(() => {
        if (this.inFlight.get(sessionId) === promise) {
          this.inFlight.delete(sessionId);
        }

        try {
          if (!this.started || this.stopping) {
            return;
          }

          const state = this.sessionStates.get(sessionId);

          if (state === undefined) return;
          if (state.oldestPendingAt !== null) {
            const reason =
              !taskDrain && (result?.status === "busy" || result?.status === "error")
                ? "retry"
                : !taskDrain && result?.hasMore === true
                  ? "has_more"
                  : "in_flight_settled";
            this.schedule(sessionId, reason);
          }
          if (state.task.oldestPendingAt !== null)
            this.schedule(sessionId, "has_more", "task_event");
        } finally {
          this.releaseSessionLeaseIfIdle(sessionId);
        }
      });

    this.inFlight.set(sessionId, promise);
    return promise;
  }

  private async runScheduledDrain(sessionId: SessionId): Promise<void> {
    try {
      await this.runTrackedDrain(sessionId);
    } catch (error) {
      this.logError(`scheduled drain failed for session ${sessionId}`, error);
    }
  }

  private handleDrainError(
    sessionId: SessionId,
    error: unknown,
    claimedPendingSince: number,
  ): DrainResult {
    if (isPoisonedStreamIndex(error)) {
      return this.handlePoisonedSessionError(sessionId, error, claimedPendingSince);
    }

    this.restoreClaimedPending(sessionId, claimedPendingSince);
    this.applyBackoff(sessionId);

    if (isSessionTurnBusy(error)) {
      return {
        sessionId,
        status: "busy",
        drained: 0,
        hasMore: true,
        error: describeError(error),
      };
    }

    this.logError(`drain failed for session ${sessionId}`, error);

    return {
      sessionId,
      status: "error",
      drained: 0,
      hasMore: true,
      error: describeError(error),
    };
  }

  private handlePoisonedSessionError(
    sessionId: SessionId,
    error: unknown,
    claimedPendingSince: number,
  ): DrainResult {
    this.markRepairOnlyPending(sessionId, claimedPendingSince);
    this.applyBackoff(sessionId);
    this.logError(`session ${sessionId} needs stream index repair`, error);

    return {
      sessionId,
      status: "error",
      drained: 0,
      hasMore: true,
      error: describeError(error),
    };
  }

  private handleRepairOnlyError(
    sessionId: SessionId,
    error: unknown,
    claimedPendingSince: number,
  ): DrainResult {
    this.markRepairOnlyPending(sessionId, claimedPendingSince);
    this.applyBackoff(sessionId);
    this.logError(`repair-only retry failed for session ${sessionId}`, error);

    return {
      sessionId,
      status: "error",
      drained: 0,
      hasMore: true,
      error: describeError(error),
    };
  }

  private markPending(sessionId: SessionId, pendingAt: number, lane: ResponseLane = "user"): void {
    const state = this.laneState(sessionId, lane);

    if (state.oldestPendingAt === null) {
      state.oldestPendingAt = pendingAt;
    }
  }

  private restoreClaimedPending(
    sessionId: SessionId,
    claimedPendingSince: number,
    lane: ResponseLane = "user",
  ): void {
    const state = this.laneState(sessionId, lane);

    state.oldestPendingAt = Math.min(
      state.oldestPendingAt ?? claimedPendingSince,
      claimedPendingSince,
    );
  }

  private markRepairOnlyPending(sessionId: SessionId, pendingAt: number): void {
    const state = this.ensureSessionState(sessionId);

    state.repairOnly = true;
    state.oldestPendingAt = Math.min(state.oldestPendingAt ?? pendingAt, pendingAt);
  }

  private ensureSessionState(sessionId: SessionId): SessionState {
    let state = this.sessionStates.get(sessionId);

    if (state === undefined) {
      state = {
        ...newLaneState(),
        task: newLaneState(),
        lease: null,
      };
      this.sessionStates.set(sessionId, state);
    }

    return state;
  }

  private laneState(sessionId: SessionId, lane: ResponseLane): LaneState {
    const state = this.ensureSessionState(sessionId);
    return lane === "user" ? state : state.task;
  }

  private delayFor(state: LaneState, reason: ScheduleReason): number {
    if (reason === "startup" || reason === "has_more") {
      return 0;
    }

    if (reason === "retry") {
      return state.backoffMs ?? this.options.config.backoffBaseMs;
    }

    const now = this.clock.now();
    const oldestPendingAt = state.oldestPendingAt ?? now;
    const quietDueAt = now + this.options.config.quietWindowMs;
    const maxWaitDueAt = oldestPendingAt + this.options.config.maxWaitMs;

    return clampNonnegativeDelay(Math.min(quietDueAt, maxWaitDueAt) - now);
  }

  private applyBackoff(sessionId: SessionId, lane: ResponseLane = "user"): void {
    const state = this.laneState(sessionId, lane);
    const previous = state.backoffMs;
    const next =
      previous === null
        ? this.options.config.backoffBaseMs
        : Math.max(this.options.config.backoffBaseMs, previous * 2);

    state.backoffMs = Math.min(next, this.options.config.maxBackoffMs);
  }

  private reconcile(sessionId: SessionId) {
    const result = this.options.coordinator.reconcile(sessionId);
    if (result.advancedThrough !== null) {
      try {
        this.options.onReconcileAdvance?.({
          sessionId,
          advancedThrough: result.advancedThrough,
        });
      } catch (error) {
        this.logError(`reconcile observer failed for session ${sessionId}`, error);
      }
    }
    return result;
  }

  private watermarkAdvanced(
    sessionId: SessionId,
    before: StreamCursor | null,
    after: StreamCursor | null,
  ): boolean {
    if (after === null) {
      return false;
    }
    return before === null || this.options.coordinator.compareCursors(sessionId, after, before) > 0;
  }

  private acquireBackgroundLease(): ChatResponseCatchUpLease | null {
    try {
      return this.options.acquireLease?.() ?? null;
    } catch (error) {
      this.logError("failed to acquire background lease", error);
      return null;
    }
  }

  private ensureSessionLease(state: SessionState): void {
    state.lease ??= this.acquireBackgroundLease();
  }

  private releaseSessionLeaseIfIdle(sessionId: SessionId): void {
    const state = this.sessionStates.get(sessionId);
    if (
      state === undefined ||
      state.timer !== null ||
      state.task.timer !== null ||
      this.inFlight.has(sessionId)
    ) {
      return;
    }
    state.lease?.release();
    state.lease = null;
  }

  private logError(message: string, error: unknown): void {
    console.error(`Chat response catch-up worker ${message}: ${describeError(error)}`);
  }

  private acceptsSession(sessionId: SessionId): boolean {
    return this.options.sessionPredicate?.(sessionId) ?? true;
  }
}
