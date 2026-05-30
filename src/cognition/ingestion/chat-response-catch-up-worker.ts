/*
 * v1 delivery contract: Inbound is durable (committed before ack). Reply generation is replay-safe (cursor-stamped response_to + reconcile-before-generate; at-least-once). External delivery is NOT auto-retried by borg (no durable outbox in v1). Accepted D1 loss window: a crash after the stamped terminal append but before the transport delivers leaves the reply recorded-but-possibly-undelivered, not retried. Dedup is per source_message_key via the single-writer daemon's in-process serialization; cross-process concurrent writers are out of v1 scope.
 */
import type { SessionsRepository } from "../../sessions/index.js";
import type { StreamEntry, StreamEntryIndexRepository } from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import { describeError, SessionBusyError } from "../../util/errors.js";
import type { SessionId } from "../../util/ids.js";
import type { TurnOrchestrator } from "../turn-orchestrator.js";

import type { ChatResponseBacklogPrefixBuilder } from "./backlog-prefix.js";
import type { ChatResponseWatermarkCoordinator } from "./chat-response-watermark.js";

type TimeoutHandle = ReturnType<typeof setTimeout>;
type SetTimeoutFn = (callback: () => void, delayMs: number) => TimeoutHandle;
type ClearTimeoutFn = (handle: TimeoutHandle) => void;

type SessionState = {
  oldestPendingAt: number | null;
  timer: TimeoutHandle | null;
  backoffMs: number | null;
  repairOnly: boolean;
};

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

export type ChatResponseCatchUpWorkerOptions = {
  coordinator: Pick<ChatResponseWatermarkCoordinator, "reconcile">;
  prefixBuilder: Pick<ChatResponseBacklogPrefixBuilder, "build">;
  entryIndex: Pick<StreamEntryIndexRepository, "listSessionIdsWithPendingResponseBacklog">;
  repairSessionStreamEntryIndex: (sessionId: SessionId) => Promise<unknown>;
  turnOrchestrator: Pick<TurnOrchestrator, "run">;
  sessionsRepository: Pick<SessionsRepository, "count" | "list">;
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

  constructor(private readonly options: ChatResponseCatchUpWorkerOptions) {
    this.clock = options.clock;
    this.setTimeoutFn =
      options.setTimeoutFn ?? ((callback, delayMs) => setTimeout(callback, delayMs));
    this.clearTimeoutFn = options.clearTimeoutFn ?? ((handle) => clearTimeout(handle));
  }

  start(): void {
    if (this.started) {
      return;
    }

    this.stopping = false;
    this.started = true;
    this.startupScan = this.runStartupScan()
      .catch((error) => {
        this.logError("startup scan failed", error);
      })
      .finally(() => {
        this.startupScan = null;
      });
  }

  async stop(options: { graceful?: boolean } = {}): Promise<void> {
    this.started = false;
    this.stopping = true;

    for (const state of this.sessionStates.values()) {
      if (state.timer !== null) {
        this.clearTimeoutFn(state.timer);
        state.timer = null;
      }
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

  onAppend(entries: readonly StreamEntry[]): void {
    if (!this.started || this.stopping) {
      return;
    }

    for (const entry of entries) {
      if (!isQueuedUserMessage(entry)) {
        continue;
      }

      this.onPendingSession(entry.session_id, entry.timestamp);
    }
  }

  onPendingSession(sessionId: SessionId, pendingAt: number): void {
    if (!this.started || this.stopping) {
      return;
    }

    this.markPending(sessionId, pendingAt);
    this.schedule(sessionId, "append");
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

      try {
        const { watermark } = this.options.coordinator.reconcile(sessionId);
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
  }

  private schedule(sessionId: SessionId, reason: ScheduleReason): void {
    if (!this.started || this.stopping) {
      return;
    }

    const state = this.ensureSessionState(sessionId);

    if (state.oldestPendingAt === null) {
      return;
    }

    if (this.inFlight.has(sessionId)) {
      return;
    }

    if (state.timer !== null) {
      this.clearTimeoutFn(state.timer);
      state.timer = null;
    }

    const delayMs = this.delayFor(state, reason);
    state.timer = this.setTimeoutFn(() => {
      state.timer = null;

      if (!this.started || this.stopping) {
        return;
      }

      void this.runScheduledDrain(sessionId);
    }, delayMs);
  }

  private async drain(sessionId: SessionId): Promise<DrainResult> {
    const state = this.ensureSessionState(sessionId);
    const claimedPendingSince = state.oldestPendingAt ?? this.clock.now();
    const repairOnly = state.repairOnly;
    state.oldestPendingAt = null;
    state.repairOnly = false;

    try {
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

      const { watermark } = this.options.coordinator.reconcile(sessionId);
      const prefix = await this.options.prefixBuilder.build({
        sessionId,
        fromCursorExclusive: watermark,
      });

      if (prefix.includedCount === 0) {
        state.backoffMs = null;

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

      await this.options.turnOrchestrator.run({
        sessionId,
        origin: "user",
        lockMode: "try",
        inboundBatch: {
          kind: "stream_backlog",
          entryIds: prefix.entryIds,
          throughCursorInclusive: prefix.throughCursorInclusive,
        },
      });

      state.backoffMs = null;

      if (prefix.hasMore) {
        this.markPending(sessionId, this.clock.now());
      }

      return {
        sessionId,
        status: "drained",
        drained: prefix.includedCount,
        hasMore: prefix.hasMore,
      };
    } catch (error) {
      if (repairOnly) {
        return this.handleRepairOnlyError(sessionId, error, claimedPendingSince);
      }

      return this.handleDrainError(sessionId, error, claimedPendingSince);
    }
  }

  private runTrackedDrain(sessionId: SessionId): Promise<DrainResult> {
    const existing = this.inFlight.get(sessionId);

    if (existing !== undefined) {
      return existing;
    }

    let result: DrainResult | undefined;
    let promise: Promise<DrainResult>;

    promise = this.drain(sessionId)
      .then((drainResult) => {
        result = drainResult;
        return drainResult;
      })
      .finally(() => {
        if (this.inFlight.get(sessionId) === promise) {
          this.inFlight.delete(sessionId);
        }

        if (!this.started || this.stopping) {
          return;
        }

        const state = this.sessionStates.get(sessionId);

        if (state === undefined || state.oldestPendingAt === null) {
          return;
        }

        if (result?.status === "busy" || result?.status === "error") {
          this.schedule(sessionId, "retry");
          return;
        }

        if (result?.hasMore === true) {
          this.schedule(sessionId, "has_more");
          return;
        }

        this.schedule(sessionId, "in_flight_settled");
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

  private markPending(sessionId: SessionId, pendingAt: number): void {
    const state = this.ensureSessionState(sessionId);

    if (state.oldestPendingAt === null) {
      state.oldestPendingAt = pendingAt;
    }
  }

  private restoreClaimedPending(sessionId: SessionId, claimedPendingSince: number): void {
    const state = this.ensureSessionState(sessionId);

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
        oldestPendingAt: null,
        timer: null,
        backoffMs: null,
        repairOnly: false,
      };
      this.sessionStates.set(sessionId, state);
    }

    return state;
  }

  private delayFor(state: SessionState, reason: ScheduleReason): number {
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

  private applyBackoff(sessionId: SessionId): void {
    const state = this.ensureSessionState(sessionId);
    const previous = state.backoffMs;
    const next =
      previous === null
        ? this.options.config.backoffBaseMs
        : Math.max(this.options.config.backoffBaseMs, previous * 2);

    state.backoffMs = Math.min(next, this.options.config.maxBackoffMs);
  }

  private logError(message: string, error: unknown): void {
    console.error(`Chat response catch-up worker ${message}: ${describeError(error)}`);
  }
}
