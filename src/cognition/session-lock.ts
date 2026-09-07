import { mkdirSync } from "node:fs";
import { join } from "node:path";

import { acquireFileLockLease, isFileLockLive } from "../stream/file-lock.js";
import { DEFAULT_SESSION_ID, type SessionId } from "../util/ids.js";
import { SessionBusyError, StreamError } from "../util/errors.js";

export type SessionLockAcquireOptions = {
  timeoutMs?: number;
};

export type SessionLockOptions = {
  dataDir: string;
  defaultTimeoutMs?: number;
  retryDelayMs?: number;
};

export type SessionLockLease = {
  release(): Promise<void>;
};

const DEFAULT_ACQUIRE_TIMEOUT_MS = 30_000;
const DEFAULT_RETRY_DELAY_MS = 20;

function normalizeTimeout(timeoutMs: number | undefined): number | undefined {
  if (timeoutMs === undefined) {
    return undefined;
  }

  if (!Number.isFinite(timeoutMs) || timeoutMs < 0) {
    throw new SessionBusyError(`Invalid session lock timeout: ${timeoutMs}`, {
      code: "SESSION_LOCK_INVALID_TIMEOUT",
    });
  }

  return timeoutMs;
}

function isLockTimeoutError(error: unknown): boolean {
  return (
    error instanceof StreamError &&
    error.message.startsWith("Timed out waiting for stream lock at ")
  );
}

export class SessionLock {
  private readonly lockDir: string;
  private readonly defaultTimeoutMs: number;
  private readonly retryDelayMs: number;

  constructor(options: SessionLockOptions) {
    this.lockDir = join(options.dataDir, "locks");
    this.defaultTimeoutMs = options.defaultTimeoutMs ?? DEFAULT_ACQUIRE_TIMEOUT_MS;
    this.retryDelayMs = options.retryDelayMs ?? DEFAULT_RETRY_DELAY_MS;
    mkdirSync(this.lockDir, { recursive: true });
  }

  async acquire(
    sessionId: SessionId = DEFAULT_SESSION_ID,
    options: SessionLockAcquireOptions = {},
  ): Promise<SessionLockLease | null> {
    const timeoutMs = normalizeTimeout(options.timeoutMs) ?? this.defaultTimeoutMs;
    return this.acquireWithTimeout(sessionId, timeoutMs);
  }

  async tryAcquire(sessionId: SessionId = DEFAULT_SESSION_ID): Promise<SessionLockLease | null> {
    return this.acquireWithTimeout(sessionId, 0);
  }

  // Advisory check: local PID liveness or a foreign heartbeat/held guard.
  // The primitive renews throughout a turn, however long it takes.
  // Stale locks (owner crashed) return false so maintenance and
  // other opt-in consumers aren't blocked indefinitely after a crash. Callers
  // should treat this as a hint, not a correctness boundary -- acquiring the
  // lock is still the only way to safely perform writes.
  isHeld(sessionId: SessionId = DEFAULT_SESSION_ID): boolean {
    return isFileLockLive(this.lockPathFor(sessionId));
  }

  private lockPathFor(sessionId: SessionId): string {
    return join(this.lockDir, `session-${sessionId}.lock`);
  }

  private async acquireWithTimeout(
    sessionId: SessionId,
    timeoutMs: number,
  ): Promise<SessionLockLease | null> {
    try {
      return await acquireFileLockLease(this.lockPathFor(sessionId), {
        timeoutMs,
        retryDelayMs: this.retryDelayMs,
      });
    } catch (error) {
      if (isLockTimeoutError(error)) return null;
      throw error;
    }
  }
}
