import {
  closeSync,
  existsSync,
  fsyncSync,
  lstatSync,
  openSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { randomUUID } from "node:crypto";
import { hostname } from "node:os";
import { dirname } from "node:path";
import type { DatabaseSync } from "node:sqlite";

import { writeFileAtomic } from "../util/atomic-write.js";
import { sleep } from "../util/clock.js";
import { StreamError } from "../util/errors.js";
import { isNodeError } from "../util/guards.js";
import { serializeJsonValue } from "../util/json-value.js";
import { FILE_LOCK_GUARD_SUFFIX, tryAcquireFileLockGuard } from "./file-lock-guard.js";

type FileLockOptions = {
  timeoutMs?: number;
  retryDelayMs?: number;
  malformedGraceMs?: number;
};

export type FileLockLease = { release(): Promise<void> };

export const FILE_LOCK_HEARTBEAT_INTERVAL_MS = 10_000;
// Twelve missed refreshes tolerate transient scheduling/I/O delays. Age alone
// never overrides the OS-backed guard of a participating live holder.
export const FILE_LOCK_STALE_MS = 120_000;

type FileLockMetadata = {
  pid: number;
  host: string;
  timestamp: number;
  heartbeat?: number;
  owner?: string;
};

const LOCAL_HOSTNAME = hostname();
const DEFAULT_MALFORMED_LOCK_GRACE_MS = 5_000;

type LockFileIdentity = {
  dev: number;
  ino: number;
  size: number;
  mtimeMs: number;
};

function lockFileIdentity(lockPath: string): LockFileIdentity | null {
  try {
    const stat = lstatSync(lockPath);
    return { dev: stat.dev, ino: stat.ino, size: stat.size, mtimeMs: stat.mtimeMs };
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

function sameLockFileIdentity(left: LockFileIdentity, right: LockFileIdentity): boolean {
  return (
    left.dev === right.dev &&
    left.ino === right.ino &&
    left.size === right.size &&
    left.mtimeMs === right.mtimeMs
  );
}

function isFileLockMetadata(value: unknown): value is FileLockMetadata {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    typeof (value as FileLockMetadata).pid === "number" &&
    Number.isInteger((value as FileLockMetadata).pid) &&
    typeof (value as FileLockMetadata).host === "string" &&
    typeof (value as FileLockMetadata).timestamp === "number" &&
    Number.isFinite((value as FileLockMetadata).timestamp) &&
    ((value as FileLockMetadata).heartbeat === undefined ||
      (typeof (value as FileLockMetadata).heartbeat === "number" &&
        Number.isFinite((value as FileLockMetadata).heartbeat))) &&
    ((value as FileLockMetadata).owner === undefined ||
      typeof (value as FileLockMetadata).owner === "string")
  );
}

function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    if (isNodeError(error) && error.code === "ESRCH") {
      return false;
    }

    if (isNodeError(error) && error.code === "EPERM") {
      return true;
    }

    throw error;
  }
}

function removeLockFileIfOwned(
  lockPath: string,
  expectedIdentity: LockFileIdentity,
  expectedContents: string,
): boolean {
  // The caller holds the stable SQLite guard throughout this compare/unlink.
  // Identity/content checks alone would have a cross-process TOCTOU race.
  try {
    const currentIdentity = lockFileIdentity(lockPath);
    if (
      currentIdentity === null ||
      !sameLockFileIdentity(currentIdentity, expectedIdentity) ||
      readFileSync(lockPath, "utf8") !== expectedContents
    ) {
      return currentIdentity === null;
    }

    unlinkSync(lockPath);
    return true;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return true;
    }

    return false;
  }
}

function reapStaleLock(lockPath: string, malformedGraceMs: number): boolean {
  const identity = lockFileIdentity(lockPath);
  if (identity === null) {
    return true;
  }
  let metadataText: string;

  try {
    metadataText = readFileSync(lockPath, "utf8");
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return true;
    }

    return false;
  }

  let metadata: unknown;

  try {
    metadata = JSON.parse(metadataText) as unknown;
  } catch {
    return Date.now() - identity.mtimeMs < malformedGraceMs
      ? false
      : removeLockFileIfOwned(lockPath, identity, metadataText);
  }

  if (!isFileLockMetadata(metadata)) {
    return Date.now() - identity.mtimeMs < malformedGraceMs
      ? false
      : removeLockFileIfOwned(lockPath, identity, metadataText);
  }

  if (metadata.host !== LOCAL_HOSTNAME) {
    // Pre-heartbeat files age from their original acquisition timestamp.
    return Date.now() - (metadata.heartbeat ?? metadata.timestamp) > FILE_LOCK_STALE_MS
      ? removeLockFileIfOwned(lockPath, identity, metadataText)
      : false;
  }

  // SIGSTOP/event-loop stalls are not death. A local live PID may still resume
  // work with open storage handles, regardless of the timestamp's age.
  if (isProcessAlive(metadata.pid)) {
    return false;
  }

  return removeLockFileIfOwned(lockPath, identity, metadataText);
}

// Advisory check: local PID liveness or fresh foreign heartbeat/held guard.
// Used by callers (e.g., MaintenanceScheduler)
// that want to skip work when a session is busy without racing to acquire the
// lock. Stale locks (crashed owner) return false so maintenance isn't blocked
// indefinitely after a crash.
export function isFileLockLive(
  lockPath: string,
  options: { malformedGraceMs?: number } = {},
): boolean {
  const malformedGraceMs = options.malformedGraceMs ?? DEFAULT_MALFORMED_LOCK_GRACE_MS;
  const identity = lockFileIdentity(lockPath);
  let metadataText: string;

  try {
    metadataText = readFileSync(lockPath, "utf8");
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return false;
    }

    return false;
  }

  let metadata: unknown;

  try {
    metadata = JSON.parse(metadataText) as unknown;
  } catch {
    return identity !== null && Date.now() - identity.mtimeMs < malformedGraceMs;
  }

  if (!isFileLockMetadata(metadata)) {
    return identity !== null && Date.now() - identity.mtimeMs < malformedGraceMs;
  }

  if (metadata.host !== LOCAL_HOSTNAME) {
    if (Date.now() - (metadata.heartbeat ?? metadata.timestamp) <= FILE_LOCK_STALE_MS) {
      return true;
    }
    // A stopped remote holder can miss heartbeats while retaining its guard.
    // Avoid creating files for this advisory check of a legacy lease.
    if (!existsSync(`${lockPath}${FILE_LOCK_GUARD_SUFFIX}`)) return false;
    try {
      const guard = tryAcquireFileLockGuard(lockPath);
      if (guard === null) return true;
      guard.close();
      return false;
    } catch {
      return true; // Unverifiable guard: keep maintenance out.
    }
  }

  return isProcessAlive(metadata.pid);
}

/** Acquire a renewable lease, retaining its guard until idempotent release. */
export async function acquireFileLockLease(
  lockPath: string,
  options: FileLockOptions = {},
): Promise<FileLockLease> {
  const timeoutMs = options.timeoutMs ?? 2_000;
  const retryDelayMs = options.retryDelayMs ?? 20;
  const malformedGraceMs = options.malformedGraceMs ?? DEFAULT_MALFORMED_LOCK_GRACE_MS;
  const deadline = Date.now() + timeoutMs;

  let guard: DatabaseSync | null = null;
  const owner = randomUUID();

  while (true) {
    try {
      guard = tryAcquireFileLockGuard(lockPath);
      if (guard !== null) {
        // Hold the guard over reaping AND wx creation. A second reaper cannot
        // pass a stale comparison and later unlink the winner's replacement.
        if (reapStaleLock(lockPath, malformedGraceMs)) {
          const lockFd = openSync(lockPath, "wx", 0o600);
          try {
            const now = Date.now();
            writeFileSync(
              lockFd,
              serializeJsonValue({
                pid: process.pid,
                host: LOCAL_HOSTNAME,
                timestamp: now,
                heartbeat: now,
                owner,
              }),
            );
            fsyncSync(lockFd);
          } catch (error) {
            // No callback has started, and the guard still excludes acquirers.
            unlinkSync(lockPath);
            throw error;
          } finally {
            closeSync(lockFd);
          }
          break;
        }
        guard.close();
        guard = null;
      }
    } catch (error) {
      guard?.close();
      guard = null;
      if (!isNodeError(error) || error.code !== "EEXIST") {
        throw new StreamError(`Failed to acquire stream lock at ${lockPath}`, {
          cause: error,
        });
      }
    }

    if (Date.now() >= deadline) {
      throw new StreamError(`Timed out waiting for stream lock at ${lockPath}`);
    }
    await sleep(retryDelayMs);
  }

  // A per-acquisition token survives atomic heartbeat inode replacements,
  // including a write that renamed successfully but failed directory fsync.
  function readOwnedLock(): { identity: LockFileIdentity; contents: string } | null {
    const identity = lockFileIdentity(lockPath);
    if (identity === null) return null;
    const contents = readFileSync(lockPath, "utf8");
    const metadata: unknown = JSON.parse(contents);
    return isFileLockMetadata(metadata) && metadata.owner === owner ? { identity, contents } : null;
  }

  const heartbeat = setInterval(() => {
    try {
      if (readOwnedLock() === null) {
        clearInterval(heartbeat);
        console.warn(`Failed to refresh stream lock at ${lockPath}: lock ownership changed`);
        return;
      }
      const now = Date.now();
      writeFileAtomic(
        lockPath,
        serializeJsonValue({
          pid: process.pid,
          host: LOCAL_HOSTNAME,
          timestamp: now,
          heartbeat: now,
          owner,
        }),
        { mode: 0o600 },
      );
    } catch (error) {
      // The retained guard still protects the work if refreshes fail or stall.
      console.warn(`Failed to refresh stream lock at ${lockPath}`, error);
    }
  }, FILE_LOCK_HEARTBEAT_INTERVAL_MS);
  heartbeat.unref();

  let released = false;
  return {
    release: async () => {
      if (released) return;
      released = true;
      clearInterval(heartbeat);
      try {
        const owned = readOwnedLock();
        if (owned === null || !removeLockFileIfOwned(lockPath, owned.identity, owned.contents)) {
          console.warn(
            `Failed to release stream lock in ${dirname(lockPath)}: lock ownership changed`,
          );
        }
      } catch (error) {
        console.warn(`Failed to release stream lock at ${lockPath}`, error);
      } finally {
        guard.close();
      }
    },
  };
}

export async function withFileLock<T>(
  lockPath: string,
  callback: () => T | Promise<T>,
  options: FileLockOptions = {},
): Promise<T> {
  const lease = await acquireFileLockLease(lockPath, options);
  try {
    return await callback();
  } finally {
    await lease.release();
  }
}
