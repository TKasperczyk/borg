import {
  closeSync,
  fsyncSync,
  lstatSync,
  openSync,
  readFileSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { hostname } from "node:os";
import { dirname } from "node:path";

import { sleep } from "../util/clock.js";
import { StreamError } from "../util/errors.js";
import { isNodeError } from "../util/guards.js";
import { serializeJsonValue } from "../util/json-value.js";

type FileLockOptions = {
  timeoutMs?: number;
  retryDelayMs?: number;
  malformedGraceMs?: number;
};

type FileLockMetadata = {
  pid: number;
  host: string;
  timestamp: number;
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
    Number.isFinite((value as FileLockMetadata).timestamp)
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
    return false;
  }

  if (isProcessAlive(metadata.pid)) {
    return false;
  }

  return removeLockFileIfOwned(lockPath, identity, metadataText);
}

// Advisory check: returns true when the given lock path exists and is held by
// a live process on this host. Used by callers (e.g., MaintenanceScheduler)
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
    // Remote holder: cannot verify liveness, treat as live to err on the
    // cautious side for cross-host setups.
    return true;
  }

  return isProcessAlive(metadata.pid);
}

export async function withFileLock<T>(
  lockPath: string,
  callback: () => T | Promise<T>,
  options: FileLockOptions = {},
): Promise<T> {
  const timeoutMs = options.timeoutMs ?? 2_000;
  const retryDelayMs = options.retryDelayMs ?? 20;
  const malformedGraceMs = options.malformedGraceMs ?? DEFAULT_MALFORMED_LOCK_GRACE_MS;
  const deadline = Date.now() + timeoutMs;

  let lockFd: number | undefined;
  let ownedIdentity: LockFileIdentity | undefined;
  let ownedContents: string | undefined;

  while (lockFd === undefined) {
    try {
      lockFd = openSync(lockPath, "wx", 0o600);
      ownedContents = serializeJsonValue({
        pid: process.pid,
        host: LOCAL_HOSTNAME,
        timestamp: Date.now(),
      });
      writeFileSync(lockFd, ownedContents);
      fsyncSync(lockFd);
      ownedIdentity = lockFileIdentity(lockPath) ?? undefined;
    } catch (error) {
      if (!isNodeError(error) || error.code !== "EEXIST") {
        if (lockFd !== undefined) {
          closeSync(lockFd);
          lockFd = undefined;
        }
        throw new StreamError(`Failed to acquire stream lock at ${lockPath}`, {
          cause: error,
        });
      }

      if (reapStaleLock(lockPath, malformedGraceMs)) {
        continue;
      }

      if (Date.now() >= deadline) {
        throw new StreamError(`Timed out waiting for stream lock at ${lockPath}`);
      }

      await sleep(retryDelayMs);
    }
  }

  try {
    return await callback();
  } finally {
    closeSync(lockFd);

    const released =
      ownedIdentity !== undefined &&
      ownedContents !== undefined &&
      lockFileIdentity(lockPath) !== null
        ? removeLockFileIfOwned(lockPath, ownedIdentity, ownedContents)
        : false;
    if (!released) {
      console.warn(`Failed to release stream lock in ${dirname(lockPath)}: lock ownership changed`);
    }
  }
}
