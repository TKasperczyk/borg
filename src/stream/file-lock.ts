import { closeSync, fsyncSync, openSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { hostname } from "node:os";
import { dirname } from "node:path";

import { StreamError } from "../util/errors.js";
import { serializeJsonValue } from "../util/json-value.js";

type FileLockOptions = {
  timeoutMs?: number;
  retryDelayMs?: number;
};

type FileLockMetadata = {
  pid: number;
  host: string;
  timestamp: number;
};

const LOCAL_HOSTNAME = hostname();

function isNodeError(error: unknown): error is NodeJS.ErrnoException & { code: string } {
  return error instanceof Error && typeof (error as NodeJS.ErrnoException).code === "string";
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
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

function removeLockFile(lockPath: string): boolean {
  try {
    unlinkSync(lockPath);
    return true;
  } catch (error) {
    if (isNodeError(error) && error.code === "ENOENT") {
      return true;
    }

    return false;
  }
}

function reapStaleLock(lockPath: string): boolean {
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
    return removeLockFile(lockPath);
  }

  if (!isFileLockMetadata(metadata)) {
    return removeLockFile(lockPath);
  }

  if (metadata.host !== LOCAL_HOSTNAME) {
    return false;
  }

  if (isProcessAlive(metadata.pid)) {
    return false;
  }

  return removeLockFile(lockPath);
}

// Advisory check: returns true when the given lock path exists and is held by
// a live process on this host. Used by callers (e.g., MaintenanceScheduler)
// that want to skip work when a session is busy without racing to acquire the
// lock. Stale locks (crashed owner) return false so maintenance isn't blocked
// indefinitely after a crash.
export function isFileLockLive(lockPath: string): boolean {
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
    return false;
  }

  if (!isFileLockMetadata(metadata)) {
    return false;
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
  const deadline = Date.now() + timeoutMs;

  let lockFd: number | undefined;

  while (lockFd === undefined) {
    try {
      lockFd = openSync(lockPath, "wx");
      writeFileSync(
        lockFd,
        serializeJsonValue({
          pid: process.pid,
          host: LOCAL_HOSTNAME,
          timestamp: Date.now(),
        }),
      );
      fsyncSync(lockFd);
    } catch (error) {
      if (!isNodeError(error) || error.code !== "EEXIST") {
        throw new StreamError(`Failed to acquire stream lock at ${lockPath}`, {
          cause: error,
        });
      }

      if (reapStaleLock(lockPath)) {
        continue;
      }

      if (Date.now() >= deadline) {
        throw new StreamError(`Timed out waiting for stream lock at ${lockPath}`);
      }

      await delay(retryDelayMs);
    }
  }

  try {
    return await callback();
  } finally {
    closeSync(lockFd);

    try {
      unlinkSync(lockPath);
    } catch (error) {
      console.warn(
        `Failed to release stream lock in ${dirname(lockPath)}: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  }
}
