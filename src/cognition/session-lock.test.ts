import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_SESSION_ID } from "../util/ids.js";
import { FILE_LOCK_STALE_MS } from "../stream/file-lock.js";

import { SessionLock } from "./session-lock.js";

describe("SessionLock", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.useRealTimers();
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("recovers a former pod's session and renews throughout a long turn", async () => {
    vi.useFakeTimers();
    const dataDir = mkdtempSync(join(tmpdir(), "borg-lock-heartbeat-"));
    tempDirs.push(dataDir);
    const lock = new SessionLock({ dataDir });
    const path = join(dataDir, "locks", `session-${DEFAULT_SESSION_ID}.lock`);
    writeFileSync(
      path,
      JSON.stringify({
        pid: 999_999,
        host: "former-pod",
        timestamp: Date.now() - FILE_LOCK_STALE_MS - 1,
      }),
    );
    expect(lock.isHeld()).toBe(true);
    expect(await lock.tryAcquire()).toBeNull();
    await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS);
    expect(lock.isHeld()).toBe(false);
    const lease = await lock.tryAcquire();
    expect(lease).not.toBeNull();
    try {
      await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS * 3);
      expect(JSON.parse(readFileSync(path, "utf8")).heartbeat).toBe(Date.now());
      expect(lock.isHeld()).toBe(true);
      expect(await lock.tryAcquire()).toBeNull();
    } finally {
      await lease?.release();
    }
    expect(existsSync(path)).toBe(false);
  });

  it("coordinates separate lock instances through the filesystem", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-lock-"));
    tempDirs.push(tempDir);

    const first = new SessionLock({
      dataDir: tempDir,
      retryDelayMs: 1,
    });
    const second = new SessionLock({
      dataDir: tempDir,
      retryDelayMs: 1,
    });

    const firstLease = await first.tryAcquire(DEFAULT_SESSION_ID);
    expect(firstLease).not.toBeNull();

    const secondLeaseWhileHeld = await second.tryAcquire(DEFAULT_SESSION_ID);
    expect(secondLeaseWhileHeld).toBeNull();

    await firstLease?.release();

    const secondLeaseAfterRelease = await second.tryAcquire(DEFAULT_SESSION_ID);
    expect(secondLeaseAfterRelease).not.toBeNull();

    await secondLeaseAfterRelease?.release();
  });
});
