import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { DEFAULT_SESSION_ID } from "../util/ids.js";

import { SessionLock } from "./session-lock.js";

describe("SessionLock", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
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
