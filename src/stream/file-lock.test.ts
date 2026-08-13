import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from "node:fs";
import { hostname } from "node:os";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { isFileLockLive, withFileLock } from "./file-lock.js";

describe("file-lock", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("reaps stale locks from dead processes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    mkdirSync(join(tempDir, "stream"), { recursive: true });
    const lockPath = join(tempDir, "stream", "default.jsonl.lock");

    writeFileSync(
      lockPath,
      JSON.stringify({
        pid: 999_999,
        host: hostname(),
        timestamp: Date.now() - 10_000,
      }),
    );

    const result = await withFileLock(lockPath, async () => "acquired");

    expect(result).toBe("acquired");
    expect(existsSync(lockPath)).toBe(false);
  });

  it("preserves callback success when lock cleanup unlink fails", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    mkdirSync(join(tempDir, "stream"), { recursive: true });
    const lockPath = join(tempDir, "stream", "default.jsonl.lock");
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    const result = await withFileLock(lockPath, async () => {
      unlinkSync(lockPath);
      return "completed";
    });

    expect(result).toBe("completed");
    expect(warnSpy).toHaveBeenCalledOnce();
  });

  it("treats malformed locks as live during a grace period, then reaps them", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const lockPath = join(tempDir, "malformed.lock");
    writeFileSync(lockPath, "partial lock metadata");

    expect(isFileLockLive(lockPath, { malformedGraceMs: 5_000 })).toBe(true);
    await expect(
      withFileLock(lockPath, () => "unreachable", {
        malformedGraceMs: 5_000,
        timeoutMs: 5,
        retryDelayMs: 1,
      }),
    ).rejects.toThrow("Timed out waiting for stream lock");

    const old = new Date(Date.now() - 10_000);
    utimesSync(lockPath, old, old);
    await expect(
      withFileLock(lockPath, () => "acquired", { malformedGraceMs: 5_000 }),
    ).resolves.toBe("acquired");
  });

  it("does not unlink a replacement lock during release", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const lockPath = join(tempDir, "replacement.lock");
    const replacement = JSON.stringify({ pid: process.pid, host: hostname(), timestamp: 42 });
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);

    await withFileLock(lockPath, () => {
      unlinkSync(lockPath);
      writeFileSync(lockPath, replacement);
    });

    expect(readFileSync(lockPath, "utf8")).toBe(replacement);
    expect(warnSpy).toHaveBeenCalledOnce();
  });
});
