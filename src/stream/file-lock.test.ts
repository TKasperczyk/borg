import { existsSync, mkdirSync, mkdtempSync, rmSync, unlinkSync, writeFileSync } from "node:fs";
import { hostname } from "node:os";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { withFileLock } from "./file-lock.js";

describe("file-lock", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
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
});
