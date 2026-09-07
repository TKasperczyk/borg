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
import { fork, type ChildProcess } from "node:child_process";
import { once } from "node:events";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  acquireFileLockLease,
  FILE_LOCK_HEARTBEAT_INTERVAL_MS,
  FILE_LOCK_STALE_MS,
  isFileLockLive,
  withFileLock,
} from "./file-lock.js";
import * as atomicWrite from "../util/atomic-write.js";

describe("file-lock", () => {
  const tempDirs: string[] = [];
  const children: ChildProcess[] = [];

  function leasePath(): string {
    const dir = mkdtempSync(join(tmpdir(), "file-lock-heartbeat-"));
    tempDirs.push(dir);
    return join(dir, "lease.lock");
  }

  async function contender(path: string, host: string): Promise<ChildProcess> {
    const child = fork(
      new URL("./test-support/file-lock-process.ts", import.meta.url),
      [path, host],
      {
        execArgv: ["--import", "tsx"],
        stdio: ["ignore", "ignore", "inherit", "ipc"],
      },
    );
    children.push(child);
    expect((await once(child, "message"))[0]).toBe("ready");
    return child;
  }

  async function acquireInChild(child: ChildProcess): Promise<unknown> {
    const response = once(child, "message");
    child.send("acquire");
    return (await response)[0];
  }

  it("holds a lease across calls, propagates contention, and releases idempotently", async () => {
    const dir = mkdtempSync(join(tmpdir(), "file-lock-lease-"));
    tempDirs.push(dir);
    const path = join(dir, "lease.lock");
    const lease = await acquireFileLockLease(path);
    expect(isFileLockLive(path)).toBe(true);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    await Promise.all([lease.release(), lease.release()]);
    expect(existsSync(path)).toBe(false);
    const next = await acquireFileLockLease(path, { timeoutMs: 0 });
    await next.release();
  });

  it("reaps a dead lease owner and surfaces acquisition I/O errors", async () => {
    const dir = mkdtempSync(join(tmpdir(), "file-lock-dead-"));
    tempDirs.push(dir);
    const path = join(dir, "lease.lock");
    writeFileSync(path, JSON.stringify({ pid: 999_999, host: hostname(), timestamp: Date.now() }));
    await (await acquireFileLockLease(path, { timeoutMs: 0 })).release();
    await expect(acquireFileLockLease(join(dir, "missing", "lease.lock"))).rejects.toThrow();
  });

  afterEach(async () => {
    for (const child of children.splice(0)) {
      if (child.exitCode === null && child.signalCode === null) {
        const exited = once(child, "exit");
        child.kill("SIGKILL");
        await exited;
      }
    }
    vi.useRealTimers();
    vi.restoreAllMocks();
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("keeps a fresh foreign heartbeat live and reaps only after the stale window", async () => {
    vi.useFakeTimers();
    const path = leasePath();
    writeFileSync(
      path,
      JSON.stringify({
        pid: process.pid,
        host: "former-pod",
        timestamp: Date.now() - FILE_LOCK_STALE_MS * 3,
        heartbeat: Date.now(),
      }),
    );
    expect(isFileLockLive(path)).toBe(true);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS);
    expect(isFileLockLive(path)).toBe(true);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    await vi.advanceTimersByTimeAsync(1);
    expect(isFileLockLive(path)).toBe(false);
    const lease = await acquireFileLockLease(path, { timeoutMs: 0 });
    await lease.release();
  });

  it("ages a legacy foreign lease from its acquisition timestamp", async () => {
    vi.useFakeTimers();
    const path = leasePath();
    writeFileSync(path, JSON.stringify({ pid: 999_999, host: "old-pod", timestamp: Date.now() }));
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS + 1);
    expect(isFileLockLive(path)).toBe(false);
    await (await acquireFileLockLease(path, { timeoutMs: 0 })).release();
  });

  it("heartbeats a long-held lease and stops refreshing after release", async () => {
    vi.useFakeTimers();
    const path = leasePath();
    const lease = await acquireFileLockLease(path);
    const initial = JSON.parse(readFileSync(path, "utf8"));
    try {
      await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS * 3);
      expect(JSON.parse(readFileSync(path, "utf8"))).toEqual({
        ...initial,
        timestamp: Date.now(),
        heartbeat: Date.now(),
      });
      await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    } finally {
      await lease.release();
    }
    await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS * 2);
    expect(existsSync(path)).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("retains a local live PID even when its legacy timestamp is ancient", async () => {
    const path = leasePath();
    writeFileSync(path, JSON.stringify({ pid: process.pid, host: hostname(), timestamp: 0 }));
    expect(isFileLockLive(path)).toBe(true);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
  });

  it.each([false, true])(
    "logs heartbeat I/O failure and still releases (renamed=%s)",
    async (renamed) => {
      vi.useFakeTimers();
      const path = leasePath();
      const lease = await acquireFileLockLease(path);
      const original = atomicWrite.writeFileAtomic;
      const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
      vi.spyOn(atomicWrite, "writeFileAtomic").mockImplementationOnce((...args) => {
        if (renamed) original(...args);
        throw new Error("simulated fsync failure");
      });
      try {
        await vi.advanceTimersByTimeAsync(FILE_LOCK_HEARTBEAT_INTERVAL_MS);
        expect(warn).toHaveBeenCalledOnce();
        // Release immediately, including when rename succeeded but fsync failed.
      } finally {
        await lease.release();
      }
      expect(existsSync(path)).toBe(false);
      expect(warn).toHaveBeenCalledOnce();
      await (await acquireFileLockLease(path, { timeoutMs: 0 })).release();
    },
  );

  it("does not refresh or delete a replacement owner", async () => {
    vi.useFakeTimers();
    const path = leasePath();
    const lease = await acquireFileLockLease(path);
    const replacement = JSON.stringify({ pid: process.pid, host: hostname(), timestamp: 42 });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    writeFileSync(path, replacement);
    await vi.advanceTimersByTimeAsync(FILE_LOCK_HEARTBEAT_INTERVAL_MS * 2);
    expect(readFileSync(path, "utf8")).toBe(replacement);
    expect(warn).toHaveBeenCalledOnce();
    await lease.release();
    expect(readFileSync(path, "utf8")).toBe(replacement);
  });

  it("allows exactly one of two processes racing to reap and acquire a stale foreign lease", async () => {
    const path = leasePath();
    writeFileSync(
      path,
      JSON.stringify({
        pid: 999_999,
        host: "dead-pod",
        timestamp: 0,
        heartbeat: 0,
      }),
    );
    const [a, b] = await Promise.all([contender(path, "pod-a"), contender(path, "pod-b")]);
    const results = await Promise.all([acquireInChild(a), acquireInChild(b)]);
    expect(results.filter((result) => result === "acquired")).toHaveLength(1);
    expect(
      results.filter((result) => typeof result === "string" && result.startsWith("Timed out")),
    ).toHaveLength(1);
    const winner = results[0] === "acquired" ? a : b;
    expect(JSON.parse(readFileSync(path, "utf8")).pid).toBe(winner.pid);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    const exited = once(winner, "exit");
    winner.send("release");
    await exited;
    expect(existsSync(path)).toBe(false);
  });

  it.skipIf(process.platform === "win32")(
    "protects a stopped foreign holder and recovers after SIGKILL",
    async () => {
      const path = leasePath();
      const child = await contender(path, "paused-pod");
      expect(await acquireInChild(child)).toBe("acquired");
      child.kill("SIGSTOP");
      vi.useFakeTimers({ toFake: ["Date"] });
      vi.setSystemTime(Date.now() + FILE_LOCK_STALE_MS * 3);
      expect(isFileLockLive(path)).toBe(true);
      await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
      const exited = once(child, "exit");
      child.kill("SIGKILL");
      await exited;
      expect(isFileLockLive(path)).toBe(false);
      await (await acquireFileLockLease(path, { timeoutMs: 0 })).release();
    },
  );

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
