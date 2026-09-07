import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  renameSync,
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

  async function commandInChild(
    child: ChildProcess,
    action: string | { advanceMs: number },
  ): Promise<unknown> {
    const response = once(child, "message");
    child.send(action);
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

  it.each([
    ["current clock", 0],
    ["one hour ahead", 3_600_000],
    ["one hour behind", -3_600_000],
    ["absurd finite timestamp", 1e30],
  ])("observes a full stale window for a foreign heartbeat (%s)", async (_label, skew) => {
    vi.useFakeTimers();
    const path = leasePath();
    writeFileSync(
      path,
      JSON.stringify({
        pid: process.pid,
        host: "former-pod",
        timestamp: Date.now() - FILE_LOCK_STALE_MS * 3,
        heartbeat: Date.now() + skew,
      }),
    );
    expect(isFileLockLive(path)).toBe(true);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS - 1);
    expect(isFileLockLive(path)).toBe(true);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    await vi.advanceTimersByTimeAsync(1);
    expect(isFileLockLive(path)).toBe(false);
    const lease = await acquireFileLockLease(path, { timeoutMs: 0 });
    await lease.release();
  });

  it.each([0, 1e30])(
    "observes legacy foreign leases for 120 seconds (timestamp=%s)",
    async (timestamp) => {
      vi.useFakeTimers();
      const path = leasePath();
      writeFileSync(path, JSON.stringify({ pid: 999_999, host: "old-pod", timestamp }));
      await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
      await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS + 1);
      expect(isFileLockLive(path)).toBe(false);
      await (await acquireFileLockLease(path, { timeoutMs: 0 })).release();
    },
  );

  it.each([-3_600_000, 3_600_000])(
    "never reaps a foreign holder whose heartbeat keeps advancing (skew=%s)",
    async (skew) => {
      vi.useFakeTimers();
      const path = leasePath();
      // Exercise observed renewal independently of the retained guard. Actual
      // primitive timers and stopped foreign holders are covered separately.
      for (
        let elapsed = 0;
        elapsed <= FILE_LOCK_STALE_MS * 3;
        elapsed += FILE_LOCK_HEARTBEAT_INTERVAL_MS
      ) {
        atomicWrite.writeFileAtomic(
          path,
          JSON.stringify({
            pid: process.pid,
            host: "live-foreign-pod",
            timestamp: Date.now() + skew,
            heartbeat: Date.now() + skew,
          }),
        );
        expect(isFileLockLive(path)).toBe(true);
        await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
        await vi.advanceTimersByTimeAsync(FILE_LOCK_HEARTBEAT_INTERVAL_MS);
      }
    },
  );

  it.each(["contents", "inode", "mtime"])(
    "restarts observation when only the %s changes",
    async (change) => {
      vi.useFakeTimers();
      const path = leasePath();
      const metadata = { pid: 999_999, host: "foreign-pod", timestamp: 0, heartbeat: 1e30 };
      writeFileSync(path, JSON.stringify(metadata));
      utimesSync(path, 0, 0);
      expect(isFileLockLive(path)).toBe(true);
      await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS - 1);
      if (change === "contents") {
        // Same inode, byte length, and mtime; only the contents change.
        writeFileSync(path, JSON.stringify({ ...metadata, heartbeat: 2e30 }));
        utimesSync(path, 0, 0);
      } else if (change === "inode") {
        atomicWrite.writeFileAtomic(path, JSON.stringify(metadata));
        utimesSync(path, 0, 0);
      } else {
        const future = new Date(Date.now() + 3_600_000);
        utimesSync(path, future, future);
      }
      expect(isFileLockLive(path)).toBe(true);
      await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS - 1);
      await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
      await vi.advanceTimersByTimeAsync(1);
      await (await acquireFileLockLease(path, { timeoutMs: 0 })).release();
    },
  );

  it("forgets observation when the file disappears, even if the same inode returns", async () => {
    vi.useFakeTimers();
    const path = leasePath();
    writeFileSync(path, JSON.stringify({ pid: 999_999, host: "foreign-pod", timestamp: 0 }));
    expect(isFileLockLive(path)).toBe(true);
    await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS);
    renameSync(path, `${path}.moved`);
    expect(isFileLockLive(path)).toBe(false);
    renameSync(`${path}.moved`, path);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS);
    await (await acquireFileLockLease(path, { timeoutMs: 0 })).release();
  });

  it("restarts the observation window if the contender's clock moves backward", async () => {
    vi.useFakeTimers();
    const path = leasePath();
    writeFileSync(path, JSON.stringify({ pid: 999_999, host: "foreign-pod", timestamp: 1e30 }));
    expect(isFileLockLive(path)).toBe(true);
    await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS - 1);
    expect(isFileLockLive(path)).toBe(true);
    vi.setSystemTime(Date.now() - 3_600_000);
    expect(isFileLockLive(path)).toBe(true);
    await vi.advanceTimersByTimeAsync(FILE_LOCK_STALE_MS - 1);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    await vi.advanceTimersByTimeAsync(1);
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
    // Every process must observe its own full window before it can reap.
    expect(await commandInChild(a, "observe")).toBe(true);
    expect(await commandInChild(b, "acquire")).toMatch(/^Timed out/);
    await Promise.all(
      [a, b].map((child) => commandInChild(child, { advanceMs: FILE_LOCK_STALE_MS })),
    );
    const newcomer = await contender(path, "new-pod");
    expect(await commandInChild(newcomer, "acquire")).toMatch(/^Timed out/);
    const results = await Promise.all([commandInChild(a, "acquire"), commandInChild(b, "acquire")]);
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
      expect(await commandInChild(child, "acquire")).toBe("acquired");
      child.kill("SIGSTOP");
      vi.useFakeTimers({ toFake: ["Date"] });
      expect(isFileLockLive(path)).toBe(true);
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

  it.each([
    ["partial lock metadata", -3_600_000],
    ["partial lock metadata", 3_600_000],
    ['{"timestamp":null}', -3_600_000],
    ['{"timestamp":null}', 3_600_000],
  ])(
    "observes malformed locks for the grace period regardless of mtime (%s, %s)",
    async (contents, skew) => {
      vi.useFakeTimers();
      const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
      tempDirs.push(tempDir);
      const lockPath = join(tempDir, "malformed.lock");
      writeFileSync(lockPath, contents);
      const mtime = new Date(Date.now() + skew);
      utimesSync(lockPath, mtime, mtime);

      expect(isFileLockLive(lockPath, { malformedGraceMs: 5_000 })).toBe(true);
      await expect(
        withFileLock(lockPath, () => "unreachable", {
          malformedGraceMs: 5_000,
          timeoutMs: 0,
          retryDelayMs: 1,
        }),
      ).rejects.toThrow("Timed out waiting for stream lock");

      await vi.advanceTimersByTimeAsync(4_999);
      expect(isFileLockLive(lockPath)).toBe(true);
      await vi.advanceTimersByTimeAsync(1);
      expect(isFileLockLive(lockPath)).toBe(false);
      await expect(
        withFileLock(lockPath, () => "acquired", { malformedGraceMs: 5_000 }),
      ).resolves.toBe("acquired");
    },
  );

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

  it("restarts malformed grace when a partial write changes", async () => {
    vi.useFakeTimers();
    const path = leasePath();
    writeFileSync(path, '{"pid":');
    expect(isFileLockLive(path)).toBe(true);
    await vi.advanceTimersByTimeAsync(4_000);
    writeFileSync(path, '{"pid":999999,');
    expect(isFileLockLive(path)).toBe(true);
    await vi.advanceTimersByTimeAsync(4_999);
    await expect(acquireFileLockLease(path, { timeoutMs: 0 })).rejects.toThrow("Timed out");
    await vi.advanceTimersByTimeAsync(1);
    await (await acquireFileLockLease(path, { timeoutMs: 0 })).release();
  });
});
