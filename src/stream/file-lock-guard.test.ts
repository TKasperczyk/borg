import { fork, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { FILE_LOCK_GUARD_SUFFIX } from "./file-lock-guard.js";

describe("file-lock-guard", () => {
  const children: ChildProcess[] = [];
  const directories: string[] = [];

  afterEach(async () => {
    for (const child of children.splice(0)) {
      if (child.exitCode === null && child.signalCode === null) {
        const exited = once(child, "exit");
        child.kill("SIGKILL");
        await exited;
      }
    }
    for (const dir of directories.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  async function contender(path: string): Promise<ChildProcess> {
    const child = fork(
      new URL("./test-support/file-lock-guard-process.ts", import.meta.url),
      [path],
      { execArgv: ["--import", "tsx"], stdio: ["ignore", "ignore", "inherit", "ipc"] },
    );
    children.push(child);
    expect((await once(child, "message"))[0]).toBe("ready");
    return child;
  }

  async function acquire(child: ChildProcess): Promise<unknown> {
    const response = once(child, "message");
    child.send("acquire");
    return (await response)[0];
  }

  it.skipIf(process.platform === "win32")(
    "refuses a readable but non-writable companion in both processes, then admits one writer",
    async () => {
      const dir = mkdtempSync(join(tmpdir(), "file-lock-readonly-"));
      directories.push(dir);
      chmodSync(dir, 0o777);
      const path = join(dir, "lease.lock");
      const companion = `${path}${FILE_LOCK_GUARD_SUFFIX}`;
      // With root-run tests, this is 0644 owned by another uid, as on the PVC.
      // Without root, 0444 gives the same readable/non-writable SQLite fallback.
      writeFileSync(companion, "", { mode: process.getuid?.() === 0 ? 0o644 : 0o444 });
      const [a, b] = await Promise.all([contender(path), contender(path)]);
      const denied = await Promise.all([acquire(a), acquire(b)]);
      expect(denied).toEqual([
        `File lock guard is not writable at ${companion}`,
        `File lock guard is not writable at ${companion}`,
      ]);

      chmodSync(companion, 0o666);
      const results = await Promise.all([acquire(a), acquire(b)]);
      expect(results.sort()).toEqual(["acquired", "busy"]);
    },
  );
});
