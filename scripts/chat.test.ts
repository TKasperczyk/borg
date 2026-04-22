import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, describe, expect, it } from "vitest";

const maybeIt = process.env.BORG_SKIP_CHAT_SPAWN === "1" ? it.skip : it;
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const pnpmCommand = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop();

    if (dir !== undefined) {
      rmSync(dir, { recursive: true, force: true });
    }
  }
});

describe("chat script", () => {
  maybeIt(
    "runs a short fake repl session",
    async () => {
      const dataDir = mkdtempSync(join(tmpdir(), "borg-chat-"));
      tempDirs.push(dataDir);

      const result = await new Promise<{
        code: number | null;
        stdout: string;
        stderr: string;
      }>((resolve, reject) => {
        const child = spawn(pnpmCommand, ["exec", "tsx", "scripts/chat.ts", "--fakes"], {
          cwd: repoRoot,
          env: {
            ...process.env,
            BORG_DATA_DIR: dataDir,
            FORCE_COLOR: "0",
          },
          stdio: ["pipe", "pipe", "pipe"],
        });
        let stdout = "";
        let stderr = "";
        const timeout = setTimeout(() => {
          child.kill("SIGTERM");
          reject(new Error("chat script exceeded 30s timeout"));
        }, 30_000);

        child.stdin.end("/help\nhello\n/who\n/exit\n");
        child.stdout.on("data", (chunk: Buffer | string) => {
          stdout += chunk.toString();
        });
        child.stderr.on("data", (chunk: Buffer | string) => {
          stderr += chunk.toString();
        });
        child.on("error", (error) => {
          clearTimeout(timeout);
          reject(error);
        });
        child.on("close", (code) => {
          clearTimeout(timeout);
          resolve({
            code,
            stdout,
            stderr,
          });
        });
      });

      expect(result.code).toBe(0);
      expect(result.stderr).toBe("");
      expect(result.stdout).toContain("borg chat ready.");
      expect(result.stdout).toContain("Commands:");
      expect(result.stdout).toContain("borg >");
      expect(result.stdout).toContain("session=chat audience=user");
      expect(result.stdout).toContain("extracted: inserted=");
      expect(result.stdout).toContain("saved working memory");
    },
    30_000,
  );
});
