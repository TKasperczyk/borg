import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const maybeIt = process.env.BORG_SKIP_DEBUG_SPAWN === "1" ? it.skip : it;
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const pnpmCommand = process.platform === "win32" ? "pnpm.cmd" : "pnpm";

describe("debug script", () => {
  maybeIt(
    "runs to completion with the fake path",
    async () => {
      const result = await new Promise<{
        code: number | null;
        stdout: string;
        stderr: string;
      }>((resolve, reject) => {
        const child = spawn(pnpmCommand, ["exec", "tsx", "scripts/debug.ts"], {
          cwd: repoRoot,
          env: {
            ...process.env,
            FORCE_COLOR: "0",
          },
          stdio: ["ignore", "pipe", "pipe"],
        });
        let stdout = "";
        let stderr = "";
        const timeout = setTimeout(() => {
          child.kill("SIGTERM");
          reject(new Error("debug script exceeded 30s timeout"));
        }, 30_000);

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
      expect(result.stdout).toContain("Using LLM: fake, Embeddings: fake");
      expect(result.stdout).toContain("=== Phase 1. Setup & self ===");
      expect(result.stdout).toContain("=== Phase 2. Stream + extraction ===");
      expect(result.stdout).toContain("=== Phase 5. Dream cycle ===");
      expect(result.stdout).toContain("=== Phase 8. Maintenance scheduler (Sprint 28) ===");
      expect(result.stdout).toContain("=== Phase 9. Retrieval confidence snapshot (Sprint 28) ===");
      expect(result.stdout).toContain("=== Phase 10. Inspection footer ===");
      expect(result.stdout).toContain("debug run complete");
    },
    30_000,
  );
});
