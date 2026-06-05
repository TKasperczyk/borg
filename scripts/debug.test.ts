import { spawn } from "node:child_process";
import { rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

const maybeIt = process.env.BORG_SKIP_DEBUG_SPAWN === "1" ? it.skip : it;
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const pnpmCommand = process.platform === "win32" ? "pnpm.cmd" : "pnpm";

async function runHeuristicsGuardScript(): Promise<{
  code: number | null;
  stdout: string;
  stderr: string;
}> {
  return new Promise((resolve, reject) => {
    const child = spawn(pnpmCommand, ["exec", "tsx", "scripts/heuristics-guard.ts"], {
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
      reject(new Error("heuristics guard exceeded 30s timeout"));
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
}

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

describe("heuristics guard", () => {
  it("fails broad frame-anomaly degraded fallback patterns", async () => {
    const fixturePath = join(repoRoot, "src/cognition/frame-anomaly/heuristics-guard-fixture.ts");
    writeFileSync(
      fixturePath,
      'const fixture = [{ pattern: "as an ai", kind: "assistant_self_claim_in_user_role" }];\n',
    );

    try {
      const result = await new Promise<{
        code: number | null;
        stdout: string;
        stderr: string;
      }>((resolve, reject) => {
        const child = spawn(pnpmCommand, ["exec", "tsx", "scripts/heuristics-guard.ts"], {
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
          reject(new Error("heuristics guard exceeded 30s timeout"));
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

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("frame-anomaly broad degraded fallback marker");
      expect(result.stderr).toContain("as an ai");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);

  it("fails disclosure search calls from cognition paths", async () => {
    const fixturePath = join(repoRoot, "src/cognition/recall-guard-fixture.ts");
    writeFileSync(
      fixturePath,
      [
        "async function unsafeRecall(pipeline: {",
        "  searchWithContextForDisclosure: (query: string) => Promise<unknown>; ",
        "}) {",
        '  return pipeline.searchWithContextForDisclosure("private planning");',
        "}",
        "",
      ].join("\n"),
    );

    try {
      const result = await new Promise<{
        code: number | null;
        stdout: string;
        stderr: string;
      }>((resolve, reject) => {
        const child = spawn(pnpmCommand, ["exec", "tsx", "scripts/heuristics-guard.ts"], {
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
          reject(new Error("heuristics guard exceeded 30s timeout"));
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

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("disclosure search symbols in cognition/recall paths");
      expect(result.stderr).toContain("searchWithContextForDisclosure");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);

  it("fails disclosure-suffixed search calls from cognition paths", async () => {
    const fixturePath = join(repoRoot, "src/cognition/retrieval-guard-fixture.ts");
    writeFileSync(
      fixturePath,
      [
        "async function unsafeRecall(pipeline: {",
        "  searchEpisodesForDisclosure: (query: string) => Promise<unknown>; ",
        "}) {",
        '  return pipeline.searchEpisodesForDisclosure("private planning");',
        "}",
        "",
      ].join("\n"),
    );

    try {
      const result = await new Promise<{
        code: number | null;
        stdout: string;
        stderr: string;
      }>((resolve, reject) => {
        const child = spawn(pnpmCommand, ["exec", "tsx", "scripts/heuristics-guard.ts"], {
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
          reject(new Error("heuristics guard exceeded 30s timeout"));
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

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("disclosure search symbols in cognition/recall paths");
      expect(result.stderr).toContain("searchEpisodesForDisclosure");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);

  it("fails aliased disclosure search calls from cognition paths", async () => {
    const fixturePath = join(repoRoot, "src/cognition/alias-guard-fixture.ts");
    writeFileSync(
      fixturePath,
      [
        'import { filterEpisodesByAudience as f } from "../memory/episodic/audience-filter.js";',
        "",
        "function unsafeRecall(episodes: never[]) {",
        '  return f(episodes, null, "recall");',
        "}",
        "",
      ].join("\n"),
    );

    try {
      const result = await new Promise<{
        code: number | null;
        stdout: string;
        stderr: string;
      }>((resolve, reject) => {
        const child = spawn(pnpmCommand, ["exec", "tsx", "scripts/heuristics-guard.ts"], {
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
          reject(new Error("heuristics guard exceeded 30s timeout"));
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

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("disclosure search symbols in cognition/recall paths");
      expect(result.stderr).toContain("f -> filterEpisodesByAudience");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);

  it("fails model-facing memory rows without disclosure labels", async () => {
    const fixturePath = join(repoRoot, "src/offline/label-coverage-guard-fixture.ts");
    writeFileSync(
      fixturePath,
      [
        "type Episode = { id: string; narrative: string };",
        "export function buildFixturePrompt(episode: Episode) {",
        "  return JSON.stringify({",
        '    task: "fixture",',
        "    episode: {",
        "      id: episode.id,",
        "      narrative: episode.narrative,",
        "    },",
        "  });",
        "}",
        "",
      ].join("\n"),
    );

    try {
      const result = await new Promise<{
        code: number | null;
        stdout: string;
        stderr: string;
      }>((resolve, reject) => {
        const child = spawn(pnpmCommand, ["exec", "tsx", "scripts/heuristics-guard.ts"], {
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
          reject(new Error("heuristics guard exceeded 30s timeout"));
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

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("model-facing memory serializers missing disclosure labels");
      expect(result.stderr).toContain("label-coverage-guard-fixture.ts");
      expect(result.stderr).toContain("narrative");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);

  it("fails nested private rows even when an ancestor has disclosure labels", async () => {
    const fixturePath = join(repoRoot, "src/offline/nested-label-coverage-guard-fixture.ts");
    writeFileSync(
      fixturePath,
      [
        "type Episode = { id: string; narrative: string };",
        "export function buildFixturePrompt(episode: Episode) {",
        "  return JSON.stringify({",
        '    disclosure: "disclosure_class=unknown",',
        "    disclosure_label: {",
        '      disclosure_class: "unknown",',
        "      origin_audience_entity_ids: [],",
        "      private_to_entity_ids: [],",
        "      public_to_entity_ids: [],",
        "    },",
        "    episode: {",
        "      id: episode.id,",
        "      narrative: episode.narrative,",
        "    },",
        "  });",
        "}",
        "",
      ].join("\n"),
    );

    try {
      const result = await runHeuristicsGuardScript();

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("model-facing memory serializers missing disclosure labels");
      expect(result.stderr).toContain("nested-label-coverage-guard-fixture.ts");
      expect(result.stderr).toContain("narrative");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);

  it("fails unlabeled private rows returned by serializer helper functions", async () => {
    const fixturePath = join(repoRoot, "src/offline/helper-label-coverage-guard-fixture.ts");
    writeFileSync(
      fixturePath,
      [
        "type Episode = { id: string; narrative: string };",
        "function episodePayload(episode: Episode) {",
        "  return {",
        "    id: episode.id,",
        "    narrative: episode.narrative,",
        "  };",
        "}",
        "export function buildFixturePrompt(episode: Episode) {",
        "  return JSON.stringify({",
        "    episode: episodePayload(episode),",
        "  });",
        "}",
        "",
      ].join("\n"),
    );

    try {
      const result = await runHeuristicsGuardScript();

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("model-facing memory serializers missing disclosure labels");
      expect(result.stderr).toContain("helper-label-coverage-guard-fixture.ts");
      expect(result.stderr).toContain("narrative");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);
});
