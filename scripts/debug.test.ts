import { spawn } from "node:child_process";
import { cpSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterEach, beforeEach, describe, expect, it } from "vitest";

const maybeIt = process.env.BORG_SKIP_DEBUG_SPAWN === "1" ? it.skip : it;
const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "..");
const pnpmCommand = process.platform === "win32" ? "pnpm.cmd" : "pnpm";
const heuristicsGuardScriptPath = join(repoRoot, "scripts/heuristics-guard.ts");

async function runHeuristicsGuardScript(cwd: string): Promise<{
  code: number | null;
  stdout: string;
  stderr: string;
}> {
  return new Promise((resolve, reject) => {
    const child = spawn(pnpmCommand, ["exec", "tsx", heuristicsGuardScriptPath], {
      cwd,
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
  let guardRoot: string;

  beforeEach(() => {
    guardRoot = mkdtempSync(join(tmpdir(), "borg-heuristics-guard-"));
    // The guard scans paths relative to cwd; keep every fixture and its source scan isolated.
    cpSync(join(repoRoot, "src"), join(guardRoot, "src"), { recursive: true });
    cpSync(join(repoRoot, "package.json"), join(guardRoot, "package.json"));
    symlinkSync(join(repoRoot, "node_modules"), join(guardRoot, "node_modules"), "junction");
  });

  afterEach(() => {
    rmSync(guardRoot, { recursive: true, force: true });
  });

  it("fails broad frame-anomaly degraded fallback patterns", async () => {
    const fixturePath = join(guardRoot, "src/cognition/frame-anomaly/heuristics-guard-fixture.ts");
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
        const child = spawn(pnpmCommand, ["exec", "tsx", heuristicsGuardScriptPath], {
          cwd: guardRoot,
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
    const fixturePath = join(guardRoot, "src/cognition/recall-guard-fixture.ts");
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
        const child = spawn(pnpmCommand, ["exec", "tsx", heuristicsGuardScriptPath], {
          cwd: guardRoot,
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
    const fixturePath = join(guardRoot, "src/cognition/retrieval-guard-fixture.ts");
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
        const child = spawn(pnpmCommand, ["exec", "tsx", heuristicsGuardScriptPath], {
          cwd: guardRoot,
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
    const fixturePath = join(guardRoot, "src/cognition/alias-guard-fixture.ts");
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
        const child = spawn(pnpmCommand, ["exec", "tsx", heuristicsGuardScriptPath], {
          cwd: guardRoot,
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
    const fixturePath = join(guardRoot, "src/offline/label-coverage-guard-fixture.ts");
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
        const child = spawn(pnpmCommand, ["exec", "tsx", heuristicsGuardScriptPath], {
          cwd: guardRoot,
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
    const fixturePath = join(guardRoot, "src/offline/nested-label-coverage-guard-fixture.ts");
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
      const result = await runHeuristicsGuardScript(guardRoot);

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
    const fixturePath = join(guardRoot, "src/offline/helper-label-coverage-guard-fixture.ts");
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
      const result = await runHeuristicsGuardScript(guardRoot);

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("model-facing memory serializers missing disclosure labels");
      expect(result.stderr).toContain("helper-label-coverage-guard-fixture.ts");
      expect(result.stderr).toContain("narrative");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);

  it("fails non-labeling raw record-copy passthroughs in model-facing payloads", async () => {
    const fixturePath = join(guardRoot, "src/offline/raw-record-label-coverage-fixture.ts");
    writeFileSync(
      fixturePath,
      [
        "function serializableRecord(value: unknown): unknown {",
        "  return value;",
        "}",
        "export function buildTargetPrompt(loaded: { target: unknown }) {",
        "  return JSON.stringify({",
        "    target: serializableRecord(loaded.target),",
        "  });",
        "}",
        "export function buildReviewPrompt(item: unknown, payload: unknown) {",
        "  return JSON.stringify({",
        "    review: serializableRecord(item),",
        "    overseer_flag: serializableRecord(payload),",
        "  });",
        "}",
        "",
      ].join("\n"),
    );

    try {
      const result = await runHeuristicsGuardScript(guardRoot);

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("model-facing memory serializers missing disclosure labels");
      expect(result.stderr).toContain("raw-record-label-coverage-fixture.ts");
      expect(result.stderr).toContain("target");
      expect(result.stderr).toContain("review");
      expect(result.stderr).toContain("overseer_flag");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);

  it("allows labeled serializers, inline labeled objects, fallback record copies, and live-turn text", async () => {
    const fixturePath = join(guardRoot, "src/offline/label-coverage-negative-fixture.ts");
    writeFileSync(
      fixturePath,
      [
        "declare function serializableRecord(value: unknown): unknown;",
        "declare function serializableRecordWithFallbackDisclosure(value: unknown): unknown;",
        "declare function semanticNodePromptPayload(node: unknown, labels: unknown): unknown;",
        "declare function memoryDisclosurePayloadFields(label: unknown): Record<string, unknown>;",
        "declare function sharedStateMemoryDisclosureLabel(entry: unknown): unknown;",
        "export function buildNodePrompt(nodes: unknown[], labels: unknown) {",
        "  return JSON.stringify({",
        "    nodes: nodes.map((node) => serializableRecord(semanticNodePromptPayload(node, labels))),",
        "  });",
        "}",
        "export function buildInlinePrompt(id: string, source_episode_ids: string[], label: unknown) {",
        "  return JSON.stringify({",
        "    row: serializableRecord({",
        "      id,",
        "      source_episode_ids,",
        "      ...memoryDisclosurePayloadFields(label),",
        "    }),",
        "  });",
        "}",
        "export function buildFallbackPrompt(input: { target: unknown }) {",
        "  return JSON.stringify({",
        "    target: serializableRecordWithFallbackDisclosure(input.target),",
        "  });",
        "}",
        "export function buildLiveTurnPrompt(text: string, responseText: string, stream_entry_id: string) {",
        "  return JSON.stringify({",
        "    current_user_turn: { stream_entry_id, text },",
        "    assistant_response: { stream_entry_id, text: responseText },",
        "  });",
        "}",
        "export function toSharedStatePromptSummaryEntry(entry: {",
        "  id: string;",
        "  state_key: string;",
        "  text: string;",
        "}) {",
        "  return {",
        "    id: entry.id,",
        "    state_key: entry.state_key,",
        "    text: entry.text,",
        "    canonicalizes_ids_count: 1,",
        "    ...memoryDisclosurePayloadFields(sharedStateMemoryDisclosureLabel(entry)),",
        "  };",
        "}",
        "",
      ].join("\n"),
    );

    try {
      const result = await runHeuristicsGuardScript(guardRoot);

      expect(result.code).toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toBe("");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);

  it("fails unlabeled shared-state text rows in prompt serializers", async () => {
    const fixturePath = join(guardRoot, "src/cognition/shared-state-label-coverage-fixture.ts");
    writeFileSync(
      fixturePath,
      [
        "export function toSharedStatePromptSummaryEntry(entry: {",
        "  id: string;",
        "  state_key: string;",
        "  text: string;",
        "}) {",
        "  return {",
        "    id: entry.id,",
        "    state_key: entry.state_key,",
        "    text: entry.text,",
        "    canonicalizes_ids_count: 1,",
        "  };",
        "}",
        "export function compactSharedStateEntryForPrompt(entry: {",
        "  id: string;",
        "  kind: string;",
        "  text: string;",
        "  canonicalizes: unknown;",
        "}) {",
        "  return {",
        "    id: entry.id,",
        "    kind: entry.kind,",
        "    text: entry.text,",
        "    canonicalizes: entry.canonicalizes,",
        "  };",
        "}",
        "",
      ].join("\n"),
    );

    try {
      const result = await runHeuristicsGuardScript(guardRoot);

      expect(result.code).not.toBe(0);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("model-facing memory serializers missing disclosure labels");
      expect(result.stderr).toContain("shared-state-label-coverage-fixture.ts");
      expect(result.stderr).toContain("text");
    } finally {
      rmSync(fixturePath, { force: true });
    }
  }, 30_000);
});
