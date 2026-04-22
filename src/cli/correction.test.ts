import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { Borg } from "../borg.js";
import { DEFAULT_CONFIG } from "../config/index.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/index.js";
import { FixedClock } from "../util/clock.js";
import { runCli } from "./app.js";

class CliEmbeddingClient implements EmbeddingClient {
  async embed(): Promise<Float32Array> {
    return Float32Array.from([1, 0, 0, 0]);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map(() => Float32Array.from([1, 0, 0, 0]));
  }
}

function createOutputBuffer() {
  let value = "";

  return {
    stream: {
      write(chunk: string) {
        value += chunk;
        return true;
      },
    },
    read() {
      return value;
    },
  };
}

function openCorrectionBorg(tempDir: string, llm = new FakeLLMClient()) {
  return Borg.open({
    config: {
      ...DEFAULT_CONFIG,
      dataDir: tempDir,
      defaultUser: "Sam",
      embedding: {
        ...DEFAULT_CONFIG.embedding,
        dims: 4,
      },
      perception: {
        useLlmFallback: false,
        modeWhenLlmAbsent: "problem_solving",
      },
      anthropic: {
        auth: "api-key",
        apiKey: "test",
        models: {
          cognition: "sonnet",
          background: "haiku",
          extraction: "haiku",
        },
      },
    },
    clock: new FixedClock(1_000),
    embeddingDimensions: 4,
    embeddingClient: new CliEmbeddingClient(),
    llmClient: llm,
  });
}

describe("cli correction commands", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("smokes correction why", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openCorrectionBorg(tempDir);
    const value = borg.self.values.add({
      label: "clarity",
      description: "Prefer explicit state.",
      priority: 3,
      provenance: { kind: "manual" },
    });
    await borg.close();

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();
    const exitCode = await runCli(["node", "borg", "correction", "why", value.id], {
      stdout: stdout.stream,
      stderr: stderr.stream,
      openBorg: async () => openCorrectionBorg(tempDir),
    });

    expect(exitCode).toBe(0);
    expect(JSON.parse(stdout.read())).toMatchObject({
      target_type: "value",
      record: {
        id: value.id,
      },
    });
    expect(stderr.read()).toBe("");
  });

  it("smokes correction correct and review resolve --accept", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openCorrectionBorg(tempDir);
    const value = borg.self.values.add({
      label: "memory",
      description: "Keep a usable trace.",
      priority: 4,
      provenance: { kind: "manual" },
    });
    await borg.close();

    const correctOut = createOutputBuffer();
    const correctErr = createOutputBuffer();
    const correctExit = await runCli(
      [
        "node",
        "borg",
        "correction",
        "correct",
        value.id,
        "--patch",
        '{"description":"Keep a durable trace."}',
      ],
      {
        stdout: correctOut.stream,
        stderr: correctErr.stream,
        openBorg: async () => openCorrectionBorg(tempDir),
      },
    );

    const queued = JSON.parse(correctOut.read()) as { id: number; kind: string };

    expect(correctExit).toBe(0);
    expect(queued.kind).toBe("correction");

    const resolveOut = createOutputBuffer();
    const resolveErr = createOutputBuffer();
    const resolveExit = await runCli(
      ["node", "borg", "review", "resolve", String(queued.id), "--accept"],
      {
        stdout: resolveOut.stream,
        stderr: resolveErr.stream,
        openBorg: async () => openCorrectionBorg(tempDir),
      },
    );

    expect(resolveExit).toBe(0);
    expect(JSON.parse(resolveOut.read())).toMatchObject({
      id: queued.id,
      resolution: "accept",
    });
    expect(correctErr.read()).toBe("");
    expect(resolveErr.read()).toBe("");
  });

  it("smokes correction forget", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openCorrectionBorg(tempDir);
    const value = borg.self.values.add({
      label: "humility",
      description: "Prefer revisability.",
      priority: 2,
      provenance: { kind: "manual" },
    });
    await borg.close();

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();
    const exitCode = await runCli(["node", "borg", "correction", "forget", value.id], {
      stdout: stdout.stream,
      stderr: stderr.stream,
      openBorg: async () => openCorrectionBorg(tempDir),
    });

    expect(exitCode).toBe(0);
    expect(JSON.parse(stdout.read())).toMatchObject({
      id: value.id,
      archived: true,
    });
    expect(stderr.read()).toBe("");
  });

  it("smokes correction about-me", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openCorrectionBorg(tempDir);
    borg.commitments.add({
      type: "boundary",
      directive: "Keep Sam informed",
      priority: 5,
      audience: "Sam",
      provenance: { kind: "manual" },
    });
    borg.social.recordInteraction("Sam", {
      provenance: { kind: "manual" },
      valence: 0.1,
    });
    await borg.close();

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();
    const exitCode = await runCli(["node", "borg", "correction", "about-me"], {
      stdout: stdout.stream,
      stderr: stderr.stream,
      openBorg: async () => openCorrectionBorg(tempDir),
    });

    const parsed = JSON.parse(stdout.read()) as {
      social_profile: { interaction_count: number };
      active_commitments: unknown[];
    };

    expect(exitCode).toBe(0);
    expect(parsed.social_profile.interaction_count).toBeGreaterThan(0);
    expect(parsed.active_commitments).toHaveLength(1);
    expect(stderr.read()).toBe("");
  });

  it("smokes correction events", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const borg = await openCorrectionBorg(tempDir);
    const value = borg.self.values.add({
      label: "grounding",
      description: "Prefer evidence.",
      priority: 6,
      provenance: { kind: "manual" },
    });
    await borg.close();

    const stdout = createOutputBuffer();
    const stderr = createOutputBuffer();
    const exitCode = await runCli(
      ["node", "borg", "correction", "events", "--record-type", "value", "--record-id", value.id],
      {
        stdout: stdout.stream,
        stderr: stderr.stream,
        openBorg: async () => openCorrectionBorg(tempDir),
      },
    );

    expect(exitCode).toBe(0);
    expect(JSON.parse(stdout.read())).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          record_type: "value",
          record_id: value.id,
        }),
      ]),
    );
    expect(stderr.read()).toBe("");
  });
});
