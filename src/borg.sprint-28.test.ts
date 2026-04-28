// End-to-end integration for Sprint 28: the offline maintenance scheduler
// and the retrieval-confidence signal feeding path selection. Opens a real
// Borg (with FakeLLMClient + deterministic embeddings), exercises the wiring
// end-to-end, and asserts observable behavior (stream entries, decisions).
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { Borg } from "./borg.js";
import { DEFAULT_CONFIG } from "./config/index.js";
import type { EmbeddingClient } from "./embeddings/index.js";
import { FakeLLMClient } from "./llm/index.js";
import { ManualClock } from "./util/clock.js";

class ConstantEmbeddingClient implements EmbeddingClient {
  async embed(): Promise<Float32Array> {
    return Float32Array.from([1, 0, 0, 0]);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map(() => Float32Array.from([1, 0, 0, 0]));
  }
}

describe("Sprint 28 integration", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  function configWith(overrides: {
    dataDir: string;
    maintenance?: Partial<typeof DEFAULT_CONFIG.maintenance>;
    perceptionMode?: "problem_solving" | "relational" | "reflective" | "idle";
  }) {
    return {
      ...DEFAULT_CONFIG,
      dataDir: overrides.dataDir,
      perception: {
        ...DEFAULT_CONFIG.perception,
        useLlmFallback: false,
        modeWhenLlmAbsent: overrides.perceptionMode ?? "idle",
      },
      embedding: {
        ...DEFAULT_CONFIG.embedding,
        dims: 4,
      },
      maintenance: {
        ...DEFAULT_CONFIG.maintenance,
        ...(overrides.maintenance ?? {}),
      },
    };
  }

  it("exposes a maintenance scheduler on the facade without starting it", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-s28-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: configWith({ dataDir: tempDir }),
      clock: new ManualClock(1_000_000),
      embeddingDimensions: 4,
      embeddingClient: new ConstantEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect(borg.maintenance.scheduler).toBeDefined();
      // Default config has maintenance.enabled = true; the scheduler exists in
      // an enabled-but-unstarted state until a runtime (daemon) calls start().
      expect(borg.maintenance.scheduler.isEnabled()).toBe(true);
    } finally {
      await borg.close();
    }
  });

  it("runs a light cadence tick and emits a dream_report to the stream", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-s28-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: configWith({
        dataDir: tempDir,
        maintenance: {
          enabled: true,
          lightIntervalMs: 60_000,
          heavyIntervalMs: 600_000,
          // Curator is LLM-free (heat decay / archival only) so the tick
          // runs to completion without scripting any Anthropic responses.
          lightProcesses: ["curator"],
          heavyProcesses: ["self-narrator"],
        },
      }),
      clock: new ManualClock(1_000_000),
      embeddingDimensions: 4,
      embeddingClient: new ConstantEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const tick = await borg.maintenance.scheduler.tick("light");

      expect(tick.status).toBe("ok");
      expect(tick.cadence).toBe("light");
      expect(tick.processes).toEqual(["curator"]);
      expect(tick.result).not.toBeNull();
      expect(tick.result?.errors).toEqual([]);

      const streamEntries = borg.stream.tail(10);
      const dreamReport = streamEntries.find((entry) => entry.kind === "dream_report");

      expect(dreamReport).toBeDefined();
      const content = dreamReport?.content as { processes: string[] };
      expect(content.processes).toContain("curator");
    } finally {
      await borg.close();
    }
  });

  it("reports skipped_busy when the isBusy hook is active", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-s28-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: configWith({
        dataDir: tempDir,
        maintenance: {
          enabled: true,
          lightIntervalMs: 60_000,
          heavyIntervalMs: 600_000,
          lightProcesses: ["curator"],
          heavyProcesses: ["self-narrator"],
        },
      }),
      clock: new ManualClock(1_000_000),
      embeddingDimensions: 4,
      embeddingClient: new ConstantEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      // Replace the busy-check on the already-constructed scheduler so we can
      // force the busy-branch without actually holding a session lock. The
      // options object is private-ish; reaching in is fine for an integration test.
      const schedulerInternals = borg.maintenance.scheduler as unknown as {
        options: { isBusy?: () => boolean };
      };
      schedulerInternals.options.isBusy = () => true;

      const tick = await borg.maintenance.scheduler.tick("light");

      expect(tick.status).toBe("skipped_busy");
      expect(tick.result).toBeNull();

      // No dream_report should have been emitted -- maintenance was skipped.
      const dreamReports = borg.stream.tail(10).filter((entry) => entry.kind === "dream_report");
      expect(dreamReports).toHaveLength(0);
    } finally {
      await borg.close();
    }
  });

  it("runs light and heavy cadences in parallel without coalescing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-s28-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: configWith({
        dataDir: tempDir,
        maintenance: {
          enabled: true,
          lightIntervalMs: 60_000,
          heavyIntervalMs: 600_000,
          lightProcesses: ["curator"],
          // Empty heavy set → skipped_empty, which still proves the heavy
          // tick is scheduled independently of the light tick.
          heavyProcesses: [],
        },
      }),
      clock: new ManualClock(1_000_000),
      embeddingDimensions: 4,
      embeddingClient: new ConstantEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const [light, heavy] = await Promise.all([
        borg.maintenance.scheduler.tick("light"),
        borg.maintenance.scheduler.tick("heavy"),
      ]);

      expect(light.cadence).toBe("light");
      expect(light.status).toBe("ok");
      expect(heavy.cadence).toBe("heavy");
      expect(heavy.status).toBe("skipped_empty");
    } finally {
      await borg.close();
    }
  });

  it("routes a no-evidence turn to S2 via low retrieval confidence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-s28-"));
    tempDirs.push(tempDir);

    // problem_solving mode is the non-short-circuit path: idle forces S1
    // and reflective forces S2 without consulting confidence. problem_solving
    // lets the confidence signal actually decide.
    const borg = await Borg.open({
      config: configWith({ dataDir: tempDir, perceptionMode: "problem_solving" }),
      clock: new ManualClock(1_000_000),
      embeddingDimensions: 4,
      embeddingClient: new ConstantEmbeddingClient(),
      llmClient: new FakeLLMClient({
        responses: [
          // S2 planner call (converse tool-use) emits EmitTurnPlan.
          [
            {
              type: "tool_use",
              id: "toolu_plan_1",
              name: "EmitTurnPlan",
              input: {
                verification_steps: ["No retrieval evidence was found for this query."],
                tensions: [],
                voice_note: "Speak honestly about uncertainty.",
                uncertainty: "high",
                referenced_episode_ids: [],
                intents: [],
              },
            },
          ],
          // Finalizer text response.
          {
            text: "I do not have evidence on that yet.",
            input_tokens: 10,
            output_tokens: 5,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
    });

    try {
      const result = await borg.turn({
        userMessage: "Tell me about the thing we never discussed.",
      });

      // The deliberator exposes the chosen path on the result. With no
      // retrieved episodes, RetrievalConfidence.overall = 0, forcing S2.
      expect(result.path).toBe("system_2");
    } finally {
      await borg.close();
    }
  });

  it("keeps maintenance isBusy hook wired to session-lock state", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-s28-"));
    tempDirs.push(tempDir);

    const borg = await Borg.open({
      config: configWith({
        dataDir: tempDir,
        maintenance: {
          enabled: true,
          lightIntervalMs: 60_000,
          heavyIntervalMs: 600_000,
          lightProcesses: ["curator"],
          heavyProcesses: [],
        },
      }),
      clock: new ManualClock(1_000_000),
      embeddingDimensions: 4,
      embeddingClient: new ConstantEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      // Default isBusy (wired in open.ts) calls sessionLock.isHeld(). No turn
      // is running, so the lock file doesn't exist -- maintenance must run.
      const tick = await borg.maintenance.scheduler.tick("light");

      expect(tick.status).toBe("ok");
    } finally {
      await borg.close();
    }
  });
});
