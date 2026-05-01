import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { Borg } from "../../borg.js";
import { DEFAULT_CONFIG } from "../../config/index.js";
import { FakeEmbeddingClient } from "../../embeddings/index.js";
import { FakeLLMClient } from "../../llm/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { JsonlTracer, NoopTracer, createTurnTracer, type TurnTracer } from "./tracer.js";

type TraceEvent = {
  ts: number;
  turnId: string;
  event: string;
  [key: string]: unknown;
};

function readTraceEvents(path: string): TraceEvent[] {
  return readFileSync(path, "utf8")
    .trim()
    .split(/\r?\n/)
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as TraceEvent);
}

function emitContractEvent(tracer: TurnTracer): void {
  if (!tracer.enabled) {
    tracer.emit("recency_compiled", {
      turnId: "turn_contract",
      messageCount: 0,
      sourceEntryIds: [],
    });
    return;
  }

  tracer.emit("recency_compiled", {
    turnId: "turn_contract",
    messageCount: 0,
    sourceEntryIds: [],
  });
}

describe("TurnTracer", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  function createTempDir(): string {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-trace-"));
    tempDirs.push(tempDir);
    return tempDir;
  }

  it("supports the minimal structured emit contract", () => {
    const tempDir = createTempDir();
    const tracePath = join(tempDir, "trace.jsonl");
    const tracer = new JsonlTracer({
      path: tracePath,
      clock: new FixedClock(42),
    });

    expect(() => emitContractEvent(new NoopTracer())).not.toThrow();
    expect(() => emitContractEvent(tracer)).not.toThrow();

    const events = readTraceEvents(tracePath);
    expect(events).toHaveLength(1);
    expect(events[0]).toMatchObject({
      ts: 42,
      turnId: "turn_contract",
      event: "recency_compiled",
      messageCount: 0,
      sourceEntryIds: [],
    });
    expect(typeof events[0]?.wallMs).toBe("number");
  });

  it("writes valid JSONL with turn correlation", () => {
    const tempDir = createTempDir();
    const tracePath = join(tempDir, "trace.jsonl");
    const tracer = new JsonlTracer({
      path: tracePath,
      clock: new FixedClock(123),
    });

    tracer.emit("retrieval_started", {
      turnId: "turn_1",
      query: "pgvector drift",
      options: {
        limit: 3,
      },
    });
    tracer.emit("retrieval_completed", {
      turnId: "turn_1",
      episodeCount: 0,
      semanticHits: 0,
      confidence: {
        overall: 0,
      },
    });

    const events = readTraceEvents(tracePath);

    expect(events).toHaveLength(2);
    expect(events.every((event) => event.ts === 123)).toBe(true);
    expect(new Set(events.map((event) => event.turnId))).toEqual(new Set(["turn_1"]));
    expect(events.map((event) => event.event)).toEqual([
      "retrieval_started",
      "retrieval_completed",
    ]);
  });

  it("supports degraded-mode observability events", () => {
    const tempDir = createTempDir();
    const tracePath = join(tempDir, "degraded.jsonl");
    const tracer = new JsonlTracer({
      path: tracePath,
      clock: new FixedClock(321),
    });

    tracer.emit("perception_classifier_degraded", {
      turnId: "turn_degraded",
      classifier: "affective_signal",
      reason: "llm_unavailable",
    });
    tracer.emit("retrieval_degraded", {
      turnId: "turn_degraded",
      subsystem: "open_questions",
      reason: "embedding_unavailable",
    });
    tracer.emit("working_memory_degraded", {
      turnId: "turn_degraded",
      subsystem: "pending_actions",
      reason: "non_action",
    });

    expect(readTraceEvents(tracePath).map((event) => event.event)).toEqual([
      "perception_classifier_degraded",
      "retrieval_degraded",
      "working_memory_degraded",
    ]);
  });

  it("keeps NoopTracer inert", () => {
    const tempDir = createTempDir();
    const tracePath = join(tempDir, "noop.jsonl");
    const tracer: TurnTracer = new NoopTracer();

    expect(tracer.enabled).toBe(false);
    expect(tracer.includePayloads).toBe(false);
    expect(
      tracer.emit("llm_call_started", {
        turnId: "turn_noop",
        label: "noop",
        model: "none",
        promptCharCount: 0,
        toolSchemas: [],
      }),
    ).toBeUndefined();
    expect(existsSync(tracePath)).toBe(false);
  });

  it("creates a JsonlTracer from BORG_TRACE env", () => {
    const tempDir = createTempDir();
    const tracePath = join(tempDir, "env-trace.jsonl");
    const tracer = createTurnTracer({
      env: {
        BORG_TRACE: tracePath,
        BORG_TRACE_PROMPTS: "1",
      },
      clock: new FixedClock(500),
    });

    expect(tracer.enabled).toBe(true);
    expect(tracer.includePayloads).toBe(true);
    tracer.emit("plan_extraction", {
      turnId: "turn_env",
      success: true,
    });

    const event = readTraceEvents(tracePath)[0];
    expect(event).toMatchObject({
      ts: 500,
      turnId: "turn_env",
      event: "plan_extraction",
      success: true,
    });
    expect(typeof event?.wallMs).toBe("number");
  });

  it("emits expected events in order for a full Borg turn", async () => {
    const tempDir = createTempDir();
    const tracePath = join(tempDir, "turn.jsonl");
    const clock = new ManualClock(1_000);
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 10,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "",
                verification_steps: [],
                tensions: [],
                voice_note: "stay concrete",
                referenced_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
                intents: [],
              },
            },
          ],
        },
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_search",
              name: "tool.episodic.search",
              input: {
                query: "pgvector",
                limit: 1,
              },
            },
          ],
          input_tokens: 11,
          output_tokens: 5,
          stop_reason: "tool_use",
        },
        {
          messageBlocks: [
            {
              type: "text",
              text: "Check the operator class first.",
            },
          ],
          input_tokens: 12,
          output_tokens: 6,
          stop_reason: "end_turn",
        },
        {
          text: "",
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_reflection",
              name: "EmitTurnReflection",
              input: {
                advanced_goals: [],
                procedural_outcomes: [],
                trait_demonstrations: [],
                intent_updates: [],
              },
            },
          ],
        },
      ],
    });
    const borg = await Borg.open({
      config: {
        ...DEFAULT_CONFIG,
        dataDir: tempDir,
        perception: {
          ...DEFAULT_CONFIG.perception,
          useLlmFallback: false,
          modeWhenLlmAbsent: "reflective",
        },
        embedding: {
          ...DEFAULT_CONFIG.embedding,
          dims: 4,
        },
      },
      clock,
      embeddingDimensions: 4,
      embeddingClient: new FakeEmbeddingClient(4),
      llmClient: llm,
      tracerPath: tracePath,
      liveExtraction: false,
    });

    try {
      const result = await borg.turn({
        userMessage: "I'm stuck again on pgvector embeddings",
        stakes: "medium",
      });

      expect(result.path).toBe("system_2");
    } finally {
      await borg.close();
    }

    const events = readTraceEvents(tracePath);

    expect(new Set(events.map((event) => event.turnId)).size).toBe(1);
    expect(events.map((event) => event.event)).toEqual([
      "recency_compiled",
      "perception_started",
      "perception_classifier_degraded",
      "perception_classifier_degraded",
      "perception_completed",
      "retrieval_started",
      "llm_call_started",
      "llm_call_response",
      "retrieval_completed",
      "path_selected",
      "llm_call_started",
      "llm_call_response",
      "plan_extraction",
      "plan_persisted",
      "llm_call_started",
      "llm_call_response",
      "tool_call_dispatched",
      "tool_call_completed",
      "llm_call_started",
      "llm_call_response",
      "commitment_check",
      "reflection_emitted",
    ]);
    expect(events.find((event) => event.event === "plan_persisted")).toMatchObject({
      streamEntryId: expect.stringMatching(/^strm_/),
    });
    expect(
      events
        .filter((event) => event.event === "perception_classifier_degraded")
        .map((event) => event.classifier),
    ).toEqual(["affective_signal", "temporal_cue"]);
    expect(events.find((event) => event.event === "tool_call_completed")).toMatchObject({
      toolName: "tool.episodic.search",
      success: true,
    });
  });
});
