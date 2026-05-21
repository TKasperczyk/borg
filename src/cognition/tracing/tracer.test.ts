import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { Borg } from "../../borg.js";
import { DEFAULT_CONFIG } from "../../config/index.js";
import { FakeEmbeddingClient } from "../../embeddings/index.js";
import { FakeLLMClient, createFakeEmitAnswerResponse } from "../../llm/test-support/fake-client.js";
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
    tracer.emit("recency.completed", {
      turnId: "turn_contract",
      messageCount: 0,
      sourceEntryIds: [],
    });
    return;
  }

  tracer.emit("recency.completed", {
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
      event: "recency.completed",
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

    tracer.emit("retrieval.started", {
      turnId: "turn_1",
      query: "pgvector drift",
      options: {
        limit: 3,
      },
    });
    tracer.emit("retrieval.completed", {
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
      "retrieval.started",
      "retrieval.completed",
    ]);
  });

  it("supports degraded-mode observability events", () => {
    const tempDir = createTempDir();
    const tracePath = join(tempDir, "degraded.jsonl");
    const tracer = new JsonlTracer({
      path: tracePath,
      clock: new FixedClock(321),
    });

    tracer.emit("perception.classifier.degraded", {
      turnId: "turn_degraded",
      classifier: "affective_signal",
      reason: "llm_unavailable",
    });
    tracer.emit("retrieval.degraded", {
      turnId: "turn_degraded",
      subsystem: "open_questions",
      reason: "embedding_unavailable",
    });
    tracer.emit("working_memory.degraded", {
      turnId: "turn_degraded",
      subsystem: "pending_actions",
      reason: "non_action",
    });

    expect(readTraceEvents(tracePath).map((event) => event.event)).toEqual([
      "perception.classifier.degraded",
      "retrieval.degraded",
      "working_memory.degraded",
    ]);
  });

  it("keeps NoopTracer inert", () => {
    const tempDir = createTempDir();
    const tracePath = join(tempDir, "noop.jsonl");
    const tracer: TurnTracer = new NoopTracer();

    expect(tracer.enabled).toBe(false);
    expect(tracer.includePayloads).toBe(false);
    expect(
      tracer.emit("llm_call.started", {
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
    tracer.emit("deliberation.plan.completed", {
      turnId: "turn_env",
      success: true,
    });

    const event = readTraceEvents(tracePath)[0];
    expect(event).toMatchObject({
      ts: 500,
      turnId: "turn_env",
      event: "deliberation.plan.completed",
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
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_entity",
              name: "EmitEntityExtraction",
              input: { entities: ["pgvector"] },
            },
          ],
        },
        {
          text: "",
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_mode",
              name: "EmitModeDetection",
              input: { mode: "reflective", is_operational: false },
            },
          ],
        },
        {
          text: "",
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_temporal",
              name: "EmitTemporalCue",
              input: { has_cue: false },
            },
          ],
        },
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
                intents: [],
              },
            },
          ],
        },
        createFakeEmitAnswerResponse("Check the operator class first.", {
          inputTokens: 12,
          outputTokens: 6,
        }),
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
          llmEnabled: true,
        },
        affective: {
          ...DEFAULT_CONFIG.affective,
          llmEnabled: false,
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
      "recency.completed",
      "perception.started",
      "perception.classifier.degraded",
      "perception.completed",
      "llm_call.started",
      "llm_call.completed",
      "frame_anomaly.completed",
      "llm_call.started",
      "llm_call.started",
      "llm_call.started",
      "llm_call.completed",
      "llm_call.completed",
      "extraction.actions.completed",
      "llm_call.completed",
      "extraction.goals.completed",
      "retrieval.started",
      "llm_call.started",
      "llm_call.completed",
      "retrieval.degraded",
      "retrieval.completed",
      "evidence_ledger.completed",
      "deliberation.contradiction_routing.completed",
      "deliberation.path.completed",
      "deliberation.planner_ledger.completed",
      "llm_call.started",
      "llm_call.completed",
      "deliberation.plan.completed",
      "deliberation.plan_persistence.completed",
      "llm_call.started",
      "llm_call.completed",
      "finalizer.completed",
      "commitment_check.completed",
      "closure_response_guard.completed",
      "reflection.completed",
      "action_archive_scan.completed",
    ]);
    expect(events.find((event) => event.event === "deliberation.plan_persistence.completed")).toMatchObject({
      streamEntryId: expect.stringMatching(/^strm_/),
    });
    expect(
      events
        .filter((event) => event.event === "perception.classifier.degraded")
        .map((event) => event.classifier),
    ).toEqual(["affective_signal"]);
    expect(events.find((event) => event.event === "finalizer.completed")).toMatchObject({
      path: "system_2",
      decision: "answer",
    });
  });
});
