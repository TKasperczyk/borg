import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { Borg } from "../../borg.js";
import { DEFAULT_CONFIG } from "../../config/index.js";
import { FakeEmbeddingClient } from "../../embeddings/index.js";
import {
  FakeLLMClient,
  createFakeEmitAnswerResponse,
  createFakeStreamingResponse,
} from "../../llm/test-support/fake-client.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { createSessionId } from "../../util/ids.js";
import {
  JsonlTracer,
  NoopTracer,
  compositeTracer,
  createTurnTracer,
  type TurnTraceData,
  type TurnTracer,
} from "./tracer.js";

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

  it("strips payload-gated keys per child tracer", () => {
    const withoutPayloads: TurnTraceData[] = [];
    const withPayloads: TurnTraceData[] = [];
    const tracer = compositeTracer([
      {
        enabled: true,
        includePayloads: false,
        emit: (_event, data) => {
          withoutPayloads.push(data);
        },
      },
      {
        enabled: true,
        includePayloads: true,
        emit: (_event, data) => {
          withPayloads.push(data);
        },
      },
    ]);

    tracer.emit("evidence_ledger.built", {
      turnId: "turn_payload",
      prompt: "full prompt",
      response: "full response",
      ledger: { sections: [] },
      record: { id: "record" },
      rawToolInput: { raw: true },
      normalizedPayload: { normalized: true },
      original_response: "before",
      rewritten_response: "after",
      dropped_facets: [{ query: "hidden", priority: 1 }],
      description: "candidate text",
      candidate_description: "goal candidate text",
      description_excerpt: "action excerpt",
      skipped_promotions: [{ description_excerpt: "skipped goal" }],
      error: "stack-like detail",
      spans: [{ text: "hidden span" }],
      reason: "ordinary_reason",
      summary: "safe metadata",
    });

    expect(withPayloads[0]).toMatchObject({
      prompt: "full prompt",
      response: "full response",
      ledger: { sections: [] },
      candidate_description: "goal candidate text",
      description_excerpt: "action excerpt",
      skipped_promotions: [{ description_excerpt: "skipped goal" }],
      summary: "safe metadata",
    });
    expect(withoutPayloads[0]).toEqual({
      turnId: "turn_payload",
      reason: "ordinary_reason",
      summary: "safe metadata",
    });

    tracer.emit("turn.token", {
      turnId: "turn_payload",
      turn_id: "turn_payload",
      phase: "final",
      chunk_text: "hidden chunk",
      sequence: 1,
    });
    tracer.emit("turn.token.flush", {
      turnId: "turn_payload",
      turn_id: "turn_payload",
      phase: "final",
      full_text: "hidden full text",
    });

    expect(withPayloads[1]).toMatchObject({
      chunk_text: "hidden chunk",
      sequence: 1,
    });
    expect(withoutPayloads[1]).toMatchObject({
      turnId: "turn_payload",
      turn_id: "turn_payload",
      phase: "final",
      sequence: 1,
    });
    expect(withoutPayloads[1]).not.toHaveProperty("chunk_text");
    expect(withPayloads[2]).toMatchObject({
      full_text: "hidden full text",
    });
    expect(withoutPayloads[2]).toMatchObject({
      turnId: "turn_payload",
      turn_id: "turn_payload",
      phase: "final",
    });
    expect(withoutPayloads[2]).not.toHaveProperty("full_text");

    tracer.emit("closure_response_guard.completed", {
      turnId: "turn_spans",
      reason: "mixed_closure_observed",
      spans: [{ text: "kept", kind: "farewell", rationale: "mixed" }],
    });

    expect(withoutPayloads[3]).toMatchObject({
      reason: "mixed_closure_observed",
      spans: [{ text: "kept" }],
    });
  });

  it("strips payload-gated keys for a lone JsonlTracer without payloads", () => {
    const tempDir = createTempDir();
    const tracePath = join(tempDir, "jsonl-no-payloads.jsonl");
    const tracer = new JsonlTracer({
      path: tracePath,
      clock: new FixedClock(700),
      includePayloads: false,
    });

    tracer.emit("evidence_ledger.built", {
      turnId: "turn_jsonl_payload",
      turn_id: "turn_jsonl_payload",
      prompt: "full prompt",
      response: "full response",
      ledger: { sections: [] },
      chunk_text: "token chunk",
      full_text: "complete answer",
      rawToolInput: { text: "raw" },
      normalizedPayload: { text: "normalized" },
      original_response: "before",
      rewritten_response: "after",
      dropped_facets: [{ query: "hidden", priority: 1 }],
      description: "candidate text",
      candidate_description: "goal candidate text",
      description_excerpt: "action excerpt",
      skipped_promotions: [{ description_excerpt: "skipped goal" }],
      record: { id: "record" },
      error: "stack-like detail",
      spans: [{ text: "hidden span" }],
      reason: "ordinary_reason",
      summary: "safe metadata",
    });

    const event = readTraceEvents(tracePath)[0];

    expect(event).toMatchObject({
      ts: 700,
      turnId: "turn_jsonl_payload",
      event: "evidence_ledger.built",
      turn_id: "turn_jsonl_payload",
      reason: "ordinary_reason",
      summary: "safe metadata",
    });
    expect(event).not.toHaveProperty("prompt");
    expect(event).not.toHaveProperty("response");
    expect(event).not.toHaveProperty("ledger");
    expect(event).not.toHaveProperty("chunk_text");
    expect(event).not.toHaveProperty("full_text");
    expect(event).not.toHaveProperty("rawToolInput");
    expect(event).not.toHaveProperty("normalizedPayload");
    expect(event).not.toHaveProperty("spans");
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
    const sessionId = createSessionId();
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
        createFakeStreamingResponse(
          ["Check ", "the operator class first."],
          createFakeEmitAnswerResponse("Check the operator class first.", {
            inputTokens: 12,
            outputTokens: 6,
          }),
        ),
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
        sessionId,
      });

      expect(result.path).toBe("system_2");
    } finally {
      await borg.close();
    }

    const events = readTraceEvents(tracePath);
    const turnEvents = events.filter((event) => event.turnId !== "startup");

    expect(events).toContainEqual(
      expect.objectContaining({
        event: "observed_event_embedding_backfill.started",
        turnId: "startup",
        recall_consistency: "topic_recall_eventual_until_complete",
      }),
    );
    expect(new Set(turnEvents.map((event) => event.turnId)).size).toBe(1);
    expect(turnEvents.map((event) => event.event)).toEqual([
      "turn_phase.started",
      "turn_phase.completed",
      "turn_phase.started",
      "turn_phase.completed",
      "turn_phase.started",
      "recency.completed",
      "perception.started",
      "perception.classifier.degraded",
      "perception.completed",
      "turn_phase.completed",
      "turn_phase.started",
      "llm_call.started",
      "llm_call.completed",
      "frame_anomaly.completed",
      "frame_anomaly.disposition",
      "turn_phase.completed",
      "turn_phase.started",
      "llm_call.started",
      "llm_call.started",
      "llm_call.started",
      "llm_call.completed",
      "extraction.commitments.degraded",
      "llm_call.completed",
      "extraction.actions.completed",
      "llm_call.completed",
      "extraction.goals.completed",
      "turn_phase.completed",
      "turn_phase.started",
      "turn_phase.completed",
      "turn_phase.started",
      "turn_phase.completed",
      "turn_phase.started",
      "retrieval.started",
      "llm_call.started",
      "llm_call.completed",
      "retrieval.degraded",
      "retrieval.completed",
      "turn_phase.started",
      "session_reentry.continuity.evaluated",
      "evidence_ledger.reverse_scan",
      "turn_phase.started",
      "turn_phase.completed",
      "evidence_ledger.completed",
      "evidence_ledger.built",
      "turn_phase.completed",
      "turn_phase.completed",
      "turn_phase.started",
      "deliberation.contradiction_routing.completed",
      "deliberation.path.completed",
      "deliberation.planner_ledger.completed",
      "llm_call.started",
      "llm_call.completed",
      "deliberation.plan.completed",
      "deliberation.plan_persistence.completed",
      "turn_phase.started",
      "llm_call.started",
      "turn.token",
      "turn.token",
      "llm_call.completed",
      "turn.token.flush",
      "finalizer.completed",
      "turn_phase.completed",
      "turn_phase.completed",
      "turn_phase.started",
      "commitment_check.completed",
      "closure_response_guard.completed",
      "turn_phase.completed",
      "turn_phase.started",
      "turn_phase.completed",
      "turn_phase.started",
      "reflection.completed",
      "turn_phase.completed",
      "action_archive_scan.completed",
      "turn.terminal",
    ]);
    expect(
      turnEvents
        .filter((event) => event.event === "turn_phase.completed")
        .map((event) => event.phase),
    ).toEqual(
      expect.arrayContaining([
        "ingest",
        "audience",
        "closure_loop",
        "generation_gate",
        "final",
        "guards",
        "persist",
      ]),
    );
    expect(turnEvents.find((event) => event.event === "turn.terminal")).toMatchObject({
      session_id: sessionId,
      outcome: "reflected",
      turn_id: expect.any(String),
      duration_ms: expect.any(Number),
    });
    expect(turnEvents.find((event) => event.event === "turn_phase.started")).toMatchObject({
      session_id: sessionId,
      phase: "ingest",
    });
    expect(turnEvents.find((event) => event.event === "turn.token")).toMatchObject({
      session_id: sessionId,
      phase: "final",
    });
    expect(
      turnEvents.some(
        (event) => event.event === "llm_call.started" && event.session_id === sessionId,
      ),
    ).toBe(true);
    expect(turnEvents.find((event) => event.event === "commitment_check.completed")).toMatchObject({
      session_id: sessionId,
    });
    expect(
      turnEvents.find((event) => event.event === "deliberation.plan_persistence.completed"),
    ).toMatchObject({
      streamEntryId: expect.stringMatching(/^strm_/),
    });
    expect(
      turnEvents
        .filter((event) => event.event === "perception.classifier.degraded")
        .map((event) => event.classifier),
    ).toEqual(["affective_signal"]);
    expect(turnEvents.find((event) => event.event === "finalizer.completed")).toMatchObject({
      path: "system_2",
      decision: "answer",
    });
  });
});
