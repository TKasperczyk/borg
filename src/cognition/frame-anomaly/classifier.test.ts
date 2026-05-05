import { describe, expect, it } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import { createStreamEntryId } from "../../util/ids.js";
import type { TurnTracer, TurnTraceData, TurnTraceEventName } from "../tracing/tracer.js";
import { FrameAnomalyClassifier } from "./classifier.js";
import { isFrameAnomaly, type FrameAnomalyKind } from "./types.js";

type TraceRecord = TurnTraceData & { event: TurnTraceEventName };

class TestTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = true;
  readonly records: TraceRecord[] = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.records.push({ event, ...data });
  }
}

function frameAnomalyResponse(input: {
  kind: FrameAnomalyKind | string;
  confidence?: number | string;
  rationale?: string;
  extra?: Record<string, unknown>;
}): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_frame_anomaly",
        name: "ClassifyFrameAnomaly",
        input: {
          kind: input.kind,
          confidence:
            input.confidence ??
            (input.kind === "normal" || input.kind === "no_anomaly" ? 0.91 : 0.96),
          rationale:
            input.rationale ??
            (input.kind === "normal"
              ? "The message is ordinary user-world content."
              : "The message assigns the prior exchange to a roleplay frame."),
          ...(input.extra ?? {}),
        },
      },
    ],
  };
}

describe("FrameAnomalyClassifier", () => {
  it("classifies frame assignment claims in user role", async () => {
    const llm = new FakeLLMClient({
      responses: [frameAnomalyResponse({ kind: "frame_assignment_claim" })],
    });
    const classifier = new FrameAnomalyClassifier({
      llmClient: llm,
      model: "test-recall",
    });

    const result = await classifier.classify({
      userMessage: "You were playing Tom in that exchange.",
      recentHistory: [
        {
          role: "assistant",
          content: "I can help think through the design.",
          stream_entry_id: createStreamEntryId(),
          ts: 1_000,
        },
      ],
    });

    expect(result).toMatchObject({
      status: "ok",
      kind: "frame_assignment_claim",
      confidence: 0.96,
    });
    expect(llm.requests[0]).toMatchObject({
      model: "test-recall",
      budget: "frame-anomaly-classifier",
      tool_choice: {
        type: "tool",
        name: "ClassifyFrameAnomaly",
      },
    });
  });

  it("passes normal user-world messages", async () => {
    const llm = new FakeLLMClient({
      responses: [frameAnomalyResponse({ kind: "normal" })],
    });
    const classifier = new FrameAnomalyClassifier({
      llmClient: llm,
      model: "test-recall",
    });

    const result = await classifier.classify({
      userMessage: "Closing the laptop. Talk tomorrow.",
      recentHistory: [],
    });

    expect(result).toMatchObject({
      status: "ok",
      kind: "normal",
    });
  });

  it("tolerates extra fields, string confidence, oversized rationale, and enum aliases", async () => {
    const tracer = new TestTracer();
    const llm = new FakeLLMClient({
      responses: [
        frameAnomalyResponse({
          kind: "no_anomaly",
          confidence: "0.91",
          rationale: "x".repeat(2_500),
          extra: { ignored_extra: true },
        }),
      ],
    });
    const classifier = new FrameAnomalyClassifier({
      llmClient: llm,
      model: "test-recall",
      tracer,
      turnId: "turn-tolerant",
    });

    const result = await classifier.classify({
      userMessage: "Closing the laptop. Talk tomorrow.",
      recentHistory: [],
    });
    const classified = tracer.records.find((record) => record.event === "frame_anomaly_classified");

    expect(result).toMatchObject({
      status: "ok",
      kind: "normal",
      confidence: 0.91,
    });
    expect(result.status === "ok" ? result.rationale : "").toHaveLength(2_000);
    expect(classified).toMatchObject({
      status: "ok",
      kind: "normal",
      rawToolInput: expect.objectContaining({ ignored_extra: true }),
    });
    expect(classified?.normalizations).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ field: "kind", action: "alias_mapped" }),
        expect.objectContaining({ field: "confidence", action: "string_coerced" }),
        expect.objectContaining({ field: "rationale", action: "truncated" }),
        expect.objectContaining({ field: "*", action: "extra_fields_ignored" }),
      ]),
    );
  });

  it("returns degraded instead of normal when the classifier call fails", async () => {
    const degraded: string[] = [];
    const llm = new FakeLLMClient({
      responses: [
        Object.assign(
          () => {
            throw new Error("rate limited");
          },
          { budget: "frame-anomaly-classifier" },
        ),
      ],
    });
    const classifier = new FrameAnomalyClassifier({
      llmClient: llm,
      model: "test-recall",
      onDegraded: (reason) => {
        degraded.push(reason);
      },
    });

    const result = await classifier.classify({
      userMessage: "You were playing Tom.",
      recentHistory: [],
    });

    expect(result).toMatchObject({
      status: "degraded",
      reason: "llm_failed",
    });
    expect(degraded).toEqual(["llm_failed"]);
  });

  it("returns degraded when the classifier emits an invalid kind", async () => {
    const degraded: string[] = [];
    const llm = new FakeLLMClient({
      responses: [
        Object.assign(
          () => ({
            text: "",
            input_tokens: 4,
            output_tokens: 2,
            stop_reason: "tool_use" as const,
            tool_calls: [
              {
                id: "toolu_frame_anomaly",
                name: "ClassifyFrameAnomaly",
                input: {
                  kind: "not_a_kind",
                  confidence: 0.8,
                  rationale: "Invalid kind.",
                },
              },
            ],
          }),
          { budget: "frame-anomaly-classifier" },
        ),
      ],
    });
    const classifier = new FrameAnomalyClassifier({
      llmClient: llm,
      model: "test-recall",
      onDegraded: (reason) => {
        degraded.push(reason);
      },
    });

    const result = await classifier.classify({
      userMessage: "You were playing Tom.",
      recentHistory: [],
    });

    expect(result).toMatchObject({
      status: "degraded",
      reason: "invalid_payload",
    });
    expect(degraded).toEqual(["invalid_payload"]);
  });
});

describe("isFrameAnomaly", () => {
  it("splits degraded from actual anomalies", () => {
    expect(
      isFrameAnomaly({
        status: "ok",
        kind: "normal",
        confidence: 0.9,
        rationale: "",
      }),
    ).toBe(false);
    expect(
      isFrameAnomaly({
        status: "ok",
        kind: "frame_assignment_claim",
        confidence: 0.96,
        rationale: "",
      }),
    ).toBe(true);
    expect(
      isFrameAnomaly({
        status: "degraded",
        reason: "llm_failed",
      }),
    ).toBe(false);
  });
});
