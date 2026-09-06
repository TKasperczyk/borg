import { describe, expect, it, vi } from "vitest";

import type { LLMClient } from "../llm/index.js";

import { createTurnExecutionMetrics, observeTurnLlmClient } from "./turn-execution-metrics.js";

describe("turn execution metrics", () => {
  it("counts finalizer requests and stall retries while preserving the existing retry observer", async () => {
    const priorRetryObserver = vi.fn();
    const client: LLMClient = {
      complete: async (options) => {
        options.onTransportRetry?.({
          attempt: 2,
          kind: "stall",
          code: "LLM_STREAM_STALLED",
          retry_transport: "unary",
        });
        return {
          text: "planned",
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
          tool_calls: [],
        };
      },
      converse: async () => ({
        messageBlocks: [],
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "end_turn",
      }),
      streamConverse: async (options) => {
        options.onTransportRetry?.({
          attempt: 2,
          kind: "stall",
          code: "LLM_STREAM_STALLED",
          retry_transport: "unary",
        });
        options.onTransportRetry?.({
          attempt: 3,
          kind: "connection",
          retry_transport: "unary",
        });
        return {
          messageBlocks: [],
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "end_turn",
        };
      },
    };
    const metrics = createTurnExecutionMetrics();
    const observed = observeTurnLlmClient(client, metrics);

    await observed.streamConverse?.({
      model: "test",
      messages: [],
      budget: "cognition-system-1",
      onTransportRetry: priorRetryObserver,
    });
    await observed.converse({
      model: "test",
      messages: [],
      budget: "cognition-system-2",
      onTransportRetry: priorRetryObserver,
    });
    await observed.converse({
      model: "test",
      messages: [],
      budget: "cognition-system-2",
      onTransportRetry: priorRetryObserver,
    });
    await observed.complete({
      model: "test",
      messages: [],
      budget: "cognition-plan",
      onTransportRetry: priorRetryObserver,
    });

    expect(metrics).toEqual({ finalizer_rounds: 3, stall_retries: 2 });
    expect(priorRetryObserver).toHaveBeenCalledTimes(3);
  });
});
