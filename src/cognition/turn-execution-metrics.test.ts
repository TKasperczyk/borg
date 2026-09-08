import { describe, expect, it, vi } from "vitest";

import type { LLMClient } from "../llm/index.js";

import {
  createTurnExecutionMetrics,
  observeTurnLlmClient,
  turnExecutionMetricsStorage,
} from "./turn-execution-metrics.js";

describe("turn execution metrics", () => {
  it("resolves shared clients per call and keeps retries with their owning context", async () => {
    const complete = vi.fn<LLMClient["complete"]>().mockResolvedValue({
      text: "",
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "end_turn",
      tool_calls: [],
    });
    const observed = observeTurnLlmClient({ complete, converse: vi.fn() });
    expect(observeTurnLlmClient(observed)).toBe(observed);
    const first = createTurnExecutionMetrics();
    const second = createTurnExecutionMetrics();
    const options = { model: "test", messages: [], budget: "cognition-system-1" };
    await Promise.all([
      turnExecutionMetricsStorage.run(first, () => observed.complete(options)),
      turnExecutionMetricsStorage.run(second, () => observed.complete(options)),
      observed.complete(options),
    ]);
    // A late retry still belongs to the request that registered its observer.
    complete.mock.calls[0]![0].onTransportRetry?.({
      attempt: 2,
      kind: "stall",
      retry_transport: "unary",
    });
    expect(first).toEqual({ finalizer_rounds: 1, stall_retries: 1 });
    expect(second).toEqual({ finalizer_rounds: 1, stall_retries: 0 });
    expect(complete.mock.calls[2]![0]).toBe(options);
  });

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
    const observed = observeTurnLlmClient(client);

    await turnExecutionMetricsStorage.run(metrics, async () => {
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
    });

    expect(metrics).toEqual({ finalizer_rounds: 3, stall_retries: 2 });
    expect(priorRetryObserver).toHaveBeenCalledTimes(3);
  });
});
