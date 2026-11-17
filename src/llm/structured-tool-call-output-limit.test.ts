import { describe, expect, it, vi } from "vitest";
import { z } from "zod";

import type { LLMCompleteResult } from "./index.js";
import { FakeLLMClient } from "./test-support/fake-client.js";
import { getLlmErrorStopReason } from "./stop-reason.js";
import { callStructuredTool } from "./structured-tool-call.js";
import { LLMToolArgumentsError } from "../util/errors.js";

const schema = z.object({ items: z.array(z.string()), note: z.string().optional() });
const request = {
  model: "claude-opus-5-5",
  max_tokens: 1_000,
  budget: "test",
  messages: [{ role: "user" as const, content: "source" }],
};
function response(
  stop: string | null,
  input: unknown = { items: ["complete"] },
): LLMCompleteResult {
  return {
    text: "discarded text",
    stop_reason: stop,
    input_tokens: 10,
    output_tokens: 5,
    tool_calls: [{ id: "tool", name: "EmitResult", input }],
  };
}

describe("generic structured output-limit repair", () => {
  it.each(["max_tokens", "length"])(
    "rejects schema-valid partial input stopped by %s",
    async (stop) => {
      const llm = new FakeLLMClient({
        responses: [response(stop, { items: [] }), response("tool_use")],
      });
      const parse = vi.fn((input: unknown) => schema.parse(input));
      const result = await callStructuredTool({
        llmClient: llm,
        request,
        toolName: "EmitResult",
        parse,
      });
      expect(result.parsed).toEqual({ items: ["complete"] });
      expect(parse).toHaveBeenCalledExactlyOnceWith({ items: ["complete"] }, expect.anything());
      expect(llm.requests.map((r) => r.max_tokens)).toEqual([1_000, 4_000]);
      expect(llm.requests[1]?.messages).toEqual([
        ...request.messages,
        {
          role: "user",
          content:
            "Your previous output was cut at the limit of 1000 output tokens and was discarded whole. Emit exactly one complete EmitResult tool call within the retry limit of 4000 output tokens, including thinking.",
        },
      ]);
    },
  );

  it.each([
    { original: 3_000, repair: undefined, hard: false, expected: 6_000 },
    { original: 1_000, repair: 9_000, hard: false, expected: 9_000 },
    { original: 12_000, repair: undefined, hard: false, expected: 16_000 },
    { original: 16_000, repair: undefined, hard: false, expected: 16_000 },
    { original: 40_000, repair: undefined, hard: false, expected: 40_000 },
    { original: 8_000, repair: 24_000, hard: false, expected: 24_000 },
    { original: 40_000, repair: 80_000, hard: false, expected: 64_000 },
    { original: 1_000, repair: 9_000, hard: true, expected: 1_000 },
  ])("uses the effective retry cap $expected", async ({ original, repair, hard, expected }) => {
    const llm = new FakeLLMClient({ responses: [response("max_tokens"), response("tool_use")] });
    await callStructuredTool({
      llmClient: llm,
      request: { ...request, max_tokens: original },
      toolName: "EmitResult",
      repairMaxTokens: repair,
      outputLimitIsHard: hard,
      parse: (v) => schema.parse(v),
    });
    expect(llm.requests[1]?.max_tokens).toBe(expected);
    expect(llm.requests[1]?.messages.at(-1)?.content).toContain(
      `retry limit of ${expected} output tokens`,
    );
  });

  it.each(["max_tokens", "length"])(
    "rejects raw %s with unknown classification without parsing or retrying at maxAttempts 1",
    async (stop) => {
      const llm = new FakeLLMClient({
        responses: [
          {
            ...response(stop),
            stop_reason_classification: { kind: "unknown", raw: stop },
          },
        ],
      });
      const parse = vi.fn((v: unknown) => schema.parse(v));
      await expect(
        callStructuredTool({
          llmClient: llm,
          request,
          toolName: "EmitResult",
          maxAttempts: 1,
          parse,
        }),
      ).rejects.toMatchObject({
        kind: "llm_failed",
        stopReason: stop,
        stopReasonClassification: { kind: "max_tokens", raw: stop },
        message: expect.stringContaining("1000"),
      });
      expect(parse).not.toHaveBeenCalled();
      expect(llm.requests).toHaveLength(1);
    },
  );

  it("accepts and parses raw tool_use despite cut metadata", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          ...response("tool_use"),
          stop_reason_classification: { kind: "max_tokens", raw: "length" },
        },
      ],
    });
    const parse = vi.fn((v: unknown) => schema.parse(v));
    const result = await callStructuredTool({
      llmClient: llm,
      request,
      toolName: "EmitResult",
      maxAttempts: 1,
      parse,
    });
    expect(result.parsed).toEqual({ items: ["complete"] });
    expect(parse).toHaveBeenCalledExactlyOnceWith({ items: ["complete"] }, expect.anything());
    expect(llm.requests).toHaveLength(1);
  });

  it("rejects cut metadata without a raw reason before parsing", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          ...response(null),
          stop_reason_classification: { kind: "max_tokens", raw: "length" },
        },
      ],
    });
    const parse = vi.fn((v: unknown) => schema.parse(v));
    await expect(
      callStructuredTool({
        llmClient: llm,
        request,
        toolName: "EmitResult",
        maxAttempts: 1,
        parse,
      }),
    ).rejects.toMatchObject({
      kind: "llm_failed",
      stopReason: null,
      stopReasonClassification: { kind: "max_tokens", raw: "length" },
    });
    expect(parse).not.toHaveBeenCalled();
    expect(llm.requests).toHaveLength(1);
  });

  it("repairs raw length on malformed arguments with the generic instruction and cap", async () => {
    const llm = new FakeLLMClient({
      responses: [
        () => {
          throw new LLMToolArgumentsError("Partial arguments", {
            toolName: "EmitResult",
            stopReason: "length",
          });
        },
        response("tool_use"),
      ],
    });
    const result = await callStructuredTool({
      llmClient: llm,
      request,
      toolName: "EmitResult",
      parse: (v) => schema.parse(v),
    });
    expect(result.parsed).toEqual({ items: ["complete"] });
    expect(llm.requests.map((r) => r.max_tokens)).toEqual([1_000, 4_000]);
    expect(llm.requests[1]?.messages).toEqual([
      ...request.messages,
      {
        role: "user",
        content:
          "Your previous output was cut at the limit of 1000 output tokens and was discarded whole. Emit exactly one complete EmitResult tool call within the retry limit of 4000 output tokens, including thinking.",
      },
    ]);
  });

  it.each(["max_tokens", "length"])(
    "ends raw %s errors with conflicting metadata as a cut",
    async (stop) => {
      const fail = () => {
        throw new LLMToolArgumentsError("Partial arguments", {
          toolName: "EmitResult",
          stopReason: stop,
          stopReasonClassification: { kind: "unknown", raw: stop },
        });
      };
      const llm = new FakeLLMClient({ responses: [fail, fail] });
      await expect(
        callStructuredTool({
          llmClient: llm,
          request,
          toolName: "EmitResult",
          parse: (v) => schema.parse(v),
        }),
      ).rejects.toMatchObject({
        kind: "llm_failed",
        stopReason: stop,
        stopReasonClassification: { kind: "max_tokens", raw: stop },
        message: expect.stringContaining("4000"),
      });
      expect(llm.requests).toHaveLength(2);
      expect(llm.requests[1]?.messages.at(-1)?.content).toContain("discarded whole");
    },
  );

  it("reports cut then cut as an LLM failure without parsing either partial payload", async () => {
    const llm = new FakeLLMClient({
      responses: [response("max_tokens", {}), response("max_tokens", { items: [] })],
    });
    const parse = vi.fn((v: unknown) => schema.parse(v));
    const error = await callStructuredTool({
      llmClient: llm,
      request,
      toolName: "EmitResult",
      parse,
    }).catch((e: unknown) => e);
    expect(error).toMatchObject({
      kind: "llm_failed",
      attemptCount: 2,
      message: expect.stringContaining("4000"),
    });
    expect(getLlmErrorStopReason(error).kind).toBe("max_tokens");
    expect(parse).not.toHaveBeenCalled();
    expect(llm.requests).toHaveLength(2);
  });

  it("preserves the original schema failure and records a cut repair separately", async () => {
    const llm = new FakeLLMClient({
      responses: [response("tool_use", {}), response("max_tokens", { items: [] })],
    });
    const parse = vi.fn((v: unknown) => schema.parse(v));
    const error = await callStructuredTool({
      llmClient: llm,
      request,
      toolName: "EmitResult",
      parse,
    }).catch((e: unknown) => e);
    expect(error).toMatchObject({
      kind: "invalid_payload",
      attemptCount: 2,
      cause: expect.any(z.ZodError),
      stopReason: "tool_use",
      repairFailure: {
        kind: "llm_failed",
        message: expect.stringContaining("1000"),
        stopReason: "max_tokens",
        stopReasonClassification: { kind: "max_tokens", raw: "max_tokens" },
      },
    });
    expect(getLlmErrorStopReason(error).kind).toBe("tool_use");
    expect(parse).toHaveBeenCalledExactlyOnceWith({}, expect.anything());
    expect(llm.requests).toHaveLength(2);
  });

  it("does not retry when maxAttempts is one and reports the clamped original limit", async () => {
    const llm = new FakeLLMClient({ responses: [response("max_tokens", { items: [] })] });
    const parse = vi.fn((v: unknown) => schema.parse(v));
    await expect(
      callStructuredTool({
        llmClient: llm,
        request: { ...request, max_tokens: 80_000 },
        toolName: "EmitResult",
        maxAttempts: 1,
        parse,
      }),
    ).rejects.toMatchObject({ kind: "llm_failed", message: expect.stringContaining("64000") });
    expect(parse).not.toHaveBeenCalled();
    expect(llm.requests).toHaveLength(1);
  });
});
