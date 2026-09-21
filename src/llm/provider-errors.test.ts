import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { describe, expect, it } from "vitest";

import { LLMError } from "../util/errors.js";
import {
  AnthropicLLMClient,
  isLlmProviderTransportError,
  LLMStructuredOutputParseError,
} from "./index.js";

describe("provider/transport failure classification", () => {
  it.each([
    "LLM_STREAM_STALLED",
    "LLM_STREAM_EVENT_STALLED",
    "LLM_CALL_TIMED_OUT",
    "LLM_CONNECTION_FAILED",
  ] as const)("recognizes nested typed %s failures without inspecting prose", (code) =>
    expect(
      isLlmProviderTransportError(
        new Error("turn failed", {
          cause: new LLMError("任意の文言", { code }),
        }),
      ),
    ).toBe(true),
  );

  for (const SDK of [Anthropic, OpenAI]) {
    it.each([408, 429, 500, 503, 529, 400, 401, 403, 422])(
      `${SDK.name} classifies HTTP %s by the SDK error status`,
      (status) =>
        expect(
          isLlmProviderTransportError(
            new LLMError("request failed", {
              cause: new SDK.APIError(status, {}, "arbitrary message", new Headers()),
            }),
          ),
        ).toBe(status === 408 || status === 429 || status >= 500),
    );
    it(`${SDK.name} recognizes connection and timeout classes`, () => {
      expect(
        isLlmProviderTransportError(new SDK.APIConnectionError({ message: "arbitrary" })),
      ).toBe(true);
      expect(isLlmProviderTransportError(new SDK.APIConnectionTimeoutError())).toBe(true);
    });
  }

  it.each([
    { type: "overloaded_error", exempt: true },
    { type: "api_error", exempt: true },
    { type: "rate_limit_error", exempt: true },
    { type: "timeout_error", exempt: true },
    { type: "invalid_request_error", exempt: false },
    { type: "authentication_error", exempt: false },
    { type: "permission_error", exempt: false },
    { type: "not_found_error", exempt: false },
    { type: "billing_error", exempt: false },
  ])(
    "classifies streamed $type through the SDK type without HTTP status",
    async ({ type, exempt }) => {
      const client = new AnthropicLLMClient({
        client: new Anthropic({
          apiKey: "test-only",
          maxRetries: 0,
          fetch: async () =>
            new Response(
              `event: error\ndata: ${JSON.stringify({
                type: "error",
                error: { type, message: "任意の文言" },
              })}\n\n`,
              { headers: { "content-type": "text/event-stream" } },
            ),
        }),
        transportStallMaxRetries: 0,
      });
      let failure: unknown;
      try {
        await client.streamComplete({
          model: "test-model",
          messages: [{ role: "user", content: "hello" }],
          max_tokens: 32,
          budget: "test",
        });
      } catch (error) {
        failure = error;
      }
      expect(failure).toBeInstanceOf(LLMError);
      expect(failure).toMatchObject({
        cause: expect.objectContaining({ status: undefined, type }),
      });
      expect(isLlmProviderTransportError(failure)).toBe(exempt);
    },
  );

  it.each([
    new Anthropic.APIConnectionError({ message: "任意の文言" }),
    new Anthropic.APIConnectionTimeoutError({ message: "任意の文言" }),
  ])("recognizes a streamed $name through the client wrapper", async (failure) => {
    const client = new AnthropicLLMClient({
      client: {
        messages: {
          stream: () =>
            ({
              async *[Symbol.asyncIterator]() {
                throw failure;
              },
            }) as never,
          create: async () => {
            throw failure;
          },
        },
      },
      transportStallMaxRetries: 0,
    });
    let wrapped: unknown;
    try {
      await client.streamComplete({
        model: "test-model",
        messages: [{ role: "user", content: "hello" }],
        max_tokens: 32,
        budget: "test",
      });
    } catch (error) {
      wrapped = error;
    }
    expect(wrapped).toBeInstanceOf(LLMError);
    expect(isLlmProviderTransportError(wrapped)).toBe(true);
  });

  it("keeps an explicit request-error HTTP status authoritative over a provider type", () => {
    expect(
      isLlmProviderTransportError(
        new Anthropic.APIError(400, {}, "任意の文言", new Headers(), "overloaded_error"),
      ),
    ).toBe(false);
  });

  it("keeps model-side and untyped errors budgeted even if their text sounds like transport", () => {
    for (const error of [
      new LLMStructuredOutputParseError("invalid JSON", new SyntaxError("parse")),
      new LLMError("upstream connection error timeout 503 rate limit"),
      new Error("LLM_CONNECTION_FAILED"),
      { status: 503, message: "not an SDK error" },
      { type: "overloaded_error", message: "not an SDK error" },
    ])
      expect(isLlmProviderTransportError(error)).toBe(false);
  });
});
