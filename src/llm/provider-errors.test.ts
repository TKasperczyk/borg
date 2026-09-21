import Anthropic from "@anthropic-ai/sdk";
import OpenAI from "openai";
import { describe, expect, it } from "vitest";

import { LLMError } from "../util/errors.js";
import { isLlmProviderTransportError, LLMStructuredOutputParseError } from "./index.js";

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

  it("keeps model-side and untyped errors budgeted even if their text sounds like transport", () => {
    for (const error of [
      new LLMStructuredOutputParseError("invalid JSON", new SyntaxError("parse")),
      new LLMError("upstream connection error timeout 503 rate limit"),
      new Error("LLM_CONNECTION_FAILED"),
      { status: 503, message: "not an SDK error" },
    ])
      expect(isLlmProviderTransportError(error)).toBe(false);
  });
});
