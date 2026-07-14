import { describe, expect, it } from "vitest";
import { z } from "zod";

import { LLMError } from "../util/errors.js";
import { serializeJsonValue } from "../util/json-value.js";

import {
  formatOfflineProcessErrorMessage,
  MAX_OFFLINE_ERROR_MESSAGE_LENGTH,
  offlineProcessError,
} from "./process-errors.js";

describe("offlineProcessError", () => {
  it("omits the code key when the thrown value has no .code property", () => {
    const err = offlineProcessError("semantic-extractor", new TypeError("boom"));

    expect("code" in err).toBe(false);
    expect(() => serializeJsonValue(err)).not.toThrow();
  });

  it("omits the code key when includeErrorCode is false and no explicit code is passed", () => {
    const wrapped = new Error("inner");
    (wrapped as Error & { code?: string }).code = "SOMETHING";
    const err = offlineProcessError("semantic-extractor", wrapped, {
      includeErrorCode: false,
    });

    expect("code" in err).toBe(false);
    expect(() => serializeJsonValue(err)).not.toThrow();
  });

  it("includes the code key when the thrown value carries one", () => {
    const wrapped = new Error("inner");
    (wrapped as Error & { code?: string }).code = "STORAGE_FAILED";
    const err = offlineProcessError("semantic-extractor", wrapped);

    expect(err.code).toBe("STORAGE_FAILED");
    expect(() => serializeJsonValue(err)).not.toThrow();
  });

  it("prefers an explicit code option over the thrown value's code", () => {
    const wrapped = new Error("inner");
    (wrapped as Error & { code?: string }).code = "FROM_ERROR";
    const err = offlineProcessError("semantic-extractor", wrapped, {
      code: "FROM_OPTIONS",
    });

    expect(err.code).toBe("FROM_OPTIONS");
  });

  it("omits target_type and target_id when not provided", () => {
    const err = offlineProcessError("semantic-extractor", new Error("x"));

    expect("target_type" in err).toBe(false);
    expect("target_id" in err).toBe(false);
  });

  it("survives serialization when the underlying error has no .code property", () => {
    const errs = [
      offlineProcessError("semantic-extractor", new TypeError("a")),
      offlineProcessError("semantic-extractor", "string thrown"),
      offlineProcessError("semantic-extractor", { weird: "object" }),
    ];

    for (const err of errs) {
      expect(() => serializeJsonValue({ content: { errors: [err] } })).not.toThrow();
    }
  });

  it("includes bounded nested zod issue paths in schema-validation errors", () => {
    const parsed = z
      .object({
        nodes: z.array(z.object({ label: z.string(), description: z.string() })),
      })
      .safeParse({
        nodes: [{ label: 42 }, { label: false, description: 10 }, { label: null, description: [] }],
      });
    expect(parsed.success).toBe(false);
    if (parsed.success) {
      return;
    }

    const wrapped = new LLMError("Semantic extractor returned invalid payload", {
      cause: parsed.error,
      code: "SEMANTIC_EXTRACTOR_INVALID",
    });
    const detail = offlineProcessError("semantic-extractor", wrapped);

    expect(detail.code).toBe("SEMANTIC_EXTRACTOR_INVALID");
    expect(detail.message).toContain("Semantic extractor returned invalid payload");
    expect(detail.message).toContain("nodes.0.label:");
    expect(detail.message).toContain("(+3 more issues)");
    expect(detail.message.length).toBeLessThanOrEqual(MAX_OFFLINE_ERROR_MESSAGE_LENGTH);
  });

  it("prefers concise issue details for direct Zod errors", () => {
    const parsed = z.object({ nodes: z.array(z.object({ label: z.string() })) }).safeParse({
      nodes: [{ label: 42 }],
    });
    expect(parsed.success).toBe(false);
    if (parsed.success) {
      return;
    }

    const message = formatOfflineProcessErrorMessage(parsed.error);

    expect(message).toContain("nodes.0.label:");
    expect(message).not.toContain('"code"');
    expect(message).not.toContain("[");
    expect(message.length).toBeLessThanOrEqual(MAX_OFFLINE_ERROR_MESSAGE_LENGTH);
  });

  it("reserves room for concise Zod details behind a long wrapper message", () => {
    const parsed = z.object({ patch: z.record(z.string(), z.number()) }).safeParse({
      patch: { confidence: "high" },
    });
    expect(parsed.success).toBe(false);
    if (parsed.success) {
      return;
    }

    const wrapped = new LLMError(`Semantic extractor failed ${"outer ".repeat(100)}`, {
      cause: parsed.error,
    });
    const message = formatOfflineProcessErrorMessage(wrapped);

    expect(message).toContain("Semantic extractor failed");
    expect(message).toContain("patch.confidence:");
    expect(message).toContain("expected number");
    expect(message.length).toBeLessThanOrEqual(MAX_OFFLINE_ERROR_MESSAGE_LENGTH);
  });

  it("includes a bounded HTTP status and response-body snippet from the cause chain", () => {
    const gatewayError = Object.assign(new Error("Bad Request"), {
      status: 400,
      error: {
        message: `Grammar error: Unimplemented keys: [\"propertyNames\"] ${"x".repeat(400)} END_SENTINEL`,
      },
    });
    const wrapped = new LLMError("OpenAI-compatible completion request failed", {
      cause: gatewayError,
    });

    const message = formatOfflineProcessErrorMessage(wrapped);

    expect(message).toContain("OpenAI-compatible completion request failed");
    expect(message).toContain("HTTP 400");
    expect(message).toContain('Grammar error: Unimplemented keys: ["propertyNames"]');
    expect(message).not.toContain("END_SENTINEL");
    expect(message.length).toBeLessThanOrEqual(MAX_OFFLINE_ERROR_MESSAGE_LENGTH);
  });

  it("uses only allowlisted HTTP body fields and redacts arbitrary objects", () => {
    let serialized = false;
    const gatewayError = Object.assign(new Error("Bad Request"), {
      status: 400,
      error: {
        request: { tool_arguments: "DO_NOT_EXPOSE" },
        toJSON() {
          serialized = true;
          throw new Error("must not serialize response bodies");
        },
      },
    });

    const message = formatOfflineProcessErrorMessage(gatewayError);

    expect(message).toBe("Bad Request: HTTP 400: [response body omitted: object]");
    expect(message).not.toContain("DO_NOT_EXPOSE");
    expect(serialized).toBe(false);
  });

  it("allows a safe scalar HTTP detail field", () => {
    const gatewayError = Object.assign(new Error("Bad Request"), {
      status: 422,
      body: { detail: "guided grammar compilation failed" },
    });

    expect(formatOfflineProcessErrorMessage(gatewayError)).toBe(
      "Bad Request: HTTP 422: guided grammar compilation failed",
    );
  });
});
