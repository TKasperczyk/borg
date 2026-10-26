import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DEFAULT_CONFIG } from "../config/index.js";
import { AnthropicLLMClient, callStructuredTool, type LLMCompleteOptions } from "../llm/index.js";
import { getLlmErrorUsage } from "../llm/usage.js";
import { getBudgetErrorTokens, withBudget, wrapLlmClientWithSink } from "../offline/budget.js";
import { JsonlTracer } from "../tracing/tracer.js";
import { SystemClock } from "../util/clock.js";
import { LLMError } from "../util/errors.js";
import { createLazyLlmClient, createLlmFactory } from "./clients.js";

const request: LLMCompleteOptions = {
  model: DEFAULT_CONFIG.anthropic.models.background,
  tools: [{ name: "EmitFacts", inputSchema: { type: "object" } }],
  tool_choice: { type: "tool", name: "EmitFacts" },
  messages: [{ role: "user", content: "確認します。" }],
  budget: "offline-reflector",
};

function response(tool: boolean) {
  return {
    id: "msg_accounting",
    type: "message",
    role: "assistant",
    model: request.model,
    content: tool
      ? [{ type: "tool_use", id: "tool_1", name: "EmitFacts", input: {} }]
      : [{ type: "thinking", thinking: "", signature: "signed-first-response" }],
    stop_reason: tool ? "tool_use" : "end_turn",
    stop_sequence: null,
    usage: {
      input_tokens: 12,
      output_tokens: 7,
      cache_creation_input_tokens: 3,
      cache_read_input_tokens: 4,
    },
  };
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function productionClient() {
  const factory = createLlmFactory(
    DEFAULT_CONFIG,
    undefined,
    { ANTHROPIC_API_KEY: "test-key" },
    new SystemClock(),
  );
  return createLazyLlmClient(factory);
}

describe("production LLM compatibility wiring", () => {
  const tempDirs: string[] = [];
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  for (const method of ["complete", "converse"] as const) {
    it.each(["request error", "deadline"])(
      `${method} budgets completed work when the reminder fails: %s`,
      async (failure) => {
        vi.spyOn(console, "warn").mockImplementation(() => undefined);
        if (failure === "deadline") vi.useFakeTimers();
        const fetch = vi.fn().mockResolvedValueOnce(json(response(false)));
        if (failure === "request error") {
          fetch.mockResolvedValueOnce(
            json(
              { type: "error", error: { type: "invalid_request_error", message: "failed retry" } },
              400,
            ),
          );
        } else {
          fetch.mockImplementationOnce(
            async (_input: unknown, init?: RequestInit) =>
              await new Promise<Response>((_resolve, reject) => {
                init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), {
                  once: true,
                });
              }),
          );
        }
        vi.stubGlobal("fetch", fetch);
        let caught: unknown;
        const run = withBudget("reflector", null, async ({ wrapClient }) => {
          // Real lazy factory + observer + budget wrapper, with no internal usage sink.
          const client = wrapClient(productionClient());
          if (method === "complete") return await client.complete({ ...request, timeoutMs: 100 });
          return await client.converse({
            ...request,
            timeoutMs: 100,
            messages: [{ role: "user", content: [{ type: "text", text: "確認します。" }] }],
          });
        }).catch((error: unknown) => {
          caught = error;
        });
        if (failure === "deadline") await vi.advanceTimersByTimeAsync(100);
        await run;
        expect(caught).toBeInstanceOf(LLMError);
        if (failure === "deadline") expect(caught).toMatchObject({ code: "LLM_CALL_TIMED_OUT" });
        expect(getBudgetErrorTokens(caught)).toBe(19);
        expect(getLlmErrorUsage(caught)).toEqual(response(false).usage);
        expect(fetch).toHaveBeenCalledTimes(2);
      },
    );
  }

  it.each([null, 30])(
    "accounts successful retries once, including when cap=%s throws",
    async (cap) => {
      vi.spyOn(console, "warn").mockImplementation(() => undefined);
      vi.stubGlobal(
        "fetch",
        vi
          .fn()
          .mockResolvedValueOnce(json(response(false)))
          .mockResolvedValueOnce(json(response(true))),
      );
      const run = withBudget(
        "reflector",
        cap,
        async ({ wrapClient }) => await wrapClient(productionClient()).complete(request),
      );
      if (cap === null) {
        await expect(run).resolves.toMatchObject({
          tokens_used: 38,
          result: { input_tokens: 24, output_tokens: 14 },
        });
      } else {
        await expect(run).rejects.toMatchObject({
          code: "OFFLINE_BUDGET_EXCEEDED",
          tokens_used: 38,
        });
      }
    },
  );

  it("exposes partial usage in structured failures through the factory and budget wrapper", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValueOnce(json(response(false)))
        .mockResolvedValueOnce(
          json({ error: { type: "invalid_request_error", message: "failed retry" } }, 400),
        ),
    );
    await expect(
      withBudget(
        "reflector",
        null,
        async ({ wrapClient }) =>
          await callStructuredTool({
            llmClient: wrapClient(productionClient()),
            request,
            toolName: "EmitFacts",
            parse: (input) => input,
          }),
      ),
    ).rejects.toMatchObject({
      kind: "llm_failed",
      tokens_used: 19,
      usage: response(false).usage,
    });
  });

  it("does not count a partial failure twice when the sink is also installed internally", async () => {
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const failure = new LLMError("retry failed");
    const create = vi.fn().mockResolvedValueOnce(response(false)).mockRejectedValueOnce(failure);
    const sink = vi.fn();
    const client = wrapLlmClientWithSink(
      new AnthropicLLMClient({ client: { messages: { create } }, usageSink: sink }),
      sink,
    );
    await expect(client.complete(request)).rejects.toMatchObject({ cause: failure });
    expect(sink).toHaveBeenCalledExactlyOnceWith({
      budget: request.budget,
      model: request.model,
      ...response(false).usage,
    });
  });

  it.each([false, true])(
    "writes translations beside started/completed in JSONL, retry=%s",
    async (retry) => {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => undefined);
      vi.stubGlobal(
        "fetch",
        vi
          .fn()
          .mockResolvedValueOnce(json(response(!retry)))
          .mockResolvedValueOnce(json(response(true))),
      );
      const dir = mkdtempSync(join(tmpdir(), "borg-translation-trace-"));
      tempDirs.push(dir);
      const path = join(dir, "turn.jsonl");
      const tracer = new JsonlTracer({ path });
      await callStructuredTool({
        llmClient: productionClient(),
        request,
        toolName: "EmitFacts",
        parse: (input) => input,
        trace: { tracer, turnId: "turn-compatibility", label: "extract_facts" },
      });
      const entries = readFileSync(path, "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line));
      expect(entries.map((entry) => entry.event)).toEqual([
        "llm_call.started",
        "llm_call.tool_choice_translated",
        "llm_call.completed",
      ]);
      expect(entries[1]).toMatchObject({
        turnId: "turn-compatibility",
        label: "extract_facts",
        model: request.model,
        expected_tool: "EmitFacts",
        retry_fired: retry,
        attempt: retry ? 2 : 1,
      });
      const first = retry ? { first_stop_reason: "end_turn", first_text_chars: 0 } : {};
      expect("first_text_chars" in entries[1]).toBe(retry);
      expect(entries[1]).toMatchObject(first);
      expect(warn).toHaveBeenCalledExactlyOnceWith("llm_call.tool_choice_translated", {
        label: request.budget,
        model: request.model,
        expected_tool: "EmitFacts",
        retry_fired: retry,
        attempt: retry ? 2 : 1,
        ...first,
      });
    },
  );
});
