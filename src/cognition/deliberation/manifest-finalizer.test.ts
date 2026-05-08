import { describe, expect, it } from "vitest";

import { FakeLLMClient, type LLMToolCall } from "../../llm/index.js";
import type { EvidenceLedger } from "../evidence-ledger/index.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";
import { runManifestFinalizer } from "./manifest-finalizer.js";

class CapturingTracer implements TurnTracer {
  readonly enabled = true;
  readonly includePayloads = true;
  readonly events: { event: TurnTraceEventName; data: TurnTraceData }[] = [];

  emit(event: TurnTraceEventName, data: TurnTraceData): void {
    this.events.push({ event, data });
  }
}

const evidenceLedger: EvidenceLedger = {
  sections: [
    {
      id: "current_user_message",
      label: "1. Current User Message",
      entries: [
        {
          id: "current_user_message:strm_aaaaaaaaaaaaaaaa",
          source_type: "current_user_message",
          session_scope: "current_session",
          actor: "user",
          trust_rank: 1,
          text: "Answer directly.",
        },
      ],
    },
  ],
  transcriptIncluded: false,
  transcriptOmittedReason: "over_budget",
  estimatedTokens: 12,
};

const validManifestInput = {
  final_text: "Done.",
  discourse_act: "answer",
  claims: [
    {
      kind: "hedge",
      rendered_span: "Done.",
    },
  ],
};

async function runWithToolCalls(toolCalls: readonly LLMToolCall[], tracer: CapturingTracer) {
  const llm = new FakeLLMClient({
    responses: [
      {
        text: "",
        input_tokens: 4,
        output_tokens: 2,
        stop_reason: "tool_use",
        tool_calls: [...toolCalls],
      },
    ],
  });

  return runManifestFinalizer({
    llmClient: llm,
    model: "sonnet",
    baseSystemPrompt: "Base prompt.",
    dialogueMessages: [{ role: "user", content: "Answer directly." }],
    evidenceLedger,
    maxTokens: 512,
    path: "system_1",
    tracer,
    turnId: "turn-manifest-parse",
  });
}

describe("manifest finalizer parser", () => {
  it("injects manifest coverage instructions into the forced tool prompt", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_manifest",
              name: "EmitManifestResponse",
              input: validManifestInput,
            },
          ],
        },
      ],
    });

    await runManifestFinalizer({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "Base prompt.",
      dialogueMessages: [{ role: "user", content: "Answer directly." }],
      evidenceLedger,
      maxTokens: 512,
      path: "system_1",
    });

    expect(llm.requests[0]?.system).toContain(
      "The claim manifest is the source contract for final_text.",
    );
    expect(llm.requests[0]?.system).toContain(
      "Do not hide factual or source-sensitive content under discourse_only, hedge, interpretation, or self_report.",
    );
    expect(llm.requests[0]?.system).toContain(
      "Use self_report for first-person expression of interior states, identity reflection, voice, or boundary -- the model's own perspective.",
    );
    expect(llm.requests[0]?.system).toContain("addresses_audience_by_name: true");
    expect(llm.requests[0]?.system).toContain(
      "When the entity is referenced by pronoun (she/he/they/it) or descriptive noun phrase",
    );
    expect(llm.requests[0]?.system).toContain(
      "A manifest with too few claims is invalid even if the prose sounds good.",
    );
  });

  it.each([
    {
      wrapper: "input",
      expectedTrace: "input_wrapper",
      input: {
        input: validManifestInput,
      },
    },
    {
      wrapper: "arguments",
      expectedTrace: "arguments_wrapper",
      input: {
        arguments: validManifestInput,
      },
    },
    {
      wrapper: "$PARAMETER_VALUE",
      expectedTrace: "parameter_value_wrapper",
      input: {
        $PARAMETER_VALUE: validManifestInput,
      },
    },
    {
      wrapper: "response",
      expectedTrace: "response_wrapper",
      input: {
        response: validManifestInput,
      },
    },
  ])("unwraps a single $wrapper manifest tool input wrapper", async ({ input, expectedTrace }) => {
    const tracer = new CapturingTracer();

    const result = await runWithToolCalls(
      [
        {
          id: "toolu_manifest",
          name: "EmitManifestResponse",
          input,
        },
      ],
      tracer,
    );

    expect(result.manifest).toEqual(validManifestInput);
    expect(
      tracer.events.find((entry) => entry.event === "manifest_finalizer_emitted")?.data,
    ).toMatchObject({
      parsed: true,
      manifest_finalizer_unwrapped: expectedTrace,
    });
  });

  it("drops leaked tool metadata before strict manifest parsing", async () => {
    const tracer = new CapturingTracer();

    const result = await runWithToolCalls(
      [
        {
          id: "toolu_manifest",
          name: "EmitManifestResponse",
          input: {
            ...validManifestInput,
            $FUNCTION_NAME: "EmitManifestResponse",
          },
        },
      ],
      tracer,
    );

    expect(result.manifest).toEqual(validManifestInput);
    expect(
      tracer.events.find((entry) => entry.event === "manifest_finalizer_emitted")?.data,
    ).toMatchObject({
      parsed: true,
      manifest_finalizer_unwrapped: "function_name_dropped",
    });
  });

  it("still rejects leaked tool metadata when required claims are missing", async () => {
    const tracer = new CapturingTracer();

    await expect(
      runWithToolCalls(
        [
          {
            id: "toolu_manifest",
            name: "EmitManifestResponse",
            input: {
              final_text: "Done.",
              discourse_act: "answer",
              $FUNCTION_NAME: "EmitManifestResponse",
            },
          },
        ],
        tracer,
      ),
    ).rejects.toThrow("Manifest finalizer returned invalid tool output");

    const parseFailed = tracer.events.find(
      (entry) => entry.event === "manifest_finalizer_parse_failed",
    );

    expect(parseFailed?.data.parsed).toBe(false);
    expect(JSON.stringify(parseFailed?.data.issues)).toContain("claims");
  });

  it.each([
    {
      name: "zero tool calls",
      toolCalls: [] as LLMToolCall[],
      error: "received 0",
      rawToolCallCount: 0,
    },
    {
      name: "multiple manifest tool calls",
      toolCalls: [
        {
          id: "toolu_manifest_1",
          name: "EmitManifestResponse",
          input: validManifestInput,
        },
        {
          id: "toolu_manifest_2",
          name: "EmitManifestResponse",
          input: validManifestInput,
        },
      ] satisfies LLMToolCall[],
      error: "received 2",
      rawToolCallCount: 2,
    },
    {
      name: "wrong tool name",
      toolCalls: [
        {
          id: "toolu_wrong",
          name: "WrongTool",
          input: validManifestInput,
        },
      ] satisfies LLMToolCall[],
      error: "unexpected tool WrongTool",
      rawToolCallCount: 1,
    },
  ])("fails loudly on $name", async ({ toolCalls, error, rawToolCallCount }) => {
    const tracer = new CapturingTracer();

    await expect(runWithToolCalls(toolCalls, tracer)).rejects.toThrow(
      "Manifest finalizer returned invalid tool output",
    );

    const parseFailed = tracer.events.find(
      (entry) => entry.event === "manifest_finalizer_parse_failed",
    );

    expect(parseFailed?.data.error).toContain(error);
    expect(parseFailed?.data.parsed).toBe(false);
    expect(Array.isArray(parseFailed?.data.raw_tool_calls)).toBe(true);
    expect(parseFailed?.data.raw_tool_calls).toHaveLength(rawToolCallCount);
  });
});
