import { describe, expect, it } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import type { EvidenceLedger } from "../evidence-ledger/index.js";
import type { TurnTraceData, TurnTraceEventName, TurnTracer } from "../tracing/tracer.js";
import { runManifestFinalizer } from "./manifest-finalizer.js";
import type { EmitManifestResponse } from "./manifest-schema.js";

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
} satisfies EmitManifestResponse;

function structuredOutputResponse(output: unknown): LLMCompleteResult {
  return {
    text: JSON.stringify(output),
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "end_turn",
    tool_calls: [],
    structured_output: output,
  };
}

function unparsedStructuredOutputResponse(text: string): LLMCompleteResult {
  return {
    text,
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "end_turn",
    tool_calls: [],
  };
}

async function runWithResponse(response: LLMCompleteResult, tracer: CapturingTracer) {
  const llm = new FakeLLMClient({
    responses: [response],
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

async function runWithStructuredOutput(output: unknown, tracer: CapturingTracer) {
  return runWithResponse(structuredOutputResponse(output), tracer);
}

describe("manifest finalizer parser", () => {
  it("injects manifest coverage instructions into the structured-output prompt", async () => {
    const llm = new FakeLLMClient({
      responses: [structuredOutputResponse(validManifestInput)],
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
    expect(llm.requests[0]?.system).toContain(
      "Return exactly one structured response matching the provided schema.",
    );
    expect(llm.requests[0]?.tools).toBeUndefined();
    expect(llm.requests[0]?.tool_choice).toBeUndefined();
    expect(llm.requests[0]?.output_config?.format.type).toBe("json_schema");
  });

  it("parses the structured output value directly", async () => {
    const tracer = new CapturingTracer();

    const result = await runWithStructuredOutput(validManifestInput, tracer);

    expect(result.manifest).toEqual(validManifestInput);
    expect(
      tracer.events.find((entry) => entry.event === "manifest_finalizer_emitted")?.data,
    ).toMatchObject({
      parsed: true,
      final_text_length: validManifestInput.final_text.length,
    });
  });

  it("fails loudly when the structured output does not match the manifest schema", async () => {
    const tracer = new CapturingTracer();

    await expect(
      runWithStructuredOutput(
        {
          final_text: "Done.",
          discourse_act: "answer",
        },
        tracer,
      ),
    ).rejects.toThrow("Manifest finalizer returned invalid structured output");

    const parseFailed = tracer.events.find(
      (entry) => entry.event === "manifest_finalizer_parse_failed",
    );

    expect(parseFailed?.data.parsed).toBe(false);
    expect(JSON.stringify(parseFailed?.data.issues)).toContain("claims");
    expect(parseFailed?.data.raw_structured_output).toEqual({
      final_text: "Done.",
      discourse_act: "answer",
    });
  });

  it("traces non-JSON structured-output text as a manifest parse failure", async () => {
    const tracer = new CapturingTracer();

    await expect(
      runWithResponse(unparsedStructuredOutputResponse("{"), tracer),
    ).rejects.toThrow("Manifest finalizer returned invalid structured output");

    const parseFailed = tracer.events.find(
      (entry) => entry.event === "manifest_finalizer_parse_failed",
    );

    expect(parseFailed?.data.parsed).toBe(false);
    expect(String(parseFailed?.data.error)).toContain("JSON");
    expect(parseFailed?.data.raw_text).toBe("{");
  });
});
