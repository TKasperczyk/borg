import { describe, expect, it } from "vitest";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import type { LLMCompleteResult } from "../llm/index.js";
import { expandRecall, resolveRecallTemporalCue } from "./recall-expansion.js";

const NOW = Date.parse("2026-09-05T10:00:00.000Z");

function planResponse(temporal_cue: unknown): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_recall_expansion",
        name: "EmitRecallQueryPlan",
        input: {
          resolved_query: "co ustaliliśmy wczoraj",
          semantic_variants: [{ strategy: "combined", query: "wczorajsze ustalenia" }],
          named_terms: [],
          typed_queries: [],
          temporal_cue,
        },
      },
    ],
  };
}

describe("resolveRecallTemporalCue", () => {
  it("parses ISO instants into a temporal cue and keeps the label", () => {
    expect(
      resolveRecallTemporalCue(
        {
          since: "2026-09-04T00:00:00+02:00",
          until: "2026-09-05T00:00:00+02:00",
          label: "wczoraj",
        },
        NOW,
      ),
    ).toEqual({
      label: "wczoraj",
      sinceTs: Date.parse("2026-09-04T00:00:00+02:00"),
      untilTs: Date.parse("2026-09-05T00:00:00+02:00"),
    });
  });

  it("keeps open-ended cues and drops malformed, inverted or NOW-less ones", () => {
    expect(
      resolveRecallTemporalCue(
        { since: "2026-07-01T00:00:00Z", until: null, label: "od lipca" },
        NOW,
      ),
    ).toEqual({
      label: "od lipca",
      sinceTs: Date.parse("2026-07-01T00:00:00Z"),
    });
    expect(
      resolveRecallTemporalCue({ since: "yesterday", until: "soon", label: "x" }, NOW),
    ).toBeNull();
    expect(
      resolveRecallTemporalCue(
        { since: "2026-09-05T00:00:00Z", until: "2026-09-04T00:00:00Z", label: "x" },
        NOW,
      ),
    ).toBeNull();
    expect(resolveRecallTemporalCue(null, NOW)).toBeNull();
    expect(resolveRecallTemporalCue(undefined, NOW)).toBeNull();
    expect(
      resolveRecallTemporalCue(
        { since: "2026-09-04T00:00:00Z", until: null, label: "x" },
        undefined,
      ),
    ).toBeNull();
  });
});

describe("expandRecall temporal cue", () => {
  it("lists temporal_cue as a required tool field while still accepting plans that omit it", async () => {
    const llmClient = new FakeLLMClient({ responses: [planResponse(undefined)] });
    const plan = await expandRecall({
      llmClient,
      model: "test-recall-expansion",
      focus: "Co wczoraj?",
      semanticVariantCount: 1,
      nowMs: NOW,
    });
    expect(plan.temporalCue).toBeNull();
    const request = llmClient.requests[0] as unknown as {
      tools: ReadonlyArray<{ inputSchema: { required?: string[] } }>;
    };
    expect(request.tools[0]!.inputSchema.required).toContain("temporal_cue");
  });

  it("renders NOW in the configured zone and resolves the emitted cue", async () => {
    const llmClient = new FakeLLMClient({
      responses: [
        planResponse({
          since: "2026-09-04T00:00:00+02:00",
          until: "2026-09-05T00:00:00+02:00",
          label: "wczoraj",
        }),
      ],
    });
    const plan = await expandRecall({
      llmClient,
      model: "test-recall-expansion",
      focus: "Co ustaliliśmy wczoraj?",
      semanticVariantCount: 1,
      nowMs: NOW,
      timeZone: "Europe/Warsaw",
    });
    expect(plan.temporalCue).toEqual({
      label: "wczoraj",
      sinceTs: Date.parse("2026-09-04T00:00:00+02:00"),
      untilTs: Date.parse("2026-09-05T00:00:00+02:00"),
    });
    const request = llmClient.requests[0] as unknown as {
      messages: ReadonlyArray<{ content: string }>;
    };
    const userMessage = request.messages[0]!.content;
    expect(userMessage).toContain("NOW (JSON data only):");
    expect(userMessage).toContain('"time_zone":"Europe/Warsaw"');
    expect(userMessage).toContain('"iso":"2026-09-05T10:00:00.000Z"');
    expect(userMessage).toContain("OWNER_LIVED_EXPERIENCE");
  });

  it("drops the cue when NOW was not supplied and tolerates a plan without one", async () => {
    const withoutNow = await expandRecall({
      llmClient: new FakeLLMClient({
        responses: [planResponse({ since: "2026-09-04T00:00:00Z", until: null, label: "wczoraj" })],
      }),
      model: "test-recall-expansion",
      focus: "Co ustaliliśmy wczoraj?",
      semanticVariantCount: 1,
    });
    expect(withoutNow.temporalCue).toBeNull();
    const legacy = await expandRecall({
      llmClient: new FakeLLMClient({ responses: [planResponse(undefined)] }),
      model: "test-recall-expansion",
      focus: "Co ustaliliśmy wczoraj?",
      semanticVariantCount: 1,
      nowMs: NOW,
    });
    expect(legacy.temporal_cue).toBeNull();
    expect(legacy.temporalCue).toBeNull();
  });

  it("renders owner lived experience rows, newest first and clipped", async () => {
    const llmClient = new FakeLLMClient({ responses: [planResponse(null)] });
    await expandRecall({
      llmClient,
      model: "test-recall-expansion",
      focus: "O które role chodziło?",
      semanticVariantCount: 1,
      nowMs: NOW,
      ownerLivedExperience: [
        {
          day: "2026-09-04",
          gist: "Porównałem role chat i reviewer w AI Ninjas. " + "x".repeat(500),
          salience: 0.8,
        },
        { day: "2026-09-03", gist: "Spokojny dzień." },
      ],
    });
    const request = llmClient.requests[0] as unknown as {
      messages: ReadonlyArray<{ content: string }>;
    };
    const userMessage = request.messages[0]!.content;
    expect(userMessage).toContain('"day": "2026-09-04"');
    expect(userMessage).toContain("Porównałem role chat i reviewer w AI Ninjas.");
    expect(userMessage).not.toContain("x".repeat(400));
    expect(userMessage.indexOf("2026-09-04")).toBeLessThan(userMessage.indexOf("2026-09-03"));
  });
});
