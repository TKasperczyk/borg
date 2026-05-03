import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import { createEntityId, createStreamEntryId } from "../../util/ids.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { CorrectivePreferenceExtractor } from "./corrective-preference-extractor.js";

function correctivePreferenceResponse(input: {
  classification: "corrective_preference" | "none";
  type?: "preference" | "rule" | "boundary" | null;
  directive?: string | null;
  priority?: number | null;
  reason?: string;
  confidence?: number;
  slot_negations?: unknown[];
}): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_corrective_preference",
        name: "EmitCorrectivePreference",
        input: {
          classification: input.classification,
          type: input.type ?? null,
          directive: input.directive ?? null,
          priority: input.priority ?? null,
          reason: input.reason ?? "Classification reason.",
          confidence: input.confidence ?? 0.9,
          supersedes_commitment_id: null,
          slot_negations: input.slot_negations ?? [],
        },
      },
    ],
  };
}

describe("CorrectivePreferenceExtractor", () => {
  it("emits a high-confidence corrective preference candidate", async () => {
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is still open.",
          priority: 8,
          reason: "The user corrected recurring future response behavior.",
          confidence: 0.9,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extract({
      userMessage: "You keep doing those closers. Stop that.",
      recentHistory: [],
      audienceEntityId: createEntityId(),
      activeCommitments: [],
    });

    expect(result).toMatchObject({
      type: "preference",
      directive: "Do not add ritual closing lines when the conversation is still open.",
      priority: 8,
      reason: "The user corrected recurring future response behavior.",
      confidence: 0.9,
    });
    expect(llm.requests[0]?.model).toBe("haiku");
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: "EmitCorrectivePreference",
    });
  });

  it("traces corrective preference extractor LLM calls on success", async () => {
    const emit = vi.fn();
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit,
    } satisfies TurnTracer;
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "preference",
          directive: "Do not add ritual closing lines when the conversation is still open.",
          priority: 8,
          reason: "The user corrected recurring future response behavior.",
          confidence: 0.9,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
      tracer,
      turnId: "turn-corrective-preference",
    });

    await expect(
      extractor.extract({
        userMessage: "You keep doing those closers. Stop that.",
        recentHistory: [],
        audienceEntityId: createEntityId(),
        activeCommitments: [],
      }),
    ).resolves.toMatchObject({
      type: "preference",
      directive: "Do not add ritual closing lines when the conversation is still open.",
    });

    expect(emit).toHaveBeenCalledWith("llm_call_started", {
      turnId: "turn-corrective-preference",
      label: "corrective_preference_extractor",
      model: "haiku",
      promptCharCount: expect.any(Number),
      toolSchemas: expect.any(Array),
    });
    expect(emit).toHaveBeenCalledWith("llm_call_response", {
      turnId: "turn-corrective-preference",
      label: "corrective_preference_extractor",
      responseShape: {
        textLength: 0,
        toolUseBlocks: [
          {
            id: "toolu_corrective_preference",
            name: "EmitCorrectivePreference",
          },
        ],
      },
      stopReason: "tool_use",
      usage: {
        inputTokens: 4,
        outputTokens: 2,
      },
    });
  });

  it("returns null for casual discussion", async () => {
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "none",
          reason: "The user is sharing a state, not correcting future behavior.",
          confidence: 0.95,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract({
        userMessage: "I'm tired.",
        recentHistory: [],
        audienceEntityId: null,
        activeCommitments: [],
      }),
    ).resolves.toBeNull();
  });

  it("returns slot negations separately from durable corrective preferences", async () => {
    const subject = createEntityId();
    const streamEntryId = createStreamEntryId();
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "none",
          reason: "The user rejected a stored relational value, not future style.",
          confidence: 0.95,
          slot_negations: [
            {
              subject_entity_id: subject,
              slot_key: "partner.name",
              rejected_value: "Sarah",
              source_stream_entry_ids: [streamEntryId],
              confidence: 0.92,
            },
          ],
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extractWithSlotNegations({
      userMessage: "Her name is not Sarah.",
      currentUserStreamEntryId: streamEntryId,
      recentHistory: [],
      audienceEntityId: null,
      activeCommitments: [],
      relationalSlots: [
        {
          subject_entity_id: subject,
          slot_key: "partner.name",
          value: "Sarah",
          state: "established",
          alternate_values: [],
        },
      ],
    });

    expect(result.preference).toBeNull();
    expect(result.slot_negations).toEqual([
      {
        subject_entity_id: subject,
        slot_key: "partner.name",
        rejected_value: "Sarah",
        source_stream_entry_ids: [streamEntryId],
        confidence: 0.92,
      },
    ]);
    expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain("relational_slots");
  });

  it("returns null for low-confidence corrective classifications", async () => {
    const llm = new FakeLLMClient({
      responses: [
        correctivePreferenceResponse({
          classification: "corrective_preference",
          type: "rule",
          directive: "Adjust future response behavior.",
          priority: 5,
          reason: "The signal is ambiguous.",
          confidence: 0.5,
        }),
      ],
    });
    const extractor = new CorrectivePreferenceExtractor({
      llmClient: llm,
      model: "haiku",
    });

    await expect(
      extractor.extract({
        userMessage: "Maybe don't do that.",
        recentHistory: [],
        audienceEntityId: null,
        activeCommitments: [],
      }),
    ).resolves.toBeNull();
  });

  it("reports degraded extraction without throwing", async () => {
    const onDegraded = vi.fn();
    const extractor = new CorrectivePreferenceExtractor({
      onDegraded,
    });

    await expect(
      extractor.extract({
        userMessage: "Keep this behavior different later.",
        recentHistory: [],
        audienceEntityId: null,
        activeCommitments: [],
      }),
    ).resolves.toBeNull();
    expect(onDegraded).toHaveBeenCalledWith("llm_unavailable", undefined);
  });

  it("keeps the extractor free of semantic string-matching shortcuts", () => {
    const source = readFileSync(
      new URL("./corrective-preference-extractor.ts", import.meta.url),
      "utf8",
    );

    const forbiddenFragments = [
      [".", "includes", "("],
      [".", "index", "Of", "("],
      [".", "starts", "With", "("],
      [".", "ends", "With", "("],
      ["new ", "Set", "("],
      ["new ", "Reg", "Exp", "("],
      ["to", "Upper", "Case", "("],
    ];

    for (const fragment of forbiddenFragments) {
      expect(source).not.toContain(fragment.join(""));
    }
  });
});
