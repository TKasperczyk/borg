import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import { createEntityId } from "../../util/ids.js";
import { CorrectivePreferenceExtractor } from "./corrective-preference-extractor.js";

function correctivePreferenceResponse(input: {
  classification: "corrective_preference" | "none";
  type?: "preference" | "rule" | "boundary" | null;
  directive?: string | null;
  priority?: number | null;
  reason?: string;
  confidence?: number;
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

    expect(source.match(/\.includes\s*\(/gu)).toBeNull();
    expect(source.match(/\.indexOf\s*\(/gu)).toBeNull();
    expect(source.match(/\.startsWith\s*\(/gu)).toBeNull();
    expect(source.match(/\.endsWith\s*\(/gu)).toBeNull();
    expect(source.match(/new Set\s*\(/gu)).toBeNull();
    expect(source.match(/new RegExp\s*\(/gu)).toBeNull();
    expect(source.match(/toUpperCase\s*\(/gu)).toBeNull();
  });
});
