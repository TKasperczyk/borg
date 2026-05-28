import { describe, expect, it, vi } from "vitest";

import type { LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { createEntityId } from "../../util/ids.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import { CREATOR_DIRECTIVE_SYSTEM_PROMPT } from "../prompts/creator-directive.js";
import { CREATOR_DIRECTIVE_TOOL_NAME, CreatorDirectiveExtractor } from "./extractor.js";

function creatorDirectiveResponse(
  candidates: readonly Record<string, unknown>[],
): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_creator_directive",
        name: CREATOR_DIRECTIVE_TOOL_NAME,
        input: {
          decision: candidates.length === 0 ? "none" : "creator_directive",
          reason:
            candidates.length === 0
              ? "No explicit durable creator disclosure directive."
              : "The creator gave explicit durable disclosure guidance.",
          candidates,
        },
      },
    ],
  };
}

function candidate(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    kind: "subject_fact",
    subject_kind: "entity",
    subject_entity_id: null,
    subject_label: "Alice",
    semantic_slot: null,
    semantic_value: null,
    canonical_fact: "Alice has blue hair.",
    operational_directive: "Answer allowed audiences with the blue-hair fact when asked.",
    disclosure_policy: {
      content_scope: "operator_only",
      allowed_entity_ids: [],
      allowed_entity_labels: [],
      excluded_entity_ids: [],
      excluded_entity_labels: [],
      subject_may_know: null,
      mention_policy: "answer_if_asked",
      denied_audience_behavior: "omit",
      boundary_prompt: null,
      topic_tags: ["Alice"],
    },
    priority: 5,
    confidence: 0.9,
    reason: "Explicit durable disclosure directive.",
    ...overrides,
  };
}

function extractorInput(
  overrides: Partial<Parameters<CreatorDirectiveExtractor["extract"]>[0]> = {},
) {
  const creatorId = createEntityId();

  return {
    userMessage: "Alice may know this if she asks.",
    recentHistory: [],
    audienceEntityId: creatorId,
    currentSenderEntityId: creatorId,
    currentSenderDisplayName: "Tom",
    currentSenderBorgRole: "creator" as const,
    sessionAudienceRole: "operator" as const,
    ...overrides,
  };
}

describe("CreatorDirectiveExtractor", () => {
  it("extracts ambiguous durable visibility as operator_only", async () => {
    const llm = new FakeLLMClient({
      responses: [creatorDirectiveResponse([candidate()])],
    });
    const extractor = new CreatorDirectiveExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extract(
      extractorInput({
        userMessage: "Keep this name note around for later.",
      }),
    );

    expect(result).toHaveLength(1);
    expect(result[0]?.disclosure_policy.content_scope).toBe("operator_only");
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: CREATOR_DIRECTIVE_TOOL_NAME,
    });
    expect(llm.requests[0]?.max_tokens).toBe(EXTRACTOR_MAX_TOKENS_DEFAULT);
  });

  it("extracts explicit public self-identity disclosure", async () => {
    const llm = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse([
          candidate({
            kind: "self_identity",
            subject_kind: "borg_self",
            subject_entity_id: null,
            subject_label: "Borg",
            semantic_slot: "public_name",
            semantic_value: "Kestrel",
            canonical_fact: "Borg's self-chosen name is Kestrel.",
            operational_directive: "Answer any audience with Borg's self-chosen name when asked.",
            disclosure_policy: {
              content_scope: "public",
              allowed_entity_ids: [],
              allowed_entity_labels: [],
              excluded_entity_ids: [],
              excluded_entity_labels: [],
              subject_may_know: true,
              mention_policy: "answer_if_asked",
              denied_audience_behavior: "omit",
              boundary_prompt: null,
              topic_tags: ["Kestrel"],
            },
          }),
        ]),
      ],
    });
    const extractor = new CreatorDirectiveExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extract(
      extractorInput({
        userMessage: "Kestrel is my name, anyone can know.",
      }),
    );

    expect(result[0]).toMatchObject({
      kind: "self_identity",
      subject_kind: "borg_self",
      semantic_slot: "public_name",
      semantic_value: "Kestrel",
      canonical_fact: "Borg's self-chosen name is Kestrel.",
      disclosure_policy: expect.objectContaining({
        content_scope: "public",
        mention_policy: "answer_if_asked",
      }),
    });
  });

  it("extracts explicit allow-list disclosure by label", async () => {
    const llm = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse([
          candidate({
            disclosure_policy: {
              content_scope: "allow_list",
              allowed_entity_ids: [],
              allowed_entity_labels: ["Alice"],
              excluded_entity_ids: [],
              excluded_entity_labels: [],
              subject_may_know: true,
              mention_policy: "answer_if_asked",
              denied_audience_behavior: "omit",
              boundary_prompt: null,
              topic_tags: ["Alice"],
            },
          }),
        ]),
      ],
    });
    const extractor = new CreatorDirectiveExtractor({
      llmClient: llm,
      model: "haiku",
    });

    const result = await extractor.extract(
      extractorInput({
        userMessage: "Tell Alice if she asks: Alice has blue hair.",
      }),
    );

    expect(result[0]?.disclosure_policy).toMatchObject({
      content_scope: "allow_list",
      allowed_entity_labels: ["Alice"],
      mention_policy: "answer_if_asked",
    });
  });

  it("rejects semantic values without semantic slots", async () => {
    const onDegraded = vi.fn();
    const llm = new FakeLLMClient({
      responses: [
        creatorDirectiveResponse([
          candidate({
            semantic_slot: null,
            semantic_value: "Vesper",
            canonical_fact: "Borg's self-chosen name is Claude.",
          }),
        ]),
      ],
    });
    const extractor = new CreatorDirectiveExtractor({
      llmClient: llm,
      model: "haiku",
      onDegraded,
    });

    const result = await extractor.extract(
      extractorInput({
        userMessage: "Vesper is the name.",
      }),
    );

    expect(result).toEqual([]);
    expect(onDegraded).toHaveBeenCalledWith(
      "invalid_payload",
      expect.any(Error),
      expect.objectContaining({ stopReason: "tool_use" }),
    );
  });
});

describe("CREATOR_DIRECTIVE_SYSTEM_PROMPT", () => {
  // Presence guard. canonical_fact must carry the fact's agency, or facts Borg
  // originated get flattened into agentless statements and misread as the creator's
  // act. The LLM is mocked here, so behavioral coverage lives in the sim battery;
  // this only ensures the instruction is not silently removed.
  it("instructs the extractor to attribute Borg-originated facts to Borg", () => {
    expect(CREATOR_DIRECTIVE_SYSTEM_PROMPT).toContain("attribute it to Borg");
  });
});
