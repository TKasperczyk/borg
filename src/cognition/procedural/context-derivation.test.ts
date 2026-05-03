import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import type { EntityId } from "../../util/ids.js";
import type { PerceptionResult } from "../types.js";
import { deriveProceduralContext } from "./context-derivation.js";

const PROCEDURAL_CONTEXT_TOOL_NAME = "EmitProceduralContext";

function makePerception(
  mode: PerceptionResult["mode"],
  entities: readonly string[] = [],
): Pick<PerceptionResult, "mode" | "entities"> {
  return {
    mode,
    entities: [...entities],
  };
}

function makeAudienceProfile(entityId: EntityId): SocialProfile {
  return {
    entity_id: entityId,
    trust: 0.7,
    attachment: 0.4,
    communication_style: null,
    shared_history_summary: null,
    last_interaction_at: 1_000,
    interaction_count: 3,
    commitment_count: 0,
    sentiment_history: [],
    notes: null,
    created_at: 100,
    updated_at: 1_000,
  };
}

function proceduralContextResponse(input: {
  problem_kind: string;
  domain_tags: string[];
  confidence?: number;
}) {
  return {
    text: "",
    input_tokens: 1,
    output_tokens: 1,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_context",
        name: PROCEDURAL_CONTEXT_TOOL_NAME,
        input: {
          confidence: input.confidence ?? 0.8,
          problem_kind: input.problem_kind,
          domain_tags: input.domain_tags,
        },
      },
    ],
  };
}

describe("deriveProceduralContext", () => {
  it("derives problem kind and canonical slug tags from the LLM tool output", async () => {
    const llm = new FakeLLMClient({
      responses: [
        proceduralContextResponse({
          problem_kind: "code_debugging",
          domain_tags: ["typescript", "deployment", "typescript"],
        }),
      ],
    });

    const recentMessages = Array.from({ length: 11 }, (_, index) => ({
      role: index % 2 === 0 ? ("user" as const) : ("assistant" as const),
      content: `prior message ${index}`,
    }));
    const context = await deriveProceduralContext(
      {
        userMessage: "修复 TypeScript 部署错误。",
        recentMessages,
        perception: makePerception("problem_solving", ["TypeScript", "Deploy"]),
        isSelfAudience: true,
        audienceEntityId: null,
      },
      {
        llmClient: llm,
        model: "haiku",
      },
    );

    expect(context).toMatchObject({
      problem_kind: "code_debugging",
      domain_tags: ["typescript", "deployment"],
      audience_scope: "self",
    });
    expect(context?.context_key).toMatch(/^v2:/);
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: PROCEDURAL_CONTEXT_TOOL_NAME,
    });
    const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
    const payload = JSON.parse(prompt.split("\n").at(-1) ?? "{}") as {
      recent_messages?: unknown;
    };
    expect(prompt).toContain("Use recent_messages as prior-turn context");
    expect(payload.recent_messages).toEqual(recentMessages.slice(-10));
  });

  it("derives audience scope from self, known, and unknown audiences", async () => {
    const knownEntityId = "ent_aaaaaaaaaaaaaaaa" as EntityId;
    const firstContactEntityId = "ent_bbbbbbbbbbbbbbbb" as EntityId;
    const llm = new FakeLLMClient({
      responses: [
        proceduralContextResponse({ problem_kind: "planning", domain_tags: ["atlas"] }),
        proceduralContextResponse({ problem_kind: "planning", domain_tags: ["atlas"] }),
        proceduralContextResponse({ problem_kind: "planning", domain_tags: ["atlas"] }),
      ],
    });

    await expect(
      deriveProceduralContext(
        {
          userMessage: "Plan the Atlas roadmap.",
          recentMessages: [],
          perception: makePerception("problem_solving", ["Atlas"]),
          isSelfAudience: true,
          audienceEntityId: null,
        },
        { llmClient: llm, model: "haiku" },
      ),
    ).resolves.toMatchObject({ audience_scope: "self" });
    await expect(
      deriveProceduralContext(
        {
          userMessage: "Plan the Atlas roadmap.",
          recentMessages: [],
          perception: makePerception("problem_solving", ["Atlas"]),
          isSelfAudience: false,
          audienceEntityId: knownEntityId,
          audienceProfile: makeAudienceProfile(knownEntityId),
        },
        { llmClient: llm, model: "haiku" },
      ),
    ).resolves.toMatchObject({ audience_scope: "known_other" });
    await expect(
      deriveProceduralContext(
        {
          userMessage: "Plan the Atlas roadmap.",
          recentMessages: [],
          perception: makePerception("problem_solving", ["Atlas"]),
          isSelfAudience: false,
          audienceEntityId: firstContactEntityId,
          audienceProfile: null,
        },
        { llmClient: llm, model: "haiku" },
      ),
    ).resolves.toMatchObject({ audience_scope: "unknown" });
  });

  it("returns null and reports degraded mode when no LLM is available", async () => {
    const onDegraded = vi.fn();

    await expect(
      deriveProceduralContext(
        {
          userMessage: "Fix this.",
          recentMessages: [],
          perception: makePerception("problem_solving"),
          isSelfAudience: false,
          audienceEntityId: null,
        },
        { onDegraded },
      ),
    ).resolves.toBeNull();
    expect(onDegraded).toHaveBeenCalledWith("llm_unavailable", undefined);
  });

  it("returns null for low-confidence or empty unknown context", async () => {
    const llm = new FakeLLMClient({
      responses: [
        proceduralContextResponse({
          problem_kind: "code_debugging",
          domain_tags: ["typescript"],
          confidence: 0.1,
        }),
        proceduralContextResponse({
          problem_kind: "other",
          domain_tags: [],
          confidence: 0.8,
        }),
      ],
    });

    await expect(
      deriveProceduralContext(
        {
          userMessage: "maybe this",
          recentMessages: [],
          perception: makePerception("problem_solving"),
          isSelfAudience: true,
          audienceEntityId: null,
        },
        { llmClient: llm, model: "haiku" },
      ),
    ).resolves.toBeNull();
    await expect(
      deriveProceduralContext(
        {
          userMessage: "",
          recentMessages: [],
          perception: makePerception("problem_solving"),
          isSelfAudience: false,
          audienceEntityId: null,
          audienceProfile: null,
        },
        { llmClient: llm, model: "haiku" },
      ),
    ).resolves.toBeNull();
  });

  it("caps domain tags after canonicalization and dedupe without generic filtering", async () => {
    const llm = new FakeLLMClient({
      responses: [
        proceduralContextResponse({
          problem_kind: "code_debugging",
          domain_tags: ["TypeScript", "typescript", "code", "debugging", "lancedb", "sqlite"],
        }),
      ],
    });

    const context = await deriveProceduralContext(
      {
        userMessage: "Fix this TypeScript LanceDB SQLite issue.",
        recentMessages: [],
        perception: makePerception("problem_solving"),
        isSelfAudience: true,
        audienceEntityId: null,
      },
      { llmClient: llm, model: "haiku" },
    );

    expect(context?.domain_tags).toEqual(["typescript", "code", "debugging"]);
  });
});
