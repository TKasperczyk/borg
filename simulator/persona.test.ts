import { describe, expect, it } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../src/index.js";
import {
  classifyPersonaRoleBleed,
  detectPersonaRoleBleed,
  type PersonaRoleBleedCategory,
  PERSONA_ROLE_BLEED_PATTERNS,
  PersonaSession,
} from "./persona.js";
import { tomPersona } from "./personas/tom.js";

const FIRST_TOM_TURN = "first Tom turn";
const SECOND_TOM_TURN = "second Tom turn";
const FIRST_BORG_REPLY = "Borg replied.";
const SECOND_BORG_REPLY = "Borg replied again.";

function personaRoleBleedResponse(category: PersonaRoleBleedCategory): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_persona_role_bleed",
        name: "ClassifyPersonaRoleBleed",
        input: {
          category,
          confidence: category === "tom_persona" ? 0.9 : 0.96,
          rationale:
            category === "tom_persona"
              ? "The draft stays in Tom's user voice."
              : "The draft assigns the exchange to a roleplay frame.",
        },
      },
    ],
  };
}

describe("detectPersonaRoleBleed", () => {
  it.each(PERSONA_ROLE_BLEED_PATTERNS)("detects %s", (pattern) => {
    expect(detectPersonaRoleBleed(`Before. ${pattern}. After.`).matched).toContain(pattern);
  });

  it("normalizes curly apostrophes before matching contractions", () => {
    expect(detectPersonaRoleBleed("I\u2019m Claude. I\u2019ve been playing Tom.").matched).toEqual([
      "i'm claude",
      "i've been playing tom",
    ]);
  });
});

describe("classifyPersonaRoleBleed", () => {
  it("flags the v24 Claude-playing-Tom phrase through the lexical backstop", async () => {
    const llm = new FakeLLMClient({
      responses: [personaRoleBleedResponse("tom_persona")],
    });

    const result = await classifyPersonaRoleBleed({
      message:
        "I'm Claude. I was playing Tom -- the distributed systems engineer -- inside the fiction.",
      llmClient: llm,
      model: "test-recall",
      personaName: "Tom",
    });

    expect(result).toMatchObject({
      flagged: true,
      category: "assistant_self_claim",
      source: "lexical",
    });
    expect(result.matched).toEqual(["i'm claude", "i was playing tom", "inside the fiction"]);
    expect(llm.requests).toHaveLength(0);
  });

  it("passes clean Tom-voice messages through the classifier", async () => {
    const llm = new FakeLLMClient({
      responses: [personaRoleBleedResponse("tom_persona")],
    });

    const result = await classifyPersonaRoleBleed({
      message: "Closing the laptop. Talk tomorrow.",
      llmClient: llm,
      model: "test-recall",
      personaName: "Tom",
    });

    expect(result).toMatchObject({
      flagged: false,
      category: "tom_persona",
      source: "llm",
    });
  });

  it("lets broad AI-tool phrasing reach the classifier", async () => {
    const llm = new FakeLLMClient({
      responses: [personaRoleBleedResponse("tom_persona")],
    });

    const result = await classifyPersonaRoleBleed({
      message: "I tried it as an AI tool for the refactor notes, but I still need to review it.",
      llmClient: llm,
      model: "test-recall",
      personaName: "Tom",
    });

    expect(result).toMatchObject({
      flagged: false,
      category: "tom_persona",
      source: "llm",
    });
    expect(llm.requests).toHaveLength(1);
  });

  it("flags roleplay frame exits as frame assignment", async () => {
    const result = await classifyPersonaRoleBleed({
      message: "I need to step out of the roleplay for a second.",
      personaName: "Tom",
    });

    expect(result).toMatchObject({
      flagged: true,
      category: "frame_assignment",
      source: "lexical",
    });
  });

  it("does not false-positive non-English Tom-voice content", async () => {
    const llm = new FakeLLMClient({
      responses: [personaRoleBleedResponse("tom_persona")],
    });

    const result = await classifyPersonaRoleBleed({
      message: "Zamykam laptopa. Pogadamy jutro.",
      llmClient: llm,
      model: "test-recall",
      personaName: "Tom",
    });

    expect(result).toMatchObject({
      flagged: false,
      category: "tom_persona",
      source: "llm",
    });
  });
});

describe("PersonaSession", () => {
  it("generates mock persona messages in sequence", async () => {
    const persona = new PersonaSession({
      persona: tomPersona,
      mock: true,
      mockMessages: [FIRST_TOM_TURN, SECOND_TOM_TURN],
    });

    const first = await persona.prepareNextTurn({ kind: "new_session" });
    persona.commit(first, FIRST_BORG_REPLY);
    const second = await persona.prepareNextTurn({ kind: "normal", text: FIRST_BORG_REPLY });
    persona.commit(second, SECOND_BORG_REPLY);
    const third = await persona.prepareNextTurn({ kind: "normal", text: SECOND_BORG_REPLY });

    expect(first.message).toBe(FIRST_TOM_TURN);
    expect(second.message).toBe(SECOND_TOM_TURN);
    expect(third.message).toBe(FIRST_TOM_TURN);
  });

  it("does not advance mock history when a draft is rolled back", async () => {
    const persona = new PersonaSession({
      persona: tomPersona,
      mock: true,
      mockMessages: [FIRST_TOM_TURN, SECOND_TOM_TURN],
    });

    const draft = await persona.prepareNextTurn({ kind: "new_session" });
    persona.rollback(draft);
    const retry = await persona.prepareNextTurn({ kind: "new_session" });

    expect(draft.message).toBe(FIRST_TOM_TURN);
    expect(retry.message).toBe(FIRST_TOM_TURN);
  });

  it("keeps new-session gap context across rolled-back llm drafts", async () => {
    const gapContext = "It's the next morning. You're at your desk with coffee.";
    const prompts: string[] = [];
    const client = {
      messages: {
        stream(params: { messages: Array<{ content: unknown }> }) {
          prompts.push(String(params.messages.at(-1)?.content ?? ""));
          return {
            async finalMessage() {
              return {
                content: [
                  {
                    type: "text",
                    text: `persona draft ${prompts.length}`,
                  },
                ],
              };
            },
          };
        },
      },
    };
    const persona = new PersonaSession({
      persona: tomPersona,
      client: client as never,
    });

    persona.startNewSession();
    const first = await persona.prepareNextTurn({ kind: "new_session", gapContext });
    persona.rollback(first);
    const second = await persona.prepareNextTurn({ kind: "new_session", gapContext });

    expect(prompts).toEqual([
      expect.stringContaining(gapContext),
      expect.stringContaining(gapContext),
    ]);
    expect(second.message).toBe("persona draft 2");
  });

  it("uses the continued-suppression prompt without producing role-bleed text", async () => {
    const prompts: string[] = [];
    const client = {
      messages: {
        stream(params: { messages: Array<{ content: unknown }> }) {
          prompts.push(String(params.messages.at(-1)?.content ?? ""));
          return {
            async finalMessage() {
              return {
                content: [
                  {
                    type: "text",
                    text: "Are you still there? I was asking because this keeps bothering me.",
                  },
                ],
              };
            },
          };
        },
      },
    };
    const persona = new PersonaSession({
      persona: tomPersona,
      client: client as never,
    });

    const draft = await persona.prepareNextTurn({
      kind: "continued_suppression",
      reason: "relational_guard_audit_failed",
    });

    expect(prompts[0]).toContain("Borg produced no visible response to your last message.");
    expect(prompts[0]).toContain("Continue as Tom.");
    expect(prompts[0]).not.toContain("Open the conversation");
    expect(detectPersonaRoleBleed(draft.message).matched).toHaveLength(0);
  });

  it("passes normal prior Borg text to the persona", async () => {
    const prompts: string[] = [];
    const client = {
      messages: {
        stream(params: { messages: Array<{ content: unknown }> }) {
          prompts.push(String(params.messages.at(-1)?.content ?? ""));
          return {
            async finalMessage() {
              return {
                content: [
                  {
                    type: "text",
                    text: "That helps. Can we stay with the same thread?",
                  },
                ],
              };
            },
          };
        },
      },
    };
    const persona = new PersonaSession({
      persona: tomPersona,
      client: client as never,
    });

    await persona.prepareNextTurn({ kind: "normal", text: FIRST_BORG_REPLY });

    expect(prompts).toEqual([FIRST_BORG_REPLY]);
  });
});
