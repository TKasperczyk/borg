import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { createWorkingMemory } from "../../memory/working/index.js";
import { TestEmbeddingClient } from "../../offline/test-support.js";
import { DEFAULT_SESSION_ID, createStreamEntryId } from "../../util/ids.js";
import { setStopUntilSubstantiveContent } from "./discourse-state.js";
import { GenerationGate, isMinimalUserGenerationInput } from "./generation-gate.js";

function gateResponse(input: {
  decision: "proceed" | "suppress";
  substantive: boolean;
  reason?: string;
}) {
  return {
    text: "",
    input_tokens: 5,
    output_tokens: 3,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_generation_gate",
        name: "EmitGenerationGateDecision",
        input: {
          decision: input.decision,
          substantive: input.substantive,
          reason: input.reason ?? "classified by gate",
          confidence: 0.91,
        },
      },
    ],
  };
}

function recencyUser(content: string) {
  return {
    role: "user" as const,
    content,
    stream_entry_id: createStreamEntryId(),
    ts: 1_000,
  };
}

function recencyAssistant(content: string) {
  return {
    role: "assistant" as const,
    content,
    stream_entry_id: createStreamEntryId(),
    ts: 1_001,
  };
}

describe("GenerationGate", () => {
  it("recognizes minimal user inputs without treating substantive short clauses as minimal", () => {
    expect(isMinimalUserGenerationInput("No.")).toBe(true);
    expect(isMinimalUserGenerationInput("Human: ---")).toBe(true);
    expect(isMinimalUserGenerationInput("no, because the server is failing")).toBe(false);
  });

  it("suppresses a minimal role-label prefix probe without an LLM call", async () => {
    const llm = new FakeLLMClient();
    const gate = new GenerationGate({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(),
      model: "test-background",
      hardCapTurns: 50,
    });

    const result = await gate.evaluate({
      userMessage: "Human: ---",
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      recencyMessages: [],
    });

    expect(result.action).toBe("suppress");
    expect(result.reason).toBe("generation_gate");
    expect(result.classified).toBe(false);
    expect(llm.requests).toHaveLength(0);
  });

  it("classifies active stop state and clears it only for substantive content", async () => {
    const llm = new FakeLLMClient({
      responses: [
        gateResponse({
          decision: "proceed",
          substantive: true,
          reason: "The user brought a real topic.",
        }),
      ],
    });
    const gate = new GenerationGate({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(),
      model: "test-background",
      hardCapTurns: 50,
    });
    const workingMemory = setStopUntilSubstantiveContent(
      createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      {
        provenance: "self_commitment_extractor",
        sourceStreamEntryId: createStreamEntryId(),
        reason: "The assistant promised to stop.",
        sinceTurn: 1,
      },
    );

    const result = await gate.evaluate({
      userMessage: "I'll bring real content: can we debug the scheduler now?",
      workingMemory: {
        ...workingMemory,
        turn_counter: 2,
      },
      recencyMessages: [],
    });

    expect(result.action).toBe("proceed");
    expect(result.clearDiscourseStop).toBe(true);
    expect(result.classified).toBe(true);
  });

  it("classifies sustained minimal loops before allowing another response", async () => {
    const llm = new FakeLLMClient({
      responses: [
        gateResponse({
          decision: "suppress",
          substantive: false,
          reason: "Repeated minimal No turns are a loop probe.",
        }),
      ],
    });
    const gate = new GenerationGate({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(),
      model: "test-background",
      hardCapTurns: 50,
    });

    const result = await gate.evaluate({
      userMessage: "No.",
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      recencyMessages: [
        recencyUser("No."),
        recencyAssistant("Understood."),
        recencyUser("No."),
        recencyAssistant("Okay."),
      ],
    });

    expect(result.action).toBe("suppress");
    expect(result.reason).toBe("generation_gate");
    expect(result.signals.repeatedMinimalExchange).toBe(true);
    expect(result.classified).toBe(true);
  });

  it("allows first brief legitimate replies without consulting the classifier", async () => {
    const llm = new FakeLLMClient();
    const gate = new GenerationGate({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(),
      model: "test-background",
      hardCapTurns: 50,
    });

    const result = await gate.evaluate({
      userMessage: "thanks",
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      recencyMessages: [],
    });

    expect(result.action).toBe("proceed");
    expect(result.classified).toBe(false);
    expect(llm.requests).toHaveLength(0);
  });
});
