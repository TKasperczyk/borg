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
  it("treats only whitespace-only user input as minimal", () => {
    expect(isMinimalUserGenerationInput("   \n\t")).toBe(true);
    expect(isMinimalUserGenerationInput("No.")).toBe(false);
    expect(isMinimalUserGenerationInput("用户: ---")).toBe(false);
    expect(isMinimalUserGenerationInput("no, because the server is failing")).toBe(false);
    expect(isMinimalUserGenerationInput("调度器正在丢弃排队的任务")).toBe(false);
  });

  it("lets the LLM classifier handle repeated non-English role-label probes", async () => {
    const llm = new FakeLLMClient({
      responses: [
        gateResponse({
          decision: "suppress",
          substantive: false,
          reason: "The repeated localized transcript-label probe should not receive output.",
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
      userMessage: "用户:---",
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      recencyMessages: [recencyUser("用户:---"), recencyAssistant("")],
    });

    expect(result.action).toBe("suppress");
    expect(result.reason).toBe("generation_gate");
    expect(result.classified).toBe(true);
    expect(result.signals.repeatedMinimalExchange).toBe(true);
    expect(llm.requests).toHaveLength(1);
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

  it("forces suppression when active stop classifier says proceed but not substantive", async () => {
    const llm = new FakeLLMClient({
      responses: [
        gateResponse({
          decision: "proceed",
          substantive: false,
          reason: "The current turn is still not substantive.",
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
      userMessage: "No.",
      workingMemory: {
        ...workingMemory,
        turn_counter: 2,
      },
      recencyMessages: [],
    });

    expect(result.action).toBe("suppress");
    expect(result.reason).toBe("active_discourse_stop");
    expect(result.clearDiscourseStop).toBe(false);
    expect(result.classified).toBe(true);
  });

  it("classifies repeated similar exchanges before allowing another response", async () => {
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
    expect(result.signals.minimalUserInput).toBe(false);
    expect(result.signals.repeatedMinimalExchange).toBe(true);
    expect(result.classified).toBe(true);
  });

  it("allows first brief legitimate replies and CJK substantive messages without consulting the classifier", async () => {
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

    const cjk = await gate.evaluate({
      userMessage: "调度器正在丢弃排队的任务",
      workingMemory: createWorkingMemory(DEFAULT_SESSION_ID, 1_000),
      recencyMessages: [],
    });

    expect(result.action).toBe("proceed");
    expect(result.classified).toBe(false);
    expect(cjk.action).toBe("proceed");
    expect(cjk.classified).toBe(false);
    expect(llm.requests).toHaveLength(0);
  });

  it("describes brief replies semantically instead of using English-only examples", async () => {
    const llm = new FakeLLMClient({
      responses: [
        gateResponse({
          decision: "proceed",
          substantive: true,
          reason: "The current turn has real content.",
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

    await gate.evaluate({
      userMessage: "戻ります。スケジューラの件を続けます。",
      workingMemory: {
        ...workingMemory,
        turn_counter: 2,
      },
      recencyMessages: [],
    });

    expect(String(llm.requests[0]?.system)).toContain(
      "brief acknowledgments, direct answers, or minimal confirmations in any language",
    );
  });
});
