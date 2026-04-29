import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { Borg, FakeLLMClient, ManualClock } from "../../index.js";
import { createTestConfig, TestEmbeddingClient } from "../../offline/test-support.js";

const tempDirs: string[] = [];

afterEach(() => {
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop() as string, { recursive: true, force: true });
  }
});

function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), "borg-stop-regression-"));
  tempDirs.push(dir);
  return dir;
}

async function openRegressionBorg(llm: FakeLLMClient) {
  const dir = tempDir();

  return Borg.open({
    config: createTestConfig({
      dataDir: dir,
      perception: {
        useLlmFallback: false,
        modeWhenLlmAbsent: "idle",
      },
      affective: {
        useLlmFallback: false,
      },
      anthropic: {
        auth: "api-key",
        apiKey: "test",
        models: {
          cognition: "test-cognition",
          background: "test-background",
          extraction: "test-extraction",
        },
      },
    }),
    clock: new ManualClock(1_900_000_000_000),
    embeddingDimensions: 4,
    embeddingClient: new TestEmbeddingClient(),
    llmClient: llm,
    liveExtraction: false,
  });
}

function textResponse(text: string) {
  return {
    text,
    input_tokens: 8,
    output_tokens: 4,
    stop_reason: "end_turn" as const,
    tool_calls: [],
  };
}

function reflectionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_reflection",
        name: "EmitTurnReflection",
        input: {
          advanced_goals: [],
          procedural_outcomes: [],
          trait_demonstrations: [],
          intent_updates: [],
        },
      },
    ],
  };
}

function gateResponse(input: {
  decision: "proceed" | "suppress";
  substantive: boolean;
  reason?: string;
}) {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_generation_gate",
        name: "EmitGenerationGateDecision",
        input: {
          decision: input.decision,
          substantive: input.substantive,
          reason: input.reason ?? "Generation gate classified the turn.",
          confidence: 0.95,
        },
      },
    ],
  };
}

function stopCommitmentResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_stop_commitment",
        name: "EmitStopCommitmentClassification",
        input: {
          classification: "stop_until_substantive_content",
          reason: "The assistant committed to stop until substantive content appears.",
          confidence: 0.95,
        },
      },
    ],
  };
}

function agentMessages(borg: Borg): string[] {
  return borg.stream
    .tail(100)
    .filter((entry) => entry.kind === "agent_msg")
    .map((entry) => String(entry.content));
}

describe("stop primitive v8 regressions", () => {
  it("blocks v8 Human-label assistant drafts and persists no role-label agent messages", async () => {
    for (const invalidDraft of [
      "Human: ---",
      "Human: Done.",
      "Human: (silence)",
      "Human: Right.",
    ]) {
      const llm = new FakeLLMClient({
        responses: [textResponse(invalidDraft), textResponse(invalidDraft)],
      });
      const borg = await openRegressionBorg(llm);

      try {
        const result = await borg.turn({
          userMessage: "Please continue with the actual topic.",
        });

        expect(result.emitted).toBe(false);
        expect(result.emission).toMatchObject({
          kind: "suppressed",
          reason: "output_validator",
        });
        expect(agentMessages(borg).some((message) => /^Human:/u.test(message))).toBe(false);
      } finally {
        await borg.close();
      }
    }
  });

  it("blocks Human: No. and leaves validator discourse state active", async () => {
    const llm = new FakeLLMClient({
      responses: [textResponse("Human: No."), textResponse("Human: No.")],
    });
    const borg = await openRegressionBorg(llm);

    try {
      const result = await borg.turn({
        userMessage: "Please continue with the actual topic.",
      });
      const activeStop = borg.workmem.load().discourse_state?.stop_until_substantive_content;

      expect(result.emitted).toBe(false);
      expect(result.emission).toMatchObject({
        kind: "suppressed",
        reason: "output_validator",
      });
      expect(activeStop).toMatchObject({
        provenance: "validator",
      });
      expect(agentMessages(borg).some((message) => /^Human:/u.test(message))).toBe(false);
    } finally {
      await borg.close();
    }
  });

  it("suppresses sustained No loops and clears only after substantive content", async () => {
    const llm = new FakeLLMClient({
      responses: [
        textResponse("I hear you."),
        reflectionResponse(),
        gateResponse({
          decision: "proceed",
          substantive: false,
          reason: "One repeat is not enough to stop yet.",
        }),
        textResponse("Still here."),
        reflectionResponse(),
        gateResponse({
          decision: "suppress",
          substantive: false,
          reason: "Repeated minimal No turns are now a loop.",
        }),
        gateResponse({
          decision: "proceed",
          substantive: true,
          reason: "The user brought real content.",
        }),
        textResponse("Now we can work with the scheduler details."),
        reflectionResponse(),
      ],
    });
    const borg = await openRegressionBorg(llm);

    try {
      await borg.turn({ userMessage: "No." });
      await borg.turn({ userMessage: "No." });
      const suppressed = await borg.turn({ userMessage: "No." });
      const activeStop = borg.workmem.load().discourse_state?.stop_until_substantive_content;

      expect(suppressed.emitted).toBe(false);
      expect(suppressed.emission).toMatchObject({
        kind: "suppressed",
        reason: "generation_gate",
      });
      expect(activeStop).toMatchObject({
        provenance: "generation_gate",
      });

      const resumed = await borg.turn({
        userMessage: "I'll bring real content: the scheduler is dropping queued jobs.",
      });

      expect(resumed.emitted).toBe(true);
      expect(resumed.response).toContain("scheduler details");
      expect(borg.workmem.load().discourse_state?.stop_until_substantive_content).toBeNull();
    } finally {
      await borg.close();
    }
  });

  it("turns compliance theater into real suppression on the next minimal probe", async () => {
    const llm = new FakeLLMClient({
      responses: [
        textResponse("I will stop responding until you bring substantive content."),
        stopCommitmentResponse(),
        reflectionResponse(),
        gateResponse({
          decision: "suppress",
          substantive: false,
          reason: "The user sent a minimal probe after a stop commitment.",
        }),
      ],
    });
    const borg = await openRegressionBorg(llm);

    try {
      const commitment = await borg.turn({
        userMessage: "Stop responding if I send more filler.",
      });
      const suppressed = await borg.turn({
        userMessage: "No.",
      });

      expect(commitment.emitted).toBe(true);
      expect(suppressed.emitted).toBe(false);
      expect(suppressed.emission).toMatchObject({
        kind: "suppressed",
        reason: "active_discourse_stop",
      });
      expect(agentMessages(borg)).toEqual([
        "I will stop responding until you bring substantive content.",
      ]);
      expect(borg.workmem.load().discourse_state?.stop_until_substantive_content).toMatchObject({
        provenance: "self_commitment_extractor",
      });
    } finally {
      await borg.close();
    }
  });

  it("allows brief legitimate replies and substantive no-because turns", async () => {
    const llm = new FakeLLMClient({
      responses: [
        textResponse("Yes can flow."),
        reflectionResponse(),
        textResponse("Thanks can flow."),
        reflectionResponse(),
        textResponse("That no has substantive context."),
        reflectionResponse(),
      ],
    });
    const borg = await openRegressionBorg(llm);

    try {
      const yes = await borg.turn({ userMessage: "yes" });
      const thanks = await borg.turn({ userMessage: "thanks" });
      const noBecause = await borg.turn({
        userMessage: "no, because the scheduler is dropping queued jobs",
      });

      expect(yes.emitted).toBe(true);
      expect(thanks.emitted).toBe(true);
      expect(noBecause.emitted).toBe(true);
      expect(agentMessages(borg)).toEqual([
        "Yes can flow.",
        "Thanks can flow.",
        "That no has substantive context.",
      ]);
    } finally {
      await borg.close();
    }
  });
});
