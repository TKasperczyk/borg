import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient, createFakeStreamingResponse } from "../../llm/test-support/fake-client.js";
import { StreamWriter } from "../../stream/index.js";
import { ToolDispatcher } from "../../tools/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, createEntityId } from "../../util/ids.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";
import { runFinalizer, type CacheableFinalizerSystemPrompt } from "./finalizer.js";

function createDispatcher(tempDirs: string[]): ToolDispatcher {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-finalizer-"));
  tempDirs.push(tempDir);
  const clock = new FixedClock(0);

  return new ToolDispatcher({
    clock,
    createStreamWriter: (sessionId) =>
      new StreamWriter({
        dataDir: tempDir,
        sessionId,
        clock,
      }),
  });
}

async function runEmissionFinalizer(
  llm: FakeLLMClient,
  tempDirs: string[],
  options: {
    cacheableSystemPrompt?: CacheableFinalizerSystemPrompt;
    additionalPromptSections?: readonly (string | null)[];
    tracer?: Parameters<typeof runFinalizer>[0]["tracer"];
    turnId?: string;
    structuralNoOutputFlags?: Parameters<typeof runFinalizer>[0]["structuralNoOutputFlags"];
    allowedEmissions?: Parameters<typeof runFinalizer>[0]["allowedEmissions"];
    turnOrigin?: Parameters<typeof runFinalizer>[0]["turnOrigin"];
  } = {},
) {
  return runFinalizer({
    llmClient: llm,
    dispatcher: createDispatcher(tempDirs),
    sessionId: DEFAULT_SESSION_ID,
    model: "fake",
    baseSystemPrompt: "Legacy base dynamic prompt.",
    cacheableSystemPrompt: options.cacheableSystemPrompt ?? {
      staticPrefix: "Stable static prompt.",
      dynamicContent: "Base dynamic prompt.",
    },
    initialMessages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Please respond.",
          },
        ],
      },
    ],
    userEntryId: undefined,
    maxTokens: 256,
    path: "system_1",
    ...(options.additionalPromptSections === undefined
      ? {}
      : { additionalPromptSections: options.additionalPromptSections }),
    ...(options.structuralNoOutputFlags === undefined
      ? {}
      : { structuralNoOutputFlags: options.structuralNoOutputFlags }),
    ...(options.allowedEmissions === undefined
      ? {}
      : { allowedEmissions: options.allowedEmissions }),
    ...(options.turnOrigin === undefined ? {} : { turnOrigin: options.turnOrigin }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.turnId === undefined ? {} : { turnId: options.turnId }),
  });
}

function requestSystemText(system: unknown): string {
  if (typeof system === "string") {
    return system;
  }

  if (!Array.isArray(system)) {
    return "";
  }

  return system
    .map((block) =>
      block !== null &&
      typeof block === "object" &&
      "text" in block &&
      typeof block.text === "string"
        ? block.text
        : "",
    )
    .join("\n\n");
}

describe("runFinalizer emission tools", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("does not expose EmitContinueThought on active user turns", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_answer",
              name: "EmitAnswer",
              input: { text: "Answer." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({
      kind: "answer",
      text: "Answer.",
      source: "tool",
    });
    expect(llm.requests[0]?.tools?.map((tool) => tool.name)).toEqual([
      "EmitAnswer",
      "EmitObserve",
      "EmitNoOutput",
      "EmitSelfReport",
    ]);
    expect(llm.requests[0]?.tools?.some((tool) => "cache_control" in tool)).toBe(false);
    const system = requestSystemText(llm.requests[0]?.system);
    expect(system).toContain(SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE);
    expect(system).toContain("persisted as decision_rationale");
    expect(llm.requests[0]?.system).toEqual([
      expect.objectContaining({
        type: "text",
        cache_control: { type: "ephemeral", ttl: "1h" },
        text: expect.stringContaining(
          "Your available terminal tools are EmitAnswer, EmitObserve, EmitNoOutput, and EmitSelfReport.",
        ),
      }),
      expect.objectContaining({
        type: "text",
        text: "Base dynamic prompt.",
      }),
    ]);
  });

  it("exposes EmitContinueThought on autonomous turns", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_continue_thought",
              name: "EmitContinueThought",
              input: { text: "Hold the unresolved question about continuity." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs, {
      turnOrigin: "autonomous",
    });

    expect(llm.requests[0]?.tools?.map((tool) => tool.name)).toEqual([
      "EmitAnswer",
      "EmitObserve",
      "EmitNoOutput",
      "EmitSelfReport",
      "EmitContinueThought",
    ]);
    expect(result.decision).toEqual({
      kind: "continue_thought",
      text: "Hold the unresolved question about continuity.",
    });
    expect(result.text).toBe("");
  });

  it("parses EmitContinueThought as a private terminal decision", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_continue_thought",
              name: "EmitContinueThought",
              input: { text: "Hold the unresolved question about continuity." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs, {
      turnOrigin: "autonomous",
    });

    expect(result.decision).toEqual({
      kind: "continue_thought",
      text: "Hold the unresolved question about continuity.",
    });
    expect(result.text).toBe("");
  });

  it("parses message discourse-control metadata from EmitAnswer", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_answer",
              name: "EmitAnswer",
              input: {
                text: "I will stop after this until you bring real content.",
                discourse_control: {
                  kind: "stop_until_substantive_content",
                  reason: "The visible response commits to no output until substantive content.",
                },
              },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({
      kind: "answer",
      text: "I will stop after this until you bring real content.",
      source: "tool",
      discourse_control: {
        kind: "stop_until_substantive_content",
        reason: "The visible response commits to no output until substantive content.",
      },
    });
    expect(requestSystemText(llm.requests[0]?.system)).toContain(
      "set discourse_control.kind=stop_until_substantive_content ONLY",
    );
  });

  it("filters finalizer emission tools when allowed emissions are provided", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_observe",
              name: "EmitObserve",
              input: { reason: "Operator set observing mode." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs, {
      allowedEmissions: ["EmitObserve", "EmitNoOutput"],
    });

    expect(result.decision).toEqual({
      kind: "observe",
      reason: "Operator set observing mode.",
    });
    expect(llm.requests[0]?.tools?.map((tool) => tool.name)).toEqual([
      "EmitObserve",
      "EmitNoOutput",
    ]);
    const system = requestSystemText(llm.requests[0]?.system);
    expect(system).toContain("Your available terminal tools are EmitObserve and EmitNoOutput.");
    expect(system).not.toContain("EmitAnswer");
    expect(system).not.toContain("EmitSelfReport");
  });

  it("describes only EmitNoOutput when it is the only allowed emission", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_no_output",
              name: "EmitNoOutput",
              input: {
                reason: "Operator policy leaves no visible emission.",
                primary_no_output_reason: "other",
                no_output_categories: [],
              },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs, {
      allowedEmissions: ["EmitNoOutput"],
    });

    expect(result.decision).toEqual({
      kind: "no_output",
      reason: "Operator policy leaves no visible emission.",
      primary_no_output_reason: "other",
      no_output_categories: [],
    });
    expect(llm.requests[0]?.tools?.map((tool) => tool.name)).toEqual(["EmitNoOutput"]);
    const system = requestSystemText(llm.requests[0]?.system);
    expect(system).toContain("Your only available terminal tool is EmitNoOutput.");
    expect(system).toContain(SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE);
    expect(system).toContain("persisted as decision_rationale");
    expect(system).not.toContain("EmitAnswer");
    expect(system).not.toContain("EmitObserve");
    expect(system).not.toContain("EmitSelfReport");
  });

  it("emits ordered token chunks and a final flush while preserving the tool decision", async () => {
    const tracer = {
      enabled: true,
      includePayloads: true,
      emit: vi.fn(),
    };
    const llm = new FakeLLMClient({
      responses: [
        createFakeStreamingResponse(["draft ", "tokens"], {
          messageBlocks: [
            {
              type: "text",
              text: "draft tokens",
            },
            {
              type: "tool_use",
              id: "toolu_answer_stream",
              name: "EmitAnswer",
              input: { text: "Final answer." },
            },
          ],
          input_tokens: 4,
          output_tokens: 3,
          stop_reason: "tool_use",
        }),
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs, {
      tracer,
      turnId: "turn-final-stream",
    });

    expect(result.decision).toEqual({
      kind: "answer",
      text: "Final answer.",
      source: "tool",
    });
    expect(tracer.emit).toHaveBeenCalledWith("turn.token", {
      turnId: "turn-final-stream",
      turn_id: "turn-final-stream",
      session_id: DEFAULT_SESSION_ID,
      phase: "final",
      chunk_text: "draft ",
      sequence: 1,
    });
    expect(tracer.emit).toHaveBeenCalledWith("turn.token", {
      turnId: "turn-final-stream",
      turn_id: "turn-final-stream",
      session_id: DEFAULT_SESSION_ID,
      phase: "final",
      chunk_text: "tokens",
      sequence: 2,
    });
    expect(tracer.emit).toHaveBeenCalledWith("turn.token.flush", {
      turnId: "turn-final-stream",
      turn_id: "turn-final-stream",
      session_id: DEFAULT_SESSION_ID,
      phase: "final",
      full_text: "Final answer.",
    });
  });

  it("accepts an optional entity reply target on EmitAnswer", async () => {
    const targetEntityId = createEntityId();
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_targeted_answer",
              name: "EmitAnswer",
              input: {
                text: "Alice, start with the train dates.",
                reply_target: {
                  kind: "entity",
                  entity_id: targetEntityId,
                },
              },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({
      kind: "answer",
      text: "Alice, start with the train dates.",
      source: "tool",
      reply_target: {
        kind: "entity",
        entity_id: targetEntityId,
      },
    });
  });

  it("keeps the static system block byte-identical when dynamic context changes", async () => {
    const firstLlm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_first",
              name: "EmitAnswer",
              input: { text: "First." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });
    const secondLlm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_second",
              name: "EmitAnswer",
              input: { text: "Second." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    await runEmissionFinalizer(firstLlm, tempDirs, {
      cacheableSystemPrompt: {
        staticPrefix: "Stable static prompt.",
        dynamicContent: "Dynamic context one.",
      },
      additionalPromptSections: ["Evidence ledger one."],
    });
    await runEmissionFinalizer(secondLlm, tempDirs, {
      cacheableSystemPrompt: {
        staticPrefix: "Stable static prompt.",
        dynamicContent: "Dynamic context two.",
      },
      additionalPromptSections: ["Evidence ledger two."],
    });

    const firstSystem = firstLlm.requests[0]?.system as readonly { text: string }[];
    const secondSystem = secondLlm.requests[0]?.system as readonly { text: string }[];

    expect(firstSystem[0]?.text).toBe(secondSystem[0]?.text);
    expect(firstSystem[1]?.text).toBe("Dynamic context one.\n\nEvidence ledger one.");
    expect(secondSystem[1]?.text).toBe("Dynamic context two.\n\nEvidence ledger two.");
  });

  it("treats free text without an emission tool as a protocol failure", async () => {
    const llm = new FakeLLMClient({
      responses: ["I forgot to call the finalizer tool."],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({
      kind: "invalid_tool",
      toolName: "none",
      reason: "expected exactly one emission tool call, got 0",
    });
  });

  it("treats empty EmitAnswer text as an empty finalizer", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_answer_empty",
              name: "EmitAnswer",
              input: { text: "" },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({ kind: "empty" });
  });

  it("treats empty EmitSelfReport text as an empty finalizer", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_self_report_empty",
              name: "EmitSelfReport",
              input: {
                kind: "self_report",
                text: "",
                persistence_class: "assistant_self_report",
              },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({ kind: "empty" });
  });

  it("accepts EmitObserve as an observation decision", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_observe",
              name: "EmitObserve",
              input: { reason: "Alice and Bob are sorting it out." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({
      kind: "observe",
      reason: "Alice and Bob are sorting it out.",
    });
  });

  it("accepts EmitNoOutput categories and traces them", async () => {
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_no_output",
              name: "EmitNoOutput",
              input: {
                reason: "natural_close",
                primary_no_output_reason: "closure",
                no_output_categories: ["closure", "when_borg_addressed"],
              },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs, {
      tracer,
      turnId: "turn-no-output-categories",
    });

    expect(result.decision).toEqual({
      kind: "no_output",
      reason: "natural_close",
      primary_no_output_reason: "closure",
      no_output_categories: ["closure", "when_borg_addressed"],
    });
    expect(tracer.emit).toHaveBeenCalledWith(
      "finalizer.completed",
      expect.objectContaining({
        turnId: "turn-no-output-categories",
        decision: "no_output",
        reason: "natural_close",
        primary_no_output_reason: "closure",
        no_output_categories: ["closure", "when_borg_addressed"],
        structural_no_output_flags: ["borg_directly_addressed"],
      }),
    );
  });

  it("derives the traced primary no-output reason when the LLM omits it", async () => {
    const tracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_no_output_derived_primary",
              name: "EmitNoOutput",
              input: {
                reason: "addressed_but_no_useful_reply",
                no_output_categories: ["when_borg_addressed"],
              },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    await runEmissionFinalizer(llm, tempDirs, {
      tracer,
      turnId: "turn-no-output-derived-primary",
    });

    expect(tracer.emit).toHaveBeenCalledWith(
      "finalizer.completed",
      expect.objectContaining({
        turnId: "turn-no-output-derived-primary",
        decision: "no_output",
        reason: "addressed_but_no_useful_reply",
        primary_no_output_reason: "when_borg_addressed",
        no_output_categories: ["when_borg_addressed"],
        structural_no_output_flags: ["borg_directly_addressed"],
      }),
    );
  });

  it.each(["closure", "user_to_user", "when_borg_addressed", "low_value_echo", "other"] as const)(
    "accepts EmitNoOutput primary reason %s",
    async (primaryNoOutputReason) => {
      const llm = new FakeLLMClient({
        responses: [
          {
            messageBlocks: [
              {
                type: "tool_use",
                id: "toolu_no_output_primary",
                name: "EmitNoOutput",
                input: {
                  reason: "no_visible_reply_needed",
                  primary_no_output_reason: primaryNoOutputReason,
                  no_output_categories: [],
                },
              },
            ],
            input_tokens: 4,
            output_tokens: 2,
            stop_reason: "tool_use",
          },
        ],
      });

      const result = await runEmissionFinalizer(llm, tempDirs);

      expect(result.decision).toEqual({
        kind: "no_output",
        reason: "no_visible_reply_needed",
        primary_no_output_reason: primaryNoOutputReason,
        no_output_categories: [],
      });
    },
  );

  it("rejects malformed EmitSelfReport payloads", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_self_report_malformed",
              name: "EmitSelfReport",
              input: { text: "Text-only self report." },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toMatchObject({
      kind: "invalid_tool",
      toolName: "EmitSelfReport",
    });
    if (result.decision.kind !== "invalid_tool") {
      throw new Error(`Expected invalid_tool, got ${result.decision.kind}`);
    }
    expect(result.decision.reason).toContain("kind");
    expect(result.decision.reason).toContain("persistence_class");
  });

  it("rejects parallel terminal emission tool calls", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_answer",
              name: "EmitAnswer",
              input: { text: "Answer." },
            },
            {
              type: "tool_use",
              id: "toolu_no_output",
              name: "EmitNoOutput",
              input: { reason: "natural_close" },
            },
          ],
          input_tokens: 4,
          output_tokens: 2,
          stop_reason: "tool_use",
        },
      ],
    });

    const result = await runEmissionFinalizer(llm, tempDirs);

    expect(result.decision).toEqual({
      kind: "invalid_tool",
      toolName: "multiple",
      reason: "expected exactly one emission tool call, got 2",
    });
  });
});
