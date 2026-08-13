import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import { AuthError, LLMError } from "../../util/errors.js";
import { runS2Planner } from "./s2-planner.js";
import type { TurnOrigin } from "../types.js";
import type { PlannerContextTraceSummary } from "./prompt/planner-context.js";

type ToolInputSchema = {
  properties: Record<string, unknown>;
  required?: string[];
  [key: string]: unknown;
};

const EXPECTED_USER_TURN_PLAN_INPUT_SCHEMA_JSON = JSON.stringify({
  $schema: "https://json-schema.org/draft/2020-12/schema",
  type: "object",
  properties: {
    uncertainty: {
      type: "string",
      description:
        "What's unclear about the current participant input that matters for the engagement decision or answer? Empty string if nothing.",
    },
    verification_steps: {
      type: "array",
      items: { type: "string" },
      description:
        "Short phrases describing what I should double-check or re-retrieve before engaging. Empty array if nothing.",
    },
    tensions: {
      type: "array",
      items: { type: "string" },
      description:
        "Conflicts or contradictions in what I already know that need to be reconciled if I respond. Empty array if none.",
    },
    voice_note: {
      type: "string",
      description:
        "How the voice and posture should land for this specific turn. Empty string if default voice fits.",
    },
    emission_recommendation: {
      default: "emit",
      description:
        "I use no_output only when the conversation has naturally closed and the correct current-turn behavior is to emit no visible message at all; otherwise I use emit and let the finalizer choose visible speech or observation.",
      type: "string",
      enum: ["emit", "no_output"],
    },
    intents: {
      type: "array",
      items: {
        type: "object",
        properties: {
          description: { type: "string", minLength: 1 },
          next_action: {
            anyOf: [{ type: "string", minLength: 1 }, { type: "null" }],
          },
        },
        required: ["description", "next_action"],
      },
      description:
        "Follow-up intent records to carry into working memory after this turn. I include only concrete future actions I actually intend to track, not stylistic next-step wording.",
    },
  },
  required: ["uncertainty", "verification_steps", "tensions", "voice_note", "intents"],
});

function createTracer() {
  const emit = vi.fn<TurnTracer["emit"]>();

  return {
    enabled: true,
    includePayloads: false,
    emit,
  } satisfies TurnTracer & { emit: typeof emit };
}

function validPlanInput(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    uncertainty: "",
    verification_steps: [],
    tensions: [],
    voice_note: "",
    emission_recommendation: "emit",
    intents: [],
    ...overrides,
  };
}

function expectToolInputSchema(schema: unknown): asserts schema is ToolInputSchema {
  expect(schema).toMatchObject({ type: "object" });
  expect(schema).toHaveProperty("properties");
}

async function capturePlannerToolInputSchema(turnOrigin?: TurnOrigin): Promise<ToolInputSchema> {
  const llm = new FakeLLMClient({
    responses: [
      {
        text: "",
        input_tokens: 1,
        output_tokens: 1,
        stop_reason: "tool_use",
        tool_calls: [
          {
            id: "toolu_plan_schema",
            name: "EmitTurnPlan",
            input: validPlanInput(),
          },
        ],
      },
    ],
  });

  await runS2Planner({
    llmClient: llm,
    model: "sonnet",
    baseSystemPrompt: "base",
    dialogueMessages: [{ role: "user", content: "Think this through." }],
    selfSnapshot: { values: [], goals: [], traits: [] },
    maxTokens: 512,
    ...(turnOrigin === undefined ? {} : { turnOrigin }),
  });

  const schema = llm.requests[0]?.tools?.[0]?.inputSchema;

  expectToolInputSchema(schema);

  return schema;
}

describe("s2 planner", () => {
  it("sends the compact system blocks unchanged and traces their surface budget", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_compact_plan",
              name: "EmitTurnPlan",
              input: validPlanInput(),
            },
          ],
        },
      ],
    });
    const tracer = createTracer();
    const system = [
      {
        type: "text" as const,
        text: "static planner head",
        cache_control: { type: "ephemeral" as const, ttl: "1h" as const },
      },
      { type: "text" as const, text: "durable self digest" },
      { type: "text" as const, text: "turn-local planner context" },
    ];
    const traceSummary = {
      variant: "compact",
      sections: {
        static_head: {
          chars: 19,
          estimatedTokens: 5,
          rowCount: 0,
          truncationCount: 0,
          omissionCount: 0,
          criticalOverflow: false,
        },
      },
      targetTokens: 40_000,
      totalChars: 73,
      totalEstimatedTokens: 19,
      rowCount: 3,
      truncationCount: 1,
      omissionCount: 2,
      criticalOverflow: false,
      overallOverflow: false,
    } satisfies PlannerContextTraceSummary;

    await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "legacy must not be sent",
      dialogueMessages: [{ role: "user", content: "Think this through." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 512,
      tracer,
      turnId: "turn-compact-planner",
      plannerSurface: { variant: "compact", system, traceSummary },
    });

    expect(llm.requests[0]?.system).toEqual(system);
    expect(llm.requests[0]?.tools?.some((tool) => tool.cache_control !== undefined)).toBe(false);
    expect(tracer.emit).toHaveBeenCalledWith(
      "llm_call.started",
      expect.objectContaining({
        turnId: "turn-compact-planner",
        label: "s2_planner",
        planner_surface_variant: "compact",
        planner_context_summary: expect.objectContaining({
          variant: "compact",
          target_tokens: 40_000,
          total_chars: 73,
          total_estimated_tokens: 19,
          row_count: 3,
          truncation_count: 1,
          omission_count: 2,
          critical_overflow: false,
          overall_overflow: false,
          sections: {
            static_head: {
              chars: 19,
              estimated_tokens: 5,
              row_count: 0,
              truncation_count: 0,
              omission_count: 0,
              critical_overflow: false,
            },
          },
        }),
      }),
    );
  });

  it("keeps the user tool schema byte-identical while adding autonomous want first", async () => {
    const defaultSchema = await capturePlannerToolInputSchema();
    const userSchema = await capturePlannerToolInputSchema("user");
    const directedOutboundSchema = await capturePlannerToolInputSchema("directed_outbound");
    const autonomousSchema = await capturePlannerToolInputSchema("autonomous");

    expect(JSON.stringify(defaultSchema)).toBe(EXPECTED_USER_TURN_PLAN_INPUT_SCHEMA_JSON);
    expect(JSON.stringify(userSchema)).toBe(EXPECTED_USER_TURN_PLAN_INPUT_SCHEMA_JSON);
    expect(JSON.stringify(directedOutboundSchema)).toBe(EXPECTED_USER_TURN_PLAN_INPUT_SCHEMA_JSON);
    expect(defaultSchema.properties).not.toHaveProperty("want");
    expect(userSchema.properties).not.toHaveProperty("want");

    expect(JSON.stringify(autonomousSchema)).not.toBe(EXPECTED_USER_TURN_PLAN_INPUT_SCHEMA_JSON);
    expect(Object.keys(autonomousSchema.properties)[0]).toBe("want");
    expect(autonomousSchema.required ?? []).not.toContain("want");
    expect(autonomousSchema.properties.want).toMatchObject({
      default: "",
      description:
        "Answered before every field below: setting aside whether it is justified, is there anything I want in this interval? Named plainly in my own words. Naming a want does not oblige me to act on it. Empty string when nothing genuinely surfaces -- an empty answer is complete, and I do not manufacture a want to fill it.",
      type: "string",
    });
  });

  it("accepts and drops a stray want key on user-origin plan input", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan_user_stray_want",
              name: "EmitTurnPlan",
              input: validPlanInput({
                want: "stray user-origin field",
                voice_note: "stay direct",
              }),
            },
          ],
        },
      ],
    });

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "Think this through." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 512,
      turnOrigin: "user",
    });

    expect(result.plan).toMatchObject({ voice_note: "stay direct" });
    // Current user-schema parse behavior strips unknown keys instead of failing.
    // Pin that compatibility while keeping the user tool schema want-free.
    expect(Object.hasOwn(result.plan ?? {}, "want")).toBe(false);
  });

  it("retries once when the first response omits EmitTurnPlan", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "I forgot to emit the tool.",
          input_tokens: 4,
          output_tokens: 3,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan_retry",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "",
                verification_steps: ["confirm rollback state"],
                tensions: [],
                voice_note: "stay direct",
                emission_recommendation: "emit",
                intents: [
                  {
                    description: "Check rollback status after the next deploy",
                    next_action: "review deploy status",
                  },
                ],
              },
            },
          ],
        },
      ],
    });

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "Think this through." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 512,
    });

    expect(result.plan).toMatchObject({
      verification_steps: ["confirm rollback state"],
      voice_note: "stay direct",
      emission_recommendation: "emit",
      intents: [
        {
          description: "Check rollback status after the next deploy",
          next_action: "review deploy status",
        },
      ],
    });
    expect(llm.requests).toHaveLength(2);
    expect(llm.requests.map((request) => request.timeoutMs)).toEqual([720_000, 720_000]);
    expect(llm.requests[1]?.messages.at(-1)).toEqual({
      role: "user",
      content:
        "My previous response did not include the required EmitTurnPlan tool_use block. I emit one now -- this is the only way to complete the plan step.",
    });
    expect(result.usage).toMatchObject({
      input_tokens: 9,
      output_tokens: 7,
      stop_reason: "tool_use",
    });
  });

  it("parses an explicit no-output emission recommendation", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan_no_output",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "",
                verification_steps: [],
                tensions: ["Conversation has closed."],
                voice_note: "Do not narrate silence.",
                emission_recommendation: "no_output",
                intents: [],
              },
            },
          ],
        },
      ],
    });

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "No." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 512,
    });

    expect(result.plan?.emission_recommendation).toBe("no_output");
  });

  it("uses unary completion and emits completion traces without streaming token traces", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "plan tokens",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "tool_use",
          tool_calls: [
            {
              id: "toolu_plan_stream",
              name: "EmitTurnPlan",
              input: {
                uncertainty: "",
                verification_steps: [],
                tensions: [],
                voice_note: "stay direct",
                emission_recommendation: "emit",
                intents: [],
              },
            },
          ],
        },
      ],
    });
    const complete = vi.spyOn(llm, "complete");
    const streamComplete = vi.spyOn(llm, "streamComplete");
    const tracer = createTracer();

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "Think this through." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 512,
      tracer,
      turnId: "turn-plan-stream",
    });

    expect(result.reasoning).toBe("plan tokens");
    expect(complete).toHaveBeenCalledTimes(1);
    expect(streamComplete).not.toHaveBeenCalled();
    expect(llm.requests[0]?.timeoutMs).toBe(720_000);
    expect(tracer.emit).toHaveBeenCalledWith(
      "llm_call.completed",
      expect.objectContaining({
        turnId: "turn-plan-stream",
        label: "s2_planner",
        stopReason: "tool_use",
      }),
    );
    expect(tracer.emit).toHaveBeenCalledWith("deliberation.plan.completed", {
      turnId: "turn-plan-stream",
      success: true,
    });
    expect(tracer.emit).not.toHaveBeenCalledWith("turn.token", expect.anything());
    expect(tracer.emit).not.toHaveBeenCalledWith("turn.token.flush", expect.anything());
  });

  it("emits exhaustion trace when both planner attempts omit EmitTurnPlan", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "First miss.",
          input_tokens: 4,
          output_tokens: 3,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        {
          text: "Second miss.",
          input_tokens: 5,
          output_tokens: 4,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const complete = vi.spyOn(llm, "complete");
    const streamComplete = vi.spyOn(llm, "streamComplete");
    const tracer = createTracer();

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "Think this through." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 512,
      tracer,
      turnId: "turn-1",
    });

    expect(result.plan).toBeNull();
    expect(complete).toHaveBeenCalledTimes(2);
    expect(streamComplete).not.toHaveBeenCalled();
    expect(llm.requests.map((request) => request.timeoutMs)).toEqual([720_000, 720_000]);
    expect(tracer.emit).toHaveBeenCalledWith("deliberation.planner.degraded", {
      turnId: "turn-1",
      attempts: 2,
      lastResponseShape: {
        textLength: "Second miss.".length,
        toolUseBlocks: [],
      },
    });
  });

  it("degrades a first-attempt transport timeout instead of failing the turn", async () => {
    const timeoutError = new LLMError("Planner transport timed out", {
      code: "LLM_CALL_TIMED_OUT",
    });
    const llm = new FakeLLMClient({
      responses: [() => Promise.reject(timeoutError)],
    });
    const tracer = createTracer();

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "Think this through." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 16_000,
      tracer,
      turnId: "turn-timeout",
    });

    expect(result).toEqual({
      plan: null,
      reasoning: "",
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        stop_reason: null,
      },
    });
    expect(llm.requests).toHaveLength(1);
    expect(llm.requests[0]?.timeoutMs).toBe(720_000);
    expect(tracer.emit).toHaveBeenCalledWith("deliberation.planner.degraded", {
      turnId: "turn-timeout",
      attempts: 1,
      lastResponseShape: {
        error: "Planner transport timed out",
        code: "LLM_CALL_TIMED_OUT",
      },
    });
  });

  it("preserves first-attempt usage when the repair attempt has a transport failure", async () => {
    const connectionError = new LLMError("Planner connection failed", {
      code: "LLM_CONNECTION_FAILED",
    });
    const llm = new FakeLLMClient({
      responses: [
        {
          text: "First attempt omitted the plan.",
          input_tokens: 7,
          output_tokens: 5,
          stop_reason: "end_turn",
          tool_calls: [],
        },
        () => Promise.reject(connectionError),
      ],
    });
    const tracer = createTracer();

    const result = await runS2Planner({
      llmClient: llm,
      model: "sonnet",
      baseSystemPrompt: "base",
      dialogueMessages: [{ role: "user", content: "Think this through." }],
      selfSnapshot: { values: [], goals: [], traits: [] },
      maxTokens: 16_000,
      tracer,
      turnId: "turn-repair-connection",
    });

    expect(result).toEqual({
      plan: null,
      reasoning: "First attempt omitted the plan.",
      usage: {
        input_tokens: 7,
        output_tokens: 5,
        stop_reason: "end_turn",
      },
    });
    expect(llm.requests.map((request) => request.timeoutMs)).toEqual([720_000, 720_000]);
    expect(tracer.emit).toHaveBeenCalledWith("deliberation.planner.degraded", {
      turnId: "turn-repair-connection",
      attempts: 2,
      lastResponseShape: {
        error: "Planner connection failed",
        code: "LLM_CONNECTION_FAILED",
      },
    });
  });

  it.each([
    [
      "authentication",
      new AuthError("Planner authentication failed", { code: "AUTH_REFRESH_FAILED" }),
    ],
    [
      "schema",
      new LLMError("Planner schema failed", { code: "LLM_STRUCTURED_OUTPUT_PARSE_FAILED" }),
    ],
  ])("does not degrade %s failures", async (_kind, error) => {
    const llm = new FakeLLMClient({
      responses: [() => Promise.reject(error)],
    });

    await expect(
      runS2Planner({
        llmClient: llm,
        model: "sonnet",
        baseSystemPrompt: "base",
        dialogueMessages: [{ role: "user", content: "Think this through." }],
        selfSnapshot: { values: [], goals: [], traits: [] },
        maxTokens: 16_000,
      }),
    ).rejects.toBe(error);
  });
});
