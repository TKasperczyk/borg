import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { FakeLLMClient, createFakeStreamingResponse } from "../../llm/test-support/fake-client.js";
import { createEpisodeFixture, createRetrievalScoreFixture } from "../../offline/test-support.js";
import { StreamWriter } from "../../stream/index.js";
import { ToolDispatcher, type ToolDefinition, type ToolOrigin } from "../../tools/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID, createEntityId } from "../../util/ids.js";
import {
  PROMPT_SURFACES,
  promptSurfaceBlocksForSurface,
} from "../prompts/prompt-surface-registry.js";
import { resolveFinalizerNonTerminalTools } from "./autonomous-finalizer-tools.js";
import { OUTBOUND_POST_TOOL_NAME } from "../../tools/internal/outbound-post-name.js";
import type { DeliberationContext } from "./types.js";
import {
  FINALIZER_REGENERATION_PROMPT_BLOCK_IDS,
  runFinalizer,
  type CacheableFinalizerSystemPrompt,
} from "./finalizer.js";

function createDispatcher(
  tempDirs: string[],
  registeredTools: readonly ToolDefinition[] = [],
): ToolDispatcher {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-finalizer-"));
  tempDirs.push(tempDir);
  const clock = new FixedClock(0);

  const dispatcher = new ToolDispatcher({
    clock,
    createStreamWriter: (sessionId) =>
      new StreamWriter({
        dataDir: tempDir,
        sessionId,
        clock,
      }),
  });

  for (const tool of registeredTools) {
    dispatcher.register(tool);
  }

  return dispatcher;
}

async function runEmissionFinalizer(
  llm: FakeLLMClient,
  tempDirs: string[],
  options: {
    cacheableSystemPrompt?: CacheableFinalizerSystemPrompt;
    additionalPromptSections?: Parameters<typeof runFinalizer>[0]["additionalPromptSections"];
    finalizerDynamicPromptCacheEnabled?: boolean;
    finalizerTransport?: Parameters<typeof runFinalizer>[0]["finalizerTransport"];
    finalizerSurfaceVariant?: Parameters<typeof runFinalizer>[0]["finalizerSurfaceVariant"];
    tracer?: Parameters<typeof runFinalizer>[0]["tracer"];
    turnId?: string;
    structuralNoOutputFlags?: Parameters<typeof runFinalizer>[0]["structuralNoOutputFlags"];
    allowedEmissions?: Parameters<typeof runFinalizer>[0]["allowedEmissions"];
    outboundToolAvailable?: boolean;
    participationPolicy?: Parameters<typeof runFinalizer>[0]["participationPolicy"];
    turnOrigin?: Parameters<typeof runFinalizer>[0]["turnOrigin"];
    registeredTools?: readonly ToolDefinition[];
    compactSurface?: Parameters<typeof runFinalizer>[0]["compactSurface"];
  } = {},
) {
  return runFinalizer({
    llmClient: llm,
    dispatcher: createDispatcher(tempDirs, options.registeredTools),
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
    ...(options.finalizerDynamicPromptCacheEnabled === undefined
      ? {}
      : { finalizerDynamicPromptCacheEnabled: options.finalizerDynamicPromptCacheEnabled }),
    ...(options.finalizerTransport === undefined
      ? {}
      : { finalizerTransport: options.finalizerTransport }),
    ...(options.finalizerSurfaceVariant === undefined
      ? {}
      : { finalizerSurfaceVariant: options.finalizerSurfaceVariant }),
    ...(options.structuralNoOutputFlags === undefined
      ? {}
      : { structuralNoOutputFlags: options.structuralNoOutputFlags }),
    ...(options.allowedEmissions === undefined
      ? {}
      : { allowedEmissions: options.allowedEmissions }),
    ...(options.outboundToolAvailable === undefined
      ? {}
      : { outboundToolAvailable: options.outboundToolAvailable }),
    ...(options.participationPolicy === undefined
      ? {}
      : { participationPolicy: options.participationPolicy }),
    ...(options.turnOrigin === undefined ? {} : { turnOrigin: options.turnOrigin }),
    ...(options.compactSurface === undefined ? {} : { compactSurface: options.compactSurface }),
    ...(options.tracer === undefined ? {} : { tracer: options.tracer }),
    ...(options.turnId === undefined ? {} : { turnId: options.turnId }),
  });
}

function fakeTool(name: string, allowedOrigins: readonly ToolOrigin[]): ToolDefinition {
  return {
    name,
    description: `Description for ${name}.`,
    allowedOrigins,
    writeScope: "read",
    inputSchema: z.object({}).passthrough(),
    outputSchema: z.object({}).strict(),
    async invoke() {
      return {};
    },
  };
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
    .join("");
}

type CapturedSystemBlock = {
  type: "text";
  text: string;
  cache_control?: {
    type: "ephemeral";
    ttl?: "5m" | "1h";
  };
};

const USER_ACTIVE_TOOL_AVAILABILITY =
  '<borg_finalizer_tool_availability turn_origin="user" participation_policy="active" outbound_post="unavailable" enabled_terminal_emissions="EmitAnswer,EmitObserve,EmitNoOutput,EmitSelfReport" />';

function createAnsweringLlm(toolUseId: string): FakeLLMClient {
  return new FakeLLMClient({
    responses: [
      {
        messageBlocks: [
          {
            type: "tool_use",
            id: toolUseId,
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
}

function legacyTwoBlockSystemText(
  systemBlocks: readonly CapturedSystemBlock[],
  dynamicPrompt: string,
): string {
  return requestSystemText([
    systemBlocks[0],
    {
      type: "text",
      text: dynamicPrompt,
    },
  ]);
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
    expect(system).toContain(
      "Self-referential memory voice: when a prompt-visible structure identifies content as self-owned, I write self-referential content",
    );
    expect(system).toContain("persisted as decision_rationale");
    expect(llm.requests[0]?.system).toEqual([
      expect.objectContaining({
        type: "text",
        cache_control: { type: "ephemeral", ttl: "1h" },
        text: expect.stringContaining(
          "The origin-static advertised terminal tools are EmitAnswer, EmitObserve, EmitNoOutput, and EmitSelfReport",
        ),
      }),
      expect.objectContaining({
        type: "text",
        cache_control: { type: "ephemeral", ttl: "5m" },
        text: ["Base dynamic prompt.", USER_ACTIVE_TOOL_AVAILABILITY].join("\n\n"),
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

  it("exposes own-record browsing on user turns while autonomous-only write tools stay hidden", async () => {
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

    await runEmissionFinalizer(llm, tempDirs, {
      registeredTools: [
        fakeTool("tool.ownRecords.list", ["autonomous", "deliberator"]),
        fakeTool("tool.episodic.search", ["autonomous", "deliberator"]),
        fakeTool("tool.openQuestions.create", ["autonomous", "deliberator"]),
        fakeTool("tool.journal.append", ["autonomous"]),
      ],
    });

    expect(llm.requests[0]?.tools?.map((tool) => tool.name)).toEqual([
      "EmitAnswer",
      "EmitObserve",
      "EmitNoOutput",
      "EmitSelfReport",
      "tool.ownRecords.list",
    ]);
  });

  it("exposes the registered autonomous interior tool set by exact name", async () => {
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

    await runEmissionFinalizer(llm, tempDirs, {
      turnOrigin: "autonomous",
      registeredTools: [
        fakeTool("tool.unlisted.autonomous", ["autonomous"]),
        fakeTool("tool.ownRecords.list", ["autonomous", "deliberator"]),
        fakeTool("tool.openQuestions.create", ["autonomous", "deliberator"]),
        fakeTool("tool.goals.retire", ["autonomous", "deliberator"]),
        fakeTool("tool.journal.append", ["autonomous"]),
        fakeTool("tool.episodic.search", ["autonomous", "deliberator"]),
        fakeTool("tool.promptSurface.changes", ["autonomous"]),
      ],
    });

    expect(llm.requests[0]?.tools?.map((tool) => tool.name)).toEqual([
      "EmitAnswer",
      "EmitObserve",
      "EmitNoOutput",
      "EmitSelfReport",
      "EmitContinueThought",
      "tool.ownRecords.list",
      "tool.journal.append",
      "tool.openQuestions.create",
      "tool.goals.retire",
      "tool.episodic.search",
      "tool.promptSurface.changes",
    ]);
  });

  it("keeps autonomous tool schemas and block 0 stable across live availability states", async () => {
    const pausedLlm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_paused",
              name: "EmitNoOutput",
              input: {
                reason: "Participation is paused.",
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
    const activeLlm = createAnsweringLlm("toolu_active");
    const outboundTool = fakeTool(OUTBOUND_POST_TOOL_NAME, ["autonomous", "deliberator"]);

    await runEmissionFinalizer(pausedLlm, tempDirs, {
      turnOrigin: "autonomous",
      participationPolicy: "paused",
      allowedEmissions: ["EmitNoOutput"],
      outboundToolAvailable: false,
      registeredTools: [outboundTool],
    });
    await runEmissionFinalizer(activeLlm, tempDirs, {
      turnOrigin: "autonomous",
      participationPolicy: "active",
      outboundToolAvailable: true,
      registeredTools: [outboundTool],
    });

    expect(pausedLlm.requests[0]?.tools).toEqual(activeLlm.requests[0]?.tools);
    expect(pausedLlm.requests[0]?.tools?.map((tool) => tool.name)).toEqual([
      "EmitAnswer",
      "EmitObserve",
      "EmitNoOutput",
      "EmitSelfReport",
      "EmitContinueThought",
      OUTBOUND_POST_TOOL_NAME,
    ]);
    const pausedSystem = pausedLlm.requests[0]?.system as readonly CapturedSystemBlock[];
    const activeSystem = activeLlm.requests[0]?.system as readonly CapturedSystemBlock[];
    expect(pausedSystem[0]).toEqual(activeSystem[0]);
    expect(pausedSystem[1]?.text).toContain(
      'participation_policy="paused" outbound_post="unavailable" enabled_terminal_emissions="EmitNoOutput"',
    );
    expect(activeSystem[1]?.text).toContain(
      'participation_policy="active" outbound_post="available" enabled_terminal_emissions="EmitAnswer,EmitObserve,EmitNoOutput,EmitSelfReport,EmitContinueThought"',
    );
  });

  it("keeps compact tools and blocks 0-2 stable across turn-local changes", async () => {
    const firstNow = Date.UTC(2026, 7, 14, 12, 0, 0);
    const secondNow = firstNow + 60_000;
    const retrievedEpisode: DeliberationContext["retrievalResult"][number] = {
      episode: createEpisodeFixture({
        id: "ep_aaaaaaaaaaaaaaaa" as DeliberationContext["retrievalResult"][number]["episode"]["id"],
        title: "Turn-local retrieval",
        narrative: "This record appears only on the second build.",
        created_at: firstNow - 10_000,
        updated_at: firstNow - 10_000,
      }),
      score: 0.9,
      rawScore: 0.9,
      scoreBreakdown: createRetrievalScoreFixture({
        similarity: 0.9,
        decayedSalience: 0.3,
        heat: 1,
      }),
      citationChain: [],
    };
    const compactContext = (
      nowMs: number,
      participationPolicy: "active" | "paused",
      retrievalResult: DeliberationContext["retrievalResult"],
      outboundAvailable: boolean,
    ): DeliberationContext => ({
      sessionId: DEFAULT_SESSION_ID,
      nowMs,
      turnOrigin: "autonomous",
      participationPolicy,
      userMessage: "Continue the autonomous reflection.",
      perception: {
        entities: [],
        mode: "reflective",
        affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
        temporalCue: null,
      },
      retrievalResult,
      workingMemory: {
        session_id: DEFAULT_SESSION_ID,
        turn_counter: 3,
        hot_entities: [],
        pending_actions: [],
        pending_social_attribution: null,
        pending_trait_attribution: null,
        suppressed: [],
        mood: null,
        pending_procedural_attempts: [],
        discourse_state: { stop_until_substantive_content: null },
        mode: "reflective",
        updated_at: firstNow - 1_000,
      },
      selfSnapshot: { values: [], goals: [], traits: [] },
      autonomousFinalizerToolMenu: outboundAvailable
        ? [{ name: OUTBOUND_POST_TOOL_NAME, menuSummary: "Post to an authorized target." }]
        : [],
    });
    const pausedLlm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_compact_paused",
              name: "EmitNoOutput",
              input: {
                reason: "Participation is paused.",
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
    const activeLlm = createAnsweringLlm("toolu_compact_active");
    const outboundTool = fakeTool(OUTBOUND_POST_TOOL_NAME, ["autonomous", "deliberator"]);

    await runEmissionFinalizer(pausedLlm, tempDirs, {
      turnOrigin: "autonomous",
      participationPolicy: "paused",
      allowedEmissions: ["EmitNoOutput"],
      outboundToolAvailable: false,
      registeredTools: [outboundTool],
      finalizerSurfaceVariant: "compact",
      compactSurface: {
        context: compactContext(firstNow, "paused", [], false),
        baseSystemPromptOptions: {
          retrievalContextBudget: 10_000,
          semanticContextBudget: 10_000,
          nowMs: firstNow,
        },
      },
    });
    await runEmissionFinalizer(activeLlm, tempDirs, {
      turnOrigin: "autonomous",
      participationPolicy: "active",
      outboundToolAvailable: true,
      registeredTools: [outboundTool],
      finalizerSurfaceVariant: "compact",
      compactSurface: {
        context: compactContext(secondNow, "active", [retrievedEpisode], true),
        baseSystemPromptOptions: {
          retrievalContextBudget: 10_000,
          semanticContextBudget: 10_000,
          nowMs: secondNow,
        },
      },
    });

    const pausedRequest = pausedLlm.requests[0]!;
    const activeRequest = activeLlm.requests[0]!;
    const pausedSystem = pausedRequest.system as readonly CapturedSystemBlock[];
    const activeSystem = activeRequest.system as readonly CapturedSystemBlock[];
    expect(JSON.stringify(pausedRequest.tools)).toBe(JSON.stringify(activeRequest.tools));
    expect(JSON.stringify(pausedSystem.slice(0, 3))).toBe(JSON.stringify(activeSystem.slice(0, 3)));
    expect(JSON.stringify(pausedSystem[3])).not.toBe(JSON.stringify(activeSystem[3]));
  });

  it("sends each tool once when the live-turn and interior menus overlap", () => {
    // tool.openQuestions.ruminations is listed in BOTH menus by design. Shipping it twice made
    // the API reject the whole request ("tools: Tool names must be unique."), which killed every
    // autonomous wake for two days while user turns -- which take only the live-turn menu --
    // kept working.
    const dispatcher = createDispatcher(tempDirs, [
      fakeTool("tool.ownRecords.list", ["autonomous", "deliberator"]),
      fakeTool("tool.openQuestions.ruminations", ["autonomous", "deliberator"]),
      fakeTool("tool.journal.append", ["autonomous"]),
    ]);

    const names = resolveFinalizerNonTerminalTools({
      dispatcher,
      turnOrigin: "autonomous",
    }).map((tool) => tool.name);

    expect(names).toContain("tool.openQuestions.ruminations");
    expect(new Set(names).size).toBe(names.length);
  });

  it("exposes prompt-surface changes only to autonomous finalizer interiors", () => {
    const dispatcher = createDispatcher(tempDirs, [
      fakeTool("tool.promptSurface.changes", ["autonomous"]),
    ]);

    expect(dispatcher.listTools("autonomous").map((tool) => tool.name)).toContain(
      "tool.promptSurface.changes",
    );
    expect(dispatcher.listTools("deliberator").map((tool) => tool.name)).not.toContain(
      "tool.promptSurface.changes",
    );
    expect(
      resolveFinalizerNonTerminalTools({
        dispatcher,
        turnOrigin: "user",
      }).map((tool) => tool.name),
    ).not.toContain("tool.promptSurface.changes");
    expect(
      resolveFinalizerNonTerminalTools({
        dispatcher,
        turnOrigin: "autonomous",
      }).map((tool) => tool.name),
    ).toEqual(["tool.promptSurface.changes"]);
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

  it("keeps emission schemas advertised while structurally gating observing participation", async () => {
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
      participationPolicy: "observing",
    });

    expect(result.decision).toEqual({
      kind: "observe",
      reason: "Operator set observing mode.",
    });
    expect(llm.requests[0]?.tools?.map((tool) => tool.name)).toEqual([
      "EmitAnswer",
      "EmitObserve",
      "EmitNoOutput",
      "EmitSelfReport",
    ]);
    const system = requestSystemText(llm.requests[0]?.system);
    expect(system).toContain('enabled_terminal_emissions="EmitObserve,EmitNoOutput"');
    expect(system).toContain("The origin-static advertised terminal tools are EmitAnswer");
  });

  it("keeps emission schemas advertised while structurally gating no-output participation", async () => {
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
      participationPolicy: "paused",
    });

    expect(result.decision).toEqual({
      kind: "no_output",
      reason: "Operator policy leaves no visible emission.",
      primary_no_output_reason: "other",
      no_output_categories: [],
    });
    expect(llm.requests[0]?.tools?.map((tool) => tool.name)).toEqual([
      "EmitAnswer",
      "EmitObserve",
      "EmitNoOutput",
      "EmitSelfReport",
    ]);
    const system = requestSystemText(llm.requests[0]?.system);
    expect(system).toContain('enabled_terminal_emissions="EmitNoOutput"');
    expect(system).toContain(
      "Self-referential memory voice: when a prompt-visible structure identifies content as self-owned, I write self-referential content",
    );
    expect(system).toContain("persisted as decision_rationale");
    expect(system).toContain("An advertised tool whose live state is unavailable");
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
    const streamConverse = vi.spyOn(llm, "streamConverse");

    const result = await runEmissionFinalizer(llm, tempDirs, {
      tracer,
      turnId: "turn-final-stream",
      finalizerTransport: "streaming",
    });

    expect(result.decision).toEqual({
      kind: "answer",
      text: "Final answer.",
      source: "tool",
    });
    expect(streamConverse).toHaveBeenCalledOnce();
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

  it("uses unary transport and emits the accepted tool text as one final chunk", async () => {
    const tracer = {
      enabled: true,
      includePayloads: true,
      emit: vi.fn(),
    };
    const llm = new FakeLLMClient({
      responses: [
        createFakeStreamingResponse(["must not stream"], {
          messageBlocks: [
            {
              type: "text",
              text: "Loose prose is not delivered.",
            },
            {
              type: "tool_use",
              id: "toolu_answer_unary",
              name: "EmitAnswer",
              input: { text: "Unary final answer." },
            },
          ],
          input_tokens: 4,
          output_tokens: 3,
          stop_reason: "tool_use",
        }),
      ],
    });
    const converse = vi.spyOn(llm, "converse");
    const streamConverse = vi.spyOn(llm, "streamConverse");

    const result = await runEmissionFinalizer(llm, tempDirs, {
      tracer,
      turnId: "turn-final-unary",
      finalizerTransport: "unary",
    });

    expect(converse).toHaveBeenCalledOnce();
    expect(streamConverse).not.toHaveBeenCalled();
    expect(result.decision).toEqual({
      kind: "answer",
      text: "Unary final answer.",
      source: "tool",
    });
    expect(tracer.emit.mock.calls.filter(([event]) => event === "turn.token")).toEqual([
      [
        "turn.token",
        {
          turnId: "turn-final-unary",
          turn_id: "turn-final-unary",
          session_id: DEFAULT_SESSION_ID,
          phase: "final",
          chunk_text: "Unary final answer.",
          sequence: 1,
        },
      ],
    ]);
    expect(tracer.emit).toHaveBeenCalledWith("turn.token.flush", {
      turnId: "turn-final-unary",
      turn_id: "turn-final-unary",
      session_id: DEFAULT_SESSION_ID,
      phase: "final",
      full_text: "Unary final answer.",
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
      additionalPromptSections: [{ blockId: "borg_evidence_ledger", text: "Evidence ledger one." }],
    });
    await runEmissionFinalizer(secondLlm, tempDirs, {
      cacheableSystemPrompt: {
        staticPrefix: "Stable static prompt.",
        dynamicContent: "Dynamic context two.",
      },
      additionalPromptSections: [{ blockId: "borg_evidence_ledger", text: "Evidence ledger two." }],
    });

    const firstSystem = firstLlm.requests[0]?.system as readonly { text: string }[];
    const secondSystem = secondLlm.requests[0]?.system as readonly { text: string }[];

    expect(firstSystem[0]?.text).toBe(secondSystem[0]?.text);
    expect(firstSystem[1]?.text).toBe(
      ["Dynamic context one.", USER_ACTIVE_TOOL_AVAILABILITY, "Evidence ledger one."].join("\n\n"),
    );
    expect(secondSystem[1]?.text).toBe(
      ["Dynamic context two.", USER_ACTIVE_TOOL_AVAILABILITY, "Evidence ledger two."].join("\n\n"),
    );
  });

  it("keeps the default and explicit legacy finalizer block serialization byte-identical", async () => {
    const defaultLlm = createAnsweringLlm("toolu_default_legacy");
    const explicitLlm = createAnsweringLlm("toolu_explicit_legacy");
    const sections = [
      { blockId: "borg_evidence_ledger", text: "Exact ledger bytes." },
      { blockId: "borg_s2_plan", text: "Exact plan bytes." },
    ];
    await runEmissionFinalizer(defaultLlm, tempDirs, { additionalPromptSections: sections });
    await runEmissionFinalizer(explicitLlm, tempDirs, {
      additionalPromptSections: sections,
      finalizerSurfaceVariant: "legacy",
    });
    expect(explicitLlm.requests[0]?.system).toEqual(defaultLlm.requests[0]?.system);
    expect(JSON.stringify(explicitLlm.requests[0]?.system)).toBe(
      JSON.stringify(defaultLlm.requests[0]?.system),
    );
  });

  it("keeps regeneration content in a trailing uncached block without changing wire text", async () => {
    const llm = createAnsweringLlm("toolu_cached_regeneration");
    const rollbackLlm = createAnsweringLlm("toolu_rollback_regeneration");
    const stableSections = [
      { blockId: "borg_session_reentry_continuity", text: "Continuity fixture." },
      { blockId: "borg_evidence_ledger", text: "Ledger fixture." },
      { blockId: "borg_additional_retrieval", text: "Retrieval fixture." },
      { blockId: "borg_s2_plan", text: "Plan fixture." },
    ];
    const regenerationSection = {
      blockId: "borg_commitment_regeneration_instruction",
      text: "Regeneration fixture.",
    };
    const legacyDynamicPrompt = [
      "Base dynamic prompt.",
      USER_ACTIVE_TOOL_AVAILABILITY,
      ...stableSections.map((section) => section.text),
      regenerationSection.text,
    ].join("\n\n");

    await runEmissionFinalizer(llm, tempDirs, {
      additionalPromptSections: [...stableSections, regenerationSection],
    });
    await runEmissionFinalizer(rollbackLlm, tempDirs, {
      additionalPromptSections: [...stableSections, regenerationSection],
      finalizerDynamicPromptCacheEnabled: false,
    });

    const systemBlocks = llm.requests[0]?.system as readonly CapturedSystemBlock[];
    const rollbackSystemBlocks = rollbackLlm.requests[0]?.system as readonly CapturedSystemBlock[];

    expect(systemBlocks).toHaveLength(3);
    expect(rollbackSystemBlocks).toHaveLength(2);
    expect(systemBlocks[0]?.cache_control).toEqual({ type: "ephemeral", ttl: "1h" });
    expect(systemBlocks[1]?.cache_control).toBeUndefined();
    expect(systemBlocks[1]?.text).toBe(
      [
        "Base dynamic prompt.",
        USER_ACTIVE_TOOL_AVAILABILITY,
        ...stableSections.map((section) => section.text),
      ].join("\n\n"),
    );
    expect(systemBlocks[1]?.text).not.toContain(regenerationSection.text);
    expect(systemBlocks[2]).toEqual({
      type: "text",
      text: rollbackSystemBlocks[1]!.text.slice(systemBlocks[1]!.text.length),
    });
    expect(
      systemBlocks
        .slice(1)
        .map((block) => block.text)
        .join(""),
    ).toBe(legacyDynamicPrompt);
    expect(requestSystemText(systemBlocks)).toBe(requestSystemText(rollbackSystemBlocks));
  });

  it("omits the trailing block without regeneration content and preserves prior rendering", async () => {
    const llm = createAnsweringLlm("toolu_cached_stable");
    const stableSections = [
      { blockId: "borg_session_reentry_continuity", text: "Continuity fixture." },
      { blockId: "borg_evidence_ledger", text: "Ledger fixture." },
      { blockId: "borg_additional_retrieval", text: "Retrieval fixture." },
      { blockId: "borg_s2_plan", text: "Plan fixture." },
    ];
    const legacyDynamicPrompt = [
      "Base dynamic prompt.",
      USER_ACTIVE_TOOL_AVAILABILITY,
      ...stableSections.map((section) => section.text),
    ].join("\n\n");

    await runEmissionFinalizer(llm, tempDirs, {
      additionalPromptSections: stableSections,
    });

    const systemBlocks = llm.requests[0]?.system as readonly CapturedSystemBlock[];

    expect(systemBlocks).toHaveLength(2);
    expect(systemBlocks[1]?.cache_control).toEqual({ type: "ephemeral", ttl: "5m" });
    expect(
      systemBlocks
        .slice(1)
        .map((block) => block.text)
        .join(""),
    ).toBe(legacyDynamicPrompt);
    expect(requestSystemText(systemBlocks)).toBe(
      legacyTwoBlockSystemText(systemBlocks, legacyDynamicPrompt),
    );
  });

  it("restores the original two-block request when dynamic prompt caching is disabled", async () => {
    const llm = createAnsweringLlm("toolu_uncached_regeneration");
    const regenerationSection = {
      blockId: "borg_commitment_regeneration_instruction",
      text: "Regeneration fixture.",
    };
    const legacyDynamicPrompt = [
      "Base dynamic prompt.",
      USER_ACTIVE_TOOL_AVAILABILITY,
      regenerationSection.text,
    ].join("\n\n");

    await runEmissionFinalizer(llm, tempDirs, {
      additionalPromptSections: [regenerationSection],
      finalizerDynamicPromptCacheEnabled: false,
    });

    const systemBlocks = llm.requests[0]?.system as readonly CapturedSystemBlock[];

    expect(systemBlocks).toHaveLength(2);
    expect(systemBlocks[0]?.cache_control).toEqual({ type: "ephemeral", ttl: "1h" });
    expect(systemBlocks[1]).toEqual({
      type: "text",
      text: legacyDynamicPrompt,
    });
    expect(systemBlocks.some((block) => block.cache_control?.ttl === "5m")).toBe(false);
    expect(requestSystemText(systemBlocks)).toBe(
      legacyTwoBlockSystemText(systemBlocks, legacyDynamicPrompt),
    );
  });

  it("pins regeneration block ids to the contiguous highest-order registry tail", () => {
    const finalizerDynamicBlockIds = promptSurfaceBlocksForSurface(
      PROMPT_SURFACES.finalizerDynamicSystem,
    ).map((block) => block.id);
    const tailStart =
      finalizerDynamicBlockIds.length - FINALIZER_REGENERATION_PROMPT_BLOCK_IDS.length;

    expect(tailStart).toBeGreaterThanOrEqual(0);
    expect(finalizerDynamicBlockIds.slice(tailStart)).toEqual([
      ...FINALIZER_REGENERATION_PROMPT_BLOCK_IDS,
    ]);
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
