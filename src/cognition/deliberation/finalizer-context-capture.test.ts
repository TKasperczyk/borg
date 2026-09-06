import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
} from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { gunzipSync } from "node:zlib";

import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import type { LLMClient } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { StreamWriter } from "../../stream/index.js";
import { ToolDispatcher } from "../../tools/index.js";
import { FixedClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import {
  FinalizerContextCapture,
  buildFinalizerContextCaptureRecord,
  parseFinalizerContextCaptureRecord,
} from "./finalizer-context-capture.js";
import {
  FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES,
  FinalizerToolTranscriptCollector,
  parseFinalizerToolTranscript,
  prepareFinalizerToolTranscript,
  type FinalizerToolTranscriptSnapshot,
} from "./finalizer-tool-transcript.js";
import {
  readContentAddressedCaptureSidecar,
  waitForContextCaptureMaintenance,
} from "./context-capture-storage.js";
import { fingerprintCanonicalRequest } from "./request-fingerprint.js";
import { replayFinalizerContextCapture } from "./finalizer-ab-replay.js";
import type { DeliberationContext } from "./types.js";
import { runFinalizer } from "./finalizer.js";

const tempDirs: string[] = [];
afterEach(async () => {
  await waitForContextCaptureMaintenance();
  for (const dir of tempDirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});

function context(): DeliberationContext {
  return {
    sessionId: DEFAULT_SESSION_ID,
    nowMs: 1_000,
    turnId: "turn_capture",
    userMessage: "unused raw user payload",
    perception: {
      entities: [{ display_name: "UNUSED_PERCEPTION_ENTITY" }] as never,
      mode: "reflective",
      affectiveSignal: { valence: 0, arousal: 0, dominant_emotion: null },
      temporalCue: { kind: "UNUSED_TEMPORAL_PAYLOAD" } as never,
    },
    retrievalResult: [],
    retrievedSemantic: {
      matched_node_ids: ["semn_capture_a"],
      matched_nodes: [
        {
          id: "semn_capture_a",
          label: "CAPTURED_NODE_LABEL",
          description: "A capture fixture node.",
          confidence: 0.7,
          source_episode_ids: [],
          embedding: { 0: 0.125, 1: -0.5 },
        },
      ],
      supports: [],
      contradicts: [],
      categories: [],
      support_hits: [
        {
          root_node_id: "semn_capture_a",
          edgePath: [],
          node: {
            id: "semn_capture_b",
            label: "HIT_NODE_LABEL",
            description: "A capture fixture hit node.",
            confidence: 0.7,
            source_episode_ids: [],
            embedding: { 0: 0.25 },
          },
        },
      ],
      causal_hits: [],
      contradiction_hits: [],
      category_hits: [],
    } as never,
    workingMemory: {
      session_id: DEFAULT_SESSION_ID,
      turn_counter: 1,
      hot_entities: [],
      pending_actions: [],
      pending_social_attribution: null,
      pending_trait_attribution: null,
      suppressed: [],
      mood: null,
      pending_procedural_attempts: [],
      discourse_state: { stop_until_substantive_content: null },
      mode: "reflective",
      updated_at: 1_000,
    },
    selfSnapshot: { values: [], goals: [], traits: [] },
    evidenceLedger: {
      sections: [],
      transcriptIncluded: false,
      transcriptCompacted: false,
      originalTranscriptTokenEstimate: 0,
      compactedTranscriptEntryCount: 0,
      rawPreservedUserTranscriptEntryCount: 0,
      estimatedTokens: 0,
    },
  };
}

const legacySystem = [
  {
    type: "text" as const,
    text: "legacy",
    cache_control: { type: "ephemeral" as const, ttl: "1h" as const },
  },
];
const compactSystem = [
  {
    type: "text" as const,
    text: "compact-static-and-global",
    cache_control: { type: "ephemeral" as const, ttl: "1h" as const },
  },
  {
    type: "text" as const,
    text: "compact-audience",
    cache_control: { type: "ephemeral" as const, ttl: "1h" as const },
  },
  {
    type: "text" as const,
    text: "compact-slow-standing",
    cache_control: { type: "ephemeral" as const, ttl: "5m" as const },
  },
  {
    type: "text" as const,
    text: "compact-slow-overlay",
    cache_control: { type: "ephemeral" as const, ttl: "5m" as const },
  },
  { type: "text" as const, text: "compact-fast-turn" },
];

function input() {
  return {
    capturedAt: 1_000,
    turnId: "turn_capture",
    sessionId: DEFAULT_SESSION_ID,
    path: "system_2" as const,
    attemptKind: "initial" as const,
    configuredSurfaceVariant: "legacy" as const,
    liveSurfaceVariant: "legacy" as const,
    context: context(),
    legacySystem,
    compactSystem,
    liveRequest: {
      model: "fake",
      system: legacySystem,
      messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "hello" }] }],
      tools: [
        {
          name: "EmitAnswer",
          description: "fake terminal",
          inputSchema: { type: "object" as const, properties: {} },
        },
        {
          name: "DangerousWrite",
          description: "must never be offered during replay",
          inputSchema: { type: "object" as const, properties: {} },
        },
      ],
      max_tokens: 100,
      budget: "cognition-system-2",
    },
    outcome: {
      status: "completed" as const,
      attempts: 1,
      structuralReason: "terminal_emission" as const,
      decisionKind: "answer",
      decision: { kind: "answer", text: "captured" },
      terminalToolCalls: [],
      reasoningText: "",
      usage: { input_tokens: 10, output_tokens: 2, stop_reason: "tool_use" },
    },
    usedNonTerminalTools: false,
  };
}

function toolTranscriptSnapshot(inputOverrides?: {
  payload?: string;
}): FinalizerToolTranscriptSnapshot {
  const collector = new FinalizerToolTranscriptCollector();
  collector.observe({
    ordinal: 1,
    iteration: 1,
    batchPosition: 1,
    callId: "toolu_recorded_result",
    toolName: "tool.episodic.search",
    rawArguments: { query: "wspomnienie 🌌" },
    disposition: "dispatched",
    result: {
      ok: true,
      output: { payload: inputOverrides?.payload ?? "full recorded result" },
    },
    durationMs: 17,
  });
  return collector.finish({
    requestBinding: fingerprintCanonicalRequest(input().liveRequest),
    expectedEventCount: 1,
    sourceCompleted: true,
  });
}

describe("finalizer context capture and replay", () => {
  it("does no capture work when the default-off sampler is disabled", () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const random = vi.fn(() => 0);
    const capture = new FinalizerContextCapture({ dataDir, sampleRate: 0, random });
    expect(capture.shouldCapture()).toBe(false);
    expect(random).not.toHaveBeenCalled();
    expect(existsSync(join(dataDir, "captures"))).toBe(false);
  });

  it("does not observe live tool results when capture sampling is disabled", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const clock = new FixedClock(1_000);
    const random = vi.fn(() => 0);
    const capture = new FinalizerContextCapture({ dataDir, sampleRate: 0, random, clock });
    const observe = vi.spyOn(FinalizerToolTranscriptCollector.prototype, "observe");
    const dispatcher = new ToolDispatcher({
      clock,
      createStreamWriter: (sessionId) => new StreamWriter({ dataDir, sessionId, clock }),
    });
    dispatcher.register({
      name: "tool.episodic.search",
      description: "Search recorded episodes.",
      allowedOrigins: ["autonomous"],
      writeScope: "read",
      inputSchema: z.object({ query: z.string() }).strict(),
      outputSchema: z.object({ matches: z.array(z.string()) }).strict(),
      async invoke() {
        return { matches: [] };
      },
    });
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_uncaptured",
            name: "tool.episodic.search",
            input: { query: "uncaptured" },
          },
        ],
        [
          {
            type: "tool_use",
            id: "toolu_uncaptured_terminal",
            name: "EmitNoOutput",
            input: { reason: "done" },
          },
        ],
      ],
    });

    try {
      await runFinalizer({
        llmClient: llm,
        dispatcher,
        sessionId: DEFAULT_SESSION_ID,
        model: "fake",
        baseSystemPrompt: "legacy dynamic",
        cacheableSystemPrompt: { staticPrefix: "static", dynamicContent: "legacy dynamic" },
        initialMessages: [{ role: "user", content: [{ type: "text", text: "wake" }] }],
        userEntryId: undefined,
        maxTokens: 100,
        path: "system_2",
        turnOrigin: "autonomous",
        compactSurface: {
          context: { ...context(), turnOrigin: "autonomous" },
          baseSystemPromptOptions: {
            retrievalContextBudget: 1_000,
            semanticContextBudget: 1_000,
            nowMs: 1_000,
          },
        },
        finalizerContextCapture: capture,
      });
      expect(observe).not.toHaveBeenCalled();
      expect(random).not.toHaveBeenCalled();
      expect(existsSync(join(dataDir, "captures"))).toBe(false);
    } finally {
      observe.mockRestore();
    }
  });

  it("round-trips both exact block serializations and omits raw unused perception payloads", () => {
    const record = buildFinalizerContextCaptureRecord(input());
    const parsed = parseFinalizerContextCaptureRecord(JSON.parse(JSON.stringify(record)));
    expect(parsed.surfaces.legacy.system).toEqual(legacySystem);
    expect(parsed.surfaces.compact.system).toEqual(compactSystem);
    expect(parsed.surfaces.compact.fingerprint).toMatchObject({
      systemBlockCount: 5,
      cacheBreakpointCount: 4,
    });
    expect(parsed.configured_surface_variant).toBe("legacy");
    expect(parsed.live_surface_variant).toBe("legacy");
    expect(parsed.schema_version).toBe(2);
    if (parsed.schema_version !== 2) throw new Error("expected V2 capture");
    expect(parsed.tool_transcript.status).toBe("none");
    expect(parsed.fidelity.verified).toBe(true);
    const serialized = JSON.stringify(parsed.projected_context);
    expect(serialized).not.toContain("UNUSED_PERCEPTION_ENTITY");
    expect(serialized).not.toContain("UNUSED_TEMPORAL_PAYLOAD");
    expect(serialized).not.toContain("unused raw user payload");
    expect(serialized).not.toContain('"embedding"');
    expect(serialized).toContain("CAPTURED_NODE_LABEL");
    expect(serialized).toContain("HIT_NODE_LABEL");
    expect(parsed.evidence_ledger).toEqual(context().evidenceLedger);
  });

  it("round-trips scoped policy and resolved variant while accepting older records", async () => {
    const scoped = buildFinalizerContextCaptureRecord({
      ...input(),
      configuredSurfaceVariant: "compact_conversational",
      liveSurfaceVariant: "compact",
      context: { ...context(), turnOrigin: "user" },
      liveRequest: { ...input().liveRequest, system: compactSystem },
    });
    const parsed = parseFinalizerContextCaptureRecord(JSON.parse(JSON.stringify(scoped)));
    expect(parsed.configured_surface_variant).toBe("compact_conversational");
    expect(parsed.live_surface_variant).toBe("compact");
    const replayed = await replayFinalizerContextCapture(parsed, { mode: "dry" });
    expect(replayed.source_configured_surface_variant).toBe("compact_conversational");
    expect(replayed.source_live_surface_variant).toBe("compact");

    const historical = JSON.parse(JSON.stringify(buildFinalizerContextCaptureRecord(input()))) as {
      schema_version: number;
      configured_surface_variant?: unknown;
      tool_transcript?: unknown;
      replay: {
        eligible: boolean;
        exclusion_reason: string | null;
        recorded_results_eligible?: boolean;
      };
    };
    historical.schema_version = 1;
    delete historical.configured_surface_variant;
    delete historical.tool_transcript;
    delete historical.replay.recorded_results_eligible;
    const parsedHistorical = parseFinalizerContextCaptureRecord(historical);
    expect(parsedHistorical.schema_version).toBe(1);
    expect(parsedHistorical.configured_surface_variant).toBeUndefined();
    expect(parsedHistorical.live_surface_variant).toBe("legacy");
  });

  it("captures the exact live request boundary and completed terminal outcome", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const clock = new FixedClock(1_000);
    const capture = new FinalizerContextCapture({ dataDir, sampleRate: 1, clock, random: () => 0 });
    const dispatcher = new ToolDispatcher({
      clock,
      createStreamWriter: (sessionId) => new StreamWriter({ dataDir, sessionId, clock }),
    });
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_capture_boundary",
              name: "EmitAnswer",
              input: { text: "captured answer" },
            },
          ],
          input_tokens: 11,
          output_tokens: 3,
          stop_reason: "tool_use",
        },
      ],
    });
    await runFinalizer({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      baseSystemPrompt: "legacy dynamic",
      cacheableSystemPrompt: { staticPrefix: "static", dynamicContent: "legacy dynamic" },
      initialMessages: [{ role: "user", content: [{ type: "text", text: "boundary message" }] }],
      userEntryId: undefined,
      maxTokens: 100,
      path: "system_1",
      finalizerSurfaceVariant: "compact_conversational",
      turnOrigin: "user",
      compactSurface: {
        context: { ...context(), turnOrigin: "user" },
        baseSystemPromptOptions: {
          retrievalContextBudget: 1_000,
          semanticContextBudget: 1_000,
          nowMs: 1_000,
        },
      },
      finalizerContextCapture: capture,
    });
    const record = parseFinalizerContextCaptureRecord(
      JSON.parse(readFileSync(join(dataDir, "captures", "finalizer-contexts.jsonl"), "utf8")),
    );
    expect(record.live_request?.system).toEqual(llm.requests[0]?.system);
    expect(record.configured_surface_variant).toBe("compact_conversational");
    expect(record.live_surface_variant).toBe("compact");
    expect(record.live_request?.messages).toEqual([
      { role: "user", content: [{ type: "text", text: "boundary message" }] },
    ]);
    expect(record.live_request?.tools?.map((tool) => tool.name)).toEqual(
      llm.requests[0]?.tools?.map((tool) => tool.name),
    );
    expect(record.fidelity.verified).toBe(true);
    expect(record.live_outcome).toMatchObject({
      status: "completed",
      structuralReason: "terminal_emission",
      decision: { kind: "answer", text: "captured answer" },
      usage: { input_tokens: 11, output_tokens: 3 },
    });
  });

  it("captures a multi-iteration autonomous tool trajectory through terminal emission", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const clock = new FixedClock(1_000);
    const capture = new FinalizerContextCapture({ dataDir, sampleRate: 1, clock });
    const dispatcher = new ToolDispatcher({
      clock,
      createStreamWriter: (sessionId) => new StreamWriter({ dataDir, sessionId, clock }),
    });
    dispatcher.register({
      name: "tool.episodic.search",
      description: "Search recorded episodes.",
      allowedOrigins: ["autonomous"],
      writeScope: "read",
      inputSchema: z.object({ query: z.string().min(1) }).strict(),
      outputSchema: z.object({ matches: z.array(z.string()) }).strict(),
      async invoke(input: { query: string }) {
        return { matches: [`match:${input.query}`] };
      },
    });
    const dispatch = dispatcher.dispatch.bind(dispatcher);
    vi.spyOn(dispatcher, "dispatch").mockImplementation(async (call) => {
      if (call.callId === "toolu_dispatch_failure") {
        throw new Error("dispatcher unavailable");
      }
      return dispatch(call);
    });
    const llm = new FakeLLMClient({
      responses: [
        [
          {
            type: "tool_use",
            id: "toolu_dispatch_success",
            name: "tool.episodic.search",
            input: { query: "pamięć 🌌" },
          },
          {
            type: "tool_use",
            id: "toolu_validation_failure",
            name: "tool.episodic.search",
            input: { query: "" },
          },
          {
            type: "tool_use",
            id: "toolu_dispatch_failure",
            name: "tool.episodic.search",
            input: { query: "dispatcher failure" },
          },
          {
            type: "tool_use",
            id: "toolu_iteration_cap",
            name: "tool.episodic.search",
            input: { query: "deferred" },
          },
        ],
        [
          {
            type: "tool_use",
            id: "toolu_unavailable",
            name: "tool.not.available",
            input: { query: "structural" },
          },
        ],
        [
          {
            type: "tool_use",
            id: "toolu_terminal_after_results",
            name: "EmitNoOutput",
            input: { reason: "trajectory complete" },
          },
        ],
      ],
    });

    await runFinalizer({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      model: "fake",
      baseSystemPrompt: "legacy dynamic",
      cacheableSystemPrompt: { staticPrefix: "static", dynamicContent: "legacy dynamic" },
      initialMessages: [{ role: "user", content: [{ type: "text", text: "autonomous wake" }] }],
      userEntryId: undefined,
      maxTokens: 100,
      path: "system_2",
      finalizerSurfaceVariant: "legacy",
      turnOrigin: "autonomous",
      compactSurface: {
        context: { ...context(), turnOrigin: "autonomous" },
        baseSystemPromptOptions: {
          retrievalContextBudget: 1_000,
          semanticContextBudget: 1_000,
          nowMs: 1_000,
        },
      },
      finalizerContextCapture: capture,
    });

    const record = parseFinalizerContextCaptureRecord(
      JSON.parse(readFileSync(join(dataDir, "captures", "finalizer-contexts.jsonl"), "utf8")),
    );
    if (record.schema_version !== 2 || record.tool_transcript.status !== "complete") {
      throw new Error("expected complete V2 tool transcript");
    }
    expect(record.live_outcome).toMatchObject({
      status: "completed",
      structuralReason: "nonterminal_tool_loop",
      terminalToolCalls: [{ id: "toolu_terminal_after_results", name: "EmitNoOutput" }],
    });
    expect(record.replay).toEqual({
      eligible: false,
      exclusion_reason: "autonomous",
      recorded_results_eligible: true,
    });
    const transcript = parseFinalizerToolTranscript(
      JSON.parse(
        Buffer.from(
          readContentAddressedCaptureSidecar({
            dataDir,
            relativePath: record.tool_transcript.relative_path,
            sha256: record.tool_transcript.canonical_sha256,
          }),
        ).toString("utf8"),
      ) as unknown,
    );
    expect(transcript.events.map((event) => event.disposition)).toEqual([
      "dispatched",
      "dispatched",
      "dispatched",
      // Autonomous turns allow 5 calls per iteration (raised for goal-wake
      // batching), so the fourth call dispatches; the cap-skip disposition is
      // pinned by the tool-loop boundary tests instead.
      "dispatched",
      "skipped_unavailable",
    ]);
    expect(transcript.events[0]).toMatchObject({
      raw_arguments: { query: "pamięć 🌌" },
      result: { ok: true, output: { matches: ["match:pamięć 🌌"] } },
    });
    expect(transcript.events[1]?.result).toMatchObject({ ok: false, error: expect.any(String) });
    expect(transcript.events[2]?.result).toEqual({
      ok: false,
      error: "Error: dispatcher unavailable",
    });
    expect(transcript.events[3]?.result).toEqual({
      ok: true,
      output: { matches: ["match:deferred"] },
    });
    expect(transcript.events[4]?.result).toEqual({
      ok: false,
      error: "tool tool.not.available not available in this context",
    });
  });

  it("captures a thrown live outcome best-effort and rethrows the original error", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const capture = new FinalizerContextCapture({
      dataDir,
      sampleRate: 1,
      clock: new FixedClock(1_000),
      random: () => 0,
    });
    const failure = new Error("provider unavailable");
    const llm: LLMClient = {
      complete: vi.fn(async () => {
        throw failure;
      }),
      converse: vi.fn(async () => {
        throw failure;
      }),
    };
    const dispatcher = new ToolDispatcher({
      clock: new FixedClock(1_000),
      createStreamWriter: (sessionId) =>
        new StreamWriter({ dataDir, sessionId, clock: new FixedClock(1_000) }),
    });

    await expect(
      runFinalizer({
        llmClient: llm,
        dispatcher,
        sessionId: DEFAULT_SESSION_ID,
        model: "fake",
        baseSystemPrompt: "legacy dynamic",
        cacheableSystemPrompt: { staticPrefix: "static", dynamicContent: "legacy dynamic" },
        initialMessages: [{ role: "user", content: [{ type: "text", text: "boundary" }] }],
        userEntryId: undefined,
        maxTokens: 100,
        path: "system_1",
        finalizerSurfaceVariant: "legacy",
        compactSurface: {
          context: context(),
          baseSystemPromptOptions: {
            retrievalContextBudget: 1_000,
            semanticContextBudget: 1_000,
            nowMs: 1_000,
          },
        },
        finalizerContextCapture: capture,
      }),
    ).rejects.toBe(failure);

    const record = parseFinalizerContextCaptureRecord(
      JSON.parse(readFileSync(join(dataDir, "captures", "finalizer-contexts.jsonl"), "utf8")),
    );
    expect(record.live_outcome).toMatchObject({
      status: "threw",
      attempts: 1,
      structuralReason: "finalizer_error",
      error: { name: "Error", message: "provider unavailable" },
    });
    if (record.schema_version !== 2) throw new Error("expected V2 capture");
    expect(record.tool_transcript).toMatchObject({
      status: "incomplete",
      event_count: 0,
      replay_eligible: false,
      incomplete_reasons: ["source_incomplete"],
    });
    expect(record.replay).toMatchObject({
      eligible: false,
      exclusion_reason: "source_threw",
      recorded_results_eligible: false,
    });
  });

  it("writes private JSONL and content-addressed image sidecars under a 0022 umask", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const attachmentId = "att_aaaaaaaaaaaaaaaa" as never;
    const capture = new FinalizerContextCapture({
      dataDir,
      sampleRate: 1,
      clock: new FixedClock(1_000),
      attachmentResolver: () => ({ mediaType: "image/png", bytes: Buffer.from("image") }),
    });
    const previous = process.umask(0o022);
    try {
      const result = await capture.capture({
        ...input(),
        liveRequest: {
          ...input().liveRequest,
          messages: [
            { role: "user", content: [{ type: "image_ref", attachment_id: attachmentId }] },
          ],
        },
      });
      expect(result.status).toBe("captured");
      if (result.status !== "captured") return;
      expect(statSync(join(dataDir, "captures")).mode & 0o777).toBe(0o700);
      expect(statSync(result.path).mode & 0o777).toBe(0o600);
      const sidecar = result.record.image_sidecars[0]!;
      const sidecarPath = join(dataDir, "captures", sidecar.relative_path);
      expect(statSync(sidecarPath).mode & 0o777).toBe(0o600);
      expect(readFileSync(sidecarPath).toString()).toBe("image");
    } finally {
      process.umask(previous);
    }
  });

  it("writes a complete V2 tool transcript sidecar and derives stage-two material eligibility", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const emit = vi.fn();
    const capture = new FinalizerContextCapture({
      dataDir,
      sampleRate: 1,
      tracer: { enabled: true, includePayloads: false, emit },
    });
    const result = await capture.capture({
      ...input(),
      outcome: {
        ...input().outcome,
        structuralReason: "nonterminal_tool_loop",
      },
      usedNonTerminalTools: true,
      toolTranscript: toolTranscriptSnapshot(),
    });

    expect(result.status).toBe("captured");
    if (result.status !== "captured" || result.record.schema_version !== 2) return;
    expect(result.record.tool_transcript).toMatchObject({
      status: "complete",
      event_count: 1,
      dispatched_count: 1,
      replay_eligible: true,
    });
    expect(result.record.replay).toEqual({
      eligible: false,
      exclusion_reason: "nonterminal_tools",
      recorded_results_eligible: true,
    });
    const manifest = result.record.tool_transcript;
    if (manifest.status !== "complete") throw new Error("expected complete transcript");
    expect(statSync(join(dataDir, "captures", "finalizer-tool-transcripts")).mode & 0o777).toBe(
      0o700,
    );
    expect(statSync(join(dataDir, "captures", manifest.relative_path)).mode & 0o777).toBe(0o600);
    const transcript = parseFinalizerToolTranscript(
      JSON.parse(
        Buffer.from(
          readContentAddressedCaptureSidecar({
            dataDir,
            relativePath: manifest.relative_path,
            sha256: manifest.canonical_sha256,
            byteSize: manifest.payload_bytes,
          }),
        ).toString("utf8"),
      ) as unknown,
    );
    expect(transcript.events[0]).toMatchObject({
      call_id: "toolu_recorded_result",
      raw_arguments: { query: "wspomnienie 🌌" },
      result: { ok: true, output: { payload: "full recorded result" } },
    });
    expect(emit).toHaveBeenCalledWith(
      "deliberation.finalizer_context_capture.captured",
      expect.objectContaining({
        turnId: "turn_capture",
        tool_transcript_status: "complete",
        tool_transcript_events: 1,
        tool_transcript_dispatched: 1,
        tool_transcript_bytes: manifest.payload_bytes,
      }),
    );
  });

  it("records an oversized transcript as omitted and ineligible without truncating it", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const capture = new FinalizerContextCapture({ dataDir, sampleRate: 1 });
    const payload = "x".repeat(FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES);
    const result = await capture.capture({
      ...input(),
      outcome: {
        ...input().outcome,
        structuralReason: "nonterminal_tool_loop",
      },
      usedNonTerminalTools: true,
      toolTranscript: toolTranscriptSnapshot({ payload }),
    });

    expect(result.status).toBe("captured");
    if (result.status !== "captured" || result.record.schema_version !== 2) return;
    expect(result.record.tool_transcript).toMatchObject({
      status: "omitted_oversized",
      event_count: 1,
      relative_path: null,
      replay_eligible: false,
    });
    expect(result.record.tool_transcript.payload_bytes).toBeGreaterThan(
      FINALIZER_TOOL_TRANSCRIPT_MAX_BYTES,
    );
    expect(result.record.replay.recorded_results_eligible).toBe(false);
    expect(existsSync(join(dataDir, "captures", "finalizer-tool-transcripts"))).toBe(false);
  });

  it("counts image and transcript payloads against one auxiliary-byte cap", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const transcript = toolTranscriptSnapshot();
    const transcriptBytes = prepareFinalizerToolTranscript({ snapshot: transcript }).manifest
      .payload_bytes;
    const imageBytes = Buffer.alloc(100, 1);
    const capture = new FinalizerContextCapture({
      dataDir,
      sampleRate: 1,
      maxSidecarBytes: transcriptBytes + imageBytes.byteLength - 1,
      attachmentResolver: () => ({ mediaType: "image/png", bytes: imageBytes }),
    });
    const result = await capture.capture({
      ...input(),
      outcome: { ...input().outcome, structuralReason: "nonterminal_tool_loop" },
      usedNonTerminalTools: true,
      toolTranscript: transcript,
      liveRequest: {
        ...input().liveRequest,
        messages: [
          {
            role: "user",
            content: [{ type: "image_ref", attachment_id: "att_dddddddddddddddd" as never }],
          },
        ],
      },
    });

    expect(result).toMatchObject({ status: "skipped", reason: "file_full" });
    expect(existsSync(join(dataDir, "captures", "finalizer-contexts.jsonl"))).toBe(false);
    expect(existsSync(join(dataDir, "captures", "finalizer-images"))).toBe(false);
    expect(existsSync(join(dataDir, "captures", "finalizer-tool-transcripts"))).toBe(false);
  });

  it("skips oversized records without creating the capture file", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const capture = new FinalizerContextCapture({ dataDir, sampleRate: 1, maxRecordBytes: 32 });
    const result = await capture.capture(input());
    expect(result).toMatchObject({ status: "skipped", reason: "record_oversized" });
    expect(existsSync(join(dataDir, "captures", "finalizer-contexts.jsonl"))).toBe(false);
  });

  it("rotates at the file cap and captures the triggering finalizer record", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const firstInput = { ...input(), captureId: "capture_a" };
    const triggeringInput = { ...input(), captureId: "capture_b" };
    const lineBytes = Buffer.byteLength(
      `${JSON.stringify(buildFinalizerContextCaptureRecord(firstInput))}\n`,
    );
    const emit = vi.fn();
    const logger = { info: vi.fn(), error: vi.fn() };
    const capture = new FinalizerContextCapture({
      dataDir,
      sampleRate: 1,
      maxRecordBytes: 32 * 1024 * 1024,
      maxFileBytes: lineBytes,
      clock: new FixedClock(1_000),
      tracer: { enabled: true, includePayloads: false, emit },
      logger,
    });
    await expect(capture.capture(firstInput)).resolves.toMatchObject({ status: "captured" });
    await expect(capture.capture(triggeringInput)).resolves.toMatchObject({ status: "captured" });

    const path = join(dataDir, "captures", "finalizer-contexts.jsonl");
    await waitForContextCaptureMaintenance(path);
    expect(JSON.parse(readFileSync(path, "utf8"))).toMatchObject({ capture_id: "capture_b" });
    const [rotatedName] = readdirSync(join(dataDir, "captures")).filter((name) =>
      name.startsWith("finalizer-contexts.jsonl.rotated-"),
    );
    expect(rotatedName).toBe("finalizer-contexts.jsonl.rotated-19700101T000001.000Z.gz");
    expect(
      JSON.parse(gunzipSync(readFileSync(join(dataDir, "captures", rotatedName!))).toString()),
    ).toMatchObject({ capture_id: "capture_a" });
    expect(emit).toHaveBeenCalledWith(
      "deliberation.finalizer_context_capture.captured",
      expect.objectContaining({ reason: "rotated" }),
    );
    expect(logger.info).toHaveBeenCalledWith(
      "Rotated deliberation context capture",
      expect.objectContaining({ capturePath: path }),
    );
    expect(logger.error).not.toHaveBeenCalled();
  });

  it("removes staged image and tool-transcript sidecars when the JSONL append fails", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    mkdirSync(join(dataDir, "captures", "finalizer-contexts.jsonl"), { recursive: true });
    const capture = new FinalizerContextCapture({
      dataDir,
      sampleRate: 1,
      attachmentResolver: () => ({ mediaType: "image/png", bytes: Buffer.from("staged-image") }),
    });
    const result = await capture.capture({
      ...input(),
      outcome: {
        ...input().outcome,
        structuralReason: "nonterminal_tool_loop",
      },
      usedNonTerminalTools: true,
      toolTranscript: toolTranscriptSnapshot(),
      liveRequest: {
        ...input().liveRequest,
        messages: [
          {
            role: "user",
            content: [{ type: "image_ref", attachment_id: "att_cccccccccccccccc" as never }],
          },
        ],
      },
    });
    expect(result.status).toBe("failed");
    expect(readdirSync(join(dataDir, "captures", "finalizer-images"))).toEqual([]);
    expect(readdirSync(join(dataDir, "captures", "finalizer-tool-transcripts"))).toEqual([]);
  });

  it("replays only the unary fake-terminal request without reaching repositories", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            { type: "tool_use", id: "toolu_compact", name: "EmitAnswer", input: { text: "a" } },
          ],
          input_tokens: 4,
          output_tokens: 1,
          stop_reason: "tool_use",
        },
        {
          messageBlocks: [
            { type: "tool_use", id: "toolu_legacy", name: "EmitAnswer", input: { text: "b" } },
          ],
          input_tokens: 5,
          output_tokens: 1,
          stop_reason: "tool_use",
        },
      ],
    });
    const result = await replayFinalizerContextCapture(
      buildFinalizerContextCaptureRecord({
        ...input(),
        configuredSurfaceVariant: "compact_conversational",
        liveSurfaceVariant: "compact",
        context: { ...context(), turnOrigin: "user" },
        liveRequest: { ...input().liveRequest, system: compactSystem },
      }),
      {
        mode: "live",
        llmClient: llm,
      },
    );
    expect(result.pairing_status).toBe("paired");
    expect(result.source_configured_surface_variant).toBe("compact_conversational");
    expect(result.source_live_surface_variant).toBe("compact");
    expect(llm.requests).toHaveLength(2);
    expect(llm.requests[0]?.system).toEqual(compactSystem);
    expect(llm.requests[1]?.system).toEqual(legacySystem);
    expect(
      llm.requests.every((request) => request.tools?.every((tool) => tool.name === "EmitAnswer")),
    ).toBe(true);
    expect(result.live?.compact.status).toBe("completed");
  });

  it("labels autonomous and nonterminal source calls as excluded", async () => {
    const autonomous = buildFinalizerContextCaptureRecord({
      ...input(),
      context: { ...context(), turnOrigin: "autonomous" },
    });
    const nonterminal = buildFinalizerContextCaptureRecord({
      ...input(),
      usedNonTerminalTools: true,
    });
    await expect(replayFinalizerContextCapture(autonomous, { mode: "dry" })).resolves.toMatchObject(
      {
        pairing_status: "excluded_autonomous",
      },
    );
    await expect(
      replayFinalizerContextCapture(nonterminal, { mode: "dry" }),
    ).resolves.toMatchObject({
      pairing_status: "excluded_nonterminal",
    });
  });

  it("skips live pairing when any canonical request field no longer matches capture", async () => {
    const record = parseFinalizerContextCaptureRecord(
      JSON.parse(JSON.stringify(buildFinalizerContextCaptureRecord(input()))),
    );
    record.live_request!.messages = [
      { role: "user", content: [{ type: "text", text: "tampered after capture" }] },
    ];
    const llm = new FakeLLMClient({ responses: [] });
    const result = await replayFinalizerContextCapture(record, { mode: "live", llmClient: llm });
    expect(result).toMatchObject({
      pairing_status: "skipped_fidelity",
      fidelity: {
        storedVerified: true,
        currentSourceSystemMatchesCapture: true,
        currentSourceRequestMatchesCapture: false,
      },
    });
    expect(llm.requests).toHaveLength(0);
  });

  it("emits capture.failed when alternate-surface assembly fails", () => {
    const emit = vi.fn();
    const capture = new FinalizerContextCapture({
      dataDir: "/unused",
      sampleRate: 1,
      tracer: { enabled: true, includePayloads: false, emit },
    });
    capture.recordAssemblyFailure(
      { turnId: "turn_capture", sessionId: DEFAULT_SESSION_ID },
      new Error("alternate failed"),
    );
    expect(emit).toHaveBeenCalledWith("deliberation.finalizer_context_capture.failed", {
      turnId: "turn_capture",
      session_id: DEFAULT_SESSION_ID,
      phase: "alternate_surface_assembly",
      reason: "alternate failed",
    });
  });

  it("reports alternate-surface assembly failure from the live finalizer boundary", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const emit = vi.fn();
    const capture = new FinalizerContextCapture({
      dataDir,
      sampleRate: 1,
      random: () => 0,
      tracer: { enabled: true, includePayloads: false, emit },
    });
    const brokenContext = context();
    Object.defineProperty(brokenContext.selfSnapshot, "values", {
      get: () => {
        throw new Error("compact alternative assembly failed");
      },
    });
    const clock = new FixedClock(1_000);
    const dispatcher = new ToolDispatcher({
      clock,
      createStreamWriter: (sessionId) => new StreamWriter({ dataDir, sessionId, clock }),
    });
    const llm = new FakeLLMClient({
      responses: [
        {
          messageBlocks: [
            {
              type: "tool_use",
              id: "toolu_alternate_failure",
              name: "EmitAnswer",
              input: { text: "live path continues" },
            },
          ],
          input_tokens: 1,
          output_tokens: 1,
          stop_reason: "tool_use",
        },
      ],
    });
    await runFinalizer({
      llmClient: llm,
      dispatcher,
      sessionId: DEFAULT_SESSION_ID,
      turnId: "turn_capture",
      model: "fake",
      baseSystemPrompt: "legacy dynamic",
      cacheableSystemPrompt: { staticPrefix: "static", dynamicContent: "legacy dynamic" },
      initialMessages: [{ role: "user", content: [{ type: "text", text: "boundary" }] }],
      userEntryId: undefined,
      maxTokens: 100,
      path: "system_1",
      finalizerSurfaceVariant: "legacy",
      compactSurface: {
        context: brokenContext,
        baseSystemPromptOptions: {
          retrievalContextBudget: 1_000,
          semanticContextBudget: 1_000,
          nowMs: 1_000,
        },
      },
      finalizerContextCapture: capture,
    });
    expect(emit).toHaveBeenCalledWith("deliberation.finalizer_context_capture.failed", {
      turnId: "turn_capture",
      session_id: DEFAULT_SESSION_ID,
      phase: "alternate_surface_assembly",
      reason: "compact alternative assembly failed",
    });
    expect(existsSync(join(dataDir, "captures", "finalizer-contexts.jsonl"))).toBe(false);
  });
});
