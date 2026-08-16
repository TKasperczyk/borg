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

import { afterEach, describe, expect, it, vi } from "vitest";

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
import { replayFinalizerContextCapture } from "./finalizer-ab-replay.js";
import type { DeliberationContext } from "./types.js";
import { runFinalizer } from "./finalizer.js";

const tempDirs: string[] = [];
afterEach(() => {
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
    text: "compact-static",
    cache_control: { type: "ephemeral" as const, ttl: "1h" as const },
  },
  {
    type: "text" as const,
    text: "compact-global",
    cache_control: { type: "ephemeral" as const, ttl: "1h" as const },
  },
  {
    type: "text" as const,
    text: "compact-audience",
    cache_control: { type: "ephemeral" as const, ttl: "1h" as const },
  },
  {
    type: "text" as const,
    text: "compact-turn",
    cache_control: { type: "ephemeral" as const, ttl: "5m" as const },
  },
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

describe("finalizer context capture and replay", () => {
  it("does no capture work when the default-off sampler is disabled", () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const capture = new FinalizerContextCapture({ dataDir, sampleRate: 0, random: () => 0 });
    expect(capture.shouldCapture()).toBe(false);
    expect(existsSync(join(dataDir, "captures"))).toBe(false);
  });

  it("round-trips both exact block serializations and omits raw unused perception payloads", () => {
    const record = buildFinalizerContextCaptureRecord(input());
    const parsed = parseFinalizerContextCaptureRecord(JSON.parse(JSON.stringify(record)));
    expect(parsed.surfaces.legacy.system).toEqual(legacySystem);
    expect(parsed.surfaces.compact.system).toEqual(compactSystem);
    expect(parsed.configured_surface_variant).toBe("legacy");
    expect(parsed.live_surface_variant).toBe("legacy");
    expect(parsed.fidelity.verified).toBe(true);
    const serialized = JSON.stringify(parsed.projected_context);
    expect(serialized).not.toContain("UNUSED_PERCEPTION_ENTITY");
    expect(serialized).not.toContain("UNUSED_TEMPORAL_PAYLOAD");
    expect(serialized).not.toContain("unused raw user payload");
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
      configured_surface_variant?: unknown;
    };
    delete historical.configured_surface_variant;
    const parsedHistorical = parseFinalizerContextCaptureRecord(historical);
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
    expect(record.replay).toEqual({ eligible: false, exclusion_reason: "source_threw" });
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

  it("skips oversized records without creating the capture file", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const capture = new FinalizerContextCapture({ dataDir, sampleRate: 1, maxRecordBytes: 32 });
    const result = await capture.capture(input());
    expect(result).toMatchObject({ status: "skipped", reason: "record_oversized" });
    expect(existsSync(join(dataDir, "captures", "finalizer-contexts.jsonl"))).toBe(false);
  });

  it("stops appending when the capture file growth cap is reached", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-finalizer-capture-"));
    tempDirs.push(dataDir);
    const capture = new FinalizerContextCapture({
      dataDir,
      sampleRate: 1,
      maxRecordBytes: 32 * 1024 * 1024,
      maxFileBytes: 1,
      attachmentResolver: () => ({ mediaType: "image/png", bytes: Buffer.from("staged-image") }),
    });
    const result = await capture.capture({
      ...input(),
      liveRequest: {
        ...input().liveRequest,
        messages: [
          {
            role: "user",
            content: [{ type: "image_ref", attachment_id: "att_bbbbbbbbbbbbbbbb" as never }],
          },
        ],
      },
    });
    expect(result).toMatchObject({ status: "skipped", reason: "file_full" });
    expect(statSync(join(dataDir, "captures", "finalizer-contexts.jsonl")).size).toBe(0);
    expect(readdirSync(join(dataDir, "captures", "finalizer-images"))).toEqual([]);
  });

  it("removes staged image sidecars when the JSONL append fails", async () => {
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
