import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { File } from "node:buffer";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { serve } from "@hono/node-server";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Borg,
  DEFAULT_SESSION_ID,
  DemoMessageConnector,
  ManualClock,
  createSessionId,
  createEpisodeId,
  createMaintenanceRunId,
  createSemanticEdgeId,
  createSemanticNodeId,
  type AttachmentId,
  type BorgOpenOptions,
  type StreamEntry,
  type TurnResult,
} from "borg";

import {
  FakeLLMClient,
  createFakeEmitAnswerResponse,
  createFakeStreamingResponse,
} from "../../../../src/llm/test-support/fake-client.js";
import type { AttachmentService } from "../../../../src/attachments/index.js";
import { IMAGE_PERCEPTION_TOOL_NAME } from "../../../../src/attachments/perception.js";
import type { Episode, EpisodicRepository } from "../../../../src/memory/episodic/index.js";
import type { RelationalSlotRepository } from "../../../../src/memory/relational-slots/repository.js";
import type { ReviewQueueRepository } from "../../../../src/memory/semantic/review-queue.js";
import { TestEmbeddingClient, createTestConfig } from "../../../../src/offline/test-support.js";
import type { AuditLog } from "../../../../src/offline/audit-log.js";
import type { StreamWriter } from "../../../../src/stream/index.js";
import { createDemoServerApp } from "../app.js";
import { LiveBroadcaster, createLiveBridge, type LiveFrame } from "../live.js";
import { createResetBorgController, type BorgHandle } from "../reset.js";

const PNG_1X1 = Uint8Array.from([
  0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
  0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x04, 0x00, 0x00, 0x00, 0xb5, 0x1c, 0x0c,
  0x02, 0x00, 0x00, 0x00, 0x0b, 0x49, 0x44, 0x41, 0x54, 0x78, 0xda, 0x63, 0xfc, 0xff, 0x1f, 0x00,
  0x03, 0x03, 0x02, 0x00, 0xef, 0xbf, 0x27, 0x8f, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44,
  0xae, 0x42, 0x60, 0x82,
]);

type BorgTestInternals = {
  deps: {
    attachmentService: AttachmentService;
    auditLog: AuditLog;
    createStreamWriter(sessionId: typeof DEFAULT_SESSION_ID): StreamWriter;
    episodicRepository: EpisodicRepository;
    relationalSlotRepository: RelationalSlotRepository;
    reviewQueueRepository: ReviewQueueRepository;
  };
};

type Deferred<T> = {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(error: unknown): void;
};

function createDeferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function createEmptyReflectionResponse() {
  return {
    text: "",
    input_tokens: 4,
    output_tokens: 2,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_reflection_empty",
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

function createImagePerceptionResponse() {
  return [
    {
      type: "tool_use" as const,
      id: "toolu_image",
      name: IMAGE_PERCEPTION_TOOL_NAME,
      input: {
        caption: "A tiny uploaded test image.",
        image_kind: "photo",
        visible_text: [],
        objects: ["single pixel"],
        people_or_roles: [],
        scene: "A minimal image fixture.",
        colors_and_visual_attributes: ["transparent or white pixel"],
        spatial_relationships: ["one pixel fills the image"],
        possible_user_relevant_details: ["multipart upload smoke test"],
        search_terms: ["test image", "uploaded pixel", "multipart attachment"],
        uncertainties: [],
      },
    },
  ];
}

function createHarnessOpenOptions(input: {
  tempDir: string;
  live: ReturnType<typeof createLiveBridge>;
  clock: ManualClock;
  llmClient?: FakeLLMClient;
  hostCapabilities?: string;
}): BorgOpenOptions {
  return {
    config: createTestConfig({
      dataDir: input.tempDir,
      ...(input.hostCapabilities === undefined
        ? {}
        : { host_capabilities: input.hostCapabilities }),
      perception: {
        llmEnabled: false,
      },
      affective: {
        llmEnabled: false,
      },
      generation: {
        evidenceLedger: {
          enabled: false,
        },
      },
      embedding: {
        baseUrl: "http://localhost:1234/v1",
        apiKey: "test",
        model: "test-embed",
        dims: 4,
      },
      anthropic: {
        auth: "api-key",
        apiKey: "test",
        models: {
          cognition: "test-cognition",
          background: "test-background",
          extraction: "test-extraction",
          recallExpansion: "test-recall",
        },
      },
    }),
    clock: input.clock,
    embeddingDimensions: 4,
    embeddingClient: new TestEmbeddingClient(),
    llmClient: input.llmClient ?? new FakeLLMClient(),
    tracer: input.live.tracer,
    onStreamAppend: input.live.onStreamAppend,
    outboundConnectors: [new DemoMessageConnector()],
    liveExtraction: false,
  };
}

async function openHarness(input: {
  tempDir: string;
  llmClient?: FakeLLMClient;
  hostCapabilities?: string;
}): Promise<{
  borg: Borg;
  clock: ManualClock;
  live: ReturnType<typeof createLiveBridge>;
}> {
  const live = createLiveBridge();
  const clock = new ManualClock(1_800_000_000_000);

  return {
    borg: await Borg.open(
      createHarnessOpenOptions({
        tempDir: input.tempDir,
        live,
        clock,
        llmClient: input.llmClient,
        hostCapabilities: input.hostCapabilities,
      }),
    ),
    clock,
    live,
  };
}

async function waitFor(predicate: () => boolean, timeoutMs = 5_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() < deadline) {
    if (predicate()) {
      return;
    }

    await new Promise((resolve) => setTimeout(resolve, 25));
  }

  throw new Error("Timed out waiting for condition");
}

function collectLiveFrames(live: ReturnType<typeof createLiveBridge>): {
  frames: LiveFrame[];
  wasClosed(): boolean;
} {
  const frames: LiveFrame[] = [];
  let closed = false;
  const client = {
    send(data: string): void {
      frames.push(JSON.parse(data) as LiveFrame);
    },
    close(): void {
      closed = true;
    },
  };

  live.broadcaster.add(client);
  live.broadcaster.handleSubscriptionMessage(client, {
    type: "subscribe",
    session_id: DEFAULT_SESSION_ID,
  });

  return {
    frames,
    wasClosed: () => closed,
  };
}

function serverPort(server: ReturnType<typeof serve>): number {
  const address = server.address() as AddressInfo | string | null;

  if (address === null || typeof address === "string") {
    throw new Error("Expected server to listen on a TCP port");
  }

  return address.port;
}

async function requestJson(
  app: ReturnType<typeof createDemoServerApp>["app"],
  path: string,
  method: "POST" | "PATCH",
  body: unknown,
): Promise<Response> {
  return app.request(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function seedCorrectionEpisode(
  borg: Borg,
  clock: ManualClock,
  input: {
    title?: string;
    narrative?: string;
  } = {},
): Promise<Episode> {
  const internal = borg as unknown as BorgTestInternals;
  const sourceEntry = await borg.stream.append({
    kind: "user_msg",
    content: input.narrative ?? "operator correction source",
    turn_id: "turn_correction_seed",
  });
  const now = clock.now();

  return internal.deps.episodicRepository.insert({
    id: createEpisodeId(),
    title: input.title ?? "Correction seed episode",
    narrative: input.narrative ?? "A correction endpoint seed episode.",
    participants: ["operator"],
    location: null,
    start_time: now,
    end_time: now,
    source_stream_ids: [sourceEntry.id],
    significance: 0.5,
    tags: ["demo"],
    confidence: 0.8,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: null,
    audience_entity_id: null,
    shared: false,
    embedding: new Float32Array([0.1, 0.2, 0.3, 0.4]),
    created_at: now,
    updated_at: now,
  });
}

async function seedP2EndpointRecords(borg: Borg, clock: ManualClock) {
  const internal = borg as unknown as BorgTestInternals;
  const sourceEntry = await borg.stream.append({
    kind: "user_msg",
    content: "seed p2 source",
    turn_id: "turn_seed",
    audience: "Alice",
  });
  const skill = await borg.skills.add({
    applies_when: "demo endpoint drills need a real skill",
    approach: "seed a source-linked skill and assert the DTO",
    sourceEpisodes: [createEpisodeId()],
  });

  borg.mood.update(DEFAULT_SESSION_ID, {
    valence: 0.4,
    arousal: 0.6,
    reason: "demo fixture",
    provenance: { kind: "manual" },
  });
  borg.social.recordInteraction("Alice", { provenance: { kind: "manual" }, valence: 0.25 });
  internal.deps.relationalSlotRepository.applyAssertion({
    subject_entity_id: borg.entities.resolve("Alice"),
    slot_key: "preferred_style",
    asserted_value: "terse",
    source_stream_entry_ids: [sourceEntry.id],
  });

  let attachmentId: AttachmentId;
  const writer = internal.deps.createStreamWriter(DEFAULT_SESSION_ID);
  try {
    const [persisted] = await internal.deps.attachmentService.persistTurnAttachments({
      attachments: [{ mediaType: "image/png", bytes: PNG_1X1 }],
      streamWriter: writer,
      parentEntry: sourceEntry,
      turnId: "turn_attachment",
      createdTurnGlobal: 12,
    });
    attachmentId = persisted!.attachmentId;
  } finally {
    writer.close();
  }
  internal.deps.attachmentService.setAttachmentActive(
    attachmentId,
    false,
    "turn_attachment_quarantine",
  );

  clock.advance(10);
  await borg.stream.append({
    kind: "dream_report",
    content: {
      processes: ["belief-reviser"],
      errors: [{ process: "belief-reviser", message: "old stream failure" }],
    },
    turn_id: "turn_dream_old",
  });

  clock.advance(10);
  const audit = internal.deps.auditLog.record({
    run_id: createMaintenanceRunId(),
    process: "belief-reviser",
    action: "revise demo belief",
    targets: { target_id: "semn_demo" },
    reversal: {},
  });
  const review = internal.deps.reviewQueueRepository.enqueue({
    kind: "belief_revision",
    refs: { target_type: "semantic_node", target_id: "semn_demo" },
    reason: "dependency invalidated",
    sourceProcess: "belief-reviser",
  });

  return {
    attachmentId,
    audit,
    review,
    skill,
  };
}

async function seedSemanticGraph(borg: Borg, clock: ManualClock) {
  const sourceEpisodeId = createEpisodeId();
  const nodes: Array<Awaited<ReturnType<Borg["semantic"]["nodes"]["add"]>>> = [];

  for (const input of [
    { kind: "entity" as const, label: "alice", description: "Alice entity" },
    { kind: "entity" as const, label: "borg", description: "Borg entity" },
    { kind: "concept" as const, label: "semantic graph", description: "Semantic graph concept" },
    { kind: "proposition" as const, label: "supports memory", description: "Memory support claim" },
    { kind: "concept" as const, label: "retrieval", description: "Retrieval concept" },
  ]) {
    nodes.push(
      await borg.semantic.nodes.add({
        ...input,
        sourceEpisodeIds: [sourceEpisodeId],
      }),
    );
  }

  const edgeInputs = [
    { from: 0, to: 1, relation: "supports" as const, confidence: 0.9 },
    { from: 0, to: 2, relation: "causes" as const, confidence: 0.7 },
    { from: 0, to: 3, relation: "related_to" as const, confidence: 0.5 },
    { from: 1, to: 2, relation: "is_a" as const, confidence: 0.6 },
    { from: 1, to: 4, relation: "prevents" as const, confidence: 0.4 },
  ];

  for (const edge of edgeInputs) {
    borg.semantic.edges.add({
      from_node_id: nodes[edge.from]!.id,
      to_node_id: nodes[edge.to]!.id,
      relation: edge.relation,
      confidence: edge.confidence,
      evidence_episode_ids: [sourceEpisodeId],
      created_at: clock.now(),
      last_verified_at: clock.now(),
    });
  }
}

describe("demo server", () => {
  const tempDirs: string[] = [];
  const closers: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (closers.length > 0) {
      await closers.pop()?.();
    }

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("closeAll attempts every live client even if one close throws", () => {
    const broadcaster = new LiveBroadcaster({ error: () => {} });
    let secondClosed = false;

    broadcaster.add({
      send(): void {},
      close(): void {
        throw new Error("close failed");
      },
    });
    broadcaster.add({
      send(): void {},
      close(): void {
        secondClosed = true;
      },
    });

    expect(() => broadcaster.closeAll()).not.toThrow();
    expect(secondClosed).toBe(true);
  });

  it("filters live frames by subscribed session and keeps global frames global", () => {
    const broadcaster = new LiveBroadcaster({ error: () => {} });
    const sessionA = createSessionId();
    const sessionB = createSessionId();
    const framesA: LiveFrame[] = [];
    const framesB: LiveFrame[] = [];
    const clientA = { send: (data: string) => framesA.push(JSON.parse(data) as LiveFrame) };
    const clientB = { send: (data: string) => framesB.push(JSON.parse(data) as LiveFrame) };

    broadcaster.add(clientA);
    broadcaster.add(clientB);
    broadcaster.handleSubscriptionMessage(clientA, { type: "subscribe", session_id: sessionA });
    broadcaster.handleSubscriptionMessage(clientB, { type: "subscribe", session_id: sessionB });

    broadcaster.broadcast({ type: "turn:terminal", ts: 1, session_id: sessionA });
    broadcaster.broadcast({ type: "turn:terminal", ts: 2, session_id: sessionB });
    broadcaster.broadcast({ type: "borg:reset", ts: 3 });

    expect(framesA.map((frame) => frame.ts)).toEqual([1, 3]);
    expect(framesB.map((frame) => frame.ts)).toEqual([2, 3]);
  });

  it("delivers reset to clients that unsubscribed from global frames", () => {
    const broadcaster = new LiveBroadcaster({ error: () => {} });
    const frames: LiveFrame[] = [];
    const client = { send: (data: string) => frames.push(JSON.parse(data) as LiveFrame) };

    broadcaster.add(client);
    broadcaster.handleSubscriptionMessage(client, { type: "unsubscribe_global" });
    broadcaster.broadcast({ type: "borg:reset", ts: 1 });

    expect(frames).toEqual([expect.objectContaining({ type: "borg:reset", ts: 1 })]);
  });

  it("flushes buffered session frames when a client subscribes", () => {
    const broadcaster = new LiveBroadcaster({ error: () => {} });
    const sessionId = createSessionId();
    const frames: LiveFrame[] = [];
    const client = { send: (data: string) => frames.push(JSON.parse(data) as LiveFrame) };

    broadcaster.broadcast({ type: "turn:phase:started", ts: Date.now(), session_id: sessionId });
    broadcaster.add(client);
    broadcaster.handleSubscriptionMessage(client, { type: "subscribe", session_id: sessionId });

    expect(frames).toEqual([
      expect.objectContaining({ type: "turn:phase:started", session_id: sessionId }),
    ]);
  });

  it("does not re-flush buffered session frames on duplicate subscribe", () => {
    const broadcaster = new LiveBroadcaster({ error: () => {} });
    const sessionId = createSessionId();
    const now = Date.now();
    const frames: LiveFrame[] = [];
    const client = { send: (data: string) => frames.push(JSON.parse(data) as LiveFrame) };

    broadcaster.broadcast({ type: "turn:phase:started", ts: now, session_id: sessionId });
    broadcaster.add(client);
    broadcaster.handleSubscriptionMessage(client, { type: "subscribe", session_id: sessionId });
    broadcaster.handleSubscriptionMessage(client, { type: "subscribe", session_id: sessionId });
    broadcaster.broadcast({ type: "turn:terminal", ts: now + 1, session_id: sessionId });

    expect(frames.map((frame) => frame.ts)).toEqual([now, now + 1]);
  });

  it("keeps final attempt frames scoped to their subscribed session", () => {
    const live = createLiveBridge();
    const sessionA = createSessionId();
    const sessionB = createSessionId();
    const framesA: LiveFrame[] = [];
    const framesB: LiveFrame[] = [];
    const clientA = { send: (data: string) => framesA.push(JSON.parse(data) as LiveFrame) };
    const clientB = { send: (data: string) => framesB.push(JSON.parse(data) as LiveFrame) };

    live.broadcaster.add(clientA);
    live.broadcaster.add(clientB);
    live.broadcaster.handleSubscriptionMessage(clientA, {
      type: "subscribe",
      session_id: sessionA,
    });
    live.broadcaster.handleSubscriptionMessage(clientB, {
      type: "subscribe",
      session_id: sessionB,
    });
    live.tracer.emit("commitment_guard.regeneration_requested", {
      turnId: "turn_final_attempt",
      session_id: sessionA,
      mode: "enforce",
      verdict: "requires_regeneration",
      violationCount: 1,
      commitmentIds: [],
      commitmentKinds: [],
      commitmentEnforcementClasses: [],
      criticalDomains: [],
    });

    expect(framesA).toEqual([
      expect.objectContaining({
        type: "turn:final_attempt",
        turn_id: "turn_final_attempt",
        session_id: sessionA,
      }),
    ]);
    expect(framesB).toEqual([]);
  });

  it("serves creator and operator session endpoints", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-creator-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const initialCreatorResponse = await app.request("/api/entities/creator");
    expect(initialCreatorResponse.status).toBe(200);
    const initialCreator = (await initialCreatorResponse.json()) as {
      id: string;
      canonical_name: string;
      borg_role: string | null;
    };
    expect(initialCreator).toMatchObject({
      canonical_name: "Tom",
      borg_role: "creator",
    });

    const updatedCreatorResponse = await requestJson(app, "/api/entities/creator", "POST", {
      name: "Dana",
    });
    expect(updatedCreatorResponse.status).toBe(200);
    const updatedCreator = (await updatedCreatorResponse.json()) as {
      id: string;
      canonical_name: string;
      borg_role: string | null;
    };
    expect(updatedCreator).toMatchObject({
      canonical_name: "Dana",
      borg_role: "creator",
    });
    expect(
      borg.entities.list().find((entity) => entity.id === initialCreator.id)?.borg_role,
    ).toBeNull();

    const operatorSessionResponse = await requestJson(app, "/api/sessions/operator", "POST", {});
    expect(operatorSessionResponse.status).toBe(200);
    expect(await operatorSessionResponse.json()).toMatchObject({
      audience_label: "Dana",
      audience_entity_id: updatedCreator.id,
      audience_role: "operator",
      label: "operator chat",
    });
  });

  it("returns 409 for operator session creation when no creator is set", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-no-creator-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const creator = borg.entities.getCreator();

    if (creator !== null) {
      borg.entities.setBorgRole(creator.id, null);
    }

    const response = await requestJson(app, "/api/sessions/operator", "POST", {});

    expect(response.status).toBe(409);
    expect(await response.json()).toMatchObject({
      error: {
        message: "Mark a creator first",
      },
    });
  });

  it("serves REST endpoint contract shapes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [
        createFakeEmitAnswerResponse("demo ok"),
        createEmptyReflectionResponse(),
        createFakeEmitAnswerResponse("custom session ok"),
        createEmptyReflectionResponse(),
        createFakeEmitAnswerResponse("custom session retry ok"),
        createEmptyReflectionResponse(),
        createFakeEmitAnswerResponse("custom session final ok"),
        createEmptyReflectionResponse(),
      ],
    });
    const { borg, clock, live } = await openHarness({ tempDir, llmClient: llm });
    closers.push(() => borg.close());
    live.ledgerCache.set("turn_cached", { sections: [] });
    borg.social.upsertProfile("Alice");
    const activeCommitment = borg.commitments.add({
      type: "rule",
      kind: "process_norm",
      enforcementClass: "advisory",
      directiveFamily: "demo",
      directive: "keep the demo endpoint shape stable",
      priority: 3,
      audience: "Alice",
      provenance: { kind: "manual" },
    });
    const revokedCommitment = borg.commitments.add({
      type: "boundary",
      kind: "audience_rule",
      enforcementClass: "critical",
      directiveFamily: "demo_revoke",
      directive: "old demo boundary",
      priority: 8,
      audience: "Alice",
      provenance: { kind: "manual" },
    });
    borg.commitments.revoke(revokedCommitment.id, "demo smoke", { kind: "manual" });
    const openQuestion = borg.self.openQuestions.add({
      question: "should the demo render resolved questions?",
      urgency: 0.5,
      source: "user",
      provenance: { kind: "manual" },
    });
    borg.self.openQuestions.abandon(
      openQuestion.id,
      "demo smoke",
      { kind: "manual" },
      { throughReview: true },
    );
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const customSessionId = createSessionId();
    borg.sessions.ensure({
      session_id: customSessionId,
      source_type: "demo",
      label: "demo custom",
      audience_label: "Alice",
      conversation_kind: "demo",
    });

    const state = await app.request("/api/state");
    expect(state.status).toBe(200);
    expect(await state.json()).toMatchObject({
      active_session: "default",
      counts: expect.objectContaining({
        turns: expect.any(Number),
        commitments: expect.any(Number),
        open_qs: expect.any(Number),
        dream_audit_rows: expect.any(Number),
      }),
      version: expect.any(String),
    });

    const customState = await app.request(`/api/state?session=${customSessionId}`);
    expect(customState.status).toBe(200);
    expect(await customState.json()).toMatchObject({
      active_session: customSessionId,
    });

    const sessions = await app.request("/api/sessions");
    expect(sessions.status).toBe(200);
    expect(await sessions.json()).toMatchObject({
      sessions: expect.arrayContaining([
        expect.objectContaining({
          session_id: DEFAULT_SESSION_ID,
          source_type: "demo",
          conversation_kind: "demo",
          privacy_level: "payload_off",
          participation_policy: "active",
        }),
        expect.objectContaining({
          session_id: customSessionId,
          label: "demo custom",
          participation_policy: "active",
        }),
      ]),
    });

    await borg.stream.append(
      { kind: "user_msg", content: "custom session seed", turn_id: "turn_custom_seed" },
      { session: customSessionId },
    );
    const customStream = await app.request(
      `/api/stream?session=${customSessionId}&kind=user_msg&limit=10`,
    );
    expect(customStream.status).toBe(200);
    expect(await customStream.json()).toMatchObject({
      entries: [expect.objectContaining({ session_id: customSessionId })],
      next_cursor: null,
    });

    const stream = await app.request("/api/stream?kind=user_msg,agent_msg&limit=10");
    expect(stream.status).toBe(200);
    expect(await stream.json()).toMatchObject({ entries: [], next_cursor: null });

    const ledger = await app.request("/api/turns/turn_cached/ledger");
    expect(ledger.status).toBe(200);
    expect(await ledger.json()).toMatchObject({ turn_id: "turn_cached", ledger: { sections: [] } });

    const seeded = await seedP2EndpointRecords(borg, clock);

    const bands = await app.request("/api/memory/bands");
    expect(bands.status).toBe(200);
    expect((await bands.json()).bands).toHaveLength(8);

    for (const band of [
      "episodic",
      "semantic",
      "procedural",
      "affective",
      "self",
      "commitments",
      "social",
      "relational",
    ]) {
      const response = await app.request(`/api/memory/bands/${band}`);
      expect(response.status).toBe(200);
      expect(await response.json()).toMatchObject({ band });
    }

    const procedural = await app.request("/api/memory/bands/procedural");
    expect(await procedural.json()).toMatchObject({
      items: [expect.objectContaining({ id: seeded.skill.id, sample_count: 1 })],
    });
    const affective = await app.request("/api/memory/bands/affective");
    expect(await affective.json()).toMatchObject({
      history: [expect.objectContaining({ trigger_reason: "demo fixture" })],
    });
    const social = await app.request("/api/memory/bands/social");
    expect(await social.json()).toMatchObject({
      items: [expect.objectContaining({ name: "Alice", history_count: 1 })],
    });
    const relational = await app.request("/api/memory/bands/relational");
    expect(await relational.json()).toMatchObject({
      items: [
        expect.objectContaining({
          slot: "Alice.preferred_style",
          state: "established",
          sources_count: 1,
          value: "terse",
        }),
      ],
    });

    const commitments = await app.request("/api/commitments?audience=Alice&state=all");
    expect(commitments.status).toBe(200);
    const commitmentBody = (await commitments.json()) as {
      commitments: Array<{
        id: string;
        state: string;
        enforcement_class: string;
        audience: string;
      }>;
    };
    expect(commitmentBody.commitments).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: activeCommitment.id,
          state: "active",
          enforcement_class: "advisory",
          audience: "Alice",
        }),
        expect.objectContaining({
          id: revokedCommitment.id,
          state: "revoked",
          enforcement_class: "critical",
          audience: "Alice",
        }),
      ]),
    );

    const criticalCommitments = await app.request(
      "/api/commitments?audience=Alice&state=all&enforcement=critical",
    );
    expect(criticalCommitments.status).toBe(200);
    expect((await criticalCommitments.json()) as { commitments: unknown[] }).toMatchObject({
      commitments: [expect.objectContaining({ id: revokedCommitment.id })],
    });

    const sharedState = await app.request("/api/shared-state?audience=Alice");
    expect(sharedState.status).toBe(200);
    expect(await sharedState.json()).toMatchObject({ audience: "Alice", entries: [] });

    const identity = await app.request("/api/identity");
    expect(identity.status).toBe(200);
    expect(await identity.json()).toMatchObject({
      values: [],
      traits: [],
      open_questions: [expect.objectContaining({ id: openQuestion.id, status: "abandoned" })],
      growth_markers: [],
      periods: [],
      open_question_events: expect.any(Array),
    });

    const audit = await app.request("/api/dream/audit?limit=5");
    expect(audit.status).toBe(200);
    expect(await audit.json()).toMatchObject({
      rows: [expect.objectContaining({ id: seeded.audit.id, action: "revise demo belief" })],
    });

    const dreamState = await app.request("/api/dream/state");
    expect(dreamState.status).toBe(200);
    expect(await dreamState.json()).toMatchObject({
      processes: expect.arrayContaining([
        expect.objectContaining({
          name: "belief-reviser",
          last_audit_id: seeded.audit.id,
          last_run_at: seeded.audit.applied_at,
          last_status: "ok",
        }),
      ]),
      schedule: expect.arrayContaining([
        expect.objectContaining({
          process: "belief-reviser",
          source: "audit",
          audit_id: seeded.audit.id,
        }),
        expect.objectContaining({ process: "belief-reviser", source: "stream" }),
      ]),
      audit_rows: [expect.objectContaining({ id: seeded.audit.id })],
      belief_revision_rows: [
        expect.objectContaining({ id: seeded.review.id, kind: "belief_revision" }),
      ],
      scheduler: expect.objectContaining({ enabled: expect.any(Boolean) }),
    });

    const attachmentMeta = await app.request(`/api/attachments/${seeded.attachmentId}`);
    expect(attachmentMeta.status).toBe(200);
    expect(await attachmentMeta.json()).toMatchObject({
      attachment: expect.objectContaining({ attachment_id: seeded.attachmentId }),
      status: expect.objectContaining({ active: false, quarantined: true, parent_active: true }),
    });

    const attachmentBatch = await app.request(`/api/attachments?ids=${seeded.attachmentId}`);
    expect(attachmentBatch.status).toBe(200);
    expect(await attachmentBatch.json()).toMatchObject([
      expect.objectContaining({
        id: seeded.attachmentId,
        status: expect.objectContaining({ quarantined: true }),
      }),
    ]);

    const missingAttachmentMeta = await app.request("/api/attachments/att_0000000000000000");
    expect(missingAttachmentMeta.status).toBe(404);

    const attachment = await app.request("/api/attachments/att_0000000000000000/bytes");
    expect(attachment.status).toBe(400);

    const missingAttachmentWithAudience = await app.request(
      "/api/attachments/att_0000000000000000/bytes?audience=Alice",
    );
    expect(missingAttachmentWithAudience.status).toBe(404);

    const badCommitmentQuery = await app.request("/api/commitments?state=missing");
    expect(badCommitmentQuery.status).toBe(400);

    const badAttachmentId = await app.request("/api/attachments/not_an_attachment");
    expect(badAttachmentId.status).toBe(400);

    const badAttachmentBatch = await app.request("/api/attachments?ids=not_an_attachment");
    expect(badAttachmentBatch.status).toBe(400);

    const malformed = await app.request("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{",
    });
    expect(malformed.status).toBe(400);
    expect(await malformed.json()).toEqual({
      error: {
        status: 400,
        message: "Malformed JSON body",
      },
    });

    const turn = await app.request("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hello", audience: "Alice", stakes: "low" }),
    });
    expect(turn.status).toBe(200);
    expect(await turn.json()).toMatchObject({ ok: true, turn_id: expect.any(String) });

    const customTurn = await app.request("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "hello custom",
        audience: "Alice",
        stakes: "low",
        session: customSessionId,
      }),
    });
    const customTurnText = await customTurn.text();
    expect(customTurn.status, customTurnText).toBe(200);
    const customTurnBody = JSON.parse(customTurnText) as { turn_id: string };
    expect(borg.sessions.get(customSessionId)).toMatchObject({
      session_id: customSessionId,
      last_turn_id: customTurnBody.turn_id,
      message_count: 1,
    });
  });

  it("POST /api/sessions/:id/participation updates session policy", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const sessionId = createSessionId();
    borg.sessions.ensure({
      session_id: sessionId,
      source_type: "demo",
      label: "demo policy",
      audience_label: "Alice",
      conversation_kind: "demo",
    });

    const updateResponse = await requestJson(
      app,
      `/api/sessions/${sessionId}/participation`,
      "POST",
      {
        policy: "observing",
        reason: "too much visible output",
      },
    );

    expect(updateResponse.status).toBe(200);
    expect(await updateResponse.json()).toMatchObject({
      session_id: sessionId,
      participation_policy: "observing",
    });

    const sessionsResponse = await app.request("/api/sessions");
    expect(sessionsResponse.status).toBe(200);
    expect(await sessionsResponse.json()).toMatchObject({
      sessions: expect.arrayContaining([
        expect.objectContaining({
          session_id: sessionId,
          participation_policy: "observing",
        }),
      ]),
    });
  });

  it("POST /api/sessions/:id/participation rejects invalid policy", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await requestJson(app, "/api/sessions/default/participation", "POST", {
      policy: "loud",
    });

    expect(response.status).toBe(400);
  });

  it("POST /api/sessions/:id/participation returns 404 for a missing session", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await requestJson(
      app,
      `/api/sessions/${createSessionId()}/participation`,
      "POST",
      { policy: "muted" },
    );

    expect(response.status).toBe(404);
  });

  it("accepts multipart turn uploads and writes image attachment stream entries", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [
        createImagePerceptionResponse(),
        createFakeEmitAnswerResponse("demo image ok"),
        createEmptyReflectionResponse(),
      ],
    });
    const { borg, live } = await openHarness({ tempDir, llmClient: llm });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const formData = new FormData();
    formData.set("message", "please look at this");
    formData.set("audience", "Alice");
    formData.set("stakes", "low");
    formData.append("attachments[]", new File([PNG_1X1], "pixel.png", { type: "image/png" }));

    const turn = await app.request("/api/turn", {
      method: "POST",
      body: formData,
    });

    expect(turn.status).toBe(200);
    expect(await turn.json()).toMatchObject({ ok: true, turn_id: expect.any(String) });

    const attachments: StreamEntry[] = [];
    for await (const entry of borg.stream.reader().iterate({ kinds: ["user_image_attachment"] })) {
      attachments.push(entry);
    }

    expect(attachments).toHaveLength(1);
    expect(attachments[0]).toMatchObject({
      kind: "user_image_attachment",
      audience: "Alice",
      content: expect.objectContaining({
        type: "image_ref",
        media_type: "image/png",
      }),
    });
  });

  it("wires operator mutation endpoints to Borg facade calls", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const dreamPlanSpy = vi.spyOn(borg.dream, "plan");
    const dreamApplySpy = vi.spyOn(borg.dream, "apply");
    const valueAddSpy = vi.spyOn(borg.self.values, "add");
    const goalAddSpy = vi.spyOn(borg.self.goals, "add");
    const goalStatusSpy = vi.spyOn(borg.self.goals, "updateStatus");
    const goalProgressSpy = vi.spyOn(borg.self.goals, "updateProgress");
    const growthAddSpy = vi.spyOn(borg.self.growthMarkers, "add");
    const questionResolveSpy = vi.spyOn(borg.self.openQuestions, "resolve");
    const questionAbandonSpy = vi.spyOn(borg.self.openQuestions, "abandon");
    const questionBumpSpy = vi.spyOn(borg.self.openQuestions, "bumpUrgency");
    const reviewResolveSpy = vi.spyOn(borg.review, "resolve");

    const plan = await requestJson(app, "/api/dream/plan", "POST", {
      processes: ["curator"],
      budget: 100,
    });
    expect(plan.status).toBe(200);
    const planBody = (await plan.json()) as { plan_id: string; processes: unknown[] };
    expect(planBody).toMatchObject({
      plan_id: expect.any(String),
      processes: [expect.objectContaining({ name: "curator" })],
    });
    expect(dreamPlanSpy).toHaveBeenCalledWith({
      processes: ["curator"],
      budget: 100,
    });

    const apply = await requestJson(app, "/api/dream/apply", "POST", {
      plan_id: planBody.plan_id,
    });
    expect(apply.status).toBe(200);
    expect(await apply.json()).toMatchObject({
      applied: [expect.objectContaining({ name: "curator" })],
      duration_ms: expect.any(Number),
    });
    expect(dreamApplySpy).toHaveBeenCalledTimes(1);

    const repeatedApply = await requestJson(app, "/api/dream/apply", "POST", {
      plan_id: planBody.plan_id,
    });
    expect(repeatedApply.status).toBe(200);
    expect(dreamApplySpy).toHaveBeenCalledTimes(1);

    const value = await requestJson(app, "/api/identity/values", "POST", {
      name: "care",
      description: "care about operator-visible state",
    });
    expect(value.status).toBe(200);
    expect(await value.json()).toMatchObject({ id: expect.stringMatching(/^val_/), label: "care" });
    expect(valueAddSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        label: "care",
        description: "care about operator-visible state",
      }),
    );

    const goal = await requestJson(app, "/api/identity/goals", "POST", {
      description: "ship sprint B",
      priority: 2,
    });
    expect(goal.status).toBe(200);
    const goalBody = (await goal.json()) as { id: string };
    expect(goalAddSpy).toHaveBeenCalledWith(
      expect.objectContaining({ description: "ship sprint B", priority: 2 }),
    );

    const completeGoal = await requestJson(app, `/api/identity/goals/${goalBody.id}`, "PATCH", {
      action: "complete",
    });
    expect(completeGoal.status).toBe(200);
    expect(await completeGoal.json()).toMatchObject({ id: goalBody.id, status: "done" });

    const blockedGoal = borg.self.goals.add({
      description: "blocked endpoint fixture",
      priority: 1,
      provenance: { kind: "manual" },
    });
    const block = await requestJson(app, `/api/identity/goals/${blockedGoal.id}`, "PATCH", {
      action: "block",
      note: "blocked by test fixture",
    });
    expect(block.status).toBe(200);
    expect(await block.json()).toMatchObject({ id: blockedGoal.id, status: "blocked" });

    const progressGoal = borg.self.goals.add({
      description: "progress endpoint fixture",
      priority: 1,
      provenance: { kind: "manual" },
    });
    const progress = await requestJson(app, `/api/identity/goals/${progressGoal.id}`, "PATCH", {
      action: "progress",
      progress: 50,
      note: "halfway",
    });
    expect(progress.status).toBe(200);
    expect(await progress.json()).toMatchObject({
      id: progressGoal.id,
      progress_notes: "progress 50%: halfway",
    });
    expect(goalStatusSpy).toHaveBeenCalledWith(
      goalBody.id,
      "done",
      { kind: "manual" },
      expect.objectContaining({ throughReview: true }),
    );
    expect(goalStatusSpy).toHaveBeenCalledWith(
      blockedGoal.id,
      "blocked",
      { kind: "manual" },
      expect.objectContaining({ throughReview: true }),
    );
    expect(goalProgressSpy).toHaveBeenCalledWith(
      progressGoal.id,
      "progress 50%: halfway",
      { kind: "manual" },
      expect.objectContaining({ throughReview: true }),
    );

    const growth = await requestJson(app, "/api/identity/growth-markers", "POST", {
      description: "operator surface exists",
      source: "demo",
    });
    expect(growth.status).toBe(200);
    expect(await growth.json()).toMatchObject({
      id: expect.stringMatching(/^grw_/),
      what_changed: "operator surface exists",
      source_process: "demo",
    });
    expect(growthAddSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        what_changed: "operator surface exists",
        evidence_episode_ids: [expect.stringMatching(/^strm_/)],
        source_process: "demo",
      }),
    );

    const resolveQuestion = borg.self.openQuestions.add({
      question: "what gets resolved?",
      urgency: 0.4,
      provenance: { kind: "manual" },
      source: "user",
    });
    const resolvedQuestion = await requestJson(
      app,
      `/api/identity/open-questions/${resolveQuestion.id}`,
      "PATCH",
      {
        action: "resolve",
        resolution: "operator supplied resolution",
      },
    );
    expect(resolvedQuestion.status).toBe(200);
    expect(await resolvedQuestion.json()).toMatchObject({
      id: resolveQuestion.id,
      status: "resolved",
      resolution_note: "operator supplied resolution",
    });
    expect(questionResolveSpy).toHaveBeenCalledWith(
      resolveQuestion.id,
      expect.objectContaining({
        resolution_note: "operator supplied resolution",
        resolution_evidence_stream_entry_ids: [expect.stringMatching(/^strm_/)],
      }),
      { kind: "manual" },
      expect.objectContaining({ throughReview: true }),
    );

    const abandonQuestion = borg.self.openQuestions.add({
      question: "what gets abandoned?",
      urgency: 0.4,
      provenance: { kind: "manual" },
      source: "user",
    });
    const abandonedQuestion = await requestJson(
      app,
      `/api/identity/open-questions/${abandonQuestion.id}`,
      "PATCH",
      {
        action: "abandon",
        reason: "operator abandoned it",
      },
    );
    expect(abandonedQuestion.status).toBe(200);
    expect(await abandonedQuestion.json()).toMatchObject({
      id: abandonQuestion.id,
      status: "abandoned",
      abandoned_reason: "operator abandoned it",
    });
    expect(questionAbandonSpy).toHaveBeenCalledWith(
      abandonQuestion.id,
      "operator abandoned it",
      { kind: "manual" },
      expect.objectContaining({ throughReview: true }),
    );

    const bumpQuestion = borg.self.openQuestions.add({
      question: "what gets bumped?",
      urgency: 0.4,
      provenance: { kind: "manual" },
      source: "user",
    });
    const bumpedQuestion = await requestJson(
      app,
      `/api/identity/open-questions/${bumpQuestion.id}`,
      "PATCH",
      {
        action: "bump",
      },
    );
    expect(bumpedQuestion.status).toBe(200);
    expect(await bumpedQuestion.json()).toMatchObject({ id: bumpQuestion.id, urgency: 0.5 });
    expect(questionBumpSpy).toHaveBeenCalledWith(
      bumpQuestion.id,
      0.1,
      { kind: "manual" },
      expect.objectContaining({ throughReview: true }),
    );

    const internal = borg as unknown as BorgTestInternals;
    const review = internal.deps.reviewQueueRepository.enqueue({
      kind: "belief_revision",
      refs: {
        target_type: "semantic_node",
        target_id: createSemanticNodeId(),
        invalidated_edge_id: createSemanticEdgeId(),
        dependency_path_edge_ids: [],
        surviving_support_edge_ids: [],
        evidence_episode_ids: [],
      },
      reason: "operator review fixture",
      sourceProcess: "belief-reviser",
    });
    const reviewResponse = await requestJson(app, `/api/dream/review/${review.id}`, "PATCH", {
      action: "dismiss",
      note: "not actionable",
    });
    expect(reviewResponse.status).toBe(200);
    expect(await reviewResponse.json()).toMatchObject({
      id: review.id,
      resolved_at: expect.any(Number),
      resolution: "dismiss",
    });
    expect(reviewResolveSpy).toHaveBeenCalledWith(review.id, {
      decision: "dismiss",
      reason: "not actionable",
    });
  });

  it("exposes correction endpoints for why, forget, correct, edge invalidation, and review resolution", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const episode = await seedCorrectionEpisode(borg, clock, {
      title: "Correction why fixture",
      narrative: "The operator needs provenance for this remembered event.",
    });

    const why = await app.request(`/api/correction/${episode.id}/why`);
    expect(why.status).toBe(200);
    expect(await why.json()).toMatchObject({
      target_type: "episode",
      record: {
        id: episode.id,
        title: "Correction why fixture",
      },
      source_stream_ids: episode.source_stream_ids,
      citation_chain: expect.any(Array),
    });

    const firstNode = await borg.semantic.nodes.add({
      kind: "concept",
      label: "correction endpoint source",
      description: "Source node for correction endpoint tests.",
      sourceEpisodeIds: [episode.id],
    });
    const secondNode = await borg.semantic.nodes.add({
      kind: "concept",
      label: "correction endpoint target",
      description: "Target node for correction endpoint tests.",
      sourceEpisodeIds: [episode.id],
    });
    const edge = borg.semantic.edges.add({
      from_node_id: firstNode.id,
      to_node_id: secondNode.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [episode.id],
      created_at: clock.now(),
      last_verified_at: clock.now(),
    });
    const invalidatedAt = clock.now() + 12_345;
    const invalidated = await requestJson(
      app,
      `/api/correction/semantic-edges/${edge.id}/invalidate`,
      "POST",
      {
        at: invalidatedAt,
        reason: "operator found edge stale",
      },
    );
    expect(invalidated.status).toBe(200);
    expect(await invalidated.json()).toMatchObject({
      id: edge.id,
      valid_to: invalidatedAt,
      invalidated_at: clock.now(),
      invalidated_by_process: "manual",
      invalidated_reason: "operator found edge stale",
    });

    const forgotten = await requestJson(app, `/api/correction/${episode.id}/forget`, "POST", {});
    expect(forgotten.status).toBe(200);
    expect(await forgotten.json()).toMatchObject({
      id: episode.id,
      target_type: "episode",
      archived: true,
      provenance: { kind: "manual" },
    });
    expect(
      borg.correction.listIdentityEvents({
        recordType: "episode",
        recordId: episode.id,
      }),
    ).toEqual([expect.objectContaining({ action: "forget", record_id: episode.id })]);

    const acceptedValue = borg.self.values.add({
      label: "accuracy",
      description: "keep memories accurate",
      priority: 1,
      provenance: { kind: "manual" },
    });
    const correct = await requestJson(app, `/api/correction/${acceptedValue.id}/correct`, "POST", {
      patch: { description: "keep corrected memories accurate" },
      reason: "operator correction",
    });
    expect(correct.status).toBe(200);
    const correctBody = (await correct.json()) as { id: number; refs: Record<string, unknown> };
    expect(correctBody.refs).toMatchObject({
      operator_reason: "operator correction",
    });
    const reviews = await app.request("/api/correction/reviews");
    expect(reviews.status).toBe(200);
    expect(await reviews.json()).toMatchObject({
      rows: [
        expect.objectContaining({
          id: correctBody.id,
          kind: "correction",
          refs: expect.objectContaining({
            target_id: acceptedValue.id,
            target_type: "value",
            patch: { description: "keep corrected memories accurate" },
            operator_reason: "operator correction",
          }),
        }),
      ],
    });

    const accepted = await requestJson(app, `/api/correction/reviews/${correctBody.id}`, "PATCH", {
      action: "accept",
      note: "looks right",
    });
    expect(accepted.status).toBe(200);
    expect(await accepted.json()).toMatchObject({
      id: correctBody.id,
      resolved_at: expect.any(Number),
      resolution: "accept",
    });
    expect(borg.self.values.get(acceptedValue.id)?.description).toBe(
      "keep corrected memories accurate",
    );
    expect(
      borg.correction.listIdentityEvents({
        recordType: "value",
        recordId: acceptedValue.id,
      }),
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          action: "correction_apply",
          reason: "operator correction",
          review_item_id: correctBody.id,
        }),
      ]),
    );

    const rejectedValue = borg.self.values.add({
      label: "unchanged",
      description: "leave this alone",
      priority: 1,
      provenance: { kind: "manual" },
    });
    const rejectCorrect = await requestJson(
      app,
      `/api/correction/${rejectedValue.id}/correct`,
      "POST",
      {
        patch: { description: "should not apply" },
      },
    );
    expect(rejectCorrect.status).toBe(200);
    const rejectCorrectBody = (await rejectCorrect.json()) as { id: number };
    const rejected = await requestJson(
      app,
      `/api/correction/reviews/${rejectCorrectBody.id}`,
      "PATCH",
      {
        action: "reject",
        note: "not valid",
      },
    );
    expect(rejected.status).toBe(200);
    expect(await rejected.json()).toMatchObject({
      id: rejectCorrectBody.id,
      resolved_at: expect.any(Number),
      resolution: "reject",
    });
    expect(borg.self.values.get(rejectedValue.id)?.description).toBe("leave this alone");
  });

  it("does not resolve non-correction review rows through the correction queue endpoint", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const internal = borg as unknown as BorgTestInternals;
    const review = internal.deps.reviewQueueRepository.enqueue({
      kind: "relationship_claim_ungrounded",
      refs: {
        target_type: "semantic_node_candidate",
        label: "relationship claim",
        description: "Ungrounded relationship claim fixture.",
        relationship_claim_label_families: ["kinship"],
        relationship_claims: [],
        ungrounded_relationship_claims: [],
      },
      reason: "non-correction review fixture",
    });

    const response = await requestJson(app, `/api/correction/reviews/${review.id}`, "PATCH", {
      action: "accept",
      note: "wrong queue",
    });

    expect(response.status).toBe(404);
    expect(await response.json()).toMatchObject({
      error: {
        message: "correction review item not found",
      },
    });
    expect(internal.deps.reviewQueueRepository.get(review.id)).toMatchObject({
      id: review.id,
      kind: "relationship_claim_ungrounded",
      resolved_at: null,
      resolution: null,
    });
  });

  it("POST /api/commitments creates an operator-authored commitment and lists it", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await requestJson(app, "/api/commitments", "POST", {
      type: "rule",
      kind: "process_norm",
      directive: "Prefer direct answers when speaking with Alice.",
      priority: 7,
      audience: "Alice",
      made_to: "Tom",
      about: "Project Atlas",
      directive_family: "creator guidance",
    });

    expect(response.status).toBe(200);
    const created = (await response.json()) as { id: string; source: string };
    expect(created).toMatchObject({
      id: expect.stringMatching(/^cmt_/),
      source: "manual",
    });

    const list = await app.request("/api/commitments?audience=Alice&state=all");
    expect(list.status).toBe(200);
    expect(await list.json()).toMatchObject({
      commitments: [
        expect.objectContaining({
          id: created.id,
          text: "Prefer direct answers when speaking with Alice.",
          state: "active",
          audience: "Alice",
          made_to: "Tom",
          about: "Project Atlas",
          directive_family: "creator_guidance",
        }),
      ],
    });
  });

  it("operator-authored commitment can be forgotten via correction (cross-sprint A+B)", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const createdResponse = await requestJson(app, "/api/commitments", "POST", {
      type: "rule",
      kind: "process_norm",
      directive: "Prefer terse status updates when speaking with Alice.",
      priority: 6,
      audience: "Alice",
      directive_family: "cross_sprint_ab",
    });

    expect(createdResponse.status).toBe(200);
    const created = (await createdResponse.json()) as { id: string };

    const forgottenResponse = await requestJson(
      app,
      `/api/correction/${created.id}/forget`,
      "POST",
      {},
    );
    expect(forgottenResponse.status).toBe(200);
    expect(await forgottenResponse.json()).toMatchObject({
      id: created.id,
      target_type: "commitment",
      archived: true,
      provenance: { kind: "manual" },
    });

    const activeResponse = await app.request("/api/commitments?state=active");
    expect(activeResponse.status).toBe(200);
    const activeBody = (await activeResponse.json()) as { commitments: Array<{ id: string }> };
    expect(activeBody.commitments.map((commitment) => commitment.id)).not.toContain(created.id);

    const allResponse = await app.request("/api/commitments?state=all");
    expect(allResponse.status).toBe(200);
    const allBody = (await allResponse.json()) as {
      commitments: Array<{ id: string; state: string; revoked_reason: string | null }>;
    };
    expect(allBody.commitments).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: created.id,
          state: "revoked",
          revoked_reason: "forgotten manually",
        }),
      ]),
    );
    expect(
      borg.correction.listIdentityEvents({
        recordType: "commitment",
        recordId: created.id,
      }),
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          action: "revoke",
          reason: "forgotten manually",
          provenance: { kind: "manual" },
        }),
      ]),
    );
  });

  it("POST /api/commitments rejects invalid bodies", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await requestJson(app, "/api/commitments", "POST", {
      type: "rule",
      kind: "process_norm",
      directive: "",
      priority: 11,
    });

    expect(response.status).toBe(400);
  });

  it("POST /api/commitments rejects critical enforcement at the operator boundary", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await requestJson(app, "/api/commitments", "POST", {
      type: "rule",
      kind: "process_norm",
      enforcement_class: "critical",
      directive: "This should not become a hard guard.",
      priority: 5,
    });

    expect(response.status).toBe(400);
  });

  it("POST /api/commitments rejects fractional or negative expiration timestamps", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const fractional = await requestJson(app, "/api/commitments", "POST", {
      type: "rule",
      kind: "process_norm",
      directive: "Fractional expiry should be rejected.",
      priority: 5,
      expires_at: 1.5,
    });
    const negative = await requestJson(app, "/api/commitments", "POST", {
      type: "rule",
      kind: "process_norm",
      directive: "Negative expiry should be rejected.",
      priority: 5,
      expires_at: -1,
    });

    expect(fractional.status).toBe(400);
    expect(negative.status).toBe(400);
  });

  it("POST /api/commitments is rejected while reset is in progress", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const resetStarted = createDeferred<void>();
    const resetRelease = createDeferred<void>();
    const resetBorg = vi.fn(async () => {
      resetStarted.resolve();
      await resetRelease.promise;
    });
    const { app } = createDemoServerApp({
      borgHandle: { current: borg },
      live,
      resetBorg,
    });

    const reset = app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    });
    await resetStarted.promise;

    const response = await requestJson(app, "/api/commitments", "POST", {
      type: "rule",
      kind: "process_norm",
      directive: "This should wait until reset completes.",
      priority: 5,
    });
    expect(response.status).toBe(503);

    resetRelease.resolve();
    expect((await reset).status).toBe(200);
  });

  it("POST /api/commitments/:id/revoke revokes an active commitment", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const commitment = borg.commitments.add({
      type: "preference",
      kind: "participant_preference",
      enforcementClass: "advisory",
      directiveFamily: "revocation_fixture",
      directive: "Use short answers for Alice.",
      priority: 4,
      audience: "Alice",
      provenance: { kind: "manual" },
    });

    const response = await requestJson(app, `/api/commitments/${commitment.id}/revoke`, "POST", {
      reason: "operator changed the standing instruction",
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      id: commitment.id,
      state: "revoked",
      revoked_reason: "operator changed the standing instruction",
      source: "manual",
    });
    const stored = borg.commitments
      .list({ activeOnly: false })
      .find((record) => record.id === commitment.id);
    expect(stored?.revoked_reason).toBe("operator changed the standing instruction");
    expect(stored?.revoke_provenance).toMatchObject({ kind: "manual" });
  });

  it("POST /api/commitments/:id/revoke returns 404 for a missing commitment", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await requestJson(
      app,
      "/api/commitments/cmt_0000000000000000/revoke",
      "POST",
      {},
    );

    expect(response.status).toBe(404);
  });

  it("paginates stream entries with same-timestamp file order", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    let older = await borg.stream.append({ kind: "internal_event", content: { index: 0 } });
    let cursorEntry = null as Awaited<ReturnType<typeof borg.stream.append>> | null;

    for (let index = 1; index < 80; index += 1) {
      const entry = await borg.stream.append({ kind: "internal_event", content: { index } });

      if (older.id.localeCompare(entry.id) > 0) {
        cursorEntry = entry;
        break;
      }

      older = entry;
    }

    expect(cursorEntry).not.toBeNull();

    const firstPage = await app.request("/api/stream?limit=1");
    expect(firstPage.status).toBe(200);
    const firstBody = (await firstPage.json()) as {
      entries: Array<{ id: string; entry_index?: number }>;
      next_cursor: string | null;
    };
    expect(firstBody.entries[0]?.id).toBe(cursorEntry?.id);
    expect(firstBody.entries[0]?.entry_index).toBe(cursorEntry?.entry_index);
    expect(firstBody.next_cursor).not.toBeNull();

    const secondPage = await app.request(`/api/stream?limit=1&before=${firstBody.next_cursor}`);
    expect(secondPage.status).toBe(200);
    const secondBody = (await secondPage.json()) as { entries: Array<{ id: string }> };
    expect(secondBody.entries[0]?.id).toBe(older.id);
  });

  it("serves capped semantic graph visualization data", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    await seedSemanticGraph(borg, clock);

    const response = await app.request("/api/semantic/graph?limit=3");

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      nodes: expect.arrayContaining([
        expect.objectContaining({
          id: expect.any(String),
          label: expect.any(String),
          status: "active",
          kind: expect.any(String),
          edge_count: expect.any(Number),
        }),
      ]),
      edges: expect.arrayContaining([
        expect.objectContaining({
          id: expect.any(String),
          source: expect.any(String),
          target: expect.any(String),
          type: expect.any(String),
          weight: expect.any(Number),
        }),
      ]),
      total_nodes: 5,
      total_edges: 5,
      rendered: { nodes: 3, edges: 3 },
    });

    const capped = await app.request("/api/semantic/graph?limit=999");

    expect(capped.status).toBe(200);
    expect(await capped.json()).toMatchObject({
      total_nodes: 5,
      rendered: { nodes: 5, edges: 5 },
    });
  });

  it("surfaces indexed entry_index for legacy stream JSONL rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const entry = await borg.stream.append({ kind: "internal_event", content: { legacy: true } });
    const streamPath = join(tempDir, "stream", `${DEFAULT_SESSION_ID}.jsonl`);
    const rawLine = readFileSync(streamPath, "utf8").trimEnd();
    const rawEntry = JSON.parse(rawLine) as Record<string, unknown>;
    delete rawEntry.entry_index;
    writeFileSync(streamPath, `${JSON.stringify(rawEntry)}\n`);

    const response = await app.request("/api/stream?limit=1");

    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      entries: Array<{ id: string; entry_index?: number }>;
    };
    expect(body.entries[0]).toMatchObject({
      id: entry.id,
      entry_index: entry.entry_index,
    });
  });

  it("broadcasts turn phases and stream appends over the live bridge", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [createFakeEmitAnswerResponse("ws ok"), createEmptyReflectionResponse()],
    });
    const { borg, live } = await openHarness({ tempDir, llmClient: llm });
    closers.push(() => borg.close());
    const { frames, wasClosed } = collectLiveFrames(live);

    const response = await borg.turn({
      userMessage: "hello ws",
      audience: "Alice",
      stakes: "low",
    });
    expect(response.turn_id).toEqual(expect.any(String));

    await waitFor(
      () =>
        frames.some((frame) => frame.type === "stream:append") &&
        frames.some((frame) => frame.type === "turn:phase:started") &&
        frames.some((frame) => frame.type === "turn:phase:completed") &&
        frames.some((frame) => frame.type === "turn:terminal"),
    );

    const phaseFrames = frames.filter((frame) => frame.type.startsWith("turn:phase:"));
    const perceptionStart = phaseFrames.findIndex(
      (frame) =>
        frame.type === "turn:phase:started" &&
        (frame.data as { phase?: unknown } | undefined)?.phase === "perception",
    );
    const perceptionComplete = phaseFrames.findIndex(
      (frame) =>
        frame.type === "turn:phase:completed" &&
        (frame.data as { phase?: unknown } | undefined)?.phase === "perception",
    );

    expect(perceptionStart).toBeGreaterThanOrEqual(0);
    expect(perceptionComplete).toBeGreaterThan(perceptionStart);
    expect(frames.find((frame) => frame.type === "turn:terminal")).toMatchObject({
      event: "turn.terminal",
      data: {
        outcome: "reflected",
        turn_id: expect.any(String),
        duration_ms: expect.any(Number),
      },
    });
    live.broadcaster.closeAll();
    expect(wasClosed()).toBe(true);
  });

  it("broadcasts token frames between finalizer phase start and completion", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [
        createFakeStreamingResponse(["ws ", "token"], createFakeEmitAnswerResponse("ws token ok")),
        createEmptyReflectionResponse(),
      ],
    });
    const { borg, live } = await openHarness({ tempDir, llmClient: llm });
    closers.push(() => borg.close());
    const { frames } = collectLiveFrames(live);

    await borg.turn({
      userMessage: "hello token ws",
      audience: "Alice",
      stakes: "low",
    });

    await waitFor(() => frames.some((frame) => frame.type === "turn:token"));

    const finalStart = frames.findIndex(
      (frame) =>
        frame.type === "turn:phase:started" &&
        (frame.data as { phase?: unknown } | undefined)?.phase === "final",
    );
    const tokenFrame = frames.findIndex((frame) => frame.type === "turn:token");
    const finalComplete = frames.findIndex(
      (frame) =>
        frame.type === "turn:phase:completed" &&
        (frame.data as { phase?: unknown } | undefined)?.phase === "final",
    );

    expect(finalStart).toBeGreaterThanOrEqual(0);
    expect(tokenFrame).toBeGreaterThan(finalStart);
    expect(finalComplete).toBeGreaterThan(tokenFrame);
    expect(frames[tokenFrame]).toMatchObject({
      type: "turn:token",
      phase: "final",
      chunk_text: "ws ",
      sequence: 1,
    });
  });

  it("broadcasts turn terminal frames to /api/live WebSocket clients after phase frames", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [createFakeEmitAnswerResponse("ws terminal ok"), createEmptyReflectionResponse()],
    });
    const { borg, live } = await openHarness({ tempDir, llmClient: llm });
    closers.push(() => borg.close());
    const { app, injectWebSocket } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const server = serve({ fetch: app.fetch, hostname: "127.0.0.1", port: 0 });
    injectWebSocket(server);
    closers.push(
      () =>
        new Promise<void>((resolve) => {
          server.close(() => resolve());
        }),
    );
    await waitFor(() => server.address() !== null);

    const frames: LiveFrame[] = [];
    const ws = new WebSocket(`ws://127.0.0.1:${serverPort(server)}/api/live`);
    closers.push(async () => ws.close());
    ws.addEventListener("message", (event) => {
      frames.push(JSON.parse(String(event.data)) as LiveFrame);
    });
    await new Promise<void>((resolve, reject) => {
      ws.addEventListener("open", () => resolve(), { once: true });
      ws.addEventListener("error", () => reject(new Error("websocket failed")), { once: true });
    });
    ws.send(JSON.stringify({ type: "subscribe", session_id: DEFAULT_SESSION_ID }));

    const result = await borg.turn({
      userMessage: "hello websocket terminal",
      audience: "Alice",
      stakes: "low",
    });

    await waitFor(() =>
      frames.some(
        (frame) =>
          frame.type === "turn:terminal" &&
          (frame.data as { turn_id?: unknown } | undefined)?.turn_id === result.turn_id,
      ),
    );

    const terminalIndex = frames.findIndex(
      (frame) =>
        frame.type === "turn:terminal" &&
        (frame.data as { turn_id?: unknown } | undefined)?.turn_id === result.turn_id,
    );
    const phaseIndices = frames
      .map((frame, index) => ({ frame, index }))
      .filter(
        ({ frame }) =>
          frame.type.startsWith("turn:phase:") &&
          (frame.data as { turn_id?: unknown } | undefined)?.turn_id === result.turn_id,
      )
      .map(({ index }) => index);

    expect(phaseIndices.length).toBeGreaterThan(0);
    expect(Math.max(...phaseIndices)).toBeLessThan(terminalIndex);
    expect(frames[terminalIndex]).toMatchObject({
      type: "turn:terminal",
      event: "turn.terminal",
      data: {
        turn_id: result.turn_id,
        outcome: "reflected",
      },
    });
  });

  it("broadcasts evidence ledger events over the live bridge", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { frames, wasClosed } = collectLiveFrames(live);

    live.tracer.emit("evidence_ledger.built", {
      turnId: "turn_ledger",
      turn_id: "turn_ledger",
      entry_counts: {},
      ledger: { sections: [] },
    });

    await waitFor(() => frames.some((frame) => frame.type === "evidence_ledger:built"));

    expect(live.ledgerCache.get("turn_ledger")).toEqual({ sections: [] });
    expect(frames.find((frame) => frame.type === "evidence_ledger:built")).toMatchObject({
      turn_id: "turn_ledger",
      ledger: { sections: [] },
    });
    live.broadcaster.closeAll();
    expect(wasClosed()).toBe(true);
  });

  it("GET /api/prompts returns 5 blocks, each defaulted and not overridden", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const hostCapabilities = "Configured host capability block.";
    const { borg, live } = await openHarness({ tempDir, hostCapabilities });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/prompts");
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      blocks: Array<{ key: string; current_text: string; overridden: boolean }>;
    };
    expect(body.blocks.map((b) => b.key)).toEqual([
      "base_identity_preamble",
      "voice_and_posture",
      "epistemic_posture",
      "identity_posture",
      "host_capabilities",
    ]);
    expect(body.blocks.every((b) => b.overridden === false)).toBe(true);
    const hostCapabilitiesBlock = body.blocks.find((b) => b.key === "host_capabilities");
    expect(hostCapabilitiesBlock?.current_text).toContain(hostCapabilities);
    expect(hostCapabilitiesBlock?.current_text).toContain(
      "Proactive outbound messaging via wired source_type connector(s): demo",
    );
  });

  it("PUT /api/prompts/:key sets an override, DELETE clears it", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const put = await app.request("/api/prompts/voice_and_posture", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "Speak crisply." }),
    });
    expect(put.status).toBe(200);
    const putBody = (await put.json()) as { current_text: string; overridden: boolean };
    expect(putBody).toMatchObject({ current_text: "Speak crisply.", overridden: true });

    const list = (await (await app.request("/api/prompts")).json()) as {
      blocks: Array<{ key: string; current_text: string; overridden: boolean }>;
    };
    expect(list.blocks.find((b) => b.key === "voice_and_posture")).toMatchObject({
      current_text: "Speak crisply.",
      overridden: true,
    });

    const del = await app.request("/api/prompts/voice_and_posture", { method: "DELETE" });
    expect(del.status).toBe(200);
    const delBody = (await del.json()) as { overridden: boolean };
    expect(delBody.overridden).toBe(false);
  });

  it("PUT /api/prompts/:key rejects unknown keys with 404", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/prompts/not_a_real_key", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "anything" }),
    });
    expect(response.status).toBe(404);
  });

  it("PUT /api/prompts/:key rejects whitespace-only prompt text", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/prompts/voice_and_posture", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "   \n\t" }),
    });
    expect(response.status).toBe(400);
  });

  it("PUT /api/prompts/:key trims prompt text before storing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/prompts/voice_and_posture", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "  hello  " }),
    });
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      current_text: "hello",
      overridden: true,
    });
    expect(borg.prompts.list().find((block) => block.key === "voice_and_posture")).toMatchObject({
      current_text: "hello",
    });
  });

  it("reset is rejected while a Borg request is in flight", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const turnStarted = createDeferred<void>();
    const turnRelease = createDeferred<TurnResult>();
    const turnSpy = vi.spyOn(borg, "turn").mockImplementation(async () => {
      turnStarted.resolve();
      return turnRelease.promise;
    });
    const resetBorg = vi.fn(async () => {});
    const { app } = createDemoServerApp({
      borgHandle: { current: borg },
      live,
      resetBorg,
    });

    const turn = app.request("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hello" }),
    });
    await turnStarted.promise;

    const reset = await app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    });
    expect(reset.status).toBe(409);
    expect(resetBorg).not.toHaveBeenCalled();

    turnRelease.resolve({ turn_id: "turn_gate" } as TurnResult);
    expect((await turn).status).toBe(200);
    turnSpy.mockRestore();
  });

  it("Borg requests are rejected during reset and accepted after reset completes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const resetStarted = createDeferred<void>();
    const resetRelease = createDeferred<void>();
    const resetBorg = vi.fn(async () => {
      resetStarted.resolve();
      await resetRelease.promise;
    });
    const { app } = createDemoServerApp({
      borgHandle: { current: borg },
      live,
      resetBorg,
    });

    const reset = app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    });
    await resetStarted.promise;

    const turn = await app.request("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hello during reset" }),
    });
    expect(turn.status).toBe(503);

    resetRelease.resolve();
    expect((await reset).status).toBe(200);
    expect((await app.request("/api/state")).status).toBe(200);
  });

  it("reset controller wipes state, reopens Borg, clears ledger cache, and broadcasts", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-reset-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    const borgHandle = { current: borg };
    closers.push(() => borgHandle.current.close());
    const { frames } = collectLiveFrames(live);

    await borg.stream.append({ kind: "user_msg", content: "before reset" });
    borg.prompts.set("voice_and_posture", "custom voice");
    live.ledgerCache.set("turn_old", { sections: [] });
    const streamPath = join(tempDir, "stream", `${DEFAULT_SESSION_ID}.jsonl`);
    expect(existsSync(streamPath)).toBe(true);
    mkdirSync(join(tempDir, "lancedb"), { recursive: true });
    const lanceMarker = join(tempDir, "lancedb", "stale-marker");
    writeFileSync(lanceMarker, "old");

    const resetBorg = createResetBorgController({
      dataDir: tempDir,
      live,
      borgHandle,
      openBorg: () =>
        Borg.open(
          createHarnessOpenOptions({
            tempDir,
            live,
            clock,
          }),
        ),
    });
    await resetBorg();

    expect(borgHandle.current).not.toBe(borg);
    expect(existsSync(streamPath)).toBe(false);
    expect(existsSync(join(tempDir, "borg.db"))).toBe(true);
    expect(existsSync(lanceMarker)).toBe(false);
    expect(borgHandle.current.stream.tail(10)).toEqual([]);
    expect(borgHandle.current.prompts.list().every((block) => !block.overridden)).toBe(true);
    expect(live.ledgerCache.size).toBe(0);
    expect(frames.some((frame) => frame.type === "borg:reset")).toBe(true);
  });

  it("reset controller clears buffered live session frames before reopening", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-reset-"));
    tempDirs.push(tempDir);
    const live = createLiveBridge();
    const sessionId = createSessionId();
    const borgHandle: BorgHandle = {
      current: { close: vi.fn(async () => {}) } as unknown as Borg,
    };
    const nextBorg = { close: vi.fn(async () => {}) } as unknown as Borg;
    const resetBorg = createResetBorgController({
      dataDir: tempDir,
      live,
      borgHandle,
      openBorg: async () => nextBorg,
    });
    const frames: LiveFrame[] = [];
    const client = { send: (data: string) => frames.push(JSON.parse(data) as LiveFrame) };

    live.broadcaster.broadcast({ type: "turn:phase:started", ts: 1, session_id: sessionId });
    await resetBorg();
    live.broadcaster.add(client);
    live.broadcaster.handleSubscriptionMessage(client, {
      type: "subscribe",
      session_id: sessionId,
    });

    expect(frames).toEqual([]);
  });

  it("reset controller rejects concurrent reset calls", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-reset-"));
    tempDirs.push(tempDir);
    const live = createLiveBridge();
    const openStarted = createDeferred<void>();
    const openRelease = createDeferred<void>();
    const nextBorg = { close: vi.fn(async () => {}) } as unknown as Borg;
    const borgHandle = {
      current: { close: vi.fn(async () => {}) } as unknown as Borg,
    };
    const resetBorg = createResetBorgController({
      dataDir: tempDir,
      live,
      borgHandle,
      openBorg: async () => {
        openStarted.resolve();
        await openRelease.promise;
        return nextBorg;
      },
    });

    const first = resetBorg();
    await openStarted.promise;
    await expect(resetBorg()).rejects.toThrow("Reset already in progress");
    openRelease.resolve();
    await first;
    expect(borgHandle.current).toBe(nextBorg);
  });

  it("POST /api/admin/reset retries reopen after a post-wipe open failure", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-reset-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    const borgHandle: BorgHandle = { current: borg };
    closers.push(() => borgHandle.current.close());
    const closeSpy = vi.spyOn(borg, "close");
    let openAttempts = 0;
    const resetBorg = createResetBorgController({
      dataDir: tempDir,
      live,
      borgHandle,
      openBorg: async () => {
        openAttempts += 1;
        if (openAttempts === 1) {
          throw new Error("reopen failed");
        }
        return Borg.open(
          createHarnessOpenOptions({
            tempDir,
            live,
            clock,
          }),
        );
      },
    });
    const { app } = createDemoServerApp({ borgHandle, live, resetBorg });

    const first = await app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    });
    expect(first.status).toBe(500);
    expect(await first.json()).toMatchObject({ error: { message: "reopen failed" } });
    expect(borgHandle.state).toBe("dead");
    expect(closeSpy).toHaveBeenCalledTimes(1);
    expect((await app.request("/api/state")).status).toBe(503);

    const second = await app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    });
    expect(second.status).toBe(200);
    expect(openAttempts).toBe(2);
    expect(closeSpy).toHaveBeenCalledTimes(1);
    expect(borgHandle.current).not.toBe(borg);
    expect(borgHandle.state).toBe("open");
    expect((await app.request("/api/state")).status).toBe(200);
  });

  it("POST /api/admin/reset rejects body without confirm token", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const resetBorg = vi.fn(async () => {});
    const { app } = createDemoServerApp({
      borgHandle: { current: borg },
      live,
      resetBorg,
    });

    const missing = await app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(missing.status).toBe(400);

    const wrongToken = await app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "reset" }),
    });
    expect(wrongToken.status).toBe(400);

    expect(resetBorg).not.toHaveBeenCalled();
  });

  it("POST /api/admin/reset invokes resetBorg and clears the dream plan cache", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const resetBorg = vi.fn(async () => {});
    const { app } = createDemoServerApp({
      borgHandle: { current: borg },
      live,
      resetBorg,
    });

    const plan = await app.request("/api/dream/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    expect(plan.status).toBe(200);
    const planBody = (await plan.json()) as { plan_id: string };

    const reset = await app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    });
    expect(reset.status).toBe(200);
    expect(await reset.json()).toEqual({ ok: true });
    expect(resetBorg).toHaveBeenCalledTimes(1);

    const apply = await app.request("/api/dream/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan_id: planBody.plan_id }),
    });
    expect(apply.status).toBe(404);
  });

  it("POST /api/admin/reset returns 501 when resetBorg is not configured", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    });
    expect(response.status).toBe(501);
  });
});
