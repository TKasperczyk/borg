import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { Buffer, File } from "node:buffer";
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
  createCreatorDirectiveId,
  createMaintenanceRunId,
  createSemanticEdgeId,
  createSemanticNodeId,
  createStreamEntryId,
  type AttachmentId,
  type BorgEnqueueMessageResult,
  type BorgOpenOptions,
  type CreatorDirective,
  type CreatorDirectiveActivationScope,
  type CreatorDirectiveContentScope,
  type CreatorDirectiveMentionPolicy,
  type StreamEntry,
} from "borg";

import {
  FakeLLMClient,
  createFakeEmitAnswerResponse,
  createFakeStreamingResponse,
} from "../../../../src/llm/test-support/fake-client.js";
import type { AttachmentService } from "../../../../src/attachments/index.js";
import { IMAGE_PERCEPTION_TOOL_NAME } from "../../../../src/attachments/perception.js";
import type { Episode, EpisodicRepository } from "../../../../src/memory/episodic/index.js";
import type { CreatorDirectiveRepository } from "../../../../src/memory/creator-directives/index.js";
import type { RelationalSlotRepository } from "../../../../src/memory/relational-slots/repository.js";
import type { ReviewQueueRepository } from "../../../../src/memory/review-queue/review-queue.js";
import type { TrainOfThoughtRepository } from "../../../../src/memory/train-of-thought/index.js";
import { TestEmbeddingClient, createTestConfig } from "../../../../src/offline/test-support.js";
import type { AuditLog } from "../../../../src/offline/audit-log.js";
import type { LanceDbTable } from "../../../../src/storage/lancedb/index.js";
import type { StreamWriter } from "../../../../src/stream/index.js";
import {
  broadcastMaintenanceTick,
  createDemoServerApp,
  ensureDemoDefaultSession,
  runtimeConfigFromConfig,
  wireMaintenanceSchedulerLiveObserver,
} from "../app.js";
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
    creatorDirectiveRepository: CreatorDirectiveRepository;
    episodicRepository: EpisodicRepository;
    relationalSlotRepository: RelationalSlotRepository;
    reviewQueueRepository: ReviewQueueRepository;
    trainOfThoughtRepository: TrainOfThoughtRepository;
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

function createBorgCloseStub(input: { start?: () => void } = {}): Borg {
  return {
    close: vi.fn(async () => {}),
    inbox: {
      catchUp: {
        start: input.start ?? vi.fn(),
      },
    },
    autonomy: {
      scheduler: {
        start: vi.fn(),
        stop: vi.fn(async () => {}),
      },
    },
    maintenance: {
      scheduler: {
        start: vi.fn(),
        stop: vi.fn(async () => {}),
      },
    },
  } as unknown as Borg;
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

function createDirectiveReconciliationResponse(
  judgments: Array<{
    member_ids: string[];
    verdict: "same_intent" | "conflicting" | "independent";
    resolution: "supersede_to_survivor" | "revoke_stale" | "keep_independent" | "escalate";
    survivor_id?: string | null;
    loser_ids?: string[];
    confidence: "high" | "medium" | "low";
    rationale: string;
  }>,
) {
  return {
    text: "",
    input_tokens: 20,
    output_tokens: 10,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_directive_reconciliation",
        name: "EmitDirectiveReconciliation",
        input: {
          judgments: judgments.map((judgment) => ({
            survivor_id: null,
            loser_ids: [],
            ...judgment,
          })),
        },
      },
    ],
  };
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

async function enqueueTextTurn(
  app: ReturnType<typeof createDemoServerApp>["app"],
  body: {
    message: string;
    external_message_id: string;
    audience?: string;
    session?: string;
  },
) {
  const response = await requestJson(app, "/api/turn", "POST", body);
  const text = await response.text();

  expect(response.status, text).toBe(200);

  const ack = JSON.parse(text) as {
    ok: boolean;
    status: "enqueued" | "duplicate";
    stream_entry_id: string;
  };

  expect(ack).toMatchObject({
    ok: true,
    status: "enqueued",
    stream_entry_id: expect.any(String),
  });

  return ack;
}

function responseToSingleSource(entry: StreamEntry) {
  return {
    kind: "stream_backlog" as const,
    from_cursor_exclusive: null,
    through_cursor_inclusive: {
      ts: entry.timestamp,
      entryId: entry.id,
    },
    source_entry_ids: [entry.id],
    count: 1,
  };
}

function localDayString(ts: number): string {
  const date = new Date(ts);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(
    date.getDate(),
  ).padStart(2, "0")}`;
}

async function seedCorrectionEpisode(
  borg: Borg,
  clock: ManualClock,
  input: {
    title?: string;
    narrative?: string;
    participants?: string[];
    audienceEntityId?: Episode["audience_entity_id"];
    originAudienceEntityIds?: Episode["origin_audience_entity_ids"];
    shared?: boolean;
  } = {},
): Promise<Episode> {
  const internal = borg as unknown as BorgTestInternals;
  const sourceEntry = await borg.stream.append({
    kind: "user_msg",
    content: input.narrative ?? "operator correction source",
    turn_id: "turn_correction_seed",
  });
  const now = clock.now();

  return internal.deps.episodicRepository.createEpisode({
    id: createEpisodeId(),
    title: input.title ?? "Correction seed episode",
    narrative: input.narrative ?? "A correction endpoint seed episode.",
    participants: input.participants ?? ["operator"],
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
    audience_entity_id: input.audienceEntityId ?? null,
    origin_audience_entity_ids: input.originAudienceEntityIds,
    shared: input.shared ?? false,
    embedding: new Float32Array([0.1, 0.2, 0.3, 0.4]),
    created_at: now,
    updated_at: now,
  });
}

function queueCreatorDirectiveFixture(
  borg: Borg,
  clock: ManualClock,
  input: {
    text?: string;
    contentScope?: CreatorDirectiveContentScope;
    mentionPolicy?: CreatorDirectiveMentionPolicy;
    activationScope?: CreatorDirectiveActivationScope;
    allowedEntityIds?: CreatorDirective["disclosure_policy"]["allowed_entity_ids"];
    excludedEntityIds?: CreatorDirective["disclosure_policy"]["excluded_entity_ids"];
    priority?: number;
  } = {},
): CreatorDirective {
  const creatorId = borg.entities.resolve("Tom");

  return borg.creatorDirectives.queue({
    kind: "response_policy",
    createdByEntityId: creatorId,
    sourceSessionId: DEFAULT_SESSION_ID,
    authorizationStreamEntryIds: [createStreamEntryId()],
    contentSourceStreamEntryIds: [createStreamEntryId()],
    subjectKind: "system",
    operationalDirective: input.text ?? "Prefer precise operator-facing review behavior.",
    disclosurePolicy: {
      content_scope: input.contentScope ?? "public",
      allowed_entity_ids: input.allowedEntityIds ?? [],
      excluded_entity_ids: input.excludedEntityIds ?? [],
      subject_may_know: null,
      mention_policy: input.mentionPolicy ?? "only_if_topic_raised",
      denied_audience_behavior: "omit",
      boundary_prompt: null,
      topic_tags: ["review"],
    },
    activationPolicy: {
      scope: input.activationScope ?? "same_as_disclosure",
      allowed_entity_ids: [],
      excluded_entity_ids: [],
    },
    priority: input.priority ?? 5,
    createdAt: clock.now(),
  });
}

function creatorDirectiveReconciliationRefs(
  directives: readonly CreatorDirective[],
  input: {
    subkind?: "conflict" | "disclosure_widening";
    verdict?: "same_intent" | "conflicting" | "independent";
  } = {},
) {
  const familyKey = {
    kind: "response_policy",
    subject_kind: "system",
    subject_entity_id: null,
  };

  return {
    target_type: "creator_directive_reconciliation",
    subkind: input.subkind ?? "conflict",
    directive_ids: directives.map((directive) => directive.id),
    family_key: familyKey,
    members: directives.map((directive) => ({
      id: directive.id,
      family_key: familyKey,
      scope_equivalence: {
        created_by_entity_id: directive.created_by_entity_id,
        disclosure_policy: directive.disclosure_policy,
        activation_policy: directive.activation_policy,
      },
    })),
    judgment: {
      member_ids: directives.map((directive) => directive.id),
      verdict: input.verdict ?? "same_intent",
      confidence: "high",
      rationale: "Fixture directives share content but differ by scope.",
    },
  };
}

async function seedCreatorDirectiveMergeAudit(
  borg: Borg,
  clock: ManualClock,
  llm: FakeLLMClient,
): Promise<{
  audit: ReturnType<AuditLog["list"]>[number];
  loser: CreatorDirective;
  survivor: CreatorDirective;
}> {
  const loser = queueCreatorDirectiveFixture(borg, clock, {
    text: "Prefer concise sleep-phase operator summaries.",
    priority: 1,
  });
  clock.advance(1);
  const survivor = queueCreatorDirectiveFixture(borg, clock, {
    text: "Prefer brief sleep-phase operator summaries.",
    priority: 9,
  });

  llm.pushResponse(
    createDirectiveReconciliationResponse([
      {
        member_ids: [loser.id, survivor.id],
        verdict: "same_intent",
        resolution: "supersede_to_survivor",
        survivor_id: survivor.id,
        loser_ids: [loser.id],
        confidence: "high",
        rationale: "The records express the same creator directive.",
      },
    ]),
  );

  const result = await borg.dream({ processes: ["creator-directive-reconciler"] });
  expect(result.errors).toEqual([]);
  const audit = borg.audit
    .list({ process: "creator-directive-reconciler" })
    .find(
      (row) => row.action === "creator_directive_merge" && row.targets.survivor_id === survivor.id,
    );

  if (audit === undefined) {
    throw new Error("Expected creator directive merge audit row");
  }

  return { audit, loser, survivor };
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
  const dreamRunId = createMaintenanceRunId();
  const dreamPlannedAt = clock.now();
  await borg.stream.append({
    kind: "dream_report",
    content: {
      run_id: dreamRunId,
      processes: ["belief-reviser"],
      dry_run: false,
      planned_at: dreamPlannedAt,
      changes: 1,
      tokens_used: 1234,
      errors: [
        {
          process: "belief-reviser",
          message: "old stream failure",
          code: "legacy_error",
          target_type: "semantic_node",
          target_id: "semn_demo",
          leaked_detail: "must not leave the server",
        },
      ],
      budget_exhausted_processes: ["belief-reviser"],
      notes: ["Budget exhausted: belief-reviser"],
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
    refs: {
      target_type: "semantic_node",
      target_id: "semn_aaaaaaaaaaaaaaaa",
      invalidated_edge_id: "seme_aaaaaaaaaaaaaaaa",
      dependency_path_edge_ids: [],
      surviving_support_edge_ids: [],
      evidence_episode_ids: [],
    },
    reason: "dependency invalidated",
    sourceProcess: "belief-reviser",
  });

  return {
    attachmentId,
    audit,
    dreamPlannedAt,
    dreamRunId,
    review,
    skill,
  };
}

async function seedSemanticGraph(borg: Borg, clock: ManualClock) {
  const sourceEpisodeId = createEpisodeId();
  const aliceEntityId = borg.entities.resolve("Alice");
  const nodes: Array<Awaited<ReturnType<Borg["semantic"]["nodes"]["add"]>>> = [];

  for (const input of [
    { kind: "entity" as const, label: aliceEntityId, description: "Alice entity" },
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

  return { aliceEntityId };
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

  it("delivers maintenance ticks to clients that unsubscribed from global frames", () => {
    const broadcaster = new LiveBroadcaster({ error: () => {} });
    const frames: LiveFrame[] = [];
    const client = { send: (data: string) => frames.push(JSON.parse(data) as LiveFrame) };

    broadcaster.add(client);
    broadcaster.handleSubscriptionMessage(client, { type: "unsubscribe_global" });
    broadcaster.broadcast({
      type: "maintenance:tick",
      ts: 2,
      cadence: "light",
      status: "ok",
      processes: ["curator"],
      changed: false,
      changes: 0,
      errors: 0,
    });

    expect(frames).toEqual([expect.objectContaining({ type: "maintenance:tick", ts: 2 })]);
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

  it("broadcasts unhandled turn-scoped trace events as phase detail frames", () => {
    const live = createLiveBridge();
    const { frames } = collectLiveFrames(live);

    live.tracer.emit("frame_anomaly.disposition", {
      turnId: "turn_detail",
      turn_id: "turn_detail",
      session_id: DEFAULT_SESSION_ID,
      phase: "frame",
      disposition: "continue",
      status: "ok",
      kind: "topic_shift",
      candidates: ["a", "b"],
      metrics: { considered: 2, accepted: 1 },
      empty: null,
    });

    const detailFrame = frames.find((frame) => frame.type === "turn:phase:detail");

    expect(detailFrame).toMatchObject({
      type: "turn:phase:detail",
      turn_id: "turn_detail",
      session_id: DEFAULT_SESSION_ID,
      phase: "frame",
      event: "frame_anomaly.disposition",
      summary: expect.any(String),
    });
    expect(detailFrame?.summary).toContain("disposition=continue");
    expect(detailFrame?.summary).toContain("status=ok");
    expect(detailFrame?.summary).toContain("kind=topic_shift");
    expect(detailFrame?.summary).toContain("candidates=[2]");
    expect(detailFrame?.summary).toContain("metrics={2}");
    expect(detailFrame?.summary).toContain("empty=null");
  });

  it("broadcasts raw stream entries and notifies observers when append serialization fails", () => {
    const live = createLiveBridge();
    const { frames } = collectLiveFrames(live);
    const observed: Array<readonly StreamEntry[]> = [];
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined);
    const entry: StreamEntry = {
      id: createStreamEntryId(),
      timestamp: 1,
      entry_index: 0,
      kind: "user_msg",
      content: "raw kept",
      turn_id: "turn_live_serializer_failure",
      audience: "alice",
      sender_entity_id: null,
      reply_target_entity_id: null,
      session_id: DEFAULT_SESSION_ID,
      compressed: false,
    };

    try {
      live.setStreamEntrySerializer(() => {
        throw new Error("serializer failed");
      });
      live.observeStreamAppend((entries) => observed.push(entries));

      live.onStreamAppend([entry]);

      expect(consoleError).toHaveBeenCalledWith(
        "Live stream append serialization failed; broadcasting raw entries",
        { cause: "serializer failed" },
      );
      expect(frames).toEqual([
        expect.objectContaining({
          type: "stream:append",
          session_id: DEFAULT_SESSION_ID,
          entries: [expect.objectContaining({ id: entry.id, content: "raw kept" })],
        }),
      ]);
      expect(observed).toEqual([[entry]]);
    } finally {
      consoleError.mockRestore();
    }
  });

  it("attributes unhandled turn-scoped trace events to the current phase", () => {
    const live = createLiveBridge();
    const { frames } = collectLiveFrames(live);

    live.tracer.emit("turn_phase.started", {
      turnId: "turn_detail",
      turn_id: "turn_detail",
      session_id: DEFAULT_SESSION_ID,
      phase: "retrieval",
      sub: "running",
    });
    live.tracer.emit("retrieval.completed", {
      turnId: "turn_detail",
      turn_id: "turn_detail",
      session_id: DEFAULT_SESSION_ID,
      episodeCount: 2,
      semanticHits: 4,
      confidence: 0.82,
    });

    const detailFrame = frames.find(
      (frame) => frame.type === "turn:phase:detail" && frame.event === "retrieval.completed",
    );

    expect(detailFrame).toMatchObject({
      type: "turn:phase:detail",
      turn_id: "turn_detail",
      session_id: DEFAULT_SESSION_ID,
      phase: "retrieval",
      event: "retrieval.completed",
      summary: expect.any(String),
    });
    expect(detailFrame?.summary).toContain("episodeCount=2");
    expect(detailFrame?.summary).toContain("semanticHits=4");
    expect(detailFrame?.summary).toContain("confidence=0.82");
  });

  it("does not broadcast unhandled trace events without a turn id", () => {
    const live = createLiveBridge();
    const { frames } = collectLiveFrames(live);
    const unscopedTraceData = {
      session_id: DEFAULT_SESSION_ID,
      disposition: "continue",
      status: "ok",
    } as unknown as Parameters<typeof live.tracer.emit>[1];

    live.tracer.emit("frame_anomaly.disposition", unscopedTraceData);

    expect(frames).toEqual([]);
  });

  it("keeps a stored default-session audience across boots instead of re-stamping the demo default", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-default-audience-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    createDemoServerApp({ borgHandle: { current: borg }, live });

    const selfEntityId = borg.entities.resolve("self", {
      kind: "self",
      provenance: "assistant_seeded",
    });
    borg.sessions.ensure({
      ...borg.sessions.get(DEFAULT_SESSION_ID)!,
      audience_label: "self",
      audience_entity_id: selfEntityId,
    });

    // A deployment whose default session is the entity's own autonomous space sets its
    // audience once; every later boot must adopt it, since records written there inherit
    // the session audience as origin provenance.
    ensureDemoDefaultSession(borg);

    const session = borg.sessions.get(DEFAULT_SESSION_ID);
    expect(session?.audience_label).toBe("self");
    expect(session?.audience_entity_id).toBe(selfEntityId);
  });

  it("serves the in-flight turn snapshot while a turn is running", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const idleBefore = await app.request("/api/inflight?session=default");
    expect(idleBefore.status).toBe(200);
    expect(await idleBefore.json()).toEqual({ inflight: null });

    live.tracer.emit("turn_phase.started", {
      turnId: "turn_live",
      turn_id: "turn_live",
      session_id: "default",
      phase: "ingest",
    });
    live.tracer.emit("turn_phase.completed", {
      turnId: "turn_live",
      turn_id: "turn_live",
      session_id: "default",
      phase: "ingest",
      duration_ms: 42,
    });
    live.tracer.emit("turn_phase.started", {
      turnId: "turn_live",
      turn_id: "turn_live",
      session_id: "default",
      phase: "delib",
    });

    const running = await app.request("/api/inflight?session=default");
    expect(running.status).toBe(200);
    expect(await running.json()).toEqual({
      inflight: {
        turn_id: "turn_live",
        session_id: "default",
        started_at: expect.any(Number),
        last_event_at: expect.any(Number),
        phases: [
          { phase: "ingest", status: "completed", duration_ms: 42 },
          { phase: "delib", status: "active", duration_ms: null },
        ],
      },
    });

    live.tracer.emit("turn.terminal", {
      turnId: "turn_live",
      turn_id: "turn_live",
      session_id: "default",
      outcome: "reflected",
      duration_ms: 100,
    });

    const idleAfter = await app.request("/api/inflight?session=default");
    expect(idleAfter.status).toBe(200);
    expect(await idleAfter.json()).toEqual({ inflight: null });
  });

  it("accepts only attention envelopes and makes reposts idempotent", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-attention-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const record = {
      record_key: "cclink:api-test",
      filed_at: 1_700_000_000_000,
      filer_entity_id: borg.entities.resolve("Counterpart"),
      subject: "件名",
    };
    const posted = await requestJson(app, "/api/operator-attention", "POST", record);
    expect(posted.status).toBe(200);
    expect(await posted.json()).toEqual({ inserted: true });
    const resent = await requestJson(app, "/api/operator-attention", "POST", {
      ...record,
      subject: null,
    });
    expect(await resent.json()).toEqual({ inserted: false });
    expect(borg.operatorAttention.snapshot()).toEqual({
      total: 1,
      records: [
        {
          ...record,
          disclosure_label: {
            disclosureClass: "operator_private",
            originAudienceEntityIds: [record.filer_entity_id],
            privateToEntityIds: [],
            publicToEntityIds: [],
          },
        },
      ],
    });
    const legacy = await requestJson(app, "/api/operator-attention", "POST", {
      ...record,
      record_key: "cclink:legacy",
      filed_at: record.filed_at - 1,
      subject: null,
    });
    expect(legacy.status).toBe(200);
    for (const bad of [
      { ...record, body: "NEVER STORE THIS BODY" },
      { ...record, reason: "NEVER STORE THIS BODY" },
      { ...record, filed_at: "yesterday" },
      { ...record, filer_entity_id: "not-an-entity" },
      { ...record, subject: "first line\nbody" },
      { ...record, subject: "x".repeat(241) },
    ]) {
      const response = await requestJson(app, "/api/operator-attention", "POST", bad);
      expect(response.status).toBe(400);
    }
    expect(borg.operatorAttention.snapshot().total).toBe(2);
    expect(JSON.stringify(borg.operatorAttention.snapshot())).not.toContain("NEVER STORE");
  });

  it("serves creator and operator session endpoints", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-creator-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
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
    const runtimeConfig = runtimeConfigFromConfig(
      createTestConfig({
        dataDir: tempDir,
        embedding: {
          baseUrl: "http://localhost:1234/v1",
          apiKey: "test",
          model: "state-embed",
          dims: 4,
        },
        anthropic: {
          auth: "api-key",
          apiKey: "test",
          models: {
            cognition: "state-cognition",
          },
        },
      }),
    );
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live, runtimeConfig });
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
        open_reviews: expect.any(Number),
        dream_audit_rows: expect.any(Number),
      }),
      runtime: {
        model: "state-cognition",
        embedding: {
          model: "state-embed",
          dims: 4,
        },
      },
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
    const limitedRelational = await app.request("/api/memory/bands/relational?limit=1");
    expect(await limitedRelational.json()).toMatchObject({
      items: [expect.objectContaining({ state: "established" })],
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
      dream_reports: [
        expect.objectContaining({
          run_id: seeded.dreamRunId,
          processes: ["belief-reviser"],
          dry_run: false,
          planned_at: seeded.dreamPlannedAt,
          changes: 1,
          tokens_used: 1234,
          errors: [
            {
              process: "belief-reviser",
              message: "old stream failure",
              code: "legacy_error",
              target_type: "semantic_node",
              target_id: "semn_demo",
            },
          ],
          budget_exhausted_processes: ["belief-reviser"],
          notes: ["Budget exhausted: belief-reviser"],
        }),
      ],
      audit_rows: [expect.objectContaining({ id: seeded.audit.id })],
      belief_revision_rows: [
        expect.objectContaining({ id: seeded.review.id, kind: "belief_revision" }),
      ],
      pending_extraction_episodes: expect.any(Number),
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
      body: JSON.stringify({
        message: "hello",
        external_message_id: "demo-text-1",
        audience: "Alice",
      }),
    });
    expect(turn.status).toBe(200);
    const turnBody = (await turn.json()) as {
      ok: boolean;
      status: string;
      stream_entry_id: string;
    };
    expect(turnBody).toMatchObject({
      ok: true,
      status: "enqueued",
      stream_entry_id: expect.any(String),
    });

    const duplicateTurn = await app.request("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "hello again",
        external_message_id: "demo-text-1",
        audience: "Alice",
      }),
    });
    expect(duplicateTurn.status).toBe(200);
    expect(await duplicateTurn.json()).toEqual({
      ok: true,
      status: "duplicate",
      stream_entry_id: turnBody.stream_entry_id,
    });
    expect(borg.sessions.get(DEFAULT_SESSION_ID)).toMatchObject({
      source_external_id: DEFAULT_SESSION_ID,
      audience_entity_id: expect.any(String),
      last_turn_id: null,
      message_count: 1,
    });
    expect(
      borg.stream
        .tail(10)
        .filter((entry) => entry.kind === "user_msg" && entry.content === "hello"),
    ).toHaveLength(1);
    expect(
      borg.stream.tail(10).find((entry) => entry.id === turnBody.stream_entry_id),
    ).toMatchObject({
      kind: "user_msg",
      source_message_key: {
        source_type: "demo",
        source_external_id: DEFAULT_SESSION_ID,
        external_message_id: "demo-text-1",
      },
      audience: "Alice",
    });

    const customTurn = await app.request("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "hello custom",
        external_message_id: "demo-custom-1",
        audience: "Alice",
        session: customSessionId,
      }),
    });
    const customTurnText = await customTurn.text();
    expect(customTurn.status, customTurnText).toBe(200);
    const customTurnBody = JSON.parse(customTurnText) as { stream_entry_id: string };
    expect(customTurnBody).toMatchObject({ stream_entry_id: expect.any(String) });
    expect(borg.sessions.get(customSessionId)).toMatchObject({
      session_id: customSessionId,
      source_external_id: customSessionId,
      last_turn_id: null,
      message_count: 1,
    });
  });

  it("keeps a connector-owned session's source type when a demo turn posts into it", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-peer-session-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const peerSessionId = createSessionId();

    // A connector owns this session and its source_type is a trust key: peer-channel
    // frame tolerance and the internal-identifier guard exemption both read it.
    borg.sessions.ensure({
      session_id: peerSessionId,
      source_type: "claude_code",
      source_external_id: "main",
      source_url: null,
      label: "peer channel",
      audience_label: "claude_code_dm:main",
      audience_entity_id: null,
      conversation_kind: "dm",
    });

    const turn = await app.request("/api/turn", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: "hello peer",
        external_message_id: "demo-peer-1",
        audience: "claude_code_dm:main",
        session: peerSessionId,
      }),
    });
    expect(turn.status, await turn.clone().text()).toBe(200);
    expect(borg.sessions.get(peerSessionId)).toMatchObject({
      session_id: peerSessionId,
      source_type: "claude_code",
      source_external_id: "main",
    });
  });

  it("enriches stream entries consistently for REST and live append frames", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-stream-labels-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const collected = collectLiveFrames(live);

    const ack = await enqueueTextTurn(app, {
      message: "hello labels",
      external_message_id: "demo-labels-1",
      audience: "Alice",
    });

    const streamResponse = await app.request("/api/stream?limit=5");
    expect(streamResponse.status).toBe(200);
    const stream = (await streamResponse.json()) as {
      entries: Array<{
        id: string;
        sender_label: string | null;
        session_label: string | null;
        audience_label: string | null;
      }>;
    };
    expect(stream.entries.find((entry) => entry.id === ack.stream_entry_id)).toMatchObject({
      sender_label: "Alice",
      session_label: "demo (default)",
      audience_label: "Alice",
    });

    await waitFor(() =>
      collected.frames.some(
        (frame) =>
          frame.type === "stream:append" &&
          Array.isArray(frame.entries) &&
          frame.entries.some(
            (entry) =>
              typeof entry === "object" &&
              entry !== null &&
              "id" in entry &&
              entry.id === ack.stream_entry_id &&
              "sender_label" in entry &&
              entry.sender_label === "Alice",
          ),
      ),
    );
  });

  it("adds display_content for BotArena connector envelopes without changing raw content", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-stream-display-content-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const envelopeContent =
      '[BotArena thread "thread-1"] message from Lunaria (bot) | bot-chain-depth: 2 | addressed to you: yes\n[message]\nInner body only';
    const markerlessContent =
      "[BotArena thread thread-1] | bot-chain-depth: 2 | addressed to you: yes\n\nMarkerless body";

    const envelopeEntry = await borg.stream.append({
      kind: "user_msg",
      content: envelopeContent,
      source_message_key: {
        source_type: "botarena",
        source_external_id: "thread-1",
        external_message_id: "message-1",
      },
    });
    const markerlessEntry = await borg.stream.append({
      kind: "user_msg",
      content: markerlessContent,
      source_message_key: {
        source_type: "botarena",
        source_external_id: "thread-1",
        external_message_id: "message-1b",
      },
    });
    const plainEntry = await borg.stream.append({
      kind: "user_msg",
      content: "plain body",
      source_message_key: {
        source_type: "demo",
        source_external_id: DEFAULT_SESSION_ID,
        external_message_id: "message-2",
      },
    });

    const streamResponse = await app.request("/api/stream?kind=user_msg&limit=5");
    expect(streamResponse.status).toBe(200);
    const stream = (await streamResponse.json()) as {
      entries: Array<{ id: string; content: unknown; display_content?: unknown }>;
    };

    expect(stream.entries.find((entry) => entry.id === envelopeEntry.id)).toMatchObject({
      content: envelopeContent,
      display_content: "Inner body only",
    });
    expect(stream.entries.find((entry) => entry.id === markerlessEntry.id)).toMatchObject({
      content: markerlessContent,
      display_content: "Markerless body",
    });
    expect(stream.entries.find((entry) => entry.id === plainEntry.id)).toEqual(
      expect.not.objectContaining({ display_content: expect.anything() }),
    );
  });

  it("pages episodic memory band details with next_cursor", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-memory-episodic-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());

    const oldest = await seedCorrectionEpisode(borg, clock, { title: "Oldest page episode" });
    clock.advance(10);
    const middle = await seedCorrectionEpisode(borg, clock, { title: "Middle page episode" });
    clock.advance(10);
    const newest = await seedCorrectionEpisode(borg, clock, { title: "Newest page episode" });
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const firstResponse = await app.request("/api/memory/bands/episodic?limit=2");
    expect(firstResponse.status).toBe(200);
    const first = (await firstResponse.json()) as {
      items: Array<{ id: string }>;
      next_cursor: string | null;
    };
    expect(first.items.map((item) => item.id)).toEqual([newest.id, middle.id]);
    expect(first.next_cursor).not.toBeNull();

    const secondResponse = await app.request(
      `/api/memory/bands/episodic?limit=2&cursor=${first.next_cursor}`,
    );
    expect(secondResponse.status).toBe(200);
    const second = (await secondResponse.json()) as {
      items: Array<{ id: string }>;
      next_cursor: string | null;
    };
    expect(second.items.map((item) => item.id)).toEqual([oldest.id]);
    expect(second.next_cursor).toBeNull();
  });

  it("adds resolved participant label refs to episodic memory payloads", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-memory-participants-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const participantId = borg.entities.resolve("Dana");
    const episode = await seedCorrectionEpisode(borg, clock, {
      title: "Participant label episode",
      participants: [participantId],
    });
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/memory/bands/episodic?limit=5");

    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      items: Array<{
        id: string;
        participants: string[];
        participant_refs?: Array<{ value: string; id: string | null; label: string | null }>;
      }>;
    };
    expect(body.items.find((item) => item.id === episode.id)).toMatchObject({
      participants: [participantId],
      participant_refs: [{ value: participantId, id: participantId, label: "Dana" }],
    });
  });

  it("serializes episodic disclosure labels and origin audience refs additively", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-memory-disclosure-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const audienceId = borg.entities.resolve("Dana");
    const unknownEpisode = await seedCorrectionEpisode(borg, clock, {
      title: "Unknown disclosure episode",
    });
    const privateEpisode = await seedCorrectionEpisode(borg, clock, {
      title: "Private disclosure episode",
      audienceEntityId: audienceId,
      originAudienceEntityIds: [audienceId],
      shared: false,
    });
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/memory/bands/episodic?limit=5");

    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      items: Array<{
        id: string;
        origin_audience_entity_ids?: string[];
        origin_audience_refs?: Array<{ value: string; id: string | null; label: string | null }>;
        shared?: boolean;
        disclosure_class?: string;
        disclosure_label?: {
          disclosure_class?: string;
          origin_audience_entity_ids?: string[];
          private_to_entity_ids?: string[];
          public_to_entity_ids?: string[];
        };
      }>;
    };
    expect(body.items.find((item) => item.id === unknownEpisode.id)).toMatchObject({
      origin_audience_entity_ids: [],
      origin_audience_refs: [],
      shared: false,
      disclosure_class: "unknown",
      disclosure_label: {
        disclosure_class: "unknown",
        origin_audience_entity_ids: [],
        private_to_entity_ids: [],
        public_to_entity_ids: [],
      },
    });
    expect(body.items.find((item) => item.id === privateEpisode.id)).toMatchObject({
      origin_audience_entity_ids: [audienceId],
      origin_audience_refs: [{ value: audienceId, id: audienceId, label: "Dana" }],
      shared: false,
      disclosure_class: "relationship_private",
      disclosure_label: {
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: [audienceId],
        private_to_entity_ids: [audienceId],
        public_to_entity_ids: [],
      },
    });
  });

  it("serves a single episode by id with episodic band serialization and returns 404 for unknown ids", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-episode-detail-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const audienceId = borg.entities.resolve("Dana");
    const episode = await seedCorrectionEpisode(borg, clock, {
      title: "Evidence episode",
      narrative: "Operator described the supporting context.",
      participants: [audienceId],
      audienceEntityId: audienceId,
      originAudienceEntityIds: [audienceId],
      shared: false,
    });
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request(`/api/episodes/${episode.id}`);

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toMatchObject({
      episode: {
        id: episode.id,
        title: "Evidence episode",
        narrative: "Operator described the supporting context.",
        participant_refs: [{ value: audienceId, id: audienceId, label: "Dana" }],
        origin_audience_refs: [{ value: audienceId, id: audienceId, label: "Dana" }],
        disclosure_class: "relationship_private",
        start_time: episode.start_time,
        end_time: episode.end_time,
      },
    });

    const missing = await app.request(`/api/episodes/${createEpisodeId()}`);

    expect(missing.status).toBe(404);
  });

  it("pages semantic memory nodes with next_cursor", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-memory-semantic-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const sourceEpisodeId = createEpisodeId();

    const oldest = await borg.semantic.nodes.add({
      kind: "concept",
      label: "Oldest node",
      description: "Oldest node description",
      sourceEpisodeIds: [sourceEpisodeId],
    });
    clock.advance(10);
    const middle = await borg.semantic.nodes.add({
      kind: "entity",
      label: "Middle node",
      description: "Middle node description",
      sourceEpisodeIds: [sourceEpisodeId],
    });
    clock.advance(10);
    const newest = await borg.semantic.nodes.add({
      kind: "proposition",
      label: "Newest node",
      description: "Newest node description",
      sourceEpisodeIds: [sourceEpisodeId],
    });
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const firstResponse = await app.request("/api/memory/bands/semantic?limit=2");
    expect(firstResponse.status).toBe(200);
    const first = (await firstResponse.json()) as {
      nodes: Array<{ id: string }>;
      next_cursor: string | null;
    };
    expect(first.nodes.map((node) => node.id)).toEqual([newest.id, middle.id]);
    expect(first.next_cursor).not.toBeNull();

    const secondResponse = await app.request(
      `/api/memory/bands/semantic?limit=2&cursor=${first.next_cursor}`,
    );
    expect(secondResponse.status).toBe(200);
    const second = (await secondResponse.json()) as {
      nodes: Array<{ id: string }>;
      next_cursor: string | null;
    };
    expect(second.nodes.map((node) => node.id)).toEqual([oldest.id]);
    expect(second.next_cursor).toBeNull();
  });

  it("returns bad request for malformed memory band cursors", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-memory-cursor-"));
    tempDirs.push(tempDir);
    const { borg, live, clock } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const staleShapeCursor = Buffer.from(
      JSON.stringify({ ts: 123, entryId: "strm_stale_cursor_shape" }),
      "utf8",
    ).toString("base64url");

    for (const band of ["episodic", "semantic"] as const) {
      for (const cursor of ["not-a-memory-cursor", staleShapeCursor]) {
        const response = await app.request(
          `/api/memory/bands/${band}?cursor=${encodeURIComponent(cursor)}`,
        );
        const body = (await response.json()) as { error: { status: number; message: string } };

        expect(response.status).toBe(400);
        expect(response.status).not.toBe(500);
        expect(body.error.status).toBe(400);
      }
    }
  });

  it("rejects unknown memory band detail query params", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-memory-query-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/memory/bands/episodic?limit=2&filters=archived");

    expect(response.status).toBe(400);
  });

  it("delegates memory band text search to facade search methods", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-memory-search-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const episode = await seedCorrectionEpisode(borg, clock, { title: "Searchable episode" });
    const semanticNode = await borg.semantic.nodes.add({
      kind: "concept",
      label: "Searchable node",
      description: "Searchable node description",
      sourceEpisodeIds: [episode.id],
    });
    const skill = await borg.skills.add({
      applies_when: "searchable procedural context",
      approach: "return the delegated skill",
      sourceEpisodes: [episode.id],
    });
    const episodicSearch = vi.spyOn(borg.episodic, "search").mockResolvedValue([
      {
        episode,
        score: 0.91,
      } as Awaited<ReturnType<Borg["episodic"]["search"]>>[number],
    ]);
    const semanticSearch = vi
      .spyOn(borg.semantic.nodes, "search")
      .mockResolvedValue([{ node: semanticNode, similarity: 0.82 }]);
    const proceduralSearch = vi
      .spyOn(borg.skills, "searchByContext")
      .mockResolvedValue([{ skill, similarity: 0.73 }]);
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const episodicResponse = await app.request("/api/memory/bands/episodic?query=meaning&limit=3");
    expect(episodicSearch).toHaveBeenCalledWith("meaning", {
      limit: 3,
      recordRetrieval: false,
    });
    expect(await episodicResponse.json()).toMatchObject({
      mode: "search",
      query: "meaning",
      next_cursor: null,
      items: [expect.objectContaining({ id: episode.id, search_score: 0.91 })],
    });

    const semanticResponse = await app.request("/api/memory/bands/semantic?query=meaning&limit=4");
    expect(semanticSearch).toHaveBeenCalledWith("meaning", { limit: 4 });
    expect(await semanticResponse.json()).toMatchObject({
      mode: "search",
      query: "meaning",
      next_cursor: null,
      nodes: [expect.objectContaining({ id: semanticNode.id, search_score: 0.82 })],
      edges: [],
    });

    const proceduralResponse = await app.request(
      "/api/memory/bands/procedural?query=meaning&limit=5",
    );
    expect(proceduralSearch).toHaveBeenCalledWith("meaning", 5);
    expect(await proceduralResponse.json()).toMatchObject({
      mode: "search",
      query: "meaning",
      next_cursor: null,
      items: [expect.objectContaining({ id: skill.id, search_score: 0.73 })],
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
    formData.set("external_message_id", "demo-image-1");
    formData.set("audience", "Alice");
    formData.append("attachments[]", new File([PNG_1X1], "pixel.png", { type: "image/png" }));

    const turn = await app.request("/api/turn", {
      method: "POST",
      body: formData,
    });

    expect(turn.status).toBe(200);
    expect(await turn.json()).toMatchObject({
      ok: true,
      status: "enqueued",
      stream_entry_id: expect.any(String),
    });

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
    const { frames } = collectLiveFrames(live);
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
    expect(frames).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "maintenance:tick",
          cadence: "manual",
          status: "ok",
          processes: ["curator"],
          errors: 0,
          pending_extraction_episodes: expect.any(Number),
        }),
      ]),
    );

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
    const goalBody = (await goal.json()) as {
      id: string;
      disclosure?: string;
      disclosure_label?: { disclosure_class?: string };
    };
    expect(goalBody).toMatchObject({
      disclosure: expect.stringContaining("disclosure_class=self_private"),
      disclosure_label: { disclosure_class: "self_private" },
    });
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

  it("POST /api/dream/audit/:id/revert reverts reversible reconciler audit rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient();
    const { borg, clock, live } = await openHarness({ tempDir, llmClient: llm });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const { audit, loser, survivor } = await seedCreatorDirectiveMergeAudit(borg, clock, llm);

    expect(borg.creatorDirectives.get(loser.id)).toMatchObject({
      status: "superseded",
      superseded_by: survivor.id,
    });

    const response = await app.request(`/api/dream/audit/${audit.id}/revert`, { method: "POST" });

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      id: audit.id,
      action: "creator_directive_merge",
      reverted_at: expect.any(Number),
      reverted_by: "demo_operator",
    });
    expect(borg.creatorDirectives.get(loser.id)).toMatchObject({
      status: "active",
      superseded_by: null,
    });
    expect(borg.audit.list().find((row) => row.id === audit.id)).toMatchObject({
      reverted_at: expect.any(Number),
      reverted_by: "demo_operator",
    });
  });

  it("POST /api/dream/audit/:id/revert reports operator conflicts", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient();
    const { borg, clock, live } = await openHarness({ tempDir, llmClient: llm });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const alreadyReverted = await seedCreatorDirectiveMergeAudit(borg, clock, llm);
    const firstResponse = await app.request(`/api/dream/audit/${alreadyReverted.audit.id}/revert`, {
      method: "POST",
    });
    expect(firstResponse.status).toBe(200);

    const secondResponse = await app.request(
      `/api/dream/audit/${alreadyReverted.audit.id}/revert`,
      { method: "POST" },
    );
    expect(secondResponse.status).toBe(409);
    expect(await secondResponse.json()).toMatchObject({
      error: { message: "audit row is already reverted" },
    });

    const drifted = await seedCreatorDirectiveMergeAudit(borg, clock, llm);
    const internal = borg as unknown as BorgTestInternals;
    const loserVersion = borg.creatorDirectives.get(drifted.loser.id)?.record_version;
    const replacement = queueCreatorDirectiveFixture(borg, clock, {
      text: "Replacement directive for drifted audit setup.",
      priority: 10,
    });

    expect(loserVersion).toEqual(expect.any(Number));
    internal.deps.creatorDirectiveRepository.reverseSupersede(
      drifted.loser.id,
      drifted.survivor.id,
      loserVersion!,
    );
    internal.deps.creatorDirectiveRepository.supersede(drifted.loser.id, replacement.id);

    const staleResponse = await app.request(`/api/dream/audit/${drifted.audit.id}/revert`, {
      method: "POST",
    });
    expect(staleResponse.status).toBe(409);
    expect(await staleResponse.json()).toMatchObject({
      error: { message: expect.stringContaining("Creator directive merge reversal is stale") },
    });

    const missingResponse = await app.request("/api/dream/audit/999999/revert", {
      method: "POST",
    });
    expect(missingResponse.status).toBe(404);
    expect(await missingResponse.json()).toMatchObject({
      error: { message: "audit row not found" },
    });
  });

  it("broadcasts maintenance tick frames from the scheduler observer", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { frames } = collectLiveFrames(live);
    const setObserverSpy = vi.spyOn(borg.maintenance.scheduler, "setObserver");
    const change = {
      process: "curator" as const,
      action: "test maintenance change",
      targets: { id: "target_1" },
    };
    const result = {
      run_id: createMaintenanceRunId(),
      dryRun: false,
      results: [
        {
          process: "curator" as const,
          dryRun: false,
          changes: [change],
          tokens_used: 3,
          errors: [],
          budget_exhausted: false,
        },
      ],
      changes: [change],
      tokens_used: 3,
      errors: [],
    };

    wireMaintenanceSchedulerLiveObserver(borg, live);
    const observer = setObserverSpy.mock.calls[0]?.[0];
    expect(observer).toBeDefined();
    await observer?.onTick?.({
      status: "ok",
      cadence: "light",
      ts: 1_800_000_000_123,
      processes: ["curator"],
      result,
    });

    expect(frames).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: "maintenance:tick",
          ts: 1_800_000_000_123,
          cadence: "light",
          status: "ok",
          processes: ["curator"],
          changed: true,
          changes: 1,
          errors: 0,
          run_id: result.run_id,
          pending_extraction_episodes: expect.any(Number),
        }),
      ]),
    );
  });

  it("counts pending semantic extraction from SQLite projections and coalesces diagnostics", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-pending-extraction-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    let now = 2_000_000_000_000;
    const dateNow = vi.spyOn(Date, "now").mockImplementation(() => now);
    closers.push(async () => dateNow.mockRestore());
    const processedEpisode = await seedCorrectionEpisode(borg, clock, {
      title: "Already extracted episode",
    });
    const pendingEpisode = await seedCorrectionEpisode(borg, clock, {
      title: "Pending extraction episode",
    });
    await borg.semantic.nodes.add({
      kind: "concept",
      label: "Extracted episode marker",
      description: "Semantic provenance marks one episode as processed.",
      sourceEpisodeIds: [processedEpisode.id],
    });
    const internal = borg as unknown as BorgTestInternals;
    await internal.deps.episodicRepository.reconcileCrossStoreState();
    const episodeLanceTable = (
      internal.deps.episodicRepository as unknown as { readonly table: LanceDbTable }
    ).table;
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const countPending = vi.spyOn(borg.maintenance, "countPendingSemanticExtractionEpisodes");
    const episodeLanceList = vi.spyOn(episodeLanceTable, "list");
    const episodicListAll = vi.spyOn(borg.episodic, "listAll");
    const semanticNodeList = vi.spyOn(borg.semantic.nodes, "list");
    const semanticEdgeList = vi.spyOn(borg.semantic.edges, "list");

    const first = await app.request("/api/dream/state");
    expect(first.status).toBe(200);
    expect(await first.json()).toMatchObject({ pending_extraction_episodes: 1 });

    await broadcastMaintenanceTick({
      borg,
      live,
      cadence: "light",
      status: "ok",
      result: null,
    });
    expect(countPending).toHaveBeenCalledTimes(1);
    expect(episodeLanceList).not.toHaveBeenCalled();
    expect(episodicListAll).not.toHaveBeenCalled();
    expect(semanticNodeList).not.toHaveBeenCalled();
    expect(semanticEdgeList).not.toHaveBeenCalled();

    internal.deps.auditLog.record({
      run_id: createMaintenanceRunId(),
      process: "semantic-extractor",
      action: "record extracted episode",
      targets: { episode_ids: [pendingEpisode.id] },
      reversal: {},
    });

    const cached = await app.request("/api/dream/state");
    expect(await cached.json()).toMatchObject({ pending_extraction_episodes: 1 });
    expect(countPending).toHaveBeenCalledTimes(1);

    now += 60_001;
    const refreshed = await app.request("/api/dream/state");
    expect(await refreshed.json()).toMatchObject({ pending_extraction_episodes: 0 });
    expect(countPending).toHaveBeenCalledTimes(2);
    expect(episodeLanceList).not.toHaveBeenCalled();
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

    const malformedWhy = await app.request("/api/correction/ep_short/why");
    expect(malformedWhy.status).toBe(400);
    expect(await malformedWhy.json()).toMatchObject({
      error: {
        status: 400,
        message: "Invalid ep identifier: ep_short",
      },
    });

    const absentWhy = await app.request("/api/correction/ep_aaaaaaaaaaaaaaaa/why");
    expect(absentWhy.status).toBe(404);
    expect(await absentWhy.json()).toMatchObject({
      error: {
        status: 404,
        message: "Unknown episode id: ep_aaaaaaaaaaaaaaaa",
      },
    });

    const unsupportedWhy = await app.request("/api/correction/unknown_abc/why");
    expect(unsupportedWhy.status).toBe(400);
    expect(await unsupportedWhy.json()).toMatchObject({
      error: {
        status: 400,
        message: "Unsupported correction target id: unknown_abc",
      },
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
      kind: "duplicate",
      refs: {
        node_ids: ["semn_aaaaaaaaaaaaaaa1", "semn_aaaaaaaaaaaaaaa2"],
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
      kind: "duplicate",
      resolved_at: null,
      resolution: null,
    });
  });

  it("GET /api/reviews lists open review rows across kinds and filters by kind", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const internal = borg as unknown as BorgTestInternals;
    const correctionReview = internal.deps.reviewQueueRepository.enqueue({
      kind: "correction",
      refs: {
        target_type: "value",
        target_id: "val_aaaaaaaaaaaaaaaa",
        patch: { description: "new" },
      },
      reason: "queued correction fixture",
    });
    const duplicateReview = internal.deps.reviewQueueRepository.enqueue({
      kind: "duplicate",
      refs: {
        node_ids: ["semn_aaaaaaaaaaaaaaa1", "semn_aaaaaaaaaaaaaaa2"],
      },
      reason: "duplicate review fixture",
    });

    const allReviews = await app.request("/api/reviews");
    expect(allReviews.status).toBe(200);
    expect(await allReviews.json()).toMatchObject({
      rows: expect.arrayContaining([
        expect.objectContaining({ id: correctionReview.id, kind: "correction" }),
        expect.objectContaining({
          id: duplicateReview.id,
          kind: "duplicate",
        }),
      ]),
    });

    const filteredReviews = await app.request("/api/reviews?kind=duplicate");
    expect(filteredReviews.status).toBe(200);
    expect(await filteredReviews.json()).toMatchObject({
      rows: [
        expect.objectContaining({
          id: duplicateReview.id,
          kind: "duplicate",
        }),
      ],
    });
  });

  it("PATCH /api/reviews resolves generic review rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
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
      reason: "generic review fixture",
      sourceProcess: "belief-reviser",
    });

    const response = await requestJson(app, `/api/reviews/${review.id}`, "PATCH", {
      action: "dismiss",
      note: "not actionable",
    });

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      id: review.id,
      resolved_at: expect.any(Number),
      resolution: "dismiss",
    });
  });

  it("PATCH /api/reviews rejects creator-directive reconciliation rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const internal = borg as unknown as BorgTestInternals;
    const first = queueCreatorDirectiveFixture(borg, clock);
    const second = queueCreatorDirectiveFixture(borg, clock, {
      text: "Second reconciliation member.",
    });
    const review = internal.deps.reviewQueueRepository.enqueue({
      kind: "creator_directive_reconciliation",
      refs: creatorDirectiveReconciliationRefs([first, second]),
      reason: "must use specialized endpoint",
    });

    const response = await requestJson(app, `/api/reviews/${review.id}`, "PATCH", {
      action: "accept",
      note: "wrong endpoint",
    });

    expect(response.status).toBe(409);
    expect(await response.json()).toMatchObject({
      error: {
        message: expect.stringContaining(
          "creator directive reconciliation reviews must be resolved",
        ),
      },
    });
    expect(internal.deps.reviewQueueRepository.get(review.id)).toMatchObject({
      resolved_at: null,
      resolution: null,
    });
  });

  it("PATCH /api/reviews returns 404 for already-resolved rows", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
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
      reason: "already resolved fixture",
      sourceProcess: "belief-reviser",
    });
    await borg.review.resolve(review.id, {
      decision: "dismiss",
      reason: "pre-resolved",
    });

    const response = await requestJson(app, `/api/reviews/${review.id}`, "PATCH", {
      action: "dismiss",
      note: "second resolution",
    });

    expect(response.status).toBe(404);
    expect(await response.json()).toMatchObject({
      error: {
        message: "review item not found",
      },
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
    const created = (await response.json()) as {
      id: string;
      source: string;
      disclosure?: string;
      disclosure_label?: { disclosure_class?: string; origin_audience_entity_ids?: string[] };
    };
    expect(created).toMatchObject({
      id: expect.stringMatching(/^cmt_/),
      source: "manual",
      disclosure: expect.stringContaining("disclosure_class=relationship_private"),
      disclosure_label: {
        disclosure_class: "relationship_private",
        origin_audience_entity_ids: expect.any(Array),
      },
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
          disclosure: expect.stringContaining("disclosure_class=relationship_private"),
          disclosure_label: expect.objectContaining({
            disclosure_class: "relationship_private",
          }),
        }),
      ],
    });
  });

  it("GET /api/creator-directives lists active creator directives", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const creatorId = borg.entities.resolve("Tom");
    const aliceId = borg.entities.resolve("Alice");
    const active = borg.creatorDirectives.queue({
      kind: "subject_fact",
      createdByEntityId: creatorId,
      sourceSessionId: DEFAULT_SESSION_ID,
      authorizationStreamEntryIds: [createStreamEntryId()],
      contentSourceStreamEntryIds: [createStreamEntryId()],
      subjectKind: "entity",
      subjectEntityId: aliceId,
      canonicalFact: "Alice is the launch reviewer.",
      disclosurePolicy: {
        content_scope: "public",
        allowed_entity_ids: [],
        excluded_entity_ids: [],
        subject_may_know: null,
        mention_policy: "only_if_topic_raised",
        denied_audience_behavior: "omit",
        boundary_prompt: null,
        topic_tags: ["demo"],
      },
      priority: 6,
      createdAt: clock.now(),
    });
    const revoked = borg.creatorDirectives.queue({
      kind: "response_policy",
      createdByEntityId: creatorId,
      sourceSessionId: DEFAULT_SESSION_ID,
      authorizationStreamEntryIds: [createStreamEntryId()],
      contentSourceStreamEntryIds: [createStreamEntryId()],
      subjectKind: "system",
      operationalDirective: "Use the old response policy.",
      disclosurePolicy: {
        content_scope: "operator_only",
        allowed_entity_ids: [],
        excluded_entity_ids: [],
        subject_may_know: null,
        mention_policy: "answer_if_asked",
        denied_audience_behavior: "omit",
        boundary_prompt: null,
        topic_tags: [],
      },
      priority: 4,
      createdAt: clock.now(),
    });
    borg.creatorDirectives.revoke(revoked.id, "demo smoke");

    const response = await app.request("/api/creator-directives");
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      directives: Array<{
        id: string;
        kind: string;
        text: string | null;
        canonical_fact: string | null;
        operational_directive: string | null;
        activation_scope: string;
        content_scope: string;
        mention_policy: string;
        status: string;
        subject_entity_name: string | null;
        superseded_by_id: string | null;
        revoked_reason: string | null;
        updated_at: number;
      }>;
    };
    expect(body.directives.map((directive) => directive.id)).not.toContain(revoked.id);
    expect(body.directives).toEqual([
      expect.objectContaining({
        id: active.id,
        kind: "subject_fact",
        text: "Alice is the launch reviewer.",
        canonical_fact: "Alice is the launch reviewer.",
        operational_directive: null,
        activation_scope: "same_as_disclosure",
        content_scope: "public",
        mention_policy: "only_if_topic_raised",
        status: "active",
        subject_entity_name: "Alice",
        superseded_by_id: null,
        revoked_reason: null,
        updated_at: expect.any(Number),
      }),
    ]);
  });

  it("GET /api/creator-directives status filter can list inactive creator directives", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const active = queueCreatorDirectiveFixture(borg, clock, {
      text: "Active creator directive.",
    });
    const revoked = queueCreatorDirectiveFixture(borg, clock, {
      text: "Revoked creator directive.",
    });
    const survivor = queueCreatorDirectiveFixture(borg, clock, {
      text: "Replacement creator directive.",
    });
    const superseded = queueCreatorDirectiveFixture(borg, clock, {
      text: "Superseded creator directive.",
    });

    borg.creatorDirectives.revoke(revoked.id, "operator retired obsolete directive");
    borg.creatorDirectives.supersede(superseded.id, survivor.id);

    const allResponse = await app.request("/api/creator-directives?status=all");
    expect(allResponse.status).toBe(200);
    const allBody = (await allResponse.json()) as {
      directives: Array<{
        id: string;
        status: string;
        superseded_by_id: string | null;
        revoked_reason: string | null;
        updated_at: number;
      }>;
    };
    expect(allBody.directives.map((directive) => directive.id)).toEqual(
      expect.arrayContaining([active.id, revoked.id, survivor.id, superseded.id]),
    );
    expect(allBody.directives.find((directive) => directive.id === revoked.id)).toMatchObject({
      status: "revoked",
      revoked_reason: "operator retired obsolete directive",
      superseded_by_id: null,
      updated_at: expect.any(Number),
    });
    expect(allBody.directives.find((directive) => directive.id === superseded.id)).toMatchObject({
      status: "superseded",
      revoked_reason: null,
      superseded_by_id: survivor.id,
      updated_at: expect.any(Number),
    });

    const revokedResponse = await app.request("/api/creator-directives?status=revoked");
    expect(revokedResponse.status).toBe(200);
    const revokedBody = (await revokedResponse.json()) as { directives: Array<{ id: string }> };
    expect(revokedBody.directives.map((directive) => directive.id)).toEqual([revoked.id]);

    const supersededResponse = await app.request("/api/creator-directives?status=superseded");
    expect(supersededResponse.status).toBe(200);
    const supersededBody = (await supersededResponse.json()) as {
      directives: Array<{ id: string }>;
    };
    expect(supersededBody.directives.map((directive) => directive.id)).toEqual([superseded.id]);
  });

  it("POST /api/creator-directives revoke and supersede mutate creator directives", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const revoked = queueCreatorDirectiveFixture(borg, clock, {
      text: "Directive to revoke from endpoint.",
    });
    const survivor = queueCreatorDirectiveFixture(borg, clock, {
      text: "Directive to survive endpoint supersede.",
    });
    const loser = queueCreatorDirectiveFixture(borg, clock, {
      text: "Directive to supersede from endpoint.",
    });

    const revokeResponse = await requestJson(
      app,
      `/api/creator-directives/${revoked.id}/revoke`,
      "POST",
      {
        reason: "operator revoked duplicate guidance",
      },
    );
    expect(revokeResponse.status).toBe(200);
    expect(await revokeResponse.json()).toMatchObject({
      id: revoked.id,
      status: "revoked",
    });
    expect(borg.creatorDirectives.get(revoked.id)).toMatchObject({
      status: "revoked",
      revoked_reason: "operator revoked duplicate guidance",
    });

    const supersedeResponse = await requestJson(
      app,
      `/api/creator-directives/${loser.id}/supersede`,
      "POST",
      {
        replacement_id: survivor.id,
      },
    );
    expect(supersedeResponse.status).toBe(200);
    expect(await supersedeResponse.json()).toMatchObject({
      id: loser.id,
      status: "superseded",
    });
    expect(borg.creatorDirectives.get(loser.id)).toMatchObject({
      status: "superseded",
      superseded_by: survivor.id,
    });
    expect(borg.creatorDirectives.get(survivor.id)).toMatchObject({ status: "active" });
  });

  it("POST /api/creator-directives/:id/supersede rejects inactive, nonexistent, and self replacements", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const loser = queueCreatorDirectiveFixture(borg, clock, {
      text: "Directive that should remain active.",
    });
    const inactiveReplacement = queueCreatorDirectiveFixture(borg, clock, {
      text: "Inactive replacement.",
    });
    borg.creatorDirectives.revoke(inactiveReplacement.id, "fixture inactive replacement");

    const inactive = await requestJson(
      app,
      `/api/creator-directives/${loser.id}/supersede`,
      "POST",
      {
        replacement_id: inactiveReplacement.id,
      },
    );
    expect(inactive.status).toBe(400);

    const nonexistent = await requestJson(
      app,
      `/api/creator-directives/${loser.id}/supersede`,
      "POST",
      {
        replacement_id: createCreatorDirectiveId(),
      },
    );
    expect(nonexistent.status).toBe(404);

    const self = await requestJson(app, `/api/creator-directives/${loser.id}/supersede`, "POST", {
      replacement_id: loser.id,
    });
    expect(self.status).toBe(400);
    expect(borg.creatorDirectives.get(loser.id)).toMatchObject({ status: "active" });
  });

  it("POST /api/reviews/:id/creator-directive-reconciliation supersedes losers then resolves", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const internal = borg as unknown as BorgTestInternals;
    const survivor = queueCreatorDirectiveFixture(borg, clock, {
      text: "Prefer the public response policy.",
      contentScope: "public",
    });
    const aliceId = borg.entities.resolve("Alice");
    const operatorOnly = queueCreatorDirectiveFixture(borg, clock, {
      text: "Prefer the public response policy.",
      contentScope: "operator_only",
    });
    const scopedLoser = queueCreatorDirectiveFixture(borg, clock, {
      text: "Prefer the public response policy.",
      contentScope: "all_except",
      excludedEntityIds: [aliceId],
    });
    const review = internal.deps.reviewQueueRepository.enqueue({
      kind: "creator_directive_reconciliation",
      refs: creatorDirectiveReconciliationRefs([survivor, operatorOnly, scopedLoser]),
      reason: "same content with different scopes",
    });

    const response = await requestJson(
      app,
      `/api/reviews/${review.id}/creator-directive-reconciliation`,
      "POST",
      {
        action: "supersede",
        survivor_id: survivor.id,
        reason: "public scope wins",
      },
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      id: review.id,
      resolution: "accept",
      resolved_at: expect.any(Number),
    });
    expect(borg.creatorDirectives.get(survivor.id)).toMatchObject({ status: "active" });
    expect(borg.creatorDirectives.get(operatorOnly.id)).toMatchObject({
      status: "superseded",
      superseded_by: survivor.id,
    });
    expect(borg.creatorDirectives.get(scopedLoser.id)).toMatchObject({
      status: "superseded",
      superseded_by: survivor.id,
    });
  });

  it("POST /api/reviews/:id/creator-directive-reconciliation returns 409 when supersede members changed", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const internal = borg as unknown as BorgTestInternals;
    const survivor = queueCreatorDirectiveFixture(borg, clock, {
      text: "Inactive survivor.",
      contentScope: "public",
    });
    const loser = queueCreatorDirectiveFixture(borg, clock, {
      text: "Loser must remain active after failed atomic supersede.",
      contentScope: "operator_only",
    });
    const review = internal.deps.reviewQueueRepository.enqueue({
      kind: "creator_directive_reconciliation",
      refs: creatorDirectiveReconciliationRefs([survivor, loser]),
      reason: "survivor went inactive",
    });
    borg.creatorDirectives.revoke(survivor.id, "fixture inactive survivor");

    const response = await requestJson(
      app,
      `/api/reviews/${review.id}/creator-directive-reconciliation`,
      "POST",
      {
        action: "supersede",
        survivor_id: survivor.id,
      },
    );

    expect(response.status).toBe(409);
    expect(borg.creatorDirectives.get(loser.id)).toMatchObject({
      status: "active",
      superseded_by: null,
    });
    expect(internal.deps.reviewQueueRepository.get(review.id)).toMatchObject({
      resolved_at: null,
      resolution: null,
    });
  });

  it("POST /api/reviews/:id/creator-directive-reconciliation keep path mutates nothing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const internal = borg as unknown as BorgTestInternals;
    const first = queueCreatorDirectiveFixture(borg, clock, {
      text: "Keep both scope-specific directives.",
      contentScope: "public",
    });
    const second = queueCreatorDirectiveFixture(borg, clock, {
      text: "Keep both scope-specific directives.",
      contentScope: "operator_only",
    });
    const review = internal.deps.reviewQueueRepository.enqueue({
      kind: "creator_directive_reconciliation",
      refs: creatorDirectiveReconciliationRefs([first, second]),
      reason: "same content with valid separate scopes",
    });
    const firstVersion = first.record_version;
    const secondVersion = second.record_version;

    const response = await requestJson(
      app,
      `/api/reviews/${review.id}/creator-directive-reconciliation`,
      "POST",
      {
        action: "keep",
        reason: "both scopes are intentional",
      },
    );

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      id: review.id,
      resolution: "keep",
      resolved_at: expect.any(Number),
    });
    expect(borg.creatorDirectives.get(first.id)).toMatchObject({
      status: "active",
      record_version: firstVersion,
    });
    expect(borg.creatorDirectives.get(second.id)).toMatchObject({
      status: "active",
      record_version: secondVersion,
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

  it("enumerates turn history from persisted stream entries with cursor pagination", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const emittedSource = await borg.stream.append({
      kind: "user_msg",
      content: "emitted source",
      turn_id: "turn_emitted",
      audience: "Alice",
    });
    clock.advance(10);
    await borg.stream.append({
      kind: "agent_msg",
      content: "emitted response",
      turn_id: "turn_emitted",
      audience: "Alice",
      response_to: responseToSingleSource(emittedSource),
    });

    clock.advance(10);
    const suppressedSource = await borg.stream.append({
      kind: "user_msg",
      content: "suppressed source",
      audience: "Bob",
    });
    clock.advance(10);
    await borg.stream.append({
      kind: "agent_suppressed",
      content: { reason: "finalizer_no_output" },
      turn_id: "turn_suppressed",
      audience: "Bob",
      response_to: responseToSingleSource(suppressedSource),
    });

    clock.advance(10);
    const failedSource = await borg.stream.append({
      kind: "user_msg",
      content: "failed source",
      turn_id: "turn_failed",
      audience: "Carol",
    });
    clock.advance(10);
    await borg.stream.append({
      kind: "internal_event",
      turn_id: "turn_failed",
      turn_status: "aborted",
      content: {
        event: "aborted_turn",
        turn_id: "turn_failed",
        reason: "test failure",
      },
    });

    const firstPage = await app.request("/api/turns?limit=2");
    expect(firstPage.status).toBe(200);
    const firstBody = (await firstPage.json()) as {
      rows: Array<{
        turn_id: string;
        started_at: number;
        audience: string | null;
        outcome: string;
        suppression_reason: string | null;
      }>;
      next_cursor: string | null;
    };

    expect(firstBody.rows).toEqual([
      {
        turn_id: "turn_failed",
        started_at: failedSource.timestamp,
        audience: "Carol",
        outcome: "failed",
        suppression_reason: null,
      },
      {
        turn_id: "turn_suppressed",
        started_at: suppressedSource.timestamp,
        audience: "Bob",
        outcome: "deliberate-silence",
        suppression_reason: "finalizer_no_output",
      },
    ]);
    expect(firstBody.next_cursor).not.toBeNull();

    const secondPage = await app.request(`/api/turns?limit=2&cursor=${firstBody.next_cursor}`);
    expect(secondPage.status).toBe(200);
    const secondBody = (await secondPage.json()) as typeof firstBody;
    expect(secondBody).toMatchObject({
      rows: [
        {
          turn_id: "turn_emitted",
          started_at: emittedSource.timestamp,
          audience: "Alice",
          outcome: "emitted",
          suppression_reason: null,
        },
      ],
      next_cursor: null,
    });
  });

  it("serves a day-filtered cross-session activity feed with structural origins and digest counts", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const internal = borg as unknown as BorgTestInternals;
    const customSessionId = createSessionId();

    borg.sessions.ensure({
      session_id: customSessionId,
      source_type: "demo",
      label: "autonomy lane",
      audience_label: "self",
      conversation_kind: "demo",
    });

    const userSource = await borg.stream.append({
      kind: "user_msg",
      content: "user source",
      turn_id: "turn_activity_user",
      audience: "Alice",
    });
    clock.advance(10);
    await borg.stream.append({
      kind: "agent_msg",
      content: "persisted emitted response",
      turn_id: "turn_activity_user",
      audience: "Alice",
      response_to: responseToSingleSource(userSource),
    });

    clock.advance(10);
    await borg.stream.append(
      {
        kind: "internal_event",
        content: {
          kind: "autonomous_wake",
          trigger_type: "trigger",
          source_name: "scheduled_reflection",
          source_category: "contemplative",
          payload: { interval_ms: 14_400_000 },
          ts: clock.now(),
        },
      },
      { session: customSessionId },
    );
    clock.advance(10);
    await borg.stream.append(
      {
        kind: "agent_suppressed",
        content: { reason: "finalizer_no_output" },
        turn_id: "turn_activity_auto",
        audience: "self",
      },
      { session: customSessionId },
    );
    await borg.stream.append(
      {
        kind: "internal_event",
        content: {
          kind: "autonomous_action",
          trigger: "scheduled_reflection",
          outcome_summary: "No output.",
          turn_result_id: null,
          ts: clock.now(),
        },
      },
      { session: customSessionId },
    );

    clock.advance(10);
    await borg.stream.append({
      kind: "dream_report",
      content: {
        run_id: createMaintenanceRunId(),
        processes: ["belief-reviser", "self-narrator"],
        dry_run: false,
        planned_at: clock.now(),
        changes: 3,
        tokens_used: 120,
        errors: [],
        notes: ["summarized the overnight maintenance run"],
      },
    });
    internal.deps.trainOfThoughtRepository.append({
      text: "private note from autonomous reflection",
      selfEntityId: borg.entities.resolve("self", {
        kind: "self",
        provenance: "assistant_seeded",
      }),
      sourceTurnId: "turn_activity_auto",
      now: clock.now(),
    });

    const day = localDayString(userSource.timestamp);
    const response = await app.request(`/api/activity?day=${day}`);

    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      days: string[];
      truncated: boolean;
      rows: Array<{
        id: string;
        kind: string;
        turn_id: string | null;
        session_label: string | null;
        origin: string;
        trigger: string | null;
        outcome: string;
        excerpt: string | null;
        dream?: { process_count: number; changes: number; errors: number };
      }>;
      digest: Record<string, number>;
    };

    expect(body.days).toContain(day);
    expect(body.truncated).toBe(false);
    expect(body.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "turn:default:turn_activity_user",
          kind: "turn",
          turn_id: "turn_activity_user",
          origin: "user",
          outcome: "emitted",
          excerpt: "persisted emitted response",
        }),
        expect.objectContaining({
          id: `turn:${customSessionId}:turn_activity_auto`,
          kind: "turn",
          turn_id: "turn_activity_auto",
          session_label: "autonomy lane",
          origin: "autonomous",
          trigger: "scheduled_reflection",
          outcome: "deliberate-silence",
          excerpt: "finalizer_no_output",
        }),
        expect.objectContaining({
          kind: "dream",
          origin: "dream",
          dream: expect.objectContaining({ process_count: 2, changes: 3, errors: 0 }),
        }),
      ]),
    );
    expect(body.digest).toMatchObject({
      turns: 2,
      autonomous_wakes: 1,
      emissions: 1,
      silences: 1,
      observations: 0,
      suppressions: 0,
      dream_changes: 3,
      journal_notes: 1,
    });

    const emptyDay = localDayString(userSource.timestamp - 24 * 60 * 60 * 1_000);
    const emptyResponse = await app.request(`/api/activity?day=${emptyDay}`);
    expect(emptyResponse.status).toBe(200);
    const emptyBody = (await emptyResponse.json()) as {
      rows: unknown[];
      digest: Record<string, number>;
    };
    expect(emptyBody.rows).toEqual([]);
    expect(emptyBody.digest.turns).toBe(0);
    expect(emptyBody.digest.journal_notes).toBe(0);
  });

  it("carries wake origin and abort reason for failed turns in the activity feed", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const customSessionId = createSessionId();

    borg.sessions.ensure({
      session_id: customSessionId,
      source_type: "demo",
      label: "autonomy lane",
      audience_label: "self",
      conversation_kind: "demo",
    });

    const userSource = await borg.stream.append({
      kind: "user_msg",
      content: "user message that hit a stalled stream",
      turn_id: "turn_failed_user",
      audience: "Alice",
    });
    clock.advance(10);
    await borg.stream.append({
      kind: "internal_event",
      turn_id: "turn_failed_user",
      turn_status: "aborted",
      content: {
        event: "aborted_turn",
        turn_id: "turn_failed_user",
        reason: "LLMError: stream stalled before completion",
      },
    });

    clock.advance(10);
    await borg.stream.append(
      {
        kind: "internal_event",
        content: {
          kind: "autonomous_wake",
          trigger_type: "trigger",
          source_name: "executive_focus_due",
          source_category: "operational",
          payload: {},
          ts: clock.now(),
        },
      },
      { session: customSessionId },
    );
    clock.advance(10);
    await borg.stream.append(
      {
        kind: "internal_event",
        turn_id: "turn_failed_auto",
        turn_status: "aborted",
        content: {
          event: "aborted_turn",
          turn_id: "turn_failed_auto",
          reason: "LLMError: Anthropic SSE stream stalled for 180000ms between message events",
        },
      },
      { session: customSessionId },
    );
    clock.advance(10);
    await borg.stream.append(
      {
        kind: "internal_event",
        content: {
          kind: "autonomous_action",
          trigger: "executive_focus_due",
          outcome_summary: "Autonomous turn failed: stream stalled",
          turn_result_id: null,
          ts: clock.now(),
        },
      },
      { session: customSessionId },
    );

    const day = localDayString(userSource.timestamp);
    const response = await app.request(`/api/activity?day=${day}`);

    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      rows: Array<{
        id: string;
        origin: string;
        trigger: string | null;
        outcome: string;
        excerpt: string | null;
      }>;
    };

    expect(body.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "turn:default:turn_failed_user",
          origin: "user",
          trigger: null,
          outcome: "failed",
          excerpt: "LLMError: stream stalled before completion",
        }),
        expect.objectContaining({
          id: `turn:${customSessionId}:turn_failed_auto`,
          origin: "autonomous",
          trigger: "executive_focus_due",
          outcome: "failed",
          excerpt: "LLMError: Anthropic SSE stream stalled for 180000ms between message events",
        }),
      ]),
    );
  });

  it("keeps activity rows isolated when sessions reuse the same turn id", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const customSessionId = createSessionId();

    borg.sessions.ensure({
      session_id: customSessionId,
      source_type: "demo",
      label: "collision lane",
      audience_label: "self",
      conversation_kind: "demo",
    });

    const defaultSource = await borg.stream.append({
      kind: "user_msg",
      content: "default source",
      turn_id: "turn_collision",
    });
    const defaultResult = await borg.stream.append({
      kind: "agent_msg",
      content: "default response",
      turn_id: "turn_collision",
      response_to: responseToSingleSource(defaultSource),
    });

    clock.advance(10);
    const customSource = await borg.stream.append(
      {
        kind: "user_msg",
        content: "custom source",
        turn_id: "turn_collision",
      },
      { session: customSessionId },
    );
    const customResult = await borg.stream.append(
      {
        kind: "agent_msg",
        content: "custom autonomous response",
        turn_id: "turn_collision",
        response_to: responseToSingleSource(customSource),
      },
      { session: customSessionId },
    );
    await borg.stream.append({
      kind: "internal_event",
      content: {
        kind: "autonomous_action",
        trigger: "goal_followup_due",
        outcome_summary: "Answered.",
        turn_result_id: customResult.id,
        ts: clock.now(),
      },
    });

    const response = await app.request(
      `/api/activity?day=${localDayString(defaultResult.timestamp)}`,
    );
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      rows: Array<{ id: string; origin: string; trigger: string | null; excerpt: string | null }>;
    };

    expect(body.rows).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          id: "turn:default:turn_collision",
          origin: "user",
          trigger: null,
          excerpt: "default response",
        }),
        expect.objectContaining({
          id: `turn:${customSessionId}:turn_collision`,
          origin: "autonomous",
          trigger: "goal_followup_due",
          excerpt: "custom autonomous response",
        }),
      ]),
    );
  });

  it("caps activity rows for a day and marks the response truncated", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    let firstSource: Awaited<ReturnType<typeof borg.stream.append>> | null = null;

    for (let index = 0; index < 205; index += 1) {
      const source = await borg.stream.append({
        kind: "user_msg",
        content: `source ${index}`,
        turn_id: `turn_activity_cap_${index}`,
      });
      firstSource ??= source;
      await borg.stream.append({
        kind: "agent_msg",
        content: `response ${index}`,
        turn_id: `turn_activity_cap_${index}`,
        response_to: responseToSingleSource(source),
      });
      clock.advance(1);
    }

    expect(firstSource).not.toBeNull();
    const response = await app.request(
      `/api/activity?day=${localDayString(firstSource!.timestamp)}`,
    );
    expect(response.status).toBe(200);
    const body = (await response.json()) as { rows: unknown[]; truncated: boolean };

    expect(body.rows).toHaveLength(200);
    expect(body.truncated).toBe(true);
  });

  it("serves read-only autonomy state from public scheduler and wake history facades", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    borg.autonomy.wakes.record({
      trigger_name: "scheduled_reflection",
      session_id: DEFAULT_SESSION_ID,
      wake_source_type: "trigger",
      source_category: "contemplative",
    });

    const response = await app.request("/api/autonomy");

    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body).toMatchObject({
      scheduler: {
        enabled: true,
        interval_ms: 60_000,
        next_tick_at: null,
      },
      wake_budget: {
        used: 1,
        limit: 6,
        window_ms: 86_400_000,
      },
      can_cancel_wakes: false,
      self_scheduled_wakes: [],
      wake_sources: expect.arrayContaining([
        expect.objectContaining({
          name: "scheduled_reflection",
          enabled: false,
          wake_source_type: "trigger",
          source_category: "contemplative",
          next_due_at: null,
          wake_count: 1,
        }),
        expect.objectContaining({
          name: "scheduled_wake",
          enabled: true,
          wake_source_type: "trigger",
          source_category: "contemplative",
          next_due_at: null,
        }),
      ]),
      recent_wakes: [
        expect.objectContaining({
          trigger_name: "scheduled_reflection",
          session_id: DEFAULT_SESSION_ID,
        }),
      ],
    });
    const conditionSource = body.wake_sources.find(
      (source: { name: string }) => source.name === "commitment_revoked",
    );
    expect(conditionSource).toMatchObject({
      enabled: true,
      wake_source_type: "condition",
      source_category: "operational",
    });
    expect(conditionSource).not.toHaveProperty("next_due_at");

    const cancel = await app.request("/api/autonomy/wakes/autw_missing/cancel", {
      method: "POST",
    });
    expect(cancel.status).toBe(404);
  });

  it("serves train-of-thought journal entries with labels", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const internal = borg as unknown as BorgTestInternals;
    const selfEntityId = borg.entities.resolve("self", {
      kind: "self",
      provenance: "assistant_seeded",
    });

    internal.deps.trainOfThoughtRepository.append({
      text: "journal entry visible through facade",
      selfEntityId,
      sourceTurnId: "turn_journal",
      now: clock.now(),
    });

    const response = await app.request(`/api/journal?limit=5&day=${localDayString(clock.now())}`);

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      entries: [
        expect.objectContaining({
          text: "journal entry visible through facade",
          disclosure_class: "self_private",
          source_turn_id: "turn_journal",
          self_label: "self",
        }),
      ],
    });

    const previousDay = await app.request(
      `/api/journal?limit=5&day=${localDayString(clock.now() - 24 * 60 * 60 * 1_000)}`,
    );
    expect(previousDay.status).toBe(200);
    expect(await previousDay.json()).toMatchObject({ entries: [] });
  });

  it("serves capped semantic graph visualization data", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, clock, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const seeded = await seedSemanticGraph(borg, clock);

    const response = await app.request("/api/semantic/graph?limit=3");

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      nodes: expect.arrayContaining([
        expect.objectContaining({
          id: expect.any(String),
          label: expect.any(String),
          display_label: expect.any(String),
          status: "active",
          kind: expect.any(String),
          edge_count: expect.any(Number),
        }),
        expect.objectContaining({
          label: seeded.aliceEntityId,
          display_label: "Alice",
          kind: "entity",
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

  it("serves semantic node and edge detail by id", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-semantic-node-"));
    tempDirs.push(tempDir);
    const { borg, live, clock } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const sourceEpisodeId = createEpisodeId();
    const node = await borg.semantic.nodes.add({
      kind: "entity",
      label: "Detail node",
      description: "Detail node description",
      sourceEpisodeIds: [sourceEpisodeId],
    });
    const target = await borg.semantic.nodes.add({
      kind: "proposition",
      label: "Target node",
      description: "Target node description",
      sourceEpisodeIds: [sourceEpisodeId],
    });
    const entityId = borg.entities.resolve("Dana");
    const entityNode = await borg.semantic.nodes.add({
      kind: "entity",
      label: entityId,
      description: "Entity id backed node",
      sourceEpisodeIds: [sourceEpisodeId],
    });
    const edge = borg.semantic.edges.add({
      from_node_id: node.id,
      to_node_id: target.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: [sourceEpisodeId],
      created_at: clock.now(),
      last_verified_at: clock.now(),
    });
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request(`/api/semantic/nodes/${node.id}`);

    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({
      node: {
        id: node.id,
        label: "Detail node",
        display_label: "Detail node",
        description: "Detail node description",
        source_count: 1,
        disclosure_class: "unknown",
        disclosure_label: {
          disclosure_class: "unknown",
          origin_audience_entity_ids: [],
          private_to_entity_ids: [],
          public_to_entity_ids: [],
        },
      },
    });

    const entityNodeResponse = await app.request(`/api/semantic/nodes/${entityNode.id}`);
    expect(entityNodeResponse.status).toBe(200);
    expect(await entityNodeResponse.json()).toMatchObject({
      node: {
        id: entityNode.id,
        label: entityId,
        display_label: "Dana",
      },
    });

    const missing = await app.request(`/api/semantic/nodes/${createSemanticNodeId()}`);
    expect(missing.status).toBe(404);

    const edgeResponse = await app.request(`/api/semantic/edges/${edge.id}`);

    expect(edgeResponse.status).toBe(200);
    expect(await edgeResponse.json()).toMatchObject({
      edge: {
        id: edge.id,
        from_node_id: node.id,
        to_node_id: target.id,
        relation: "supports",
        confidence: 0.7,
        evidence_episode_ids: [sourceEpisodeId],
        source_count: 1,
        disclosure_class: "unknown",
        disclosure_label: {
          disclosure_class: "unknown",
          origin_audience_entity_ids: [],
          private_to_entity_ids: [],
          public_to_entity_ids: [],
        },
      },
    });

    const missingEdge = await app.request(`/api/semantic/edges/${createSemanticEdgeId()}`);
    expect(missingEdge.status).toBe(404);

    const invalidEdge = await app.request("/api/semantic/edges/not-an-edge-id");
    expect(invalidEdge.status).toBe(400);
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
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const { frames, wasClosed } = collectLiveFrames(live);

    await enqueueTextTurn(app, {
      message: "hello ws",
      external_message_id: "ws-phases-1",
      audience: "Alice",
    });
    await expect(borg.inbox.catchUp.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "drained",
      drained: 1,
      hasMore: false,
    });

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
    const terminalFrame = frames.find((frame) => frame.type === "turn:terminal");
    expect(terminalFrame).toMatchObject({
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

  it("updates demo session last_turn_id from terminal stream appends", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    createDemoServerApp({ borgHandle: { current: borg }, live });

    await borg.stream.append({
      kind: "agent_msg",
      content: "terminal",
      turn_id: "turn_terminal_observer",
    });

    expect(borg.sessions.get(DEFAULT_SESSION_ID)).toMatchObject({
      last_turn_id: "turn_terminal_observer",
      message_count: 0,
    });
  });

  it("broadcasts the buffered unary finalizer text between phase start and completion", async () => {
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
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });
    const { frames } = collectLiveFrames(live);

    await enqueueTextTurn(app, {
      message: "hello token ws",
      external_message_id: "ws-token-1",
      audience: "Alice",
    });
    await expect(borg.inbox.catchUp.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "drained",
      drained: 1,
      hasMore: false,
    });

    await waitFor(() => frames.some((frame) => frame.type === "turn:token"));

    const finalStart = frames.findIndex(
      (frame) =>
        frame.type === "turn:phase:started" &&
        (frame.data as { phase?: unknown } | undefined)?.phase === "final",
    );
    const tokenFrame = frames.findIndex((frame) => frame.type === "turn:token");
    const flushFrame = frames.findIndex((frame) => frame.type === "turn:token:flush");
    const finalComplete = frames.findIndex(
      (frame) =>
        frame.type === "turn:phase:completed" &&
        (frame.data as { phase?: unknown } | undefined)?.phase === "final",
    );

    expect(finalStart).toBeGreaterThanOrEqual(0);
    expect(tokenFrame).toBeGreaterThan(finalStart);
    expect(flushFrame).toBeGreaterThan(tokenFrame);
    expect(finalComplete).toBeGreaterThan(flushFrame);
    expect(frames[tokenFrame]).toMatchObject({
      type: "turn:token",
      phase: "final",
      chunk_text: "ws token ok",
      sequence: 1,
    });
    expect(frames[flushFrame]).toMatchObject({
      type: "turn:token:flush",
      phase: "final",
      full_text: "ws token ok",
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

    await enqueueTextTurn(app, {
      message: "hello websocket terminal",
      external_message_id: "ws-terminal-1",
      audience: "Alice",
    });
    await expect(borg.inbox.catchUp.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "drained",
      drained: 1,
      hasMore: false,
    });

    await waitFor(() => frames.some((frame) => frame.type === "turn:terminal"));

    const terminalIndex = frames.findIndex((frame) => frame.type === "turn:terminal");
    const turnId = (frames[terminalIndex]?.data as { turn_id?: unknown } | undefined)?.turn_id;
    expect(turnId).toEqual(expect.any(String));
    const phaseIndices = frames
      .map((frame, index) => ({ frame, index }))
      .filter(
        ({ frame }) =>
          frame.type.startsWith("turn:phase:") &&
          (frame.data as { turn_id?: unknown } | undefined)?.turn_id === turnId,
      )
      .map(({ index }) => index);

    expect(phaseIndices.length).toBeGreaterThan(0);
    expect(Math.max(...phaseIndices)).toBeLessThan(terminalIndex);
    expect(frames[terminalIndex]).toMatchObject({
      type: "turn:terminal",
      event: "turn.terminal",
      data: {
        turn_id: turnId,
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
    expect(
      frames.some(
        (frame) => frame.type === "turn:phase:detail" && frame.event === "evidence_ledger.built",
      ),
    ).toBe(false);
    live.broadcaster.closeAll();
    expect(wasClosed()).toBe(true);
  });

  it("GET /api/prompts returns 7 blocks, each defaulted and not overridden", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const hostCapabilities = "Configured host capability block.";
    const { borg, live } = await openHarness({ tempDir, hostCapabilities });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/prompts");
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      blocks: Array<{
        key: string;
        current_text: string;
        current_text_kind: string;
        overridden: boolean;
      }>;
    };
    expect(body.blocks.map((b) => b.key)).toEqual([
      "base_identity_preamble",
      "self_architecture",
      "voice_and_posture",
      "epistemic_posture",
      "identity_posture",
      "participation_posture",
      "host_capabilities",
    ]);
    expect(body.blocks.every((b) => b.overridden === false)).toBe(true);
    const hostCapabilitiesBlock = body.blocks.find((b) => b.key === "host_capabilities");
    expect(hostCapabilitiesBlock?.current_text).toContain(hostCapabilities);
    expect(hostCapabilitiesBlock?.current_text_kind).toBe("runtime_composed");
    expect(hostCapabilitiesBlock?.current_text).toContain(
      "Proactive outbound messaging via wired source_type connector(s): demo",
    );
  });

  it("GET /api/prompts/assembled returns the library-composed framing prompt", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borgHandle: { current: borg }, live });

    const response = await app.request("/api/prompts/assembled");
    expect(response.status).toBe(200);
    const body = (await response.json()) as {
      text: string;
      sections: string[];
      segments: Array<{ id: string; label: string; start: number; end: number }>;
    };

    expect(body.sections).toEqual([
      "base_identity_preamble",
      "self_architecture",
      "voice_and_posture",
      "epistemic_posture",
      "identity_posture",
      "participation_posture",
      "loop_breaking_posture",
      "trusted_guidance_preamble",
      "borg_host_capabilities",
      "live_turn_read_tool_menu",
    ]);
    expect(body.segments.map((segment) => segment.id)).toEqual(body.sections);
    const hostSegment = body.segments.find((segment) => segment.id === "borg_host_capabilities");
    expect(hostSegment).toBeDefined();
    expect(body.text.slice(hostSegment!.start, hostSegment!.end)).toContain(
      "<borg_host_capabilities>",
    );
    expect(body.text).toContain("<borg_host_capabilities>");
    expect(body.text).toContain("</borg_host_capabilities>");
    expect(body.text).toContain("<borg_live_turn_read_tools>");
    expect(body.text).toContain("tool.ownRecords.list");
    expect(body.text).toContain(
      "Proactive outbound messaging via wired source_type connector(s): demo",
    );
    expect(body.text.indexOf("Voice and posture:")).toBeLessThan(
      body.text.indexOf("<borg_host_capabilities>"),
    );
    expect(body.text).not.toContain("The most recent user-role message is the current turn");
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
    const putBody = (await put.json()) as {
      current_text: string;
      current_text_kind: string;
      overridden: boolean;
    };
    expect(putBody).toMatchObject({
      current_text: "Speak crisply.",
      current_text_kind: "stored_override",
      overridden: true,
    });

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
    const enqueueRelease = createDeferred<BorgEnqueueMessageResult>();
    const enqueueSpy = vi.spyOn(borg, "enqueueMessage").mockImplementation(async () => {
      turnStarted.resolve();
      return enqueueRelease.promise;
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
      body: JSON.stringify({ message: "hello", external_message_id: "busy-1" }),
    });
    await turnStarted.promise;

    const reset = await app.request("/api/admin/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confirm: "RESET" }),
    });
    expect(reset.status).toBe(409);
    expect(resetBorg).not.toHaveBeenCalled();

    enqueueRelease.resolve({
      status: "enqueued",
      sessionId: DEFAULT_SESSION_ID,
      streamEntryId: "se_busy",
    } as BorgEnqueueMessageResult);
    expect((await turn).status).toBe(200);
    enqueueSpy.mockRestore();
  });

  it("server entrypoint starts the catch-up worker after opening Borg", async () => {
    vi.resetModules();
    const previousPort = process.env.PORT;
    process.env.PORT = "7781";

    const start = vi.fn();
    const borg = createBorgCloseStub({ start });
    const open = vi.fn(async () => borg);
    const serveMock = vi.fn(() => ({
      close: vi.fn((callback?: () => void) => callback?.()),
    }));
    const liveBridgeMock = {
      broadcaster: { closeAll: vi.fn() },
      tracer: {},
      ledgerCache: new Map(),
      setStreamEntrySerializer: vi.fn(),
      observeStreamAppend: vi.fn(),
      onStreamAppend: vi.fn(),
    };
    const createDemoServerAppMock = vi.fn(() => ({
      app: { fetch: vi.fn() },
      injectWebSocket: vi.fn(),
    }));
    const wireMaintenanceSchedulerLiveObserverMock = vi.fn();

    vi.doMock("borg", () => ({
      Borg: { open },
      DemoMessageConnector: class DemoMessageConnector {},
      loadConfig: vi.fn(() => ({
        dataDir: ".borg-data/test-entry",
        anthropic: { models: { cognition: "entry-cognition" } },
        embedding: { model: "entry-embed", dims: 4 },
      })),
    }));
    vi.doMock("@hono/node-server", () => ({ serve: serveMock }));
    vi.doMock("../app.js", () => ({
      createDemoServerApp: createDemoServerAppMock,
      ensureDemoDefaultSession: vi.fn(),
      runtimeConfigFromConfig: vi.fn(() => ({
        model: "entry-cognition",
        embedding: { model: "entry-embed", dims: 4 },
      })),
      serializeStreamEntries: vi.fn((_borg, entries) => entries),
      wireMaintenanceSchedulerLiveObserver: wireMaintenanceSchedulerLiveObserverMock,
    }));
    vi.doMock("../live.js", () => ({
      createLiveBridge: vi.fn(() => liveBridgeMock),
    }));
    vi.doMock("../reset.js", () => ({
      createResetBorgController: vi.fn(() => vi.fn()),
    }));
    vi.spyOn(console, "log").mockImplementation(() => {});

    try {
      await import("../index.js?server-start");
      expect(open).toHaveBeenCalledTimes(1);
      expect(start).toHaveBeenCalledTimes(1);
      expect(wireMaintenanceSchedulerLiveObserverMock).toHaveBeenCalledWith(borg, liveBridgeMock);
      expect(serveMock).toHaveBeenCalledTimes(1);
    } finally {
      if (previousPort === undefined) {
        delete process.env.PORT;
      } else {
        process.env.PORT = previousPort;
      }
      vi.doUnmock("borg");
      vi.doUnmock("@hono/node-server");
      vi.doUnmock("../app.js");
      vi.doUnmock("../live.js");
      vi.doUnmock("../reset.js");
      vi.resetModules();
    }
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
      body: JSON.stringify({ message: "hello during reset", external_message_id: "reset-busy-1" }),
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
      current: createBorgCloseStub(),
    };
    const nextStart = vi.fn(() => {
      expect(borgHandle.current).toBe(nextBorg);
      expect(borgHandle.state).toBe("open");
    });
    const nextBorg = createBorgCloseStub({ start: nextStart });
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
    expect(nextStart).toHaveBeenCalledTimes(1);
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
    const nextBorg = createBorgCloseStub();
    const borgHandle = {
      current: createBorgCloseStub(),
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
