import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { File } from "node:buffer";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { serve } from "@hono/node-server";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Borg,
  DEFAULT_SESSION_ID,
  ManualClock,
  createEpisodeId,
  createMaintenanceRunId,
  createSemanticEdgeId,
  createSemanticNodeId,
  type AttachmentId,
  type BorgOpenOptions,
  type StreamEntry,
} from "borg";

import {
  FakeLLMClient,
  createFakeEmitAnswerResponse,
  createFakeStreamingResponse,
} from "../../../../src/llm/test-support/fake-client.js";
import type { AttachmentService } from "../../../../src/attachments/index.js";
import { IMAGE_PERCEPTION_TOOL_NAME } from "../../../../src/attachments/perception.js";
import type { RelationalSlotRepository } from "../../../../src/memory/relational-slots/repository.js";
import type { ReviewQueueRepository } from "../../../../src/memory/semantic/review-queue.js";
import { TestEmbeddingClient, createTestConfig } from "../../../../src/offline/test-support.js";
import type { AuditLog } from "../../../../src/offline/audit-log.js";
import type { StreamWriter } from "../../../../src/stream/index.js";
import { createDemoServerApp } from "../app.js";
import { LiveBroadcaster, createLiveBridge, type LiveFrame } from "../live.js";

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
    relationalSlotRepository: RelationalSlotRepository;
    reviewQueueRepository: ReviewQueueRepository;
  };
};

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

async function openHarness(input: { tempDir: string; llmClient?: FakeLLMClient }): Promise<{
  borg: Borg;
  clock: ManualClock;
  live: ReturnType<typeof createLiveBridge>;
}> {
  const live = createLiveBridge();
  const clock = new ManualClock(1_800_000_000_000);
  const options: BorgOpenOptions = {
    config: createTestConfig({
      dataDir: input.tempDir,
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
    clock,
    embeddingDimensions: 4,
    embeddingClient: new TestEmbeddingClient(),
    llmClient: input.llmClient ?? new FakeLLMClient(),
    tracer: live.tracer,
    onStreamAppend: live.onStreamAppend,
    liveExtraction: false,
  };

  return {
    borg: await Borg.open(options),
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

  live.broadcaster.add({
    send(data: string): void {
      frames.push(JSON.parse(data) as LiveFrame);
    },
    close(): void {
      closed = true;
    },
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

  it("serves REST endpoint contract shapes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [createFakeEmitAnswerResponse("demo ok"), createEmptyReflectionResponse()],
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
    const { app } = createDemoServerApp({ borg, live });

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
    expect(attachment.status).toBe(404);

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
    const { app } = createDemoServerApp({ borg, live });
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
    const { app } = createDemoServerApp({ borg, live });
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

  it("paginates stream entries with same-timestamp file order", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app } = createDemoServerApp({ borg, live });
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
    const { app } = createDemoServerApp({ borg, live });
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
    const { app } = createDemoServerApp({ borg, live });
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
    const { app, injectWebSocket } = createDemoServerApp({ borg, live });
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
});
