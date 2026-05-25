import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AddressInfo } from "node:net";

import { serve } from "@hono/node-server";
import { afterEach, describe, expect, it } from "vitest";
import {
  Borg,
  DEFAULT_SESSION_ID,
  ManualClock,
  createEpisodeId,
  createMaintenanceRunId,
  type AttachmentId,
  type BorgOpenOptions,
} from "borg";

import {
  FakeLLMClient,
  createFakeEmitAnswerResponse,
} from "../../../../src/llm/test-support/fake-client.js";
import type { AttachmentService } from "../../../../src/attachments/index.js";
import type { RelationalSlotRepository } from "../../../../src/memory/relational-slots/repository.js";
import type { ReviewQueueRepository } from "../../../../src/memory/semantic/review-queue.js";
import { TestEmbeddingClient, createTestConfig } from "../../../../src/offline/test-support.js";
import type { AuditLog } from "../../../../src/offline/audit-log.js";
import type { StreamWriter } from "../../../../src/stream/index.js";
import { createDemoServerApp } from "../app.js";
import { createLiveBridge, type LiveFrame } from "../live.js";

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
  internal.deps.attachmentService.setAttachmentActive(attachmentId, false, "turn_attachment_quarantine");

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
    borg.self.openQuestions.abandon(openQuestion.id, "demo smoke", { kind: "manual" }, { throughReview: true });
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
      commitments: Array<{ id: string; state: string; enforcement_class: string; audience: string }>;
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
        expect.objectContaining({ process: "belief-reviser", source: "audit", audit_id: seeded.audit.id }),
        expect.objectContaining({ process: "belief-reviser", source: "stream" }),
      ]),
      audit_rows: [expect.objectContaining({ id: seeded.audit.id })],
      belief_revision_rows: [expect.objectContaining({ id: seeded.review.id, kind: "belief_revision" })],
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
      entries: Array<{ id: string }>;
      next_cursor: string | null;
    };
    expect(firstBody.entries[0]?.id).toBe(cursorEntry?.id);
    expect(firstBody.next_cursor).not.toBeNull();

    const secondPage = await app.request(`/api/stream?limit=1&before=${firstBody.next_cursor}`);
    expect(secondPage.status).toBe(200);
    const secondBody = (await secondPage.json()) as { entries: Array<{ id: string }> };
    expect(secondBody.entries[0]?.id).toBe(older.id);
  });

  it("broadcasts turn phases and stream appends over WebSocket", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const llm = new FakeLLMClient({
      responses: [createFakeEmitAnswerResponse("ws ok"), createEmptyReflectionResponse()],
    });
    const { borg, live } = await openHarness({ tempDir, llmClient: llm });
    closers.push(() => borg.close());
    const { app, injectWebSocket } = createDemoServerApp({ borg, live });
    const server = serve({ fetch: app.fetch, port: 0 });
    injectWebSocket(server);
    closers.push(
      () =>
        new Promise<void>((resolve) => {
          server.close(() => resolve());
        }),
    );
    const port = (server.address() as AddressInfo).port;
    const frames: LiveFrame[] = [];
    const ws = new WebSocket(`ws://127.0.0.1:${port}/api/live`);
    closers.push(async () => ws.close());

    ws.addEventListener("message", (event) => {
      frames.push(JSON.parse(String(event.data)) as LiveFrame);
    });
    await new Promise<void>((resolve, reject) => {
      ws.addEventListener("open", () => resolve(), { once: true });
      ws.addEventListener("error", () => reject(new Error("websocket failed")), { once: true });
    });

    const response = await fetch(`http://127.0.0.1:${port}/api/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: "hello ws", audience: "Alice", stakes: "low" }),
    });
    expect(response.status).toBe(200);

    await waitFor(
      () =>
        frames.some((frame) => frame.type === "stream:append") &&
        frames.some((frame) => frame.type === "turn:phase:started") &&
        frames.some((frame) => frame.type === "turn:phase:completed"),
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
  });

  it("broadcasts evidence ledger events over WebSocket", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-demo-server-"));
    tempDirs.push(tempDir);
    const { borg, live } = await openHarness({ tempDir });
    closers.push(() => borg.close());
    const { app, injectWebSocket } = createDemoServerApp({ borg, live });
    const server = serve({ fetch: app.fetch, port: 0 });
    injectWebSocket(server);
    closers.push(
      () =>
        new Promise<void>((resolve) => {
          server.close(() => resolve());
        }),
    );
    const port = (server.address() as AddressInfo).port;
    const frames: LiveFrame[] = [];
    const ws = new WebSocket(`ws://127.0.0.1:${port}/api/live`);
    closers.push(async () => ws.close());

    ws.addEventListener("message", (event) => {
      frames.push(JSON.parse(String(event.data)) as LiveFrame);
    });
    await new Promise<void>((resolve, reject) => {
      ws.addEventListener("open", () => resolve(), { once: true });
      ws.addEventListener("error", () => reject(new Error("websocket failed")), { once: true });
    });

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
  });
});
