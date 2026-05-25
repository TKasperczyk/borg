import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AddressInfo } from "node:net";

import { serve } from "@hono/node-server";
import { afterEach, describe, expect, it } from "vitest";
import { Borg, ManualClock, type BorgOpenOptions } from "borg";

import {
  FakeLLMClient,
  createFakeEmitAnswerResponse,
} from "../../../../src/llm/test-support/fake-client.js";
import { TestEmbeddingClient, createTestConfig } from "../../../../src/offline/test-support.js";
import { createDemoServerApp } from "../app.js";
import { createLiveBridge, type LiveFrame } from "../live.js";

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
  live: ReturnType<typeof createLiveBridge>;
}> {
  const live = createLiveBridge();
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
    clock: new ManualClock(1_800_000_000_000),
    embeddingDimensions: 4,
    embeddingClient: new TestEmbeddingClient(),
    llmClient: input.llmClient ?? new FakeLLMClient(),
    tracer: live.tracer,
    onStreamAppend: live.onStreamAppend,
    liveExtraction: false,
  };

  return {
    borg: await Borg.open(options),
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
    const { borg, live } = await openHarness({ tempDir, llmClient: llm });
    closers.push(() => borg.close());
    live.ledgerCache.set("turn_cached", { sections: [] });
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

    const bands = await app.request("/api/memory/bands");
    expect(bands.status).toBe(200);
    expect((await bands.json()).bands).toHaveLength(8);

    for (const band of ["episodic", "semantic", "commitments", "self", "affective"]) {
      const response = await app.request(`/api/memory/bands/${band}`);
      expect(response.status).toBe(200);
      expect(await response.json()).toMatchObject({ band });
    }

    const commitments = await app.request("/api/commitments?audience=Alice");
    expect(commitments.status).toBe(200);
    expect(await commitments.json()).toMatchObject({ commitments: [] });

    const sharedState = await app.request("/api/shared-state?audience=Alice");
    expect(sharedState.status).toBe(200);
    expect(await sharedState.json()).toMatchObject({ audience: "Alice", entries: [] });

    const identity = await app.request("/api/identity");
    expect(identity.status).toBe(200);
    expect(await identity.json()).toMatchObject({
      values: [],
      traits: [],
      open_questions: [],
      growth_markers: [],
      periods: [],
    });

    const audit = await app.request("/api/dream/audit?limit=5");
    expect(audit.status).toBe(200);
    expect(await audit.json()).toMatchObject({ rows: [] });

    const attachment = await app.request("/api/attachments/att_missing/bytes");
    expect(attachment.status).toBe(404);

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
