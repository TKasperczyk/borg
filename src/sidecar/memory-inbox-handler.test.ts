import { mkdtempSync, rmSync } from "node:fs";
import { createServer, request as httpRequest, type Server } from "node:http";
import { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { Borg } from "../borg.js";
import { FakeEmbeddingClient } from "../embeddings/index.js";
import { FakeLLMClient } from "../llm/test-support/fake-client.js";
import type { EntityRecord } from "../memory/commitments/index.js";
import type { StreamEntry } from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { createEntityId, createSessionId, createStreamEntryId } from "../util/ids.js";
import { createMemoryHandler, type MemoryPool } from "./memory-handler.js";
import { ResponseWaiterRegistry } from "./response-waiter-registry.js";
import { sessionFromCaller } from "./team-agent-identity.js";
import { TeamAgentTurnRunner } from "./team-agent-turn-runner.js";

const TOKEN = "secret";
const servers: Server[] = [];
const borgs: Borg[] = [];
const tempDirs: string[] = [];

afterEach(async () => {
  await Promise.all(
    servers
      .splice(0)
      .map((server) => new Promise<void>((resolve) => server.close(() => resolve()))),
  );
  while (borgs.length > 0) {
    await borgs.pop()!.close();
  }
  while (tempDirs.length > 0) {
    rmSync(tempDirs.pop()!, { recursive: true, force: true });
  }
});

function makeHarness() {
  const senderId = createEntityId();
  const groupId = createEntityId();
  const sessionId = createSessionId();
  const entryId = createStreamEntryId();
  const entities = new Map<string, EntityRecord>();
  const entity = (id: typeof senderId, name: string, kind: "person" | "group") => ({
    id,
    canonical_name: name,
    aliases: [],
    kind,
    borg_role: null,
    name_provenance: "transport_sender" as const,
    created_at: 1,
  });
  entities.set(senderId, entity(senderId, "Sender", "person"));
  entities.set(groupId, entity(groupId, "Room", "group"));
  const enqueueMessage = vi.fn(async () => ({
    status: "duplicate" as const,
    sessionId,
    streamEntryId: entryId,
  }));
  const sealPendingBacklog = vi.fn(async () => null);
  let lookup: Borg["inbox"]["findTerminalCoveringEntry"] = () => ({ status: "pending" });
  let sessionExists = false;
  const getSession = vi.fn(() =>
    sessionExists ? ({ source_type: "teams_inbox" } as never) : null,
  );
  const findTerminalCoveringEntry = vi.fn(
    (input: Parameters<Borg["inbox"]["findTerminalCoveringEntry"]>[0]) => lookup(input),
  );
  const borg = {
    entities: {
      resolveExternal: (input: { kind: string }) => (input.kind === "group" ? groupId : senderId),
      get: (id: string) => entities.get(id) ?? null,
    },
    sessions: { get: getSession },
    enqueueMessage,
    inbox: {
      sealPendingBacklog,
      findTerminalCoveringEntry,
    },
  } as unknown as Borg;
  const exclusives: Array<boolean | undefined> = [];
  let beforeWithTenant: (() => Promise<void>) | undefined;
  const pool: MemoryPool = {
    listTenantIds: async () => [],
    async withTenant(_tenant, fn, options) {
      exclusives.push(options?.exclusive);
      await beforeWithTenant?.();
      return fn(borg);
    },
  };
  const waiters = new ResponseWaiterRegistry();
  return {
    pool,
    waiters,
    enqueueMessage,
    sealPendingBacklog,
    findTerminalCoveringEntry,
    exclusives,
    sessionId,
    entryId,
    setLookup(next: typeof lookup) {
      lookup = next;
    },
    setSessionExists(next: boolean) {
      sessionExists = next;
    },
    setBeforeWithTenant(next: (() => Promise<void>) | undefined) {
      beforeWithTenant = next;
    },
  };
}

async function start(harness: ReturnType<typeof makeHarness>) {
  const server = createServer(
    createMemoryHandler({ pool: harness.pool, token: TOKEN, inboxWaiters: harness.waiters }),
  );
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return `http://127.0.0.1:${port}`;
}

function post(base: string, path: string, body: unknown) {
  return fetch(`${base}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-borg-token": TOKEN },
    body: JSON.stringify(body),
  });
}

function enqueueBody() {
  return {
    tenant: "tenant",
    session: "raw-thread",
    conversation: { external_id: "conversation", type: "groupChat", name: "Room" },
    sender: { external_id: "sender", display_name: "Sender", bot: false, operator: false },
    text: "hello",
    external_message_id: "message",
    observed_at: "2026-09-03T15:43:25.707Z",
    flags: { mentioned: true, quotes_bot: false },
  };
}

describe("memory inbox routes", () => {
  it("returns the enqueue duplicate and persists the transport envelope under the exclusive chain", async () => {
    const harness = makeHarness();
    const base = await start(harness);
    const response = await post(base, "/memory/enqueue", enqueueBody());

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      status: "duplicate",
      sidecar_session_id: harness.sessionId,
      entry_id: harness.entryId,
    });
    expect(harness.exclusives).toEqual([true]);
    expect(harness.enqueueMessage).toHaveBeenCalledWith(
      expect.objectContaining({
        observedAt: Date.parse("2026-09-03T15:43:25.707Z"),
        metadata: {
          teams_inbox: expect.objectContaining({ thread_id: "raw-thread", mentioned: true }),
        },
      }),
    );
    expect(harness.sealPendingBacklog).toHaveBeenCalledWith({
      sessionId: expect.any(String),
      reason: "Legacy append-turn backlog sealed when the session joined Teams inbox",
    });
    expect(harness.sealPendingBacklog.mock.invocationCallOrder[0]).toBeLessThan(
      harness.enqueueMessage.mock.invocationCallOrder[0]!,
    );
  });

  it("seals legacy append-turn history on claim before the real worker sends the fresh message", async () => {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-inbox-transition-"));
    tempDirs.push(dataDir);
    const rawThread = "tenant::shared::groupChat::conversation";
    const sessionId = sessionFromCaller(rawThread);
    const now = Date.parse("2026-09-03T15:43:25.707Z");
    const clock = new ManualClock(now);
    const fetchFn = vi.fn(
      async () =>
        new Response(JSON.stringify({ action: "reply", content: "fresh answer" }), {
          status: 200,
        }),
    ) as unknown as typeof fetch;
    const borg = await Borg.open({
      dataDir,
      embeddingDimensions: 4,
      embeddingClient: new FakeEmbeddingClient(4),
      llmClient: new FakeLLMClient(),
      liveExtraction: false,
      clock,
      inbox: {
        runner: ({ terminal, entityRepository }) =>
          new TeamAgentTurnRunner({
            tenant: "tenant",
            baseUrl: "http://team-agent:8080",
            apiToken: "secret",
            timeoutMs: 1_000,
            staleMs: 60_000,
            terminal,
            entityRepository,
            clock,
            fetchFn,
          }),
        sessionPredicate: (session) => session?.source_type === "teams_inbox",
        settleMs: 0,
        maxSettleMs: 1,
      },
    });
    borgs.push(borg);
    const legacySender = borg.entities.resolveExternal({
      source: "team-agent.sender",
      externalId: "sender",
      canonicalName: "Sender",
      kind: "person",
      provenance: "transport_sender",
    });
    const room = borg.entities.resolveExternal({
      source: "team-agent.conversation",
      externalId: "conversation",
      canonicalName: "Room",
      kind: "group",
      provenance: "transport_audience_label",
    });
    borg.sessions.ensure({
      session_id: sessionId,
      source_type: "team_agent",
      source_external_id: rawThread,
      label: "Room",
      audience_label: "Room",
      audience_entity_id: room,
      conversation_kind: "thread",
      audience_role: "participant",
      status: "active",
    });
    const legacy = await borg.stream.append(
      {
        kind: "user_msg",
        content: "legacy append-turn message",
        observed_at: now - 1_000,
        sender_entity_id: legacySender,
        audience: "Room",
        conversation: { type: "groupChat", name: "Room" },
      },
      { session: sessionId },
    );
    borg.inbox.catchUp.start();
    await new Promise((resolve) => setTimeout(resolve, 0));

    const waiters = new ResponseWaiterRegistry();
    const pool: MemoryPool = {
      listTenantIds: async () => [],
      withTenant: async (_tenant, fn) => fn(borg),
    };
    const server = createServer(createMemoryHandler({ pool, token: TOKEN, inboxWaiters: waiters }));
    servers.push(server);
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    const { port } = server.address() as AddressInfo;
    const body = enqueueBody();
    body.session = rawThread;
    const response = await post(`http://127.0.0.1:${port}`, "/memory/enqueue", body);
    const enqueued = (await response.json()) as {
      status: string;
      sidecar_session_id: string;
      entry_id: string;
    };

    expect(response.status).toBe(200);
    expect(enqueued.status).toBe("enqueued");
    await vi.waitFor(() => expect(fetchFn).toHaveBeenCalledTimes(1));
    const request = JSON.parse(
      String((fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]?.[1]?.body),
    );
    expect(request.messages.map((message: { entry_id: string }) => message.entry_id)).toEqual([
      enqueued.entry_id,
    ]);
    expect(borg.inbox.findTerminalCoveringEntry({ sessionId, entryId: legacy.id })).toMatchObject({
      status: "found",
      terminalEntry: { kind: "agent_observed" },
    });
    await vi.waitFor(() =>
      expect(
        borg.inbox.findTerminalCoveringEntry({
          sessionId,
          entryId: enqueued.entry_id as typeof legacy.id,
        }),
      ).toMatchObject({ status: "found", terminalEntry: { kind: "agent_msg" } }),
    );
  });

  it("authenticates inbox progress before parsing or mutation", async () => {
    const harness = makeHarness();
    const base = await start(harness);
    const response = await fetch(`${base}/memory/inbox-progress`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        tenant: "tenant",
        sidecar_session_id: harness.sessionId,
        entry_ids: [harness.entryId],
        phase: "generating",
      }),
    });

    expect(response.status).toBe(401);
    expect(harness.exclusives).toEqual([]);
  });

  it("returns 404 when inbox progress names an unknown session", async () => {
    const harness = makeHarness();
    const base = await start(harness);
    const response = await post(base, "/memory/inbox-progress", {
      tenant: "tenant",
      sidecar_session_id: harness.sessionId,
      entry_ids: [harness.entryId],
      phase: "generating",
    });

    expect(response.status).toBe(404);
    await expect(response.json()).resolves.toEqual({ error: "session not found" });
    expect(harness.exclusives).toEqual([undefined]);
  });

  it("idempotently marks inbox progress and wakes all current waiters", async () => {
    const harness = makeHarness();
    harness.setSessionExists(true);
    const first = harness.waiters.register({
      tenant: "tenant",
      sessionId: harness.sessionId,
      entryId: harness.entryId,
      timeoutMs: 1_000,
    });
    const second = harness.waiters.register({
      tenant: "tenant",
      sessionId: harness.sessionId,
      entryId: harness.entryId,
      timeoutMs: 1_000,
    });
    const base = await start(harness);
    const body = {
      tenant: "tenant",
      sidecar_session_id: harness.sessionId,
      entry_ids: [harness.entryId, harness.entryId],
      phase: "generating",
    };
    const firstResponse = await post(base, "/memory/inbox-progress", body);
    const secondResponse = await post(base, "/memory/inbox-progress", body);

    expect(firstResponse.status).toBe(200);
    await expect(firstResponse.json()).resolves.toEqual({ ok: true });
    expect(secondResponse.status).toBe(200);
    await expect(secondResponse.json()).resolves.toEqual({ ok: true });
    await expect(first.promise).resolves.toEqual({ status: "generating" });
    await expect(second.promise).resolves.toEqual({ status: "generating" });
    expect(harness.findTerminalCoveringEntry).not.toHaveBeenCalled();
    expect(harness.exclusives).toEqual([undefined, undefined]);
  });

  it("returns remembered generating progress as an interim await result", async () => {
    const harness = makeHarness();
    harness.waiters.markGenerating({
      tenant: "tenant",
      sessionId: harness.sessionId,
      entryIds: [harness.entryId],
    });
    const base = await start(harness);
    const response = await post(base, "/memory/await-response", {
      tenant: "tenant",
      sidecar_session_id: harness.sessionId,
      entry_id: harness.entryId,
      timeout_ms: 1_000,
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ status: "generating" });
    expect(harness.findTerminalCoveringEntry).toHaveBeenCalledTimes(2);
  });

  it("returns a terminal rather than remembered generating progress", async () => {
    const harness = makeHarness();
    const terminalId = createStreamEntryId();
    harness.waiters.markGenerating({
      tenant: "tenant",
      sessionId: harness.sessionId,
      entryIds: [harness.entryId],
    });
    harness.setLookup(() => ({
      status: "found",
      terminalEntry: {
        id: terminalId,
        session_id: harness.sessionId,
        timestamp: 2,
        kind: "agent_msg",
        content: "answer",
        sender_entity_id: null,
        reply_target_entity_id: null,
        compressed: false,
        response_to: {
          kind: "stream_backlog",
          from_cursor_exclusive: null,
          through_cursor_inclusive: { ts: 1, entryId: harness.entryId },
          source_entry_ids: [harness.entryId],
          count: 1,
        },
      },
      responseTo: {
        kind: "stream_backlog",
        from_cursor_exclusive: null,
        through_cursor_inclusive: { ts: 1, entryId: harness.entryId },
        source_entry_ids: [harness.entryId],
        count: 1,
      },
    }));
    const base = await start(harness);
    const response = await post(base, "/memory/await-response", {
      tenant: "tenant",
      sidecar_session_id: harness.sessionId,
      entry_id: harness.entryId,
      timeout_ms: 1_000,
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      status: "answered",
      terminal_id: terminalId,
      entry_ids: [harness.entryId],
      reply: "answer",
    });
  });

  it("returns 404 for an unknown entry without registering a waiter", async () => {
    const harness = makeHarness();
    harness.setLookup(() => ({ status: "unknown_entry" }));
    const base = await start(harness);
    const response = await post(base, "/memory/await-response", {
      tenant: "tenant",
      sidecar_session_id: harness.sessionId,
      entry_id: harness.entryId,
      timeout_ms: 5,
    });
    expect(response.status).toBe(404);
    expect(harness.waiters.size()).toBe(0);
  });

  it("returns pending at timeout", async () => {
    const harness = makeHarness();
    const base = await start(harness);
    const response = await post(base, "/memory/await-response", {
      tenant: "tenant",
      sidecar_session_id: harness.sessionId,
      entry_id: harness.entryId,
      timeout_ms: 5,
    });
    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ status: "pending" });
    expect(harness.waiters.size()).toBe(0);
  });

  it("resolves multiple route waiters for one durable terminal", async () => {
    const harness = makeHarness();
    const base = await start(harness);
    const body = {
      tenant: "tenant",
      sidecar_session_id: harness.sessionId,
      entry_id: harness.entryId,
      timeout_ms: 1_000,
    };
    const first = post(base, "/memory/await-response", body);
    const second = post(base, "/memory/await-response", body);
    while (harness.waiters.size() < 2) {
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
    const terminalId = createStreamEntryId();
    harness.waiters.resolveTerminal("tenant", {
      id: terminalId,
      session_id: harness.sessionId,
      timestamp: 2,
      kind: "agent_observed",
      content: { reason: "silent" },
      sender_entity_id: null,
      reply_target_entity_id: null,
      compressed: false,
      response_to: {
        kind: "stream_backlog",
        from_cursor_exclusive: null,
        through_cursor_inclusive: { ts: 1, entryId: harness.entryId },
        source_entry_ids: [harness.entryId],
        count: 1,
      },
    } satisfies StreamEntry);
    await expect((await first).json()).resolves.toMatchObject({
      status: "observed",
      terminal_id: terminalId,
    });
    await expect((await second).json()).resolves.toMatchObject({
      status: "observed",
      terminal_id: terminalId,
    });
  });

  it("does not leak a waiter when the client disconnects during the first scan", async () => {
    const harness = makeHarness();
    let signalScanStarted: () => void = () => {};
    const scanStarted = new Promise<void>((resolve) => {
      signalScanStarted = resolve;
    });
    let releaseScan: () => void = () => {};
    const scanGate = new Promise<void>((resolve) => {
      releaseScan = resolve;
    });
    harness.setBeforeWithTenant(async () => {
      harness.setBeforeWithTenant(undefined);
      signalScanStarted();
      await scanGate;
    });
    const base = new URL(await start(harness));
    const body = JSON.stringify({
      tenant: "tenant",
      sidecar_session_id: harness.sessionId,
      entry_id: harness.entryId,
      timeout_ms: 1_000,
    });
    const req = httpRequest({
      hostname: base.hostname,
      port: base.port,
      path: "/memory/await-response",
      method: "POST",
      headers: {
        "content-type": "application/json",
        "content-length": String(Buffer.byteLength(body)),
        "x-borg-token": TOKEN,
      },
    });
    req.on("error", () => undefined);
    req.end(body);
    await scanStarted;
    req.destroy();
    releaseScan();
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(harness.waiters.size()).toBe(0);
  });
});
