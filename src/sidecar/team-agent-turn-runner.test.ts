import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  BacklogTerminalService,
  ChatResponseBacklogPrefixBuilder,
  ChatResponseCatchUpWorker,
  ChatResponseWatermarkCoordinator,
} from "../cognition/ingestion/index.js";
import { composeMigrations, openDatabase } from "../storage/sqlite/index.js";
import {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWatermarkRepository,
  StreamWriter,
  streamEntryIndexMigrations,
  streamWatermarkMigrations,
  type StreamEntry,
} from "../stream/index.js";
import { FixedClock, ManualClock } from "../util/clock.js";
import {
  createEntityId,
  createSessionId,
  createStreamEntryId,
  type SessionId,
} from "../util/ids.js";
import { ResponseWaiterRegistry } from "./response-waiter-registry.js";
import { TeamAgentTurnRunner } from "./team-agent-turn-runner.js";

const cleanups: Array<() => void> = [];

afterEach(() => {
  vi.restoreAllMocks();
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

function harness(fetchResponse: () => Promise<Response>) {
  const sessionId = createSessionId();
  const sourceId = createStreamEntryId();
  const senderId = createEntityId();
  const sourceEntry: StreamEntry = {
    id: sourceId,
    session_id: sessionId,
    entry_index: 0,
    timestamp: 900,
    observed_at: 900,
    kind: "user_msg",
    content: "hello",
    sender_entity_id: senderId,
    reply_target_entity_id: null,
    compressed: false,
    conversation: { type: "groupChat", name: "Room" },
    source_message_key: {
      source_type: "teams_inbox",
      source_external_id: "conversation",
      external_message_id: "message",
    },
    metadata: {
      teams_inbox: {
        thread_id: "thread",
        sender: { external_id: "sender", display_name: "Sender", bot: false },
        mentioned: true,
        quotes_bot: false,
      },
    },
  };
  const appendBacklogTerminal = vi.fn(
    async (input: {
      terminal: { kind: "agent_msg"; content: string } | { kind: "agent_observed"; reason: string };
    }) => {
      const terminalId = createStreamEntryId();
      const terminalEntry = {
        id: terminalId,
        session_id: sessionId,
        timestamp: 1_000,
        kind: input.terminal.kind,
        content:
          input.terminal.kind === "agent_msg"
            ? input.terminal.content
            : { reason: input.terminal.reason },
        sender_entity_id: null,
        reply_target_entity_id: null,
        compressed: false,
        response_to: {
          kind: "stream_backlog",
          from_cursor_exclusive: null,
          through_cursor_inclusive: { ts: sourceEntry.timestamp, entryId: sourceId },
          source_entry_ids: [sourceId],
          count: 1,
        },
      } satisfies StreamEntry;
      return {
        terminalEntry,
        responseTo: terminalEntry.response_to,
        sourceEntries: [sourceEntry],
      };
    },
  );
  const terminal = {
    findTerminalCoveringEntry: vi.fn(() => ({ status: "pending" as const })),
    hydrateBacklogBatch: vi.fn(async () => ({ sourceEntries: [sourceEntry], records: [] })),
    sealStaleBacklog: vi.fn(async () => null),
    sealBacklogPrefix: vi.fn(async () => null),
    appendBacklogTerminal,
  } as unknown as BacklogTerminalService;
  const fetchFn = vi.fn(fetchResponse) as unknown as typeof fetch;
  const runner = new TeamAgentTurnRunner({
    tenant: "tenant",
    baseUrl: "http://team-agent:8080",
    apiToken: "secret",
    timeoutMs: 1_000,
    staleMs: 600,
    terminal,
    entityRepository: {
      get: () => ({ canonical_name: "Sender" }) as never,
    },
    clock: new FixedClock(1_000),
    fetchFn,
  });
  const input = {
    sessionId,
    inboundBatch: {
      kind: "stream_backlog" as const,
      entryIds: [sourceId],
      throughCursorInclusive: { ts: sourceEntry.timestamp, entryId: sourceId },
    },
  };
  return { runner, input, sourceEntry, terminal, appendBacklogTerminal, fetchFn };
}

function openRealHarness(input: {
  fetchFn: typeof fetch;
  clock?: ManualClock;
  staleMs?: number;
  onTerminalCommitted?: (entry: StreamEntry) => void;
  onReconcileAdvance?: (event: {
    sessionId: SessionId;
    advancedThrough: { ts: number; entryId: ReturnType<typeof createStreamEntryId> };
  }) => void;
}) {
  const dataDir = mkdtempSync(join(tmpdir(), "borg-team-agent-runner-"));
  const db = openDatabase(join(dataDir, "borg.db"), {
    migrations: composeMigrations(streamEntryIndexMigrations, streamWatermarkMigrations),
  });
  cleanups.push(() => {
    db.close();
    rmSync(dataDir, { recursive: true, force: true });
  });
  const clock = input.clock ?? new ManualClock(1_000);
  const entryIndex = new StreamEntryIndexRepository({ db, dataDir });
  const coordinator = new ChatResponseWatermarkCoordinator({
    watermarkRepository: new StreamWatermarkRepository({ db, clock }),
    entryIndex,
    logger: { warn() {} },
  });
  const writer = (sessionId: SessionId) =>
    new StreamWriter({ dataDir, sessionId, entryIndex, clock });
  const ingest = vi.fn(async () => ({ ran: true, processedEntries: 0 }));
  const terminal = new BacklogTerminalService({
    dataDir,
    entryIndex,
    createStreamReader: (sessionId) => new StreamReader({ dataDir, sessionId, entryIndex }),
    createStreamWriter: writer,
    coordinator,
    streamIngestionCoordinator: { ingest } as never,
    ...(input.onTerminalCommitted === undefined
      ? {}
      : { onTerminalCommitted: input.onTerminalCommitted }),
  });
  const runner = new TeamAgentTurnRunner({
    tenant: "tenant",
    baseUrl: "http://team-agent:8080",
    apiToken: "secret",
    timeoutMs: 1_000,
    staleMs: input.staleMs ?? 600,
    terminal,
    entityRepository: {
      get: () => ({ canonical_name: "Sender" }) as never,
    },
    clock,
    fetchFn: input.fetchFn,
  });
  const worker = new ChatResponseCatchUpWorker({
    coordinator,
    prefixBuilder: new ChatResponseBacklogPrefixBuilder({
      entryIndex,
      createStreamReader: (sessionId) => new StreamReader({ dataDir, sessionId, entryIndex }),
    }),
    entryIndex,
    repairSessionStreamEntryIndex: async () => ({ inserted: 0 }),
    runner,
    ...(input.onReconcileAdvance === undefined
      ? {}
      : { onReconcileAdvance: input.onReconcileAdvance }),
    clock,
    config: {
      quietWindowMs: 0,
      maxWaitMs: 1,
      backoffBaseMs: 1,
      maxBackoffMs: 10,
    },
  });

  const appendInbox = async (sessionId: SessionId, observedAt: number, content: string) => {
    const streamWriter = writer(sessionId);
    try {
      return await streamWriter.append({
        kind: "user_msg",
        content,
        observed_at: observedAt,
        sender_entity_id: createEntityId(),
        conversation: { type: "groupChat", name: "Room" },
        source_message_key: {
          source_type: "teams_inbox",
          source_external_id: "conversation",
          external_message_id: createStreamEntryId(),
        },
        metadata: {
          teams_inbox: {
            thread_id: "thread",
            sender: { external_id: "sender", display_name: "Sender", bot: false },
            mentioned: true,
            quotes_bot: false,
          },
        },
      });
    } finally {
      streamWriter.close();
    }
  };

  return {
    dataDir,
    clock,
    entryIndex,
    coordinator,
    ingest,
    terminal,
    worker,
    appendInbox,
  };
}

describe("TeamAgentTurnRunner", () => {
  it("retries the same real prefix, resolves only after commit, ingests exactly, and stops cleanly", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const waiters = new ResponseWaiterRegistry();
    const events: string[] = [];
    const requests: Array<{ messages: Array<{ entry_id: string }> }> = [];
    let fetchCount = 0;
    let signalSecondFetch: () => void = () => {};
    const secondFetchStarted = new Promise<void>((resolve) => {
      signalSecondFetch = resolve;
    });
    let resolveSecondFetch: (response: Response) => void = () => {};
    const secondFetch = new Promise<Response>((resolve) => {
      resolveSecondFetch = resolve;
    });
    const fetchFn = vi.fn(async (_url: Parameters<typeof fetch>[0], init?: RequestInit) => {
      fetchCount += 1;
      requests.push(JSON.parse(String(init?.body)) as (typeof requests)[number]);
      if (fetchCount === 1) {
        throw new Error("transport crash");
      }
      signalSecondFetch();
      return secondFetch;
    }) as unknown as typeof fetch;
    let real!: ReturnType<typeof openRealHarness>;
    real = openRealHarness({
      fetchFn,
      onTerminalCommitted: (entry) => {
        expect(real.entryIndex.lookup(entry.id)).not.toBeNull();
        events.push("durable-commit");
        waiters.resolveTerminal("tenant", entry);
      },
    });
    const sessionId = createSessionId();
    const source = await real.appendInbox(sessionId, 900, "hello");

    await expect(real.worker.tick(sessionId)).resolves.toMatchObject({ status: "error" });
    const waiter = waiters.register({
      tenant: "tenant",
      sessionId,
      entryId: source.id,
      timeoutMs: 1_000,
    });
    let waiterSettled = false;
    void waiter.promise.then(() => {
      waiterSettled = true;
      events.push("waiter-resolved");
    });
    const secondDrain = real.worker.tick(sessionId);
    await secondFetchStarted;
    const stop = real.worker.stop();
    let stopSettled = false;
    void stop.then(() => {
      stopSettled = true;
    });
    await Promise.resolve();

    expect(waiterSettled).toBe(false);
    expect(stopSettled).toBe(false);
    expect(requests.map((request) => request.messages.map((message) => message.entry_id))).toEqual([
      [source.id],
      [source.id],
    ]);

    resolveSecondFetch(
      new Response(JSON.stringify({ action: "reply", content: "answer" }), { status: 200 }),
    );
    await expect(secondDrain).resolves.toMatchObject({ status: "drained", drained: 1 });
    await stop;
    const answered = await waiter.promise;
    expect(answered).toMatchObject({ status: "answered", reply: "answer" });
    if (answered.status !== "answered") {
      throw new Error("expected an answered waiter result");
    }
    expect(events).toEqual(["durable-commit", "waiter-resolved"]);
    expect(real.ingest).toHaveBeenCalledWith(sessionId, {
      answeredWindow: {
        responseTo: {
          kind: "stream_backlog",
          from_cursor_exclusive: null,
          through_cursor_inclusive: { ts: source.timestamp, entryId: source.id },
          source_entry_ids: [source.id],
          count: 1,
        },
        terminalCursor: {
          ts: real.entryIndex.lookup(answered.terminal_id)!.timestamp,
          entryId: answered.terminal_id,
        },
      },
    });

    await expect(real.worker.tick(sessionId)).resolves.toMatchObject({ status: "empty" });
    expect(fetchFn).toHaveBeenCalledTimes(2);
  });

  it("resolves waiters when real reconciliation discovers a previously committed stamp", async () => {
    const waiters = new ResponseWaiterRegistry();
    let real!: ReturnType<typeof openRealHarness>;
    const onReconcileAdvance = vi.fn(
      (event: {
        sessionId: SessionId;
        advancedThrough: { ts: number; entryId: ReturnType<typeof createStreamEntryId> };
      }) => {
        const found = real.terminal.findTerminalCoveringEntry({
          sessionId: event.sessionId,
          entryId: event.advancedThrough.entryId,
        });
        if (found.status === "found") {
          waiters.resolveTerminal("tenant", found.terminalEntry);
        }
      },
    );
    real = openRealHarness({
      fetchFn: vi.fn() as unknown as typeof fetch,
      onReconcileAdvance,
    });
    const sessionId = createSessionId();
    const source = await real.appendInbox(sessionId, 900, "already answered");
    const writer = new StreamWriter({
      dataDir: real.dataDir,
      sessionId,
      entryIndex: real.entryIndex,
      clock: real.clock,
    });
    try {
      await writer.append({
        kind: "agent_observed",
        content: { reason: "committed before crash" },
        response_to: {
          kind: "stream_backlog",
          from_cursor_exclusive: null,
          through_cursor_inclusive: { ts: source.timestamp, entryId: source.id },
          source_entry_ids: [source.id],
          count: 1,
        },
      });
    } finally {
      writer.close();
    }
    const waiter = waiters.register({
      tenant: "tenant",
      sessionId,
      entryId: source.id,
      timeoutMs: 1_000,
    });

    await expect(real.worker.tick(sessionId)).resolves.toMatchObject({ status: "empty" });
    await expect(waiter.promise).resolves.toMatchObject({ status: "observed" });
    expect(onReconcileAdvance).toHaveBeenCalledWith({
      sessionId,
      advancedThrough: { ts: source.timestamp, entryId: source.id },
    });
  });

  it("turns a silent response into an observed terminal", async () => {
    const h = harness(
      async () =>
        new Response(JSON.stringify({ action: "silent", reason: "not addressed" }), {
          status: 200,
        }),
    );
    await h.runner.run(h.input);
    expect(h.appendBacklogTerminal).toHaveBeenCalledWith(
      expect.objectContaining({
        terminal: { kind: "agent_observed", reason: "not addressed" },
      }),
    );
  });

  it("seals a prefix with any legacy entry instead of repeatedly rejecting its metadata", async () => {
    const h = harness(
      async () =>
        new Response(JSON.stringify({ action: "reply", content: "unused" }), { status: 200 }),
    );
    delete h.sourceEntry.metadata;

    await h.runner.run(h.input);

    expect(h.terminal.sealBacklogPrefix).toHaveBeenCalledWith({
      sessionId: h.input.sessionId,
      sourceEntryIds: h.input.inboundBatch.entryIds,
      reason: "Legacy inbox backlog sealed because transport metadata is unavailable",
    });
    expect(h.fetchFn).not.toHaveBeenCalled();
  });

  it("sends a real mixed stale/fresh/stale prefix to Team Agent whole", async () => {
    const fetchFn = vi.fn(
      async () =>
        new Response(JSON.stringify({ action: "reply", content: "mixed answer" }), {
          status: 200,
        }),
    ) as unknown as typeof fetch;
    const real = openRealHarness({
      fetchFn,
      clock: new ManualClock(1_000),
      staleMs: 500,
    });
    const sealStale = vi.spyOn(real.terminal, "sealStaleBacklog");
    const sessionId = createSessionId();
    const staleFirst = await real.appendInbox(sessionId, 100, "stale first");
    const fresh = await real.appendInbox(sessionId, 900, "fresh middle");
    const staleLast = await real.appendInbox(sessionId, 100, "stale last");

    await expect(real.worker.tick(sessionId)).resolves.toMatchObject({
      status: "drained",
      drained: 3,
    });

    const request = JSON.parse(
      String((fetchFn as ReturnType<typeof vi.fn>).mock.calls[0]?.[1]?.body),
    );
    expect(request.messages.map((message: { entry_id: string }) => message.entry_id)).toEqual([
      staleFirst.id,
      fresh.id,
      staleLast.id,
    ]);
    expect(sealStale).not.toHaveBeenCalled();
    expect(real.coordinator.getWatermark(sessionId)).toEqual({
      ts: staleLast.timestamp,
      entryId: staleLast.id,
    });
  });

  it("seals a 4xx response as observed", async () => {
    const h = harness(async () => new Response("bad request", { status: 422 }));
    await h.runner.run(h.input);
    expect(h.appendBacklogTerminal).toHaveBeenCalledWith(
      expect.objectContaining({
        terminal: {
          kind: "agent_observed",
          reason: "Team Agent rejected inbox batch with HTTP 422",
        },
      }),
    );
  });

  it("throws on 5xx without stamping", async () => {
    const h = harness(async () => new Response("failed", { status: 503 }));
    await expect(h.runner.run(h.input)).rejects.toThrow("HTTP 503");
    expect(h.appendBacklogTerminal).not.toHaveBeenCalled();
  });

  it("throws on a malformed 2xx response without stamping", async () => {
    const h = harness(
      async () => new Response(JSON.stringify({ action: "reply" }), { status: 200 }),
    );
    await expect(h.runner.run(h.input)).rejects.toThrow("invalid 2xx response");
    expect(h.appendBacklogTerminal).not.toHaveBeenCalled();
  });
});
