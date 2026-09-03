import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import {
  DEFAULT_SESSION_ID,
  StreamEntryIndexRepository,
  StreamReader,
  StreamWatermarkRepository,
  StreamWriter,
  streamEntryIndexMigrations,
  streamWatermarkMigrations,
  type StreamEntry,
} from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { createSessionId, type SessionId } from "../../util/ids.js";
import { ResponseWaiterRegistry } from "../../sidecar/response-waiter-registry.js";
import { BacklogTerminalService, ChatResponseWatermarkCoordinator } from "./index.js";

describe("BacklogTerminalService", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop()!, { recursive: true, force: true });
    }
  });

  function openHarness() {
    const dataDir = mkdtempSync(join(tmpdir(), "borg-backlog-terminal-"));
    tempDirs.push(dataDir);
    const db = openDatabase(join(dataDir, "borg.db"), {
      migrations: composeMigrations(streamEntryIndexMigrations, streamWatermarkMigrations),
    });
    const entryIndex = new StreamEntryIndexRepository({ db, dataDir });
    const clock = new ManualClock(100);
    const coordinator = new ChatResponseWatermarkCoordinator({
      watermarkRepository: new StreamWatermarkRepository({ db, clock }),
      entryIndex,
      logger: { warn() {} },
    });
    const ingest = vi.fn(async () => ({ ran: true, processedEntries: 0 }));
    const writer = (sessionId: SessionId) =>
      new StreamWriter({ dataDir, sessionId, entryIndex, clock });
    const service = new BacklogTerminalService({
      dataDir,
      entryIndex,
      createStreamReader: (sessionId) => new StreamReader({ dataDir, sessionId, entryIndex }),
      createStreamWriter: writer,
      coordinator,
      streamIngestionCoordinator: { ingest } as never,
    });
    return { dataDir, db, entryIndex, clock, coordinator, ingest, service, writer };
  }

  async function appendUser(
    harness: ReturnType<typeof openHarness>,
    sessionId: SessionId,
    observedAt: number,
    receiptPending = false,
  ) {
    const writer = harness.writer(sessionId);
    try {
      return await writer.append({
        kind: "user_msg",
        content: `message-${observedAt}`,
        observed_at: observedAt,
        ...(receiptPending ? { receipt_pending: true } : {}),
      });
    } finally {
      writer.close();
    }
  }

  it("appends one stamped terminal, advances the watermark, and finds it by covered entry", async () => {
    const harness = openHarness();
    try {
      const first = await appendUser(harness, DEFAULT_SESSION_ID, 10);
      const second = await appendUser(harness, DEFAULT_SESSION_ID, 20);
      const result = await harness.service.appendBacklogTerminal({
        sessionId: DEFAULT_SESSION_ID,
        sourceEntryIds: [first.id, second.id],
        terminal: { kind: "agent_msg", content: "reply" },
      });

      expect(result.terminalEntry).toMatchObject({
        kind: "agent_msg",
        content: "reply",
        response_to: {
          source_entry_ids: [first.id, second.id],
          count: 2,
        },
      });
      expect(harness.coordinator.getWatermark(DEFAULT_SESSION_ID)).toEqual({
        ts: second.timestamp,
        entryId: second.id,
      });
      expect(
        harness.service.findTerminalCoveringEntry({
          sessionId: DEFAULT_SESSION_ID,
          entryId: first.id,
        }),
      ).toMatchObject({ status: "found", terminalEntry: { id: result.terminalEntry.id } });
      expect(harness.ingest).toHaveBeenCalledWith(
        DEFAULT_SESSION_ID,
        expect.objectContaining({ answeredWindow: expect.any(Object) }),
      );
    } finally {
      harness.db.close();
    }
  });

  it("rejects a second terminal stamp for an already covered batch", async () => {
    const harness = openHarness();
    try {
      const entry = await appendUser(harness, DEFAULT_SESSION_ID, 10);
      await harness.service.appendBacklogTerminal({
        sessionId: DEFAULT_SESSION_ID,
        sourceEntryIds: [entry.id],
        terminal: { kind: "agent_observed", reason: "silent" },
      });
      await expect(
        harness.service.appendBacklogTerminal({
          sessionId: DEFAULT_SESSION_ID,
          sourceEntryIds: [entry.id],
          terminal: { kind: "agent_msg", content: "duplicate" },
        }),
      ).rejects.toMatchObject({ code: "INBOUND_BATCH_ALREADY_RESPONDED" });
    } finally {
      harness.db.close();
    }
  });

  it("seals the uncapped stale prefix but stops before fresh and receipt-pending entries", async () => {
    const harness = openHarness();
    const freshSession = createSessionId();
    const receiptSession = createSessionId();
    try {
      const stale1 = await appendUser(harness, freshSession, 10);
      const stale2 = await appendUser(harness, freshSession, 20);
      await appendUser(harness, freshSession, 100);
      await appendUser(harness, freshSession, 5);
      const freshSeal = await harness.service.sealStaleBacklog({
        sessionId: freshSession,
        staleBefore: 50,
      });
      expect(freshSeal?.responseTo.source_entry_ids).toEqual([stale1.id, stale2.id]);

      const beforeReceipt = await appendUser(harness, receiptSession, 10);
      await appendUser(harness, receiptSession, 20, true);
      await appendUser(harness, receiptSession, 30);
      const receiptSeal = await harness.service.sealStaleBacklog({
        sessionId: receiptSession,
        staleBefore: 50,
      });
      expect(receiptSeal?.responseTo.source_entry_ids).toEqual([beforeReceipt.id]);
      expect(harness.ingest).toHaveBeenCalledWith(freshSession);
      expect(harness.ingest).toHaveBeenCalledWith(receiptSession);
    } finally {
      harness.db.close();
    }
  });

  it("seals the whole pending prefix with ordinary catch-up ingestion", async () => {
    const harness = openHarness();
    try {
      const first = await appendUser(harness, DEFAULT_SESSION_ID, 10);
      const second = await appendUser(harness, DEFAULT_SESSION_ID, 20);
      const result = await harness.service.sealPendingBacklog({
        sessionId: DEFAULT_SESSION_ID,
        reason: "claimed",
      });

      expect(result?.responseTo.source_entry_ids).toEqual([first.id, second.id]);
      expect(result?.terminalEntry).toMatchObject({
        kind: "agent_observed",
        content: expect.objectContaining({ reason: "claimed" }),
      });
      expect(harness.ingest).toHaveBeenCalledWith(DEFAULT_SESSION_ID);
    } finally {
      harness.db.close();
    }
  });

  it("notifies a committed terminal even when watermark advancement fails afterwards", async () => {
    const harness = openHarness();
    const waiters = new ResponseWaiterRegistry();
    const onTerminalCommitted = vi.fn((entry: StreamEntry) => {
      expect(harness.entryIndex.lookup(entry.id)).not.toBeNull();
      waiters.resolveTerminal("tenant", entry);
    });
    const failingCoordinator = {
      getWatermark: harness.coordinator.getWatermark.bind(harness.coordinator),
      reconcile: harness.coordinator.reconcile.bind(harness.coordinator),
      findTerminalStampForBatch: harness.coordinator.findTerminalStampForBatch.bind(
        harness.coordinator,
      ),
      advanceThrough: vi.fn(() => {
        throw new Error("advance failed");
      }),
    };
    const service = new BacklogTerminalService({
      dataDir: harness.dataDir,
      entryIndex: harness.entryIndex,
      createStreamReader: (sessionId) =>
        new StreamReader({
          dataDir: harness.dataDir,
          sessionId,
          entryIndex: harness.entryIndex,
        }),
      createStreamWriter: harness.writer,
      coordinator: failingCoordinator as never,
      onTerminalCommitted,
    });
    try {
      const source = await appendUser(harness, DEFAULT_SESSION_ID, 10);
      const waiter = waiters.register({
        tenant: "tenant",
        sessionId: DEFAULT_SESSION_ID,
        entryId: source.id,
        timeoutMs: 1_000,
      });
      await expect(
        service.appendBacklogTerminal({
          sessionId: DEFAULT_SESSION_ID,
          sourceEntryIds: [source.id],
          terminal: { kind: "agent_msg", content: "committed" },
        }),
      ).rejects.toThrow("advance failed");

      expect(onTerminalCommitted).toHaveBeenCalledTimes(1);
      await expect(waiter.promise).resolves.toMatchObject({
        status: "answered",
        reply: "committed",
      });
      const committed = onTerminalCommitted.mock.calls[0]?.[0] as StreamEntry;
      expect(harness.entryIndex.lookup(committed.id)).not.toBeNull();
      expect(
        new StreamReader({
          dataDir: harness.dataDir,
          sessionId: DEFAULT_SESSION_ID,
          entryIndex: harness.entryIndex,
        }).tail(1),
      ).toEqual([committed]);
    } finally {
      harness.db.close();
    }
  });

  it("distinguishes unknown, wrong-session, and pending entries", async () => {
    const harness = openHarness();
    const otherSession = createSessionId();
    try {
      const entry = await appendUser(harness, DEFAULT_SESSION_ID, 10);
      expect(
        harness.service.findTerminalCoveringEntry({
          sessionId: DEFAULT_SESSION_ID,
          entryId: entry.id,
        }),
      ).toEqual({ status: "pending" });
      expect(
        harness.service.findTerminalCoveringEntry({
          sessionId: otherSession,
          entryId: entry.id,
        }),
      ).toEqual({ status: "session_mismatch" });
      expect(
        harness.service.findTerminalCoveringEntry({
          sessionId: DEFAULT_SESSION_ID,
          entryId: "se_missing" as typeof entry.id,
        }),
      ).toEqual({ status: "unknown_entry" });
    } finally {
      harness.db.close();
    }
  });
});
