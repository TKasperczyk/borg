import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  AttachmentRepository,
  attachmentMigrations,
  type StoredAttachmentRecord,
} from "../../attachments/index.js";
import { backfillSessionStreamEntryIndexAndAttachments } from "../../borg/reconciliation.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import {
  StreamEntryIndexRepository,
  StreamWriter,
  streamEntryIndexMigrations,
  type StreamCursor,
  type StreamEntry,
} from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { SessionBusyError, StreamError } from "../../util/errors.js";
import {
  DEFAULT_SESSION_ID,
  createAttachmentId,
  createSessionId,
  createStreamEntryId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { TurnResult } from "../turn-orchestrator.js";

import type { BacklogPrefixResult } from "./backlog-prefix.js";
import {
  ChatResponseCatchUpWorker,
  TurnOrchestratorChatResponseCatchUpRunner,
  type ChatResponseCatchUpWorkerConfig,
} from "./index.js";

const DEFAULT_CONFIG: ChatResponseCatchUpWorkerConfig = {
  quietWindowMs: 10,
  maxWaitMs: 100,
  backoffBaseMs: 25,
  maxBackoffMs: 100,
};
const tempDirs: string[] = [];

function turnResult(): TurnResult {
  return {
    turn_id: "turn-test",
    mode: "problem_solving",
    path: "system_2",
    response: "ok",
    emitted: true,
    emission: { kind: "text", content: "ok" },
    thoughts: [],
    usage: {
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "end_turn",
    },
    retrievedEpisodeIds: [],
    referencedEpisodeIds: [],
    intents: [],
    toolCalls: [],
  } as unknown as TurnResult;
}

function cursor(entryId: StreamEntryId, ts = 1): StreamCursor {
  return {
    entryId,
    ts,
  };
}

function emptyPrefix(): BacklogPrefixResult {
  return {
    fromCursorExclusive: null,
    entryIds: [],
    throughCursorInclusive: null,
    includedCount: 0,
    remainingCount: 0,
    hasMore: false,
    estimatedTokens: 0,
    estimatedChars: 0,
  };
}

function prefix(input: { count?: number; hasMore?: boolean } = {}): BacklogPrefixResult {
  const count = input.count ?? 1;
  const entryIds = Array.from({ length: count }, () => createStreamEntryId());
  const throughEntryId = entryIds[entryIds.length - 1];

  if (throughEntryId === undefined) {
    return emptyPrefix();
  }

  return {
    fromCursorExclusive: null,
    entryIds,
    throughCursorInclusive: cursor(throughEntryId),
    includedCount: count,
    remainingCount: input.hasMore === true ? 1 : 0,
    hasMore: input.hasMore ?? false,
    estimatedTokens: count,
    estimatedChars: count,
  };
}

function entry(
  input: {
    kind?: StreamEntry["kind"];
    sessionId?: SessionId;
    timestamp?: number;
    turnId?: string;
  } = {},
): StreamEntry {
  return {
    id: createStreamEntryId(),
    timestamp: input.timestamp ?? 0,
    kind: input.kind ?? "user_msg",
    content: "test",
    session_id: input.sessionId ?? DEFAULT_SESSION_ID,
    sender_entity_id: null,
    reply_target_entity_id: null,
    compressed: false,
    ...(input.turnId === undefined ? {} : { turn_id: input.turnId }),
  };
}

function deferred<T = void>() {
  let resolve: (value: T | PromiseLike<T>) => void = () => {};
  let reject: (reason?: unknown) => void = () => {};
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return {
    promise,
    resolve,
    reject,
  };
}

function createHarness(
  options: {
    config?: Partial<ChatResponseCatchUpWorkerConfig>;
    pendingSessionIds?: readonly SessionId[];
    prefixes?: readonly BacklogPrefixResult[];
    build?: () => Promise<BacklogPrefixResult>;
    run?: (input: unknown) => Promise<TurnResult>;
    backfill?: (sessionId: SessionId) => Promise<{ inserted: number }>;
    sessionPredicate?: (sessionId: SessionId) => boolean;
    stampAfterRun?: boolean;
    customRunner?: (input: unknown) => Promise<void>;
    watermarkAfterRun?: (input: unknown, runCount: number) => StreamCursor | null;
    onReconcileAdvance?: (event: { sessionId: SessionId; advancedThrough: StreamCursor }) => void;
    acquireLease?: () => { release(): void };
    reconcileAdvance?: StreamCursor;
  } = {},
) {
  const clock = new ManualClock(0);
  const prefixQueue = [...(options.prefixes ?? [])];
  const pendingSessionIds = [...(options.pendingSessionIds ?? [])];
  let durableWatermark: StreamCursor | null = null;
  let reconcileAdvance = options.reconcileAdvance ?? null;
  const coordinator = {
    reconcile: vi.fn(() => {
      const advancedThrough = reconcileAdvance;
      if (advancedThrough !== null) {
        durableWatermark = advancedThrough;
        reconcileAdvance = null;
      }
      return {
        watermark: durableWatermark,
        advancedThrough,
        appliedStamps: advancedThrough === null ? 0 : 1,
      };
    }),
    compareCursors: vi.fn((_sessionId: SessionId, left: StreamCursor, right: StreamCursor) =>
      left.entryId === right.entryId && left.ts === right.ts ? 0 : -1,
    ),
  };
  const prefixBuilder = {
    build: vi.fn(options.build ?? (async () => prefixQueue.shift() ?? emptyPrefix())),
  };
  const entryIndex = {
    listSessionIdsWithPendingResponseBacklog: vi.fn(() => pendingSessionIds),
  };
  const repairSessionStreamEntryIndex = vi.fn(options.backfill ?? (async () => ({ inserted: 0 })));
  const underlyingRun = options.run ?? (async () => turnResult());
  let completedRuns = 0;
  const turnOrchestrator = {
    run: vi.fn(async (input: unknown) => {
      const result = await underlyingRun(input);
      if (options.stampAfterRun !== false) {
        completedRuns += 1;
        durableWatermark =
          options.watermarkAfterRun?.(input, completedRuns) ??
          (input as { inboundBatch: { throughCursorInclusive: StreamCursor } }).inboundBatch
            .throughCursorInclusive;
      }
      return result;
    }),
  };
  const customRunner =
    options.customRunner === undefined
      ? undefined
      : {
          run: vi.fn(async (input: unknown) => {
            await options.customRunner!(input);
            if (options.stampAfterRun !== false) {
              completedRuns += 1;
              durableWatermark =
                options.watermarkAfterRun?.(input, completedRuns) ??
                (input as { inboundBatch: { throughCursorInclusive: StreamCursor } }).inboundBatch
                  .throughCursorInclusive;
            }
          }),
        };
  const worker = new ChatResponseCatchUpWorker({
    coordinator,
    prefixBuilder,
    entryIndex,
    repairSessionStreamEntryIndex,
    ...(customRunner === undefined ? { turnOrchestrator } : { runner: customRunner }),
    ...(options.sessionPredicate === undefined
      ? {}
      : { sessionPredicate: options.sessionPredicate }),
    ...(options.onReconcileAdvance === undefined
      ? {}
      : { onReconcileAdvance: options.onReconcileAdvance }),
    ...(options.acquireLease === undefined ? {} : { acquireLease: options.acquireLease }),
    clock,
    setTimeoutFn: (callback, delayMs) => setTimeout(callback, delayMs),
    clearTimeoutFn: (handle) => clearTimeout(handle),
    config: {
      ...DEFAULT_CONFIG,
      ...options.config,
    },
  });

  return {
    clock,
    coordinator,
    prefixBuilder,
    entryIndex,
    repairSessionStreamEntryIndex,
    turnOrchestrator,
    customRunner,
    worker,
  };
}

async function flushAsync(): Promise<void> {
  await Promise.resolve();
  await Promise.resolve();
}

async function advance(clock: ManualClock, ms: number): Promise<void> {
  clock.advance(ms);
  await vi.advanceTimersByTimeAsync(ms);
  await flushAsync();
}

async function runPendingTimers(): Promise<void> {
  await vi.runOnlyPendingTimersAsync();
  await flushAsync();
}

describe("ChatResponseCatchUpWorker", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("wakes only daemon-enqueued user messages without turn ids", async () => {
    const otherSession = createSessionId();
    const harness = createHarness({
      prefixes: [prefix()],
    });

    harness.worker.start();
    await flushAsync();

    harness.worker.onAppend([
      entry({ kind: "agent_msg", sessionId: otherSession }),
      entry({ kind: "internal_event", sessionId: otherSession }),
      entry({ kind: "user_msg", sessionId: otherSession, turnId: "turn-direct" }),
      entry({ kind: "user_msg", sessionId: DEFAULT_SESSION_ID }),
    ]);

    await advance(harness.clock, DEFAULT_CONFIG.quietWindowMs);

    expect(harness.prefixBuilder.build).toHaveBeenCalledTimes(1);
    expect(harness.prefixBuilder.build).toHaveBeenCalledWith({
      sessionId: DEFAULT_SESSION_ID,
      fromCursorExclusive: null,
    });
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);
  });

  it("does not wake from append before start", async () => {
    const harness = createHarness({
      prefixes: [prefix()],
    });

    harness.worker.onAppend([entry({ kind: "user_msg" })]);
    await advance(harness.clock, DEFAULT_CONFIG.maxWaitMs);

    expect(harness.prefixBuilder.build).not.toHaveBeenCalled();
    expect(harness.turnOrchestrator.run).not.toHaveBeenCalled();
  });

  it("drains append-triggered backlog immediately when quietWindowMs is zero", async () => {
    const harness = createHarness({
      config: {
        quietWindowMs: 0,
      },
      prefixes: [prefix()],
    });

    harness.worker.start();
    await flushAsync();
    harness.worker.onAppend([entry({ kind: "user_msg" })]);
    await advance(harness.clock, 0);

    expect(harness.prefixBuilder.build).toHaveBeenCalledTimes(1);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);
  });

  it("startup scan finds backlog and drains immediately", async () => {
    const harness = createHarness({
      pendingSessionIds: [DEFAULT_SESSION_ID],
      prefixes: [prefix(), prefix()],
    });

    harness.worker.start();
    await flushAsync();
    await runPendingTimers();

    expect(harness.entryIndex.listSessionIdsWithPendingResponseBacklog).toHaveBeenCalledTimes(1);
    expect(harness.coordinator.reconcile).toHaveBeenCalledWith(DEFAULT_SESSION_ID);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);
  });

  it("startup scan drains a pending session outside the default active-session list cap", async () => {
    const activeSessionIds = Array.from({ length: 101 }, () => createSessionId());
    const cappedOutPendingSessionId = activeSessionIds[100]!;
    const harness = createHarness({
      pendingSessionIds: [cappedOutPendingSessionId],
      prefixes: [prefix(), prefix()],
    });

    harness.worker.start();
    await flushAsync();
    await runPendingTimers();

    expect(harness.entryIndex.listSessionIdsWithPendingResponseBacklog).toHaveBeenCalledTimes(1);
    expect(harness.coordinator.reconcile).toHaveBeenCalledWith(cappedOutPendingSessionId);
    expect(harness.prefixBuilder.build).toHaveBeenCalledWith({
      sessionId: cappedOutPendingSessionId,
      fromCursorExclusive: null,
    });
    expect(harness.turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: cappedOutPendingSessionId,
        inboundBatch: expect.objectContaining({
          kind: "stream_backlog",
        }),
      }),
    );
  });

  it("serializes drains for a session when wakes arrive during an in-flight turn", async () => {
    const firstRun = deferred();
    let activeRuns = 0;
    let maxActiveRuns = 0;
    let runCount = 0;
    const runStarted = deferred();
    const harness = createHarness({
      config: {
        quietWindowMs: 0,
      },
      prefixes: [prefix(), prefix()],
      run: async () => {
        runCount += 1;
        activeRuns += 1;
        maxActiveRuns = Math.max(maxActiveRuns, activeRuns);

        if (runCount === 1) {
          runStarted.resolve();
          await firstRun.promise;
        }

        activeRuns -= 1;
        return turnResult();
      },
    });

    harness.worker.start();
    await flushAsync();
    harness.worker.onAppend([entry()]);
    await advance(harness.clock, 0);
    await runStarted.promise;

    harness.worker.onAppend([entry()]);
    await advance(harness.clock, 0);

    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);

    firstRun.resolve();
    await flushAsync();
    await advance(harness.clock, 0);

    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(2);
    expect(maxActiveRuns).toBe(1);
  });

  it("does not lose a wake that arrives while prefix build is pending", async () => {
    const firstBuild = deferred<BacklogPrefixResult>();
    let buildCount = 0;
    const harness = createHarness({
      config: {
        quietWindowMs: 0,
      },
      build: async () => {
        buildCount += 1;
        return buildCount === 1 ? firstBuild.promise : prefix();
      },
    });

    harness.worker.start();
    await flushAsync();
    harness.worker.onAppend([entry({ timestamp: 0 })]);
    await advance(harness.clock, 0);

    expect(harness.prefixBuilder.build).toHaveBeenCalledTimes(1);

    harness.worker.onAppend([entry({ timestamp: 1 })]);
    await advance(harness.clock, 0);

    expect(harness.prefixBuilder.build).toHaveBeenCalledTimes(1);

    firstBuild.resolve(prefix({ hasMore: false }));
    await flushAsync();
    await advance(harness.clock, 0);

    expect(harness.coordinator.reconcile).toHaveBeenCalledTimes(4);
    expect(harness.prefixBuilder.build).toHaveBeenCalledTimes(2);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(2);
  });

  it("batches a quiet-window burst into one drain", async () => {
    const harness = createHarness({
      config: {
        quietWindowMs: 50,
        maxWaitMs: 1_000,
      },
      prefixes: [prefix({ count: 3 })],
    });

    harness.worker.start();
    await flushAsync();
    harness.worker.onAppend([entry({ timestamp: 0 })]);
    await advance(harness.clock, 20);
    harness.worker.onAppend([entry({ timestamp: 20 })]);
    await advance(harness.clock, 20);
    harness.worker.onAppend([entry({ timestamp: 40 })]);
    await advance(harness.clock, 49);

    expect(harness.turnOrchestrator.run).not.toHaveBeenCalled();

    await advance(harness.clock, 1);

    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledWith(
      expect.objectContaining({
        inboundBatch: expect.objectContaining({
          kind: "stream_backlog",
          entryIds: expect.arrayContaining([
            expect.any(String),
            expect.any(String),
            expect.any(String),
          ]),
        }),
      }),
    );
  });

  it("drains a never-quiet session at oldest pending plus maxWait", async () => {
    const harness = createHarness({
      config: {
        quietWindowMs: 50,
        maxWaitMs: 120,
      },
      prefixes: [prefix()],
    });

    harness.worker.start();
    await flushAsync();
    harness.worker.onAppend([entry({ timestamp: 0 })]);
    await advance(harness.clock, 40);
    harness.worker.onAppend([entry({ timestamp: 40 })]);
    await advance(harness.clock, 40);
    harness.worker.onAppend([entry({ timestamp: 80 })]);
    await advance(harness.clock, 39);
    harness.worker.onAppend([entry({ timestamp: 119 })]);

    expect(harness.turnOrchestrator.run).not.toHaveBeenCalled();

    await advance(harness.clock, 1);

    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);
  });

  it("backs off exponentially after SESSION_TURN_BUSY and retries", async () => {
    let runCount = 0;
    const harness = createHarness({
      config: {
        quietWindowMs: 0,
        backoffBaseMs: 100,
        maxBackoffMs: 250,
      },
      prefixes: [prefix(), prefix(), prefix()],
      run: async () => {
        runCount += 1;

        if (runCount <= 2) {
          throw new SessionBusyError("busy", {
            code: "SESSION_TURN_BUSY",
          });
        }

        return turnResult();
      },
    });

    harness.worker.start();
    await flushAsync();
    harness.worker.onAppend([entry()]);
    await advance(harness.clock, 0);

    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);

    await advance(harness.clock, 99);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);

    await advance(harness.clock, 1);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(2);

    await advance(harness.clock, 199);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(2);

    await advance(harness.clock, 1);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(3);
  });

  it("uses a repair-only retry after a poisoned stream index error", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const poison = new StreamError("committed stream append is not indexed", {
      code: "STREAM_INDEX_POISONED",
    });
    const harness = createHarness({
      prefixes: [prefix(), emptyPrefix()],
      run: async () => {
        throw poison;
      },
    });

    await expect(harness.worker.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "error",
      drained: 0,
      hasMore: true,
    });
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);
    expect(harness.repairSessionStreamEntryIndex).not.toHaveBeenCalled();

    await expect(harness.worker.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "drained",
      drained: 0,
      hasMore: true,
    });
    expect(harness.repairSessionStreamEntryIndex).toHaveBeenCalledTimes(1);
    expect(harness.prefixBuilder.build).toHaveBeenCalledTimes(1);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);

    await expect(harness.worker.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "empty",
      drained: 0,
      hasMore: false,
    });
    expect(harness.prefixBuilder.build).toHaveBeenCalledTimes(2);
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);
  });

  it("repair-only retry backfills and reconciles committed image attachment stream entries", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const tempDir = mkdtempSync(join(tmpdir(), "borg-worker-repair-attachment-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(streamEntryIndexMigrations, attachmentMigrations),
    });
    const entryIndex = new StreamEntryIndexRepository({
      db,
      dataDir: tempDir,
    });
    const attachmentRepository = new AttachmentRepository(db);
    const clock = new ManualClock(1_000);
    const indexedWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
      entryIndex,
    });
    const parentEntry = await indexedWriter.append({
      kind: "user_msg",
      content: "parent",
    });
    indexedWriter.close();
    const attachmentId = createAttachmentId();
    const attachment: StoredAttachmentRecord = {
      attachment_id: attachmentId,
      sha256: "0".repeat(64),
      media_type: "image/gif",
      byte_size: 10,
      width: 1,
      height: 1,
      storage_ref: "attachments/test.gif",
      thumbnail_ref: null,
      perception_id: null,
      text_embedding_ref: null,
      visual_embedding_ref: null,
      active: false,
      audience: null,
      audience_entity_id: null,
      created_turn_global: null,
      parent_entry_id: parentEntry.id,
      stream_entry_id: null,
      parent_turn_id: null,
      created_at: parentEntry.timestamp,
    };
    attachmentRepository.insert(attachment);
    const unindexedWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: DEFAULT_SESSION_ID,
      clock,
    });
    const imageEntry = await unindexedWriter.append({
      kind: "user_image_attachment",
      content: {
        type: "image_ref",
        attachment_id: attachmentId,
        media_type: "image/gif",
        parent_entry_id: parentEntry.id,
      },
    });
    unindexedWriter.close();
    const poison = new StreamError("committed image attachment append is not indexed", {
      code: "STREAM_INDEX_POISONED",
    });
    const harness = createHarness({
      prefixes: [prefix()],
      run: async () => {
        throw poison;
      },
      backfill: (sessionId) =>
        backfillSessionStreamEntryIndexAndAttachments({
          dataDir: tempDir,
          sessionId,
          entryIndex,
          attachmentRepository,
        }),
    });

    try {
      expect(entryIndex.lookup(imageEntry.id)).toBeNull();
      expect(attachmentRepository.get(attachmentId)).toMatchObject({
        stream_entry_id: null,
        active: false,
      });

      await expect(harness.worker.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
        status: "error",
        drained: 0,
        hasMore: true,
      });
      await expect(harness.worker.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
        status: "drained",
        drained: 0,
        hasMore: true,
      });

      expect(harness.repairSessionStreamEntryIndex).toHaveBeenCalledTimes(1);
      expect(entryIndex.lookup(imageEntry.id)).not.toBeNull();
      expect(attachmentRepository.get(attachmentId)).toMatchObject({
        stream_entry_id: imageEntry.id,
        active: true,
      });

      await backfillSessionStreamEntryIndexAndAttachments({
        dataDir: tempDir,
        sessionId: DEFAULT_SESSION_ID,
        entryIndex,
        attachmentRepository,
      });
      expect(attachmentRepository.listByParentEntry(parentEntry.id)).toHaveLength(1);
      expect(attachmentRepository.get(attachmentId)).toMatchObject({
        stream_entry_id: imageEntry.id,
        active: true,
      });
    } finally {
      db.close();
    }
  });

  it("immediately re-drains when the prefix reports hasMore", async () => {
    const harness = createHarness({
      config: {
        quietWindowMs: 50,
      },
      prefixes: [prefix({ hasMore: true }), prefix()],
    });

    harness.worker.start();
    await flushAsync();
    harness.worker.onAppend([entry()]);
    await advance(harness.clock, 50);
    await runPendingTimers();

    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(2);
  });

  it("coalesces mid-turn arrivals into the next drain", async () => {
    const firstRun = deferred();
    const runStarted = deferred();
    let runCount = 0;
    const harness = createHarness({
      config: {
        quietWindowMs: 50,
        maxWaitMs: 500,
      },
      prefixes: [prefix(), prefix()],
      run: async () => {
        runCount += 1;

        if (runCount === 1) {
          runStarted.resolve();
          await firstRun.promise;
        }

        return turnResult();
      },
    });

    harness.worker.start();
    await flushAsync();
    harness.worker.onAppend([entry({ timestamp: 0 })]);
    await advance(harness.clock, 50);
    await runStarted.promise;

    harness.worker.onAppend([entry({ timestamp: 50 })]);
    firstRun.resolve();
    await flushAsync();
    await advance(harness.clock, 49);

    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);

    await advance(harness.clock, 1);

    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(2);
  });

  it("applies the session predicate at startup, append, pending notification, and drain", async () => {
    const startupSession = createSessionId();
    const predicate = vi.fn(() => false);
    const startup = createHarness({
      pendingSessionIds: [startupSession],
      sessionPredicate: predicate,
    });
    startup.worker.start();
    await flushAsync();
    expect(predicate).toHaveBeenCalledWith(startupSession);
    expect(startup.coordinator.reconcile).not.toHaveBeenCalled();

    startup.worker.onAppend([entry({ sessionId: DEFAULT_SESSION_ID })]);
    startup.worker.onPendingSession(DEFAULT_SESSION_ID, 0);
    await advance(startup.clock, DEFAULT_CONFIG.maxWaitMs);
    expect(predicate).toHaveBeenCalledWith(DEFAULT_SESSION_ID);
    expect(startup.prefixBuilder.build).not.toHaveBeenCalled();

    let accepted = true;
    const beforeDrain = createHarness({
      prefixes: [prefix()],
      sessionPredicate: () => accepted,
    });
    beforeDrain.worker.start();
    await flushAsync();
    beforeDrain.worker.onPendingSession(DEFAULT_SESSION_ID, 0);
    accepted = false;
    await expect(beforeDrain.worker.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "empty",
      drained: 0,
    });
    expect(beforeDrain.prefixBuilder.build).not.toHaveBeenCalled();
  });

  it("treats a runner return without a covering durable stamp as an error", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    const harness = createHarness({ prefixes: [prefix()], stampAfterRun: false });

    await expect(harness.worker.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "error",
      drained: 0,
      hasMore: true,
      error: expect.stringContaining("did not durably cover"),
    });
    expect(harness.turnOrchestrator.run).toHaveBeenCalledTimes(1);
  });

  it("treats a short watermark advance as partial progress and immediately re-drains", async () => {
    const first = prefix({ count: 3 });
    const second = prefix({ count: 2 });
    const run = vi.fn(async (_input: unknown) => undefined);
    const harness = createHarness({
      config: { quietWindowMs: 0 },
      prefixes: [first, second],
      customRunner: run,
      watermarkAfterRun: (input, runCount) =>
        runCount === 1
          ? cursor(first.entryIds[0]!, first.throughCursorInclusive!.ts)
          : (input as { inboundBatch: { throughCursorInclusive: StreamCursor } }).inboundBatch
              .throughCursorInclusive,
    });

    harness.worker.start();
    await flushAsync();
    harness.worker.onPendingSession(DEFAULT_SESSION_ID, 0);
    await advance(harness.clock, 0);
    await runPendingTimers();

    expect(run).toHaveBeenCalledTimes(2);
    expect(run.mock.calls[0]?.[0]).toMatchObject({
      inboundBatch: { entryIds: first.entryIds },
    });
    expect(run.mock.calls[1]?.[0]).toMatchObject({
      inboundBatch: { entryIds: second.entryIds },
    });
  });

  it("notifies when reconciliation discovers and advances a durable stamp", async () => {
    const advancedThrough = cursor(createStreamEntryId(), 10);
    const onReconcileAdvance = vi.fn();
    const harness = createHarness({
      prefixes: [emptyPrefix()],
      reconcileAdvance: advancedThrough,
      onReconcileAdvance,
    });

    await expect(harness.worker.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "empty",
    });
    expect(onReconcileAdvance).toHaveBeenCalledWith({
      sessionId: DEFAULT_SESSION_ID,
      advancedThrough,
    });
  });

  it("holds a background lease while a drain is scheduled", async () => {
    const release = vi.fn();
    const acquireLease = vi.fn(() => ({ release }));
    const harness = createHarness({ acquireLease });

    harness.worker.start();
    await flushAsync();
    acquireLease.mockClear();
    release.mockClear();
    harness.worker.onPendingSession(DEFAULT_SESSION_ID, 0);

    expect(acquireLease).toHaveBeenCalledTimes(1);
    expect(release).not.toHaveBeenCalled();
    await harness.worker.stop({ graceful: false });
    expect(release).toHaveBeenCalledTimes(1);
  });

  it("keeps the default adapter's TurnOrchestrator invocation unchanged", async () => {
    const run = vi.fn(async () => turnResult());
    const adapter = new TurnOrchestratorChatResponseCatchUpRunner({ run });
    const batch = prefix();

    await adapter.run({
      sessionId: DEFAULT_SESSION_ID,
      inboundBatch: {
        kind: "stream_backlog",
        entryIds: batch.entryIds,
        throughCursorInclusive: batch.throughCursorInclusive!,
      },
    });

    expect(run).toHaveBeenCalledWith({
      sessionId: DEFAULT_SESSION_ID,
      origin: "user",
      lockMode: "try",
      inboundBatch: {
        kind: "stream_backlog",
        entryIds: batch.entryIds,
        throughCursorInclusive: batch.throughCursorInclusive,
      },
    });
  });

  it("uses a supplied runner instead of the default turn adapter", async () => {
    const run = vi.fn(async () => undefined);
    const harness = createHarness({ prefixes: [prefix()], customRunner: run });

    await expect(harness.worker.tick(DEFAULT_SESSION_ID)).resolves.toMatchObject({
      status: "drained",
      drained: 1,
    });
    expect(run).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: DEFAULT_SESSION_ID,
        inboundBatch: expect.objectContaining({ kind: "stream_backlog" }),
      }),
    );
    expect(harness.turnOrchestrator.run).not.toHaveBeenCalled();
  });
});
