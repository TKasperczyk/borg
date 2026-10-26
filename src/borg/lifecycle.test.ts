import { afterEach, describe, expect, it, vi } from "vitest";

import type { BorgDependencies } from "./types.js";
import { closeBestEffort, closeBorgDependencies, drainBorgDependencies } from "./lifecycle.js";

describe("borg lifecycle", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("forced store close never re-enters a stalled runtime drain", async () => {
    const stop = vi.fn(async () => await new Promise<void>(() => {}));
    const sqliteClose = vi.fn();
    const lanceClose = vi.fn(async () => {});
    const releaseEmbeddingBankAccess = vi.fn(async () => {});
    const deps = {
      releaseEmbeddingBankAccess,
      autonomyScheduler: { stop },
      maintenanceScheduler: { stop },
      chatResponseCatchUpWorker: { stop },
      sqlite: { close: sqliteClose },
      lance: { close: lanceClose },
    } as unknown as BorgDependencies;
    await closeBorgDependencies(deps, { drain: false });
    expect(stop).not.toHaveBeenCalled();
    expect(releaseEmbeddingBankAccess).not.toHaveBeenCalled();
    expect(sqliteClose).toHaveBeenCalledOnce();
    expect(lanceClose).toHaveBeenCalledOnce();
  });

  it("waits for startup before snapshotting repository embeddings and releasing the lease", async () => {
    let finishStartup!: () => void;
    let finishEmbedding!: () => void;
    const startupGate = new Promise<void>((done) => {
      finishStartup = done;
    });
    const embeddingTail = new Promise<void>((done) => {
      finishEmbedding = done;
    });
    let pendingEmbedding = Promise.resolve();
    const waitForPendingEmbeddings = vi.fn(async () => await pendingEmbedding);
    const startupTask = (async () => {
      await startupGate;
      expect(waitForPendingEmbeddings).not.toHaveBeenCalled();
      pendingEmbedding = embeddingTail;
    })();
    const stop = vi.fn(async () => {});
    const sqliteClose = vi.fn();
    const releaseEmbeddingBankAccess = vi.fn(async () => {});
    const deps = {
      autonomyScheduler: { stop },
      maintenanceScheduler: { stop },
      chatResponseCatchUpWorker: { stop },
      sqlite: { close: sqliteClose },
      lance: { close: vi.fn(async () => {}) },
      pendingStartupTasks: [startupTask],
      actionRepository: { waitForPendingEmbeddings },
      releaseEmbeddingBankAccess,
    } as unknown as BorgDependencies;
    const closing = closeBorgDependencies(deps);
    // Let shutdown reach the pending startup task before allowing it to enqueue
    // the final embedding write; a premature queue snapshot would now be visible.
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(waitForPendingEmbeddings).not.toHaveBeenCalled();
    expect(sqliteClose).not.toHaveBeenCalled();
    expect(releaseEmbeddingBankAccess).not.toHaveBeenCalled();
    finishStartup();
    await vi.waitFor(() => expect(waitForPendingEmbeddings).toHaveBeenCalledOnce());
    expect(sqliteClose).not.toHaveBeenCalled();
    expect(releaseEmbeddingBankAccess).not.toHaveBeenCalled();
    finishEmbedding();
    await closing;
    expect(sqliteClose).toHaveBeenCalledOnce();
    expect(releaseEmbeddingBankAccess).toHaveBeenCalledOnce();
    expect(sqliteClose.mock.invocationCallOrder[0]).toBeLessThan(
      releaseEmbeddingBankAccess.mock.invocationCallOrder[0]!,
    );
  });

  it("surfaces a rejected startup task and retains the bank lease", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const stop = vi.fn(async () => {});
    const failure = new Error("startup backfill failed");
    const releaseEmbeddingBankAccess = vi.fn(async () => {});
    const deps = {
      autonomyScheduler: { stop },
      maintenanceScheduler: { stop },
      chatResponseCatchUpWorker: { stop },
      sqlite: { close: vi.fn() },
      lance: { close: vi.fn(async () => {}) },
      pendingStartupTasks: [Promise.reject(failure)],
      releaseEmbeddingBankAccess,
    } as unknown as BorgDependencies;
    await expect(closeBorgDependencies(deps)).rejects.toThrow(AggregateError);
    expect(releaseEmbeddingBankAccess).not.toHaveBeenCalled();
    await expect(drainBorgDependencies(deps)).rejects.toMatchObject({ errors: [failure] });
  });

  it("reports failed partial-open cleanup while still attempting both stores", async () => {
    const sqlite = {
      close: vi.fn(() => {
        throw new Error("still open");
      }),
    };
    const lance = { close: vi.fn(async () => {}) };
    expect(await closeBestEffort(sqlite as never, lance as never)).toBe(false);
    expect(lance.close).toHaveBeenCalledOnce();
    expect(await closeBestEffort(undefined, undefined)).toBe(true);
  });

  it("attempts LanceDB close when SQLite close throws", async () => {
    const sqliteError = new Error("sqlite close failed");
    const flushEnqueueHooks = vi.fn().mockResolvedValue(undefined);
    const sqliteClose = vi.fn(() => {
      throw sqliteError;
    });
    const lanceClose = vi.fn().mockResolvedValue(undefined);
    vi.spyOn(console, "error").mockImplementation(() => {});

    const deps = {
      autonomyScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      maintenanceScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      chatResponseCatchUpWorker: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      reviewQueueRepository: {
        flushEnqueueHooks,
      },
      sqlite: {
        close: sqliteClose,
      },
      lance: {
        close: lanceClose,
      },
    } as unknown as BorgDependencies;

    await expect(closeBorgDependencies(deps)).rejects.toThrow(AggregateError);
    expect(flushEnqueueHooks.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER).toBeLessThan(
      sqliteClose.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER,
    );
    expect(lanceClose).toHaveBeenCalledTimes(1);
  });

  it("attempts both scheduler stops when one stop throws", async () => {
    const autonomyError = new Error("autonomy stop failed");
    const maintenanceStop = vi.fn().mockResolvedValue(undefined);
    const catchUpStop = vi.fn().mockResolvedValue(undefined);
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});

    const deps = {
      autonomyScheduler: {
        stop: vi.fn(() => {
          throw autonomyError;
        }),
      },
      maintenanceScheduler: {
        stop: maintenanceStop,
      },
      chatResponseCatchUpWorker: {
        stop: catchUpStop,
      },
      reviewQueueRepository: {
        flushEnqueueHooks: vi.fn().mockResolvedValue(undefined),
      },
      sqlite: {
        close: vi.fn(),
      },
      lance: {
        close: vi.fn().mockResolvedValue(undefined),
      },
    } as unknown as BorgDependencies;

    await expect(closeBorgDependencies(deps)).rejects.toThrow(AggregateError);
    expect(maintenanceStop).toHaveBeenCalledWith({ graceful: true });
    expect(catchUpStop).toHaveBeenCalledWith({ graceful: true });
    expect(consoleError).toHaveBeenCalledWith("Failed to close autonomy scheduler", autonomyError);
  });

  it("drains semantic reviews before their newly enqueued review hooks and store closure", async () => {
    let finishReview!: () => void;
    let finishHook!: () => void;
    const review = new Promise<void>((resolve) => {
      finishReview = resolve;
    });
    const hook = new Promise<void>((resolve) => {
      finishHook = resolve;
    });
    let hookEnqueued = false;
    const flushDuplicateReviews = vi.fn(async () => {
      await review;
      hookEnqueued = true;
    });
    const flushEnqueueHooks = vi.fn(async () => {
      expect(hookEnqueued).toBe(true);
      await hook;
    });
    const close = vi.fn();
    const release = vi.fn();
    const stop = vi.fn(async () => {});
    const deps = {
      autonomyScheduler: { stop },
      maintenanceScheduler: { stop },
      chatResponseCatchUpWorker: { stop },
      semanticReviewService: { flushDuplicateReviews },
      reviewQueueRepository: { flushEnqueueHooks },
      sqlite: { close },
      lance: { close },
      releaseEmbeddingBankAccess: release,
    } as unknown as BorgDependencies;
    const closing = closeBorgDependencies(deps);
    try {
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(flushDuplicateReviews).toHaveBeenCalledOnce();
      expect(flushEnqueueHooks).not.toHaveBeenCalled();
      finishReview();
      await new Promise<void>((resolve) => setImmediate(resolve));
      expect(flushEnqueueHooks).toHaveBeenCalledOnce();
      expect(close).not.toHaveBeenCalled();
      expect(release).not.toHaveBeenCalled();
      finishHook();
      await closing;
      expect(close).toHaveBeenCalledTimes(2);
      expect(release).toHaveBeenCalledOnce();
    } finally {
      finishReview();
      finishHook();
      await closing;
    }
  });

  it("drains review queue enqueue hooks before closing SQLite", async () => {
    const flushEnqueueHooks = vi.fn().mockResolvedValue(undefined);
    const sqliteClose = vi.fn();

    const deps = {
      autonomyScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      maintenanceScheduler: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      chatResponseCatchUpWorker: {
        stop: vi.fn().mockResolvedValue(undefined),
      },
      reviewQueueRepository: {
        flushEnqueueHooks,
      },
      sqlite: {
        close: sqliteClose,
      },
      lance: {
        close: vi.fn().mockResolvedValue(undefined),
      },
    } as unknown as BorgDependencies;

    await closeBorgDependencies(deps);

    expect(flushEnqueueHooks).toHaveBeenCalledTimes(1);
    expect(flushEnqueueHooks.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER).toBeLessThan(
      sqliteClose.mock.invocationCallOrder[0] ?? Number.MAX_SAFE_INTEGER,
    );
  });
});
