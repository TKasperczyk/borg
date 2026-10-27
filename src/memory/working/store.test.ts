import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { WorkingMemoryError } from "../../util/errors.js";
import {
  DEFAULT_SESSION_ID,
  createSessionId,
  createEntityId,
  createStreamEntryId,
  pendingActionIdHelpers,
} from "../../util/ids.js";
import { SessionLock } from "../../cognition/session-lock.js";
import { WorkingMemoryStore, pendingActionResolutionId } from "./store.js";
import { ActionRepository, actionMigrations, actionRecordSchema } from "../actions/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { createWorkingMemory, workingMemorySchema } from "./types.js";

class MapEmbeddingClient implements EmbeddingClient {
  constructor(private readonly vectors: ReadonlyMap<string, readonly number[]>) {}

  async embed(text: string): Promise<Float32Array> {
    return Float32Array.from(this.vectors.get(text) ?? [0, 0]);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map((text) => this.embed(text)));
  }
}

function actionText(text: string): string {
  return `${text}\n${text}`;
}

describe("working memory store", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("loads defaults, persists state, and clears it", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const store = new WorkingMemoryStore({
      dataDir: tempDir,
      clock: new FixedClock(100),
    });
    const initial = store.load(DEFAULT_SESSION_ID);

    expect(initial).toMatchObject({
      session_id: DEFAULT_SESSION_ID,
      turn_counter: 0,
      hot_entities: [],
      pending_actions: [],
    });

    store.save({
      ...initial,
      turn_counter: 2,
      hot_entities: ["Atlas"],
      updated_at: 200,
    });

    const reloaded = new WorkingMemoryStore({
      dataDir: tempDir,
      clock: new FixedClock(300),
    }).load(DEFAULT_SESSION_ID);

    expect(reloaded).toMatchObject({
      turn_counter: 2,
      hot_entities: ["Atlas"],
    });

    store.clear(DEFAULT_SESSION_ID);
    expect(store.load(DEFAULT_SESSION_ID).turn_counter).toBe(0);
  });

  it.each([
    { kind: "invalid shape", code: "WORKING_MEMORY_INVALID" },
    { kind: "invalid JSON", code: "WORKING_MEMORY_LOAD_FAILED" },
    { kind: "session mismatch", code: "WORKING_MEMORY_SESSION_MISMATCH" },
  ])("overwrites and traces $kind only on the session's own save", async ({ kind, code }) => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-working-save-invalid-"));
    tempDirs.push(tempDir);
    mkdirSync(join(tempDir, "working"));
    const path = join(tempDir, "working", "default.json");
    const other = createSessionId();
    const otherPath = join(tempDir, "working", `${other}.json`);
    const otherState = createWorkingMemory(other, 100);
    const otherBytes = JSON.stringify(otherState);
    const invalidBytes =
      kind === "invalid JSON"
        ? "{"
        : kind === "session mismatch"
          ? otherBytes
          : JSON.stringify({ audience: createEntityId(), nodes: [] });
    writeFileSync(path, invalidBytes);
    writeFileSync(otherPath, otherBytes);
    const emit = vi.fn();
    const store = new WorkingMemoryStore({ dataDir: tempDir, tracer: { emit } });
    const sessionLock = new SessionLock({ dataDir: tempDir });

    // The tolerance must not reach projection, locked migration/recovery, or
    // a foreign update: none may use invalid persisted state as an origin.
    expect(() => store.load(DEFAULT_SESSION_ID)).toThrow(WorkingMemoryError);
    await expect(
      store.migratePendingActionIds(DEFAULT_SESSION_ID, sessionLock),
    ).rejects.toMatchObject({ code });
    const apply = vi.fn(async () => true);
    await expect(
      store.resolvePendingAction({
        sessionId: DEFAULT_SESSION_ID,
        pendingActionId: pendingActionIdHelpers.create(),
        lockedSessionId: other,
        sessionLock,
        apply,
      }),
    ).rejects.toMatchObject({ code });
    expect(apply).not.toHaveBeenCalled();
    expect(sessionLock.isHeld(DEFAULT_SESSION_ID)).toBe(false);
    const projection = await store.projectPendingActions({
      current: otherState,
      sessionLock,
      includeOtherSessions: true,
    });
    expect(projection.degraded).toEqual([DEFAULT_SESSION_ID]);
    expect(readFileSync(path, "utf8")).toBe(invalidBytes);

    emit.mockClear();
    const lease = await sessionLock.acquire(DEFAULT_SESSION_ID);
    try {
      const saved = store.save({
        ...createWorkingMemory(DEFAULT_SESSION_ID, 200),
        turn_counter: 3,
        pending_actions: [{ description: "次の起動で確認", next_action: null }],
      });
      expect(JSON.parse(readFileSync(path, "utf8"))).toEqual(saved);
      expect(new WorkingMemoryStore({ dataDir: tempDir }).load(DEFAULT_SESSION_ID)).toEqual(saved);
      expect(readFileSync(otherPath, "utf8")).toBe(otherBytes);
      expect(emit).toHaveBeenCalledWith("working_memory.degraded", {
        turnId: "working_memory",
        session_id: DEFAULT_SESSION_ID,
        operation: "save_persisted_read",
        reason: code,
      });
    } finally {
      await lease!.release();
    }
  });

  it("does not treat a persisted-file I/O failure as absent during save", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-working-save-io-"));
    tempDirs.push(tempDir);
    mkdirSync(join(tempDir, "working", "default.json"), { recursive: true });
    const emit = vi.fn();
    const store = new WorkingMemoryStore({ dataDir: tempDir, tracer: { emit } });
    expect(() => store.save(createWorkingMemory(DEFAULT_SESSION_ID, 100))).toThrow(
      expect.objectContaining({ code: "WORKING_MEMORY_LOAD_FAILED" }),
    );
    expect(emit).not.toHaveBeenCalledWith(
      "working_memory.degraded",
      expect.objectContaining({ operation: "save_persisted_read" }),
    );
  });

  it("reads legacy JSON without writes and migrates under the origin lock without losing fields", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-pending-migration-"));
    tempDirs.push(tempDir);
    mkdirSync(join(tempDir, "working"));
    const path = join(tempDir, "working", "default.json");
    const legacy = {
      ...createWorkingMemory(DEFAULT_SESSION_ID, 100),
      pending_actions: [
        {
          description: "次の自律起動で確認",
          source_stream_ids: ["se_legacy"],
          next_action: "例外も確認する",
          created_at: 42,
        },
        { description: "Поиск", next_action: null },
      ],
    };
    writeFileSync(path, JSON.stringify(legacy));
    const store = new WorkingMemoryStore({ dataDir: tempDir });
    const sessionLock = new SessionLock({ dataDir: tempDir });
    const preview = store.load(DEFAULT_SESSION_ID);
    expect(JSON.parse(readFileSync(path, "utf8"))).toEqual(legacy);
    const first = store.load(DEFAULT_SESSION_ID);
    expect(first).toEqual(preview);
    expect(first.pending_actions).toMatchObject(legacy.pending_actions);
    expect(
      first.pending_actions.every((a) => pendingActionIdHelpers.is(a.pending_action_id!)),
    ).toBe(true);
    expect(JSON.parse(readFileSync(path, "utf8"))).toEqual(legacy);
    const conversation = await sessionLock.acquire(DEFAULT_SESSION_ID);
    expect(await store.migratePendingActionIds(DEFAULT_SESSION_ID, sessionLock)).toBe(false);
    // An ordinary read, including one while a writer owns the lock, changes no bytes.
    const newer = { ...legacy, turn_counter: 4 };
    writeFileSync(path, JSON.stringify(newer));
    expect(store.load(DEFAULT_SESSION_ID).turn_counter).toBe(4);
    expect(readFileSync(path, "utf8")).toBe(JSON.stringify(newer));
    await conversation!.release();
    expect(await store.migratePendingActionIds(DEFAULT_SESSION_ID, sessionLock)).toBe(true);
    const migrated = JSON.parse(readFileSync(path, "utf8"));
    expect(migrated).toEqual({ ...newer, pending_actions: first.pending_actions });
    expect(Object.keys(migrated.pending_actions[0])).toEqual([
      ...Object.keys(legacy.pending_actions[0]!),
      "pending_action_id",
    ]);
    const bytes = readFileSync(path, "utf8");
    expect(await store.migratePendingActionIds(DEFAULT_SESSION_ID, sessionLock)).toBe(true);
    expect(new WorkingMemoryStore({ dataDir: tempDir }).load(DEFAULT_SESSION_ID)).toEqual({
      ...first,
      turn_counter: 4,
    });
    expect(readFileSync(path, "utf8")).toBe(bytes);
    // A normal save on the session's next locked write also retains extension fields.
    const lease = await sessionLock.acquire(DEFAULT_SESSION_ID);
    store.save(store.load(DEFAULT_SESSION_ID));
    await lease!.release();
    expect(JSON.parse(readFileSync(path, "utf8")).pending_actions[0].source_stream_ids).toEqual([
      "se_legacy",
    ]);
  });

  it("rejects a filename/session mismatch without writing either origin", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-pending-identity-"));
    tempDirs.push(tempDir);
    mkdirSync(join(tempDir, "working"));
    const other = createSessionId();
    const path = join(tempDir, "working", "default.json");
    const otherPath = join(tempDir, "working", `${other}.json`);
    const misplaced = {
      ...createWorkingMemory(other, 100),
      pending_actions: [{ description: "legacy", next_action: null }],
    };
    const otherBytes = JSON.stringify(createWorkingMemory(other, 200));
    writeFileSync(path, JSON.stringify(misplaced));
    writeFileSync(otherPath, otherBytes);
    const emit = vi.fn();
    const store = new WorkingMemoryStore({ dataDir: tempDir, tracer: { emit } });
    const sessionLock = new SessionLock({ dataDir: tempDir });
    expect(() => store.load(DEFAULT_SESSION_ID)).toThrow("session identity");
    await expect(
      store.migratePendingActionIds(DEFAULT_SESSION_ID, sessionLock),
    ).rejects.toMatchObject({ code: "WORKING_MEMORY_SESSION_MISMATCH" });
    expect(readFileSync(path, "utf8")).toBe(JSON.stringify(misplaced));
    expect(readFileSync(otherPath, "utf8")).toBe(otherBytes);
    expect(sessionLock.isHeld(DEFAULT_SESSION_ID)).toBe(false);
    expect(emit).toHaveBeenCalledWith(
      "working_memory.degraded",
      expect.objectContaining({
        session_id: DEFAULT_SESSION_ID,
        reason: "WORKING_MEMORY_SESSION_MISMATCH",
      }),
    );
  });

  it("skips malformed foreign files, reports degradation, and retries without projection writes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-pending-degraded-"));
    tempDirs.push(tempDir);
    mkdirSync(join(tempDir, "working"));
    const good = createSessionId();
    const invalidJson = createSessionId();
    const invalidShape = createSessionId();
    const wrongIdentity = createSessionId();
    const goodPath = join(tempDir, "working", `${good}.json`);
    const goodBytes = JSON.stringify({
      ...createWorkingMemory(good, 100),
      pending_actions: [{ description: "有効な意図", next_action: null }],
    });
    writeFileSync(goodPath, goodBytes);
    writeFileSync(join(tempDir, "working", `${invalidJson}.json`), "invalid JSON");
    writeFileSync(join(tempDir, "working", `${invalidShape}.json`), "{}");
    writeFileSync(join(tempDir, "working", `${wrongIdentity}.json`), goodBytes);
    const emit = vi.fn();
    const store = new WorkingMemoryStore({ dataDir: tempDir, tracer: { emit } });
    const sessionLock = new SessionLock({ dataDir: tempDir });
    const acquire = vi.spyOn(sessionLock, "tryAcquire");
    const input = {
      current: store.load(DEFAULT_SESSION_ID),
      sessionLock,
      includeOtherSessions: true,
    };
    const projection = await store.projectPendingActions(input);
    expect(projection.actions).toMatchObject([
      { session_id: good, description: "有効な意図", disclosure: expect.any(String) },
    ]);
    expect(projection.degraded).toEqual([invalidJson, invalidShape, wrongIdentity].sort());
    expect(readFileSync(goodPath, "utf8")).toBe(goodBytes);
    expect(acquire).not.toHaveBeenCalled();
    for (const id of projection.degraded) {
      expect(emit).toHaveBeenCalledWith(
        "working_memory.degraded",
        expect.objectContaining({ session_id: id, operation: "pending_action_projection" }),
      );
      writeFileSync(
        join(tempDir, "working", `${id}.json`),
        JSON.stringify(createWorkingMemory(id, 100)),
      );
    }
    expect((await store.projectPendingActions(input)).degraded).toEqual([]);
    expect(readFileSync(goodPath, "utf8")).toBe(goodBytes);
  });

  it("projects all origins, preserves concurrent writes under origin locks, and resolves once by handle", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-global-pending-"));
    tempDirs.push(tempDir);
    const origin = createSessionId();
    const audience = createEntityId();
    const store = new WorkingMemoryStore({
      dataDir: tempDir,
      audienceForSession: (id) => (id === origin ? audience : null),
    });
    const sessionLock = new SessionLock({ dataDir: tempDir });
    const own = store.save({
      ...createWorkingMemory(DEFAULT_SESSION_ID, 100),
      pending_actions: [{ description: "wake intention", next_action: null }],
    });
    const foreign = store.save({
      ...createWorkingMemory(origin, 100),
      pending_actions: [{ description: "次の起動へ", next_action: null }],
    });
    const snapshot = await store.projectPendingActions({
      current: own,
      sessionLock,
      includeOtherSessions: true,
    });
    expect(snapshot.actions).toHaveLength(2);
    expect(snapshot.degraded).toEqual([]);
    expect(snapshot.actions.find((a) => a.session_id === origin)).toMatchObject({
      description: "次の起動へ",
      audience_entity_id: audience,
      disclosure: expect.stringContaining("self_private"),
      disclosure_label: {
        origin_audience_entity_ids: [audience],
        private_to_entity_ids: [audience],
      },
    });
    expect(own.pending_actions).toHaveLength(1);
    expect(
      (
        await store.projectPendingActions({
          current: own,
          sessionLock,
          includeOtherSessions: false,
        })
      ).actions,
    ).toHaveLength(1);
    const conversation = await sessionLock.acquire(origin);
    expect(conversation).not.toBeNull();
    const apply = vi.fn(async () => {
      expect(sessionLock.isHeld(origin)).toBe(true);
      return true;
    });
    const input = {
      sessionId: origin,
      pendingActionId: foreign.pending_actions[0]!.pending_action_id!,
      lockedSessionId: DEFAULT_SESSION_ID,
      sessionLock,
      apply,
    };
    expect(await store.resolvePendingAction(input)).toBe("busy");
    expect(
      (
        await store.projectPendingActions({ current: own, sessionLock, includeOtherSessions: true })
      ).actions.find((a) => a.session_id === origin),
    ).toBeDefined();
    store.save({
      ...store.load(origin),
      hot_entities: ["concurrent write"],
      pending_actions: [
        ...foreign.pending_actions,
        { description: "new conversational intention", next_action: null },
      ],
    });
    await conversation!.release();
    expect(await store.resolvePendingAction(input)).toBe("applied");
    expect(await store.resolvePendingAction(input)).toBe("missing");
    expect(apply).toHaveBeenCalledTimes(1);
    expect(store.load(origin)).toMatchObject({
      hot_entities: ["concurrent write"],
      pending_actions: [{ description: "new conversational intention" }],
    });
    expect(store.load(DEFAULT_SESSION_ID)).toEqual(own);
  });

  it.each([false, true])(
    "leaves the origin untouched when the action insert fails (throws=%s)",
    async (throws) => {
      const tempDir = mkdtempSync(join(tmpdir(), "borg-pending-insert-"));
      tempDirs.push(tempDir);
      const origin = createSessionId();
      const store = new WorkingMemoryStore({ dataDir: tempDir });
      const sessionLock = new SessionLock({ dataDir: tempDir });
      const original = store.save({
        ...createWorkingMemory(origin, 100),
        pending_actions: [{ description: "再試行", next_action: null }],
      });
      const input = {
        sessionId: origin,
        pendingActionId: original.pending_actions[0]!.pending_action_id!,
        lockedSessionId: DEFAULT_SESSION_ID,
        sessionLock,
        apply: async () => {
          expect(sessionLock.isHeld(origin)).toBe(true);
          expect(store.load(origin)).toEqual(original);
          if (throws) throw new Error("injected action insert failure");
          return false;
        },
      };
      const save = vi.spyOn(store, "save");
      expect(await store.resolvePendingAction(input)).toBe("failed");
      expect(save).not.toHaveBeenCalled();
      expect(store.load(origin)).toEqual(original);
      expect(sessionLock.isHeld(origin)).toBe(false);
      expect(await store.resolvePendingAction({ ...input, apply: async () => true })).toBe(
        "applied",
      );
      expect(await store.resolvePendingAction(input)).toBe("missing");
    },
  );

  it.each(["migration", "wake"])(
    "recovers a rollback marker through locked %s, never while its turn holds the lock",
    async (recoveryPath) => {
      const dataDir = mkdtempSync(join(tmpdir(), "borg-pending-rollback-"));
      tempDirs.push(dataDir);
      const db = openDatabase(join(dataDir, "actions.db"), { migrations: actionMigrations });
      try {
        const sessionId = createSessionId();
        const sessionLock = new SessionLock({ dataDir });
        const actionRepository = new ActionRepository({ db });
        const store = new WorkingMemoryStore({ dataDir, actionRepository });
        const original = store.save({
          ...createWorkingMemory(sessionId, 100),
          pending_actions: [{ description: "あとで確認", next_action: null }],
        });
        const pending = original.pending_actions[0]!;
        const id = pendingActionResolutionId({
          session_id: sessionId,
          pending_action_id: pending.pending_action_id!,
        });
        const completion = actionRecordSchema.parse({
          id,
          description: pending.description,
          actor: "borg",
          audience_entity_id: null,
          state: "completed",
          confidence: 1,
          provenance_episode_ids: [],
          provenance_stream_entry_ids: [createStreamEntryId()],
          created_at: 200,
          updated_at: 200,
          considering_at: null,
          committed_at: null,
          scheduled_at: null,
          completed_at: 200,
          not_done_at: null,
          unknown_at: null,
        });
        actionRepository.add(completion);
        const marker = { action_id: id, turn_id: "current-turn" };
        const deleteAction = vi.spyOn(actionRepository, "delete");
        const lease = await sessionLock.acquire(sessionId);
        try {
          store.save({
            ...original,
            pending_actions: [{ ...pending, restored_from_action: marker }],
          });
          expect(await store.migratePendingActionIds(sessionId, sessionLock)).toBe(false);
          await store.migratePendingActionIds(sessionId, sessionLock, {
            lockedSessionId: sessionId,
            turnId: "current-turn",
          });
          expect(deleteAction).not.toHaveBeenCalled();
          expect(store.load(sessionId).pending_actions[0]?.restored_from_action).toEqual(marker);
        } finally {
          await lease?.release();
        }
        const apply = vi.fn(async () => {
          expect(sessionLock.isHeld(sessionId)).toBe(true);
          expect(actionRepository.get(id)).toBeNull();
          expect(store.load(sessionId).pending_actions).toEqual([pending]);
          actionRepository.add(completion);
          return true;
        });
        if (recoveryPath === "migration") {
          expect(await store.migratePendingActionIds(sessionId, sessionLock)).toBe(true);
          expect(actionRepository.get(id)).toBeNull();
          expect(store.load(sessionId).pending_actions).toEqual([pending]);
        } else {
          expect(
            await store.resolvePendingAction({
              sessionId,
              pendingActionId: pending.pending_action_id!,
              lockedSessionId: DEFAULT_SESSION_ID,
              sessionLock,
              turnId: "later-wake",
              apply,
            }),
          ).toBe("applied");
          expect(apply).toHaveBeenCalledOnce();
          expect(store.load(sessionId).pending_actions).toEqual([]);
        }
        expect(deleteAction).toHaveBeenCalledExactlyOnceWith(id);
      } finally {
        db.close();
      }
    },
  );

  it.each(["save", "migration", "wake"] as const)(
    "recovers an action-first crash through a locked %s after reopening SQLite and the store",
    async (cleanupPath) => {
      const tempDir = mkdtempSync(join(tmpdir(), "borg-pending-crash-"));
      tempDirs.push(tempDir);
      const databasePath = join(tempDir, "actions.db");
      let db = openDatabase(databasePath, { migrations: actionMigrations });
      try {
        const origin = createSessionId();
        const sessionLock = new SessionLock({ dataDir: tempDir });
        const actionRepository = new ActionRepository({ db });
        const emit = vi.fn();
        const store = new WorkingMemoryStore({
          dataDir: tempDir,
          actionRepository,
          tracer: { emit },
        });
        const current = createWorkingMemory(DEFAULT_SESSION_ID, 100);
        const original = store.save({
          ...createWorkingMemory(origin, 100),
          pending_actions: [{ description: "確認を完了", next_action: null }],
        });
        const pendingActionId = original.pending_actions[0]!.pending_action_id!;
        const actionId = pendingActionResolutionId({
          session_id: origin,
          pending_action_id: pendingActionId,
        });
        const apply = vi.fn(async () => {
          expect(sessionLock.isHeld(origin)).toBe(true);
          expect(store.load(origin)).toEqual(original);
          actionRepository.add(
            actionRecordSchema.parse({
              id: actionId,
              description: "確認を完了",
              actor: "borg",
              audience_entity_id: null,
              state: "completed",
              confidence: 1,
              provenance_episode_ids: [],
              provenance_stream_entry_ids: [createStreamEntryId()],
              created_at: 200,
              updated_at: 200,
              considering_at: null,
              committed_at: null,
              scheduled_at: null,
              completed_at: 200,
              not_done_at: null,
              unknown_at: null,
            }),
          );
          return true;
        });
        const input = {
          sessionId: origin,
          pendingActionId,
          lockedSessionId: DEFAULT_SESSION_ID,
          sessionLock,
          apply,
        };
        vi.spyOn(store, "save").mockImplementationOnce(() => {
          throw new Error("crash before JSON save");
        });
        expect(await store.resolvePendingAction(input)).toBe("applied_cleanup_pending");
        expect(actionRepository.get(actionId)?.state).toBe("completed");
        expect(store.load(origin)).toEqual(original);
        expect(emit).toHaveBeenCalledWith(
          "working_memory.pending_action_resolved",
          expect.objectContaining({ action_id: actionId, step: "action", outcome: "committed" }),
        );
        expect(emit).toHaveBeenCalledWith(
          "working_memory.degraded",
          expect.objectContaining({ operation: "pending_action_resolution_save" }),
        );
        db.close();
        db = openDatabase(databasePath, { migrations: actionMigrations });
        const reopenedActions = new ActionRepository({ db });
        const restarted = new WorkingMemoryStore({
          dataDir: tempDir,
          actionRepository: reopenedActions,
          tracer: { emit },
        });
        const originPath = join(tempDir, "working", `${origin}.json`);
        const beforeProjection = readFileSync(originPath, "utf8");
        const projection = await restarted.projectPendingActions({
          current,
          sessionLock,
          includeOtherSessions: true,
        });
        expect(projection.actions).toEqual([]);
        expect(projection.degraded).toEqual([]);
        expect(projection.completed_pending_cleanup).toMatchObject([
          {
            session_id: origin,
            pending_action_id: pendingActionId,
            action_id: actionId,
            disclosure: expect.any(String),
            disclosure_label: { disclosure_class: "self_private" },
          },
        ]);
        expect(readFileSync(originPath, "utf8")).toBe(beforeProjection);
        if (cleanupPath === "save") {
          const lease = await sessionLock.acquire(origin);
          try {
            // The conversational path rereads under its origin lock before save.
            const latest = restarted.load(origin);
            const saved = restarted.save({ ...latest, hot_entities: ["concurrent origin state"] });
            expect(saved.pending_actions).toEqual([]);
            expect(saved.hot_entities).toEqual(["concurrent origin state"]);
          } finally {
            await lease?.release();
          }
        } else if (cleanupPath === "migration") {
          expect(await restarted.migratePendingActionIds(origin, sessionLock)).toBe(true);
        } else {
          expect(await restarted.resolvePendingAction(input)).toBe("applied");
        }
        expect(restarted.load(origin).pending_actions).toEqual([]);
        expect(
          (
            await restarted.projectPendingActions({
              current,
              sessionLock,
              includeOtherSessions: true,
            })
          ).completed_pending_cleanup,
        ).toEqual([]);
        expect(await restarted.resolvePendingAction(input)).toBe("applied");
        expect(apply).toHaveBeenCalledTimes(1);
        expect(reopenedActions.list({ state: "completed" })).toHaveLength(1);
        expect(emit).toHaveBeenCalledWith(
          "working_memory.pending_action_resolved",
          expect.objectContaining({ action_id: actionId, step: "cleanup", outcome: "saved" }),
        );
      } finally {
        db.close();
      }
    },
  );

  it("does not persist a failed save", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const store = new WorkingMemoryStore({
      dataDir: tempDir,
      clock: new FixedClock(100),
    });
    const initial = store.load(DEFAULT_SESSION_ID);
    const writeSpy = vi
      .spyOn(store as unknown as { writePersisted(state: unknown): void }, "writePersisted")
      .mockImplementation(() => {
        throw new WorkingMemoryError("disk full", {
          code: "WORKING_MEMORY_SAVE_FAILED",
        });
      });

    expect(() =>
      store.save({
        ...initial,
        turn_counter: 1,
        updated_at: 200,
      }),
    ).toThrow(WorkingMemoryError);
    expect(store.load(DEFAULT_SESSION_ID).turn_counter).toBe(0);

    writeSpy.mockRestore();
  });

  it("rejects obsolete extra fields in nested persisted shapes", () => {
    const parsed = workingMemorySchema.safeParse({
      ...createWorkingMemory(DEFAULT_SESSION_ID, 100),
      pending_trait_attribution: {
        trait_label: "careful",
        strength_delta: 0.1,
        source_stream_entry_ids: ["strm_aaaaaaaaaaaaaaaa"],
        source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
        turn_completed_ts: 100,
        audience_entity_id: null,
      },
    });

    expect(parsed.success).toBe(false);
    expect(parsed.error?.issues.map((issue) => issue.path.join("."))).toContain(
      "pending_trait_attribution",
    );
  });

  it("reloads persisted state on every load instead of returning stale cache", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const firstStore = new WorkingMemoryStore({
      dataDir: tempDir,
      clock: new FixedClock(100),
    });
    const secondStore = new WorkingMemoryStore({
      dataDir: tempDir,
      clock: new FixedClock(200),
    });
    const initial = firstStore.load(DEFAULT_SESSION_ID);

    secondStore.save({
      ...initial,
      turn_counter: 7,
      updated_at: 300,
    });

    expect(firstStore.load(DEFAULT_SESSION_ID).turn_counter).toBe(7);
  });

  it("caps pending actions to the most recent unique next actions", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const store = new WorkingMemoryStore({
      dataDir: tempDir,
      clock: new FixedClock(100),
    });
    const initial = store.load(DEFAULT_SESSION_ID);

    const saved = store.save({
      ...initial,
      pending_actions: Array.from({ length: 20 }, (_, index) => ({
        description: `Intent ${index}`,
        next_action: `action-${index}`,
      })).concat([
        {
          description: "Older duplicate",
          next_action: "action-19",
        },
      ]),
      updated_at: 200,
    });

    expect(saved.pending_actions).toHaveLength(16);
    expect(saved.pending_actions.map((intent) => intent.next_action)).toEqual([
      "action-4",
      "action-5",
      "action-6",
      "action-7",
      "action-8",
      "action-9",
      "action-10",
      "action-11",
      "action-12",
      "action-13",
      "action-14",
      "action-15",
      "action-16",
      "action-17",
      "action-18",
      "action-19",
    ]);
  });

  it("rewrites pending actions that mention quarantined relational slot values", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const store = new WorkingMemoryStore({
      dataDir: tempDir,
      clock: new FixedClock(500),
    });
    const initial = store.load(DEFAULT_SESSION_ID);

    store.save({
      ...initial,
      pending_actions: [
        {
          description: "Track whether Tom raises the planning comment with Sarah directly",
          next_action: "Ask Sarah about the planning comment if Tom brings it up",
        },
      ],
      updated_at: 200,
    });

    const sanitized = store.sanitizePendingActionsForRelationalSlot({
      sessionId: DEFAULT_SESSION_ID,
      values: ["Sarah"],
      neutralPhrase: "your partner",
    });

    expect(sanitized.pending_actions).toMatchObject([
      {
        description: "Track whether Tom raises the planning comment with your partner directly",
        next_action: "Ask your partner about the planning comment if Tom brings it up",
      },
    ]);
    expect(sanitized.updated_at).toBe(500);
  });

  it("semantically merges similar pending actions and refreshes the timestamp", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(100);
    const first = "check whether Saturday Spanish neighborhood dry run happened";
    const second = "follow up on whether Saturday Spanish dry run actually took place";
    const store = new WorkingMemoryStore({
      dataDir: tempDir,
      clock,
    });
    const embeddingClient = new MapEmbeddingClient(
      new Map([
        [actionText(first), [1, 0]],
        [actionText(second), [0.9, 0.1]],
      ]),
    );

    await store.addPendingAction({
      sessionId: DEFAULT_SESSION_ID,
      action: {
        description: first,
        next_action: first,
      },
      embeddingClient,
    });
    clock.set(200);
    const saved = await store.addPendingAction({
      sessionId: DEFAULT_SESSION_ID,
      action: {
        description: second,
        next_action: second,
      },
      embeddingClient,
    });

    expect(saved.pending_actions).toHaveLength(1);
    expect(saved.pending_actions[0]?.description).toBe(first);
    expect(saved.pending_actions[0]?.created_at).toBe(200);
  });

  it("keeps semantically distinct pending actions separate", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(100);
    const first = "check whether Saturday Spanish neighborhood dry run happened";
    const second = "prepare the backpressure design review";
    const store = new WorkingMemoryStore({
      dataDir: tempDir,
      clock,
    });
    const embeddingClient = new MapEmbeddingClient(
      new Map([
        [actionText(first), [1, 0]],
        [actionText(second), [0, 1]],
      ]),
    );

    await store.addPendingAction({
      sessionId: DEFAULT_SESSION_ID,
      action: {
        description: first,
        next_action: first,
      },
      embeddingClient,
    });
    clock.set(200);
    const saved = await store.addPendingAction({
      sessionId: DEFAULT_SESSION_ID,
      action: {
        description: second,
        next_action: second,
      },
      embeddingClient,
    });

    expect(saved.pending_actions.map((action) => action.description)).toEqual([first, second]);
  });
});
