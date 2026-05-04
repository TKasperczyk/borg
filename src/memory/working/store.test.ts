import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { WorkingMemoryError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { WorkingMemoryStore } from "./store.js";
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

    expect(sanitized.pending_actions).toEqual([
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
