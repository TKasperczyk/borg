import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { FixedClock } from "../../util/clock.js";
import { WorkingMemoryError } from "../../util/errors.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { WorkingMemoryStore } from "./store.js";

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
      current_focus: null,
      hot_entities: [],
      pending_intents: [],
    });

    store.save({
      ...initial,
      turn_counter: 2,
      current_focus: "Atlas",
      hot_entities: ["Atlas"],
      updated_at: 200,
    });

    const reloaded = new WorkingMemoryStore({
      dataDir: tempDir,
      clock: new FixedClock(300),
    }).load(DEFAULT_SESSION_ID);

    expect(reloaded).toMatchObject({
      turn_counter: 2,
      current_focus: "Atlas",
      hot_entities: ["Atlas"],
    });

    store.clear(DEFAULT_SESSION_ID);
    expect(store.load(DEFAULT_SESSION_ID).turn_counter).toBe(0);
  });

  it("does not update the in-memory cache when persistence fails", () => {
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

  it("caps pending intents to the most recent unique next actions", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);

    const store = new WorkingMemoryStore({
      dataDir: tempDir,
      clock: new FixedClock(100),
    });
    const initial = store.load(DEFAULT_SESSION_ID);

    const saved = store.save({
      ...initial,
      pending_intents: Array.from({ length: 20 }, (_, index) => ({
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

    expect(saved.pending_intents).toHaveLength(16);
    expect(saved.pending_intents.map((intent) => intent.next_action)).toEqual([
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
});
