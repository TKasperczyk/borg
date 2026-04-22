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
});
