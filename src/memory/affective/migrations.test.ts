import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";

import { affectiveMigrations } from "./migrations.js";
import { MoodRepository } from "./mood.js";

describe("affective migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("backfills mood history provenance from legacy trigger_episode_id", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "affective.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: affectiveMigrations.filter((migration) => migration.id < 230),
    });

    try {
      legacyDb
        .prepare(
          `
            INSERT INTO mood_history (
              id, session_id, ts, valence, arousal, trigger_episode_id, trigger_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
          `,
        )
        .run(1, DEFAULT_SESSION_ID, 1_000, -0.4, 0.6, "ep_aaaaaaaaaaaaaaaa", "Atlas incident");
      legacyDb
        .prepare(
          `
            INSERT INTO mood_history (
              id, session_id, ts, valence, arousal, trigger_episode_id, trigger_reason
            ) VALUES (?, ?, ?, ?, ?, NULL, ?)
          `,
        )
        .run(2, DEFAULT_SESSION_ID, 2_000, 0.1, 0.2, "background reset");
    } finally {
      legacyDb.close();
    }

    const db = openDatabase(dbPath, {
      migrations: affectiveMigrations,
    });
    const repository = new MoodRepository({ db });

    try {
      expect(repository.history(DEFAULT_SESSION_ID, { limit: 10 })).toEqual([
        expect.objectContaining({
          id: 2,
          provenance: { kind: "system" },
        }),
        expect.objectContaining({
          id: 1,
          provenance: {
            kind: "episodes",
            episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
          },
        }),
      ]);
    } finally {
      db.close();
    }
  });
});
