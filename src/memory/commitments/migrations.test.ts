import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";

import { commitmentMigrations } from "./migrations.js";
import { CommitmentRepository } from "./repository.js";

describe("commitment migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("backfills provenance from legacy source_episode_ids", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "commitments.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: commitmentMigrations.filter((migration) => migration.id < 210),
    });

    try {
      legacyDb
        .prepare(
          `
            INSERT INTO commitments (
              id, type, directive, priority, made_to_entity, restricted_audience, about_entity,
              source_episode_ids, created_at, expires_at, revoked_at, superseded_by
            ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?, NULL, NULL, NULL)
          `,
        )
        .run(
          "cmt_aaaaaaaaaaaaaaaa",
          "boundary",
          "Keep Atlas details anchored",
          10,
          '["ep_aaaaaaaaaaaaaaaa","ep_bbbbbbbbbbbbbbbb"]',
          1_000,
        );
      legacyDb
        .prepare(
          `
            INSERT INTO commitments (
              id, type, directive, priority, made_to_entity, restricted_audience, about_entity,
              source_episode_ids, created_at, expires_at, revoked_at, superseded_by
            ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?, NULL, NULL, NULL)
          `,
        )
        .run("cmt_bbbbbbbbbbbbbbbb", "promise", "Follow up later", 5, "[]", 2_000);
    } finally {
      legacyDb.close();
    }

    const db = openDatabase(dbPath, {
      migrations: commitmentMigrations,
    });
    const repository = new CommitmentRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      expect(repository.list()).toEqual([
        expect.objectContaining({
          id: "cmt_aaaaaaaaaaaaaaaa",
          provenance: {
            kind: "episodes",
            episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
          },
        }),
        expect.objectContaining({
          id: "cmt_bbbbbbbbbbbbbbbb",
          provenance: {
            kind: "system",
          },
        }),
      ]);
    } finally {
      db.close();
    }
  });
});
