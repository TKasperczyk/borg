import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";

import { socialMigrations } from "./migrations.js";
import { SocialRepository } from "./repository.js";

describe("social migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("backfills a system baseline event for legacy profiles", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "social.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: socialMigrations.filter((migration) => migration.id < 240),
    });

    try {
      legacyDb
        .prepare(
          `
            INSERT INTO social_profiles (
              entity_id, trust, attachment, communication_style, shared_history_summary,
              last_interaction_at, interaction_count, commitment_count, sentiment_history, notes,
              created_at, updated_at
            ) VALUES (?, ?, ?, NULL, NULL, ?, ?, 0, '[]', NULL, ?, ?)
          `,
        )
        .run("ent_aaaaaaaaaaaaaaaa", 0.8, 0.3, 2_500, 4, 1_000, 3_000);
    } finally {
      legacyDb.close();
    }

    const db = openDatabase(dbPath, {
      migrations: socialMigrations,
    });
    const repository = new SocialRepository({ db });

    try {
      expect(repository.listEvents("ent_aaaaaaaaaaaaaaaa" as never)).toEqual([
        expect.objectContaining({
          kind: "baseline",
          provenance: { kind: "system" },
          trust_delta: 0.8,
          attachment_delta: 0.3,
          interaction_delta: 4,
        }),
      ]);
    } finally {
      db.close();
    }
  });

  it("upgrades social event provenance checks to allow online writes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "social.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: socialMigrations.filter((migration) => migration.id < 241),
    });
    legacyDb.close();

    const db = openDatabase(dbPath, {
      migrations: socialMigrations,
    });
    const repository = new SocialRepository({ db });

    try {
      repository.recordInteraction("ent_onlineeeeeeeeeee" as never, {
        provenance: {
          kind: "online",
          process: "reflector",
        },
      });

      expect(repository.listEvents("ent_onlineeeeeeeeeee" as never)).toEqual([
        expect.objectContaining({
          provenance: {
            kind: "online",
            process: "reflector",
          },
        }),
      ]);
    } finally {
      db.close();
    }
  });
});
