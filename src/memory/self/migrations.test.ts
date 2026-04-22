import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { parseEpisodeId } from "../../util/ids.js";

import { AutobiographicalRepository } from "./autobiographical.js";
import { selfMigrations } from "./migrations.js";
import { OpenQuestionsRepository, buildOpenQuestionDedupeKey } from "./open-questions.js";
import { GoalsRepository, TraitsRepository, ValuesRepository } from "./repository.js";

describe("self migrations", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("backfills provenance across legacy self tables", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "self.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: selfMigrations.filter((migration) => migration.id < 220),
    });
    const relatedEpisodeId = parseEpisodeId("ep_aaaaaaaaaaaaaaaa");

    try {
      legacyDb
        .prepare(
          `
            INSERT INTO "values" (id, label, description, priority, created_at, last_affirmed)
            VALUES (?, ?, ?, ?, ?, NULL)
          `,
        )
        .run("val_aaaaaaaaaaaaaaaa", "clarity", "Prefer explicit state.", 1, 1_000);
      legacyDb
        .prepare(
          `
            INSERT INTO "values" (id, label, description, priority, created_at, last_affirmed)
            VALUES (?, ?, ?, ?, ?, NULL)
          `,
        )
        .run("val_bbbbbbbbbbbbbbbb", "curiosity", "Keep asking why.", 2, 2_000);
      legacyDb
        .prepare("INSERT INTO value_sources (value_id, episode_id) VALUES (?, ?)")
        .run("val_aaaaaaaaaaaaaaaa", relatedEpisodeId);

      legacyDb
        .prepare(
          `
            INSERT INTO goals (
              id, description, priority, parent_goal_id, status, progress_notes, created_at, target_at
            ) VALUES (?, ?, ?, NULL, 'active', NULL, ?, NULL)
          `,
        )
        .run("goal_aaaaaaaaaaaaaaaa", "Ship Sprint 6", 10, 3_000);

      legacyDb
        .prepare(
          `
            INSERT INTO traits (label, strength, last_reinforced, last_decayed)
            VALUES (?, ?, ?, NULL)
          `,
        )
        .run("patient", 0.6, 4_000);

      legacyDb
        .prepare(
          `
            INSERT INTO autobiographical_periods (
              id, label, start_ts, end_ts, narrative, key_episode_ids, themes, created_at, last_updated
            ) VALUES (?, ?, ?, NULL, ?, '[]', '[]', ?, ?)
          `,
        )
        .run("abp_aaaaaaaaaaaaaaaa", "2026-Q2", 5_000, "Implementation quarter.", 5_000, 5_000);

      legacyDb
        .prepare(
          `
            INSERT INTO open_questions (
              id, question, urgency, status, related_episode_ids, related_semantic_node_ids, source,
              created_at, last_touched, resolution_episode_id, resolution_note, resolved_at,
              abandoned_reason, abandoned_at, dedupe_key
            ) VALUES (?, ?, ?, 'open', ?, '[]', ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?)
          `,
        )
        .run(
          "oq_aaaaaaaaaaaaaaaa",
          "Why did Atlas fail?",
          0.8,
          JSON.stringify([relatedEpisodeId]),
          "reflection",
          6_000,
          6_000,
          buildOpenQuestionDedupeKey({
            question: "Why did Atlas fail?",
            relatedEpisodeIds: [relatedEpisodeId],
            relatedSemanticNodeIds: [],
          }),
        );
      legacyDb
        .prepare(
          `
            INSERT INTO open_questions (
              id, question, urgency, status, related_episode_ids, related_semantic_node_ids, source,
              created_at, last_touched, resolution_episode_id, resolution_note, resolved_at,
              abandoned_reason, abandoned_at, dedupe_key
            ) VALUES (?, ?, ?, 'open', '[]', '[]', ?, ?, ?, NULL, NULL, NULL, NULL, NULL, ?)
          `,
        )
        .run(
          "oq_bbbbbbbbbbbbbbbb",
          "What principle applies here?",
          0.5,
          "user",
          7_000,
          7_000,
          buildOpenQuestionDedupeKey({
            question: "What principle applies here?",
            relatedEpisodeIds: [],
            relatedSemanticNodeIds: [],
          }),
        );
    } finally {
      legacyDb.close();
    }

    const db = openDatabase(dbPath, {
      migrations: selfMigrations,
    });
    const values = new ValuesRepository({ db, clock: new FixedClock(10_000) });
    const goals = new GoalsRepository({ db, clock: new FixedClock(10_000) });
    const traits = new TraitsRepository({ db, clock: new FixedClock(10_000) });
    const periods = new AutobiographicalRepository({ db, clock: new FixedClock(10_000) });
    const openQuestions = new OpenQuestionsRepository({ db, clock: new FixedClock(10_000) });

    try {
      expect(values.list()).toEqual([
        expect.objectContaining({
          id: "val_bbbbbbbbbbbbbbbb",
          provenance: { kind: "system" },
        }),
        expect.objectContaining({
          id: "val_aaaaaaaaaaaaaaaa",
          provenance: {
            kind: "episodes",
            episode_ids: [relatedEpisodeId],
          },
        }),
      ]);
      expect(goals.list({ status: "active" })[0]).toEqual(
        expect.objectContaining({
          id: "goal_aaaaaaaaaaaaaaaa",
          provenance: { kind: "system" },
        }),
      );
      expect(traits.list()[0]).toEqual(
        expect.objectContaining({
          label: "patient",
          provenance: { kind: "system" },
        }),
      );
      expect(traits.list()[0]?.id).toMatch(/^trt_/);
      expect(periods.listPeriods({ limit: 10 })[0]).toEqual(
        expect.objectContaining({
          id: "abp_aaaaaaaaaaaaaaaa",
          provenance: { kind: "system" },
        }),
      );
      expect(openQuestions.list({ limit: 10 })).toEqual([
        expect.objectContaining({
          id: "oq_aaaaaaaaaaaaaaaa",
          provenance: {
            kind: "episodes",
            episode_ids: [relatedEpisodeId],
          },
        }),
        expect.objectContaining({
          id: "oq_bbbbbbbbbbbbbbbb",
          provenance: { kind: "system" },
        }),
      ]);
    } finally {
      db.close();
    }
  });
});
