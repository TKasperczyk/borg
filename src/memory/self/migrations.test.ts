import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { parseEpisodeId } from "../../util/ids.js";
import { identityMigrations } from "../identity/index.js";

import { AutobiographicalRepository } from "./autobiographical.js";
import { GrowthMarkersRepository } from "./growth-markers.js";
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

  it("backfills evidence-backed fields from reinforcement history", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "self.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: selfMigrations.filter((migration) => migration.id < 260),
    });
    const relatedEpisodeIds = [
      parseEpisodeId("ep_aaaaaaaaaaaaaaaa"),
      parseEpisodeId("ep_bbbbbbbbbbbbbbbb"),
      parseEpisodeId("ep_cccccccccccccccc"),
    ];

    try {
      legacyDb
        .prepare(
          `
            INSERT INTO "values" (
              id, label, description, priority, created_at, last_affirmed,
              provenance_kind, provenance_episode_ids, provenance_process, state, established_at
            ) VALUES (?, ?, ?, ?, ?, NULL, 'episodes', ?, NULL, 'established', ?)
          `,
        )
        .run(
          "val_aaaaaaaaaaaaaaaa",
          "clarity",
          "Prefer explicit state.",
          1,
          1_000,
          JSON.stringify([relatedEpisodeIds[0]]),
          3_000,
        );
      legacyDb
        .prepare(
          `
            INSERT INTO value_reinforcement_events (
              value_id, ts, provenance_kind, provenance_episode_ids, provenance_process
            ) VALUES (?, ?, 'episodes', ?, NULL)
          `,
        )
        .run("val_aaaaaaaaaaaaaaaa", 1_000, JSON.stringify([relatedEpisodeIds[0]]));
      legacyDb
        .prepare(
          `
            INSERT INTO value_reinforcement_events (
              value_id, ts, provenance_kind, provenance_episode_ids, provenance_process
            ) VALUES (?, ?, 'episodes', ?, NULL)
          `,
        )
        .run("val_aaaaaaaaaaaaaaaa", 2_000, JSON.stringify([relatedEpisodeIds[1]]));
      legacyDb
        .prepare(
          `
            INSERT INTO value_reinforcement_events (
              value_id, ts, provenance_kind, provenance_episode_ids, provenance_process
            ) VALUES (?, ?, 'episodes', ?, NULL)
          `,
        )
        .run("val_aaaaaaaaaaaaaaaa", 3_000, JSON.stringify([relatedEpisodeIds[2]]));

      legacyDb
        .prepare(
          `
            INSERT INTO traits (
              id, label, strength, last_reinforced, last_decayed,
              provenance_kind, provenance_episode_ids, provenance_process, state, established_at
            ) VALUES (?, ?, ?, ?, NULL, 'episodes', ?, NULL, 'candidate', NULL)
          `,
        )
        .run(
          "trt_aaaaaaaaaaaaaaaa",
          "warm",
          0.4,
          4_000,
          JSON.stringify([relatedEpisodeIds[0]]),
        );
      legacyDb
        .prepare(
          `
            INSERT INTO trait_reinforcement_events (
              trait_id, delta, ts, provenance_kind, provenance_episode_ids, provenance_process
            ) VALUES (?, ?, ?, 'episodes', ?, NULL)
          `,
        )
        .run("trt_aaaaaaaaaaaaaaaa", 0.1, 4_000, JSON.stringify([relatedEpisodeIds[0]]));
      legacyDb
        .prepare(
          `
            INSERT INTO trait_reinforcement_events (
              trait_id, delta, ts, provenance_kind, provenance_episode_ids, provenance_process
            ) VALUES (?, ?, ?, 'episodes', ?, NULL)
          `,
        )
        .run("trt_aaaaaaaaaaaaaaaa", 0.1, 5_000, JSON.stringify([relatedEpisodeIds[1]]));
    } finally {
      legacyDb.close();
    }

    const db = openDatabase(dbPath, {
      migrations: selfMigrations,
    });
    const values = new ValuesRepository({ db, clock: new FixedClock(10_000) });
    const traits = new TraitsRepository({ db, clock: new FixedClock(10_000) });

    try {
      expect(values.get("val_aaaaaaaaaaaaaaaa" as never)).toEqual(
        expect.objectContaining({
          support_count: 3,
          contradiction_count: 0,
          last_tested_at: 3_000,
          last_contradicted_at: null,
          confidence: 5 / 6,
          evidence_episode_ids: [
            relatedEpisodeIds[2],
            relatedEpisodeIds[1],
            relatedEpisodeIds[0],
          ],
        }),
      );

      expect(traits.get("trt_aaaaaaaaaaaaaaaa" as never)).toEqual(
        expect.objectContaining({
          support_count: 2,
          contradiction_count: 0,
          last_tested_at: 5_000,
          last_contradicted_at: null,
          confidence: 4 / 5,
          evidence_episode_ids: [relatedEpisodeIds[1], relatedEpisodeIds[0]],
        }),
      );
    } finally {
      db.close();
    }
  });

  it("backfills goal last_progress_ts from identity events when progress notes changed", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "self.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: [...selfMigrations.filter((migration) => migration.id < 262), ...identityMigrations],
    });

    try {
      legacyDb
        .prepare(
          `
            INSERT INTO goals (
              id, description, priority, parent_goal_id, status, progress_notes, created_at, target_at,
              provenance_kind, provenance_episode_ids, provenance_process
            ) VALUES (?, ?, ?, NULL, 'active', ?, ?, NULL, 'manual', '[]', NULL)
          `,
        )
        .run("goal_aaaaaaaaaaaaaaaa", "Ship Sprint 11", 10, "Made initial progress.", 1_000);
      legacyDb
        .prepare(
          `
            INSERT INTO goals (
              id, description, priority, parent_goal_id, status, progress_notes, created_at, target_at,
              provenance_kind, provenance_episode_ids, provenance_process
            ) VALUES (?, ?, ?, NULL, 'active', NULL, ?, NULL, 'manual', '[]', NULL)
          `,
        )
        .run("goal_bbbbbbbbbbbbbbbb", "Keep notes tidy", 4, 2_000);

      legacyDb
        .prepare(
          `
            INSERT INTO identity_events (
              record_type, record_id, action, old_value_json, new_value_json, reason,
              provenance_kind, provenance_episode_ids, provenance_process, review_item_id,
              overwrite_without_review, ts
            ) VALUES ('goal', ?, ?, ?, ?, NULL, 'manual', '[]', NULL, NULL, 0, ?)
          `,
        )
        .run(
          "goal_aaaaaaaaaaaaaaaa",
          "update",
          JSON.stringify({
            progress_notes: null,
          }),
          JSON.stringify({
            progress_notes: "Made initial progress.",
          }),
          3_000,
        );
      legacyDb
        .prepare(
          `
            INSERT INTO identity_events (
              record_type, record_id, action, old_value_json, new_value_json, reason,
              provenance_kind, provenance_episode_ids, provenance_process, review_item_id,
              overwrite_without_review, ts
            ) VALUES ('goal', ?, ?, ?, ?, NULL, 'manual', '[]', NULL, NULL, 0, ?)
          `,
        )
        .run(
          "goal_aaaaaaaaaaaaaaaa",
          "update_progress",
          JSON.stringify({
            progress_notes: "Made initial progress.",
          }),
          JSON.stringify({
            progress_notes: "Made initial progress.\nClosed the remaining gaps.",
          }),
          5_000,
        );
      legacyDb
        .prepare(
          `
            INSERT INTO identity_events (
              record_type, record_id, action, old_value_json, new_value_json, reason,
              provenance_kind, provenance_episode_ids, provenance_process, review_item_id,
              overwrite_without_review, ts
            ) VALUES ('goal', ?, ?, ?, ?, NULL, 'manual', '[]', NULL, NULL, 0, ?)
          `,
        )
        .run(
          "goal_bbbbbbbbbbbbbbbb",
          "update",
          JSON.stringify({
            progress_notes: null,
          }),
          JSON.stringify({
            progress_notes: null,
          }),
          4_000,
        );
    } finally {
      legacyDb.close();
    }

    const db = openDatabase(dbPath, {
      migrations: [...selfMigrations, ...identityMigrations],
    });
    const goals = new GoalsRepository({ db, clock: new FixedClock(10_000) });

    try {
      expect(goals.get("goal_aaaaaaaaaaaaaaaa" as never)).toEqual(
        expect.objectContaining({
          last_progress_ts: 5_000,
        }),
      );
      expect(goals.get("goal_bbbbbbbbbbbbbbbb" as never)).toEqual(
        expect.objectContaining({
          last_progress_ts: null,
        }),
      );
    } finally {
      db.close();
    }
  });

  it("backfills growth marker provenance from evidence episodes and source process", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "self.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: selfMigrations.filter((migration) => migration.id < 263),
    });
    const evidenceEpisodeId = parseEpisodeId("ep_aaaaaaaaaaaaaaaa");

    try {
      legacyDb
        .prepare(
          `
            INSERT INTO growth_markers (
              id, ts, category, what_changed, before_description, after_description,
              evidence_episode_ids, confidence, source_process, created_at
            ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
          `,
        )
        .run(
          "grw_aaaaaaaaaaaaaaaa",
          1_000,
          "understanding",
          "Episode-backed marker",
          JSON.stringify([evidenceEpisodeId]),
          0.7,
          "self-narrator",
          1_000,
        );
      legacyDb
        .prepare(
          `
            INSERT INTO growth_markers (
              id, ts, category, what_changed, before_description, after_description,
              evidence_episode_ids, confidence, source_process, created_at
            ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
          `,
        )
        .run(
          "grw_bbbbbbbbbbbbbbbb",
          2_000,
          "habit",
          "Offline-only marker",
          "[]",
          0.5,
          "ruminator",
          2_000,
        );
      legacyDb
        .prepare(
          `
            INSERT INTO growth_markers (
              id, ts, category, what_changed, before_description, after_description,
              evidence_episode_ids, confidence, source_process, created_at
            ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
          `,
        )
        .run(
          "grw_cccccccccccccccc",
          3_000,
          "skill",
          "Fallback marker",
          "[]",
          0.4,
          "",
          3_000,
        );
    } finally {
      legacyDb.close();
    }

    const db = openDatabase(dbPath, {
      migrations: selfMigrations,
    });
    const markers = new GrowthMarkersRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      expect(markers.get("grw_aaaaaaaaaaaaaaaa" as never)).toEqual(
        expect.objectContaining({
          provenance: {
            kind: "episodes",
            episode_ids: [evidenceEpisodeId],
          },
        }),
      );
      expect(markers.get("grw_bbbbbbbbbbbbbbbb" as never)).toEqual(
        expect.objectContaining({
          provenance: {
            kind: "offline",
            process: "ruminator",
          },
        }),
      );
      expect(markers.get("grw_cccccccccccccccc" as never)).toEqual(
        expect.objectContaining({
          provenance: {
            kind: "offline",
            process: "growth-marker-detector",
          },
        }),
      );
    } finally {
      db.close();
    }
  });

  it("upgrades self evidence-event provenance checks to allow online writes", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const dbPath = join(tempDir, "self.db");
    tempDirs.push(tempDir);

    const legacyDb = openDatabase(dbPath, {
      migrations: selfMigrations.filter((migration) => migration.id < 264),
    });
    legacyDb.close();

    const db = openDatabase(dbPath, {
      migrations: selfMigrations,
    });
    const clock = new FixedClock(12_000);
    const values = new ValuesRepository({
      db,
      clock,
    });
    const traits = new TraitsRepository({
      db,
      clock,
    });

    try {
      const value = values.add({
        label: "clarity",
        description: "Prefer explicit state.",
        priority: 1,
        provenance: {
          kind: "manual",
        },
      });
      values.reinforce(value.id, {
        kind: "online",
        process: "reflector",
      });
      values.recordContradiction({
        valueId: value.id,
        provenance: {
          kind: "online",
          process: "overseer",
        },
      });

      traits.reinforce({
        label: "patient",
        delta: 0.2,
        provenance: {
          kind: "manual",
        },
      });
      const trait = traits.reinforce({
        label: "patient",
        delta: 0.1,
        provenance: {
          kind: "online",
          process: "reflector",
        },
      });
      traits.recordContradiction({
        label: "patient",
        provenance: {
          kind: "online",
          process: "overseer",
        },
      });

      expect(values.listReinforcementEvents(value.id)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            provenance: {
              kind: "online",
              process: "reflector",
            },
          }),
        ]),
      );
      expect(values.listContradictionEvents(value.id)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            provenance: {
              kind: "online",
              process: "overseer",
            },
          }),
        ]),
      );
      expect(traits.listReinforcementEvents(trait.id)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            provenance: {
              kind: "online",
              process: "reflector",
            },
          }),
        ]),
      );
      expect(traits.listContradictionEvents(trait.id)).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            provenance: {
              kind: "online",
              process: "overseer",
            },
          }),
        ]),
      );
    } finally {
      db.close();
    }
  });
});
