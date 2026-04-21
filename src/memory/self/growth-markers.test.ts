import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEpisodeId } from "../../util/ids.js";

import { AutobiographicalRepository } from "./autobiographical.js";
import { GrowthMarkersRepository } from "./growth-markers.js";
import { selfMigrations } from "./migrations.js";

describe("GrowthMarkersRepository", () => {
  it("adds markers, filters them, and summarizes by period", () => {
    const clock = new FixedClock(10_000);
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const periods = new AutobiographicalRepository({
      db,
      clock,
    });
    const repository = new GrowthMarkersRepository({
      db,
      clock,
    });
    const episodeId = createEpisodeId();
    const period = periods.upsertPeriod({
      label: "2026-Q2",
      start_ts: 1_000,
      end_ts: 20_000,
      narrative: "A growth period.",
      key_episode_ids: [episodeId],
      themes: ["learning"],
    });

    repository.add({
      ts: 5_000,
      category: "understanding",
      what_changed: "Understood the release graph.",
      evidence_episode_ids: [episodeId],
      confidence: 0.6,
      source_process: "manual",
    });
    repository.add({
      ts: 6_000,
      category: "skill",
      what_changed: "Improved rollback drills.",
      evidence_episode_ids: [createEpisodeId()],
      confidence: 0.5,
      source_process: "manual",
    });

    expect(repository.list({ category: "understanding" })).toHaveLength(1);
    expect(repository.summarize({ periodId: period.id })).toMatchObject({
      counts: {
        understanding: 1,
      },
    });

    db.close();
  });

  it("rejects empty evidence", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new GrowthMarkersRepository({
      db,
      clock: new FixedClock(10_000),
    });

    expect(() =>
      repository.add({
        ts: 5_000,
        category: "understanding",
        what_changed: "Invalid marker",
        evidence_episode_ids: [],
        confidence: 0.4,
        source_process: "manual",
      }),
    ).toThrow();

    db.close();
  });
});
