import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEpisodeId } from "../../util/ids.js";

import { AutobiographicalRepository } from "./autobiographical.js";
import { selfMigrations } from "./migrations.js";

describe("AutobiographicalRepository", () => {
  it("upserts, lists, closes, and updates periods", () => {
    const clock = new FixedClock(10_000);
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new AutobiographicalRepository({
      db,
      clock,
    });

    const initial = repository.upsertPeriod({
      label: "2026-Q2",
      start_ts: 1_000,
      narrative: "A planning-heavy quarter began.",
      key_episode_ids: [createEpisodeId()],
      themes: ["planning"],
    });

    expect(repository.currentPeriod()?.id).toBe(initial.id);
    expect(repository.getByLabel("2026-Q2")?.narrative).toContain("planning-heavy");
    expect(repository.listPeriods({ limit: 10 })).toHaveLength(1);

    repository.updateNarrative(
      initial.id,
      "The quarter shifted toward implementation.",
      [...initial.key_episode_ids, createEpisodeId()],
      ["planning", "implementation"],
    );
    repository.closePeriod(initial.id, 5_000);

    const updated = repository.getPeriod(initial.id);

    expect(updated).toMatchObject({
      id: initial.id,
      end_ts: 5_000,
      themes: ["planning", "implementation"],
    });
    expect(updated?.narrative).toContain("implementation");

    db.close();
  });

  it("preserves history when reopening the same label and auto-closes prior open periods", () => {
    const clock = new FixedClock(20_000);
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new AutobiographicalRepository({
      db,
      clock,
    });

    const first = repository.upsertPeriod({
      label: "2026-Q1",
      start_ts: 1_000,
      narrative: "First quarter arc.",
    });
    const second = repository.upsertPeriod({
      label: "2026-Q1",
      start_ts: 5_000,
      narrative: "A fresh chapter with the same human label.",
    });

    const periods = repository.listPeriods({
      limit: 10,
    });

    expect(periods).toHaveLength(2);
    expect(repository.currentPeriod()?.id).toBe(second.id);
    expect(repository.getPeriod(first.id)?.end_ts).toBe(5_000);
    expect(repository.getByLabel("2026-Q1")?.id).toBe(second.id);

    db.close();
  });

  it("enforces a single open period at the schema level", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });

    db.prepare(
      `
        INSERT INTO autobiographical_periods (
          id, label, start_ts, end_ts, narrative, key_episode_ids, themes, created_at, last_updated
        ) VALUES (?, ?, ?, NULL, ?, '[]', '[]', ?, ?)
      `,
    ).run("abp_aaaaaaaaaaaaaaaa", "2026-Q1", 1_000, "First open period", 1_000, 1_000);

    expect(() =>
      db
        .prepare(
          `
            INSERT INTO autobiographical_periods (
              id, label, start_ts, end_ts, narrative, key_episode_ids, themes, created_at, last_updated
            ) VALUES (?, ?, ?, NULL, ?, '[]', '[]', ?, ?)
          `,
        )
        .run("abp_bbbbbbbbbbbbbbbb", "2026-Q2", 2_000, "Second open period", 2_000, 2_000),
    ).toThrow();

    db.close();
  });
});
