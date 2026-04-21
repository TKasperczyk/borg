import { describe, expect, it } from "vitest";

import { FixedClock, ManualClock } from "../../util/clock.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { selfMigrations } from "./migrations.js";
import { GoalsRepository, TraitsRepository, ValuesRepository } from "./repository.js";

describe("self repositories", () => {
  it("manages values and episode bindings", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const values = new ValuesRepository({
      db,
      clock: new FixedClock(100),
    });

    try {
      const value = values.add({
        label: "curiosity",
        description: "Prefer learning over stasis.",
        priority: 10,
      });

      values.bindToEpisode(value.id, "ep_aaaaaaaaaaaaaaaa" as never);
      values.affirm(value.id, 200);

      expect(values.list()).toEqual([
        expect.objectContaining({
          id: value.id,
          label: "curiosity",
          last_affirmed: 200,
          source_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
        }),
      ]);

      expect(values.remove(value.id)).toBe(true);
      expect(values.list()).toEqual([]);
    } finally {
      db.close();
    }
  });

  it("manages hierarchical goals and progress", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const goals = new GoalsRepository({
      db,
      clock: new FixedClock(100),
    });

    try {
      const parent = goals.add({
        description: "Ship Sprint 2",
        priority: 10,
      });
      const child = goals.add({
        description: "Write extractor tests",
        priority: 8,
        parentId: parent.id,
      });

      goals.updateProgress(child.id, "Covered happy path and dedup.");
      goals.updateStatus(child.id, "done");

      expect(goals.list()).toEqual([
        expect.objectContaining({
          id: parent.id,
          children: [
            expect.objectContaining({
              id: child.id,
              status: "done",
              progress_notes: "Covered happy path and dedup.",
            }),
          ],
        }),
      ]);
      expect(goals.list({ status: "done" })).toEqual([
        expect.objectContaining({
          id: child.id,
        }),
      ]);
    } finally {
      db.close();
    }
  });

  it("reinforces, decays, and culls traits", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const clock = new ManualClock(0);
    const traits = new TraitsRepository({
      db,
      clock,
    });

    try {
      traits.reinforce("patient", 0.8, 0);
      clock.advance(24 * 3_600_000);
      traits.decay(24, clock.now());
      traits.reinforce("decisive", 0.2, clock.now());

      const listed = traits.list();
      expect(listed[0]).toEqual(
        expect.objectContaining({
          label: "patient",
        }),
      );
      expect(listed.find((trait) => trait.label === "patient")?.strength).toBeLessThan(0.8);
      expect(traits.cull(0.3)).toBe(1);
      expect(traits.list()).toEqual([
        expect.objectContaining({
          label: "patient",
        }),
      ]);
    } finally {
      db.close();
    }
  });

  it("rejects invalid stored value source episode ids", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const values = new ValuesRepository({
      db,
      clock: new FixedClock(100),
    });

    try {
      const value = values.add({
        label: "clarity",
        description: "Prefer explicit state.",
        priority: 1,
      });

      db.prepare("INSERT INTO value_sources (value_id, episode_id) VALUES (?, ?)").run(
        value.id,
        "not-an-episode-id",
      );

      expect(() => values.list()).toThrow();
    } finally {
      db.close();
    }
  });
});
