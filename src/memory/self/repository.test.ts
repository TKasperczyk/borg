import { describe, expect, it } from "vitest";

import { FixedClock, ManualClock } from "../../util/clock.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { ProvenanceError } from "../../util/errors.js";
import { selfMigrations } from "./migrations.js";
import { GoalsRepository, TraitsRepository, ValuesRepository } from "./repository.js";

describe("self repositories", () => {
  const manualProvenance = { kind: "manual" } as const;
  const episodeProvenance = {
    kind: "episodes",
    episode_ids: ["ep_aaaaaaaaaaaaaaaa" as const],
  } as const;

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
        provenance: manualProvenance,
      });

      values.bindToEpisode(value.id, "ep_aaaaaaaaaaaaaaaa" as never);
      values.affirm(value.id, 200);

      expect(values.list()).toEqual([
        expect.objectContaining({
          id: value.id,
          label: "curiosity",
          last_affirmed: 200,
          provenance: episodeProvenance,
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
        provenance: manualProvenance,
      });
      const child = goals.add({
        description: "Write extractor tests",
        priority: 8,
        parentId: parent.id,
        provenance: manualProvenance,
      });

      goals.updateProgress(child.id, "Covered happy path and dedup.", manualProvenance);
      goals.updateStatus(child.id, "done", manualProvenance);

      expect(goals.list()).toEqual([
        expect.objectContaining({
          id: parent.id,
          children: [
            expect.objectContaining({
              id: child.id,
              status: "done",
              progress_notes: "Covered happy path and dedup.",
              provenance: manualProvenance,
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
      traits.reinforce({
        label: "patient",
        delta: 0.8,
        provenance: manualProvenance,
        timestamp: 0,
      });
      clock.advance(24 * 3_600_000);
      traits.decay(24, clock.now());
      traits.reinforce({
        label: "decisive",
        delta: 0.2,
        provenance: episodeProvenance,
        timestamp: clock.now(),
      });

      const listed = traits.list();
      expect(listed[0]).toEqual(
        expect.objectContaining({
          label: "patient",
          provenance: manualProvenance,
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

  it("rejects invalid stored value provenance episode ids", () => {
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
        provenance: manualProvenance,
      });

      db.prepare(
        `
          UPDATE "values"
          SET provenance_kind = 'episodes', provenance_episode_ids = ?
          WHERE id = ?
        `,
      ).run(
        '["not-an-episode-id"]',
        value.id,
      );

      expect(() => values.list()).toThrow();
    } finally {
      db.close();
    }
  });

  it("rejects provenance-less creates and updates", () => {
    const db = openDatabase(":memory:", {
      migrations: [...selfMigrations],
    });
    const values = new ValuesRepository({ db, clock: new FixedClock(100) });
    const goals = new GoalsRepository({ db, clock: new FixedClock(100) });
    const traits = new TraitsRepository({ db, clock: new FixedClock(100) });

    try {
      expect(() =>
        values.add({
          label: "clarity",
          description: "Prefer explicit state.",
          priority: 1,
          provenance: undefined as never,
        }),
      ).toThrow(ProvenanceError);

      const goal = goals.add({
        description: "Ship Sprint 6",
        priority: 1,
        provenance: manualProvenance,
      });

      expect(() =>
        goals.updateProgress(goal.id, "Updated", undefined as never),
      ).toThrow(ProvenanceError);
      expect(() =>
        goals.updateStatus(goal.id, "done", undefined as never),
      ).toThrow(ProvenanceError);
      expect(() =>
        traits.reinforce({
          label: "patient",
          delta: 0.2,
          provenance: undefined as never,
        }),
      ).toThrow(ProvenanceError);
    } finally {
      db.close();
    }
  });
});
