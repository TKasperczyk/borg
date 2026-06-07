import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { createEntityId } from "../../util/ids.js";
import { trainOfThoughtMigrations } from "./migrations.js";
import { TrainOfThoughtRepository } from "./repository.js";

describe("TrainOfThoughtRepository", () => {
  it("upserts a single self-private train of thought row", () => {
    const clock = new ManualClock(1_000);
    const db = openDatabase(":memory:", {
      migrations: trainOfThoughtMigrations,
    });
    const repository = new TrainOfThoughtRepository({
      db,
      clock,
    });
    const selfEntityId = createEntityId();

    try {
      expect(repository.get()).toBeNull();

      const first = repository.upsert({
        text: "First private thought.",
        selfEntityId,
      });

      expect(first).toMatchObject({
        text: "First private thought.",
        self_entity_id: selfEntityId,
        disclosure_class: "self_private",
        created_at: 1_000,
        updated_at: 1_000,
      });

      clock.set(2_000);
      const second = repository.upsert({
        text: "Second private thought.",
        selfEntityId,
      });

      expect(second).toMatchObject({
        text: "Second private thought.",
        self_entity_id: selfEntityId,
        disclosure_class: "self_private",
        created_at: 1_000,
        updated_at: 2_000,
      });
      expect(repository.get()).toEqual(second);
      expect(
        db.prepare("SELECT COUNT(*) AS count FROM train_of_thought").get() as { count: number },
      ).toMatchObject({ count: 1 });
    } finally {
      db.close();
    }
  });
});
