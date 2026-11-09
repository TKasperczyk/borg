import { describe, expect, it } from "vitest";
import { openDatabase } from "../../storage/sqlite/index.js";
import { actionMigrations } from "./migrations.js";
import { createActionId, createStreamEntryId } from "../../util/ids.js";
import { makeActionRecord } from "../../test-support/factories/memory.js";
import { ActionRepository } from "./repository.js";

describe("action declined migration", () => {
  it("preserves legacy rows, deployment columns, indexes, triggers and inbound references", () => {
    const db = openDatabase(":memory:", { migrations: actionMigrations.slice(0, 1) });
    try {
      db.exec(
        "ALTER TABLE action_records ADD COLUMN deployment_note TEXT; CREATE INDEX deployment_action_note ON action_records(deployment_note); CREATE TABLE action_observer(id TEXT REFERENCES action_records(id)); CREATE TABLE action_changes(id TEXT); CREATE TRIGGER observe_action_change AFTER UPDATE ON action_records BEGIN INSERT INTO action_changes VALUES(NEW.id); END;",
      );
      const id = createActionId();
      db.prepare(
        "INSERT INTO action_records(id, description, actor, state, confidence, provenance_episode_ids, provenance_stream_entry_ids, created_at, updated_at, not_done_at, deployment_note) VALUES (?, 'opaque', 'borg', 'not_done', 0.8, '[]', ?, 100, 200, 200, 'keep')",
      ).run(id, JSON.stringify([createStreamEntryId()]));
      db.prepare("INSERT INTO action_observer VALUES (?)").run(id);
      const before = db.prepare("SELECT * FROM action_records").get();
      db.pragma("foreign_keys = OFF");
      db.transaction(() => actionMigrations[1].up(db))();
      db.pragma("foreign_keys = ON");
      expect(db.prepare("SELECT * FROM action_records WHERE id = ?").get(id)).toEqual({
        ...before,
        declined_at: null,
      });
      expect(db.pragma("foreign_key_check")).toEqual([]);
      expect(
        db.prepare("SELECT name FROM sqlite_master WHERE name = 'deployment_action_note'").get(),
      ).toBeDefined();
      const repo = new ActionRepository({ db });
      const declined = makeActionRecord({ state: "declined", declined_at: 300 });
      repo.add(declined);
      expect(repo.get(declined.id)?.declined_at).toBe(300);
      repo.update(declined.id, { confidence: 0.9 });
      expect(
        db.prepare("SELECT id FROM action_changes WHERE id = ?").get(declined.id),
      ).toBeDefined();
      expect(repo.get(id)?.state).toBe("not_done");
    } finally {
      db.close();
    }
  });
});
