import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { composeMigrations, openDatabase } from "../storage/sqlite/index.js";
import { streamWatermarkMigrations } from "../stream/index.js";
import { ManualClock } from "../util/clock.js";
import { autonomyMigrations } from "./migrations.js";
import { ScheduledWakesRepository } from "./scheduled-wakes-repository.js";

const NOW = 1_000_000;
const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function createRepo() {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-scheduled-wakes-"));
  tempDirs.push(tempDir);
  const clock = new ManualClock(NOW);
  const db = openDatabase(join(tempDir, "sw.db"), {
    migrations: composeMigrations(autonomyMigrations, streamWatermarkMigrations),
  });
  const repository = new ScheduledWakesRepository({ db, clock });
  return { repository, clock, db };
}

describe("ScheduledWakesRepository", () => {
  it("schedules a one-time wake at now + delay and surfaces it only once due", () => {
    const { repository, db } = createRepo();
    const wake = repository.schedule({
      delaySeconds: 3600,
      note: "Revisit the Wren naming question",
      originSessionId: "default",
    });

    expect(wake.fire_at).toBe(NOW + 3_600_000);
    expect(wake.status).toBe("pending");
    expect(wake.note).toBe("Revisit the Wren naming question");
    expect(wake.origin_session_id).toBe("default");

    expect(repository.listDuePending(NOW)).toEqual([]);
    expect(repository.listDuePending(NOW + 3_600_000)).toHaveLength(1);
    db.close();
  });

  it("rejects a non-positive delay and an empty note", () => {
    const { repository, db } = createRepo();
    expect(() => repository.schedule({ delaySeconds: 0, note: "x" })).toThrow();
    expect(() => repository.schedule({ delaySeconds: -5, note: "x" })).toThrow();
    expect(() => repository.schedule({ delaySeconds: 10, note: "   " })).toThrow();
    db.close();
  });

  it("marks fired and excludes from due-pending", () => {
    const { repository, db } = createRepo();
    const wake = repository.schedule({ delaySeconds: 60, note: "n" });
    repository.markFired([wake.id], NOW + 60_000);

    expect(repository.get(wake.id)?.status).toBe("fired");
    expect(repository.get(wake.id)?.fired_at).toBe(NOW + 60_000);
    expect(repository.listDuePending(NOW + 60_000)).toEqual([]);
    db.close();
  });

  it("cancels a pending wake and no-ops on an already-resolved one", () => {
    const { repository, db } = createRepo();
    const wake = repository.schedule({ delaySeconds: 60, note: "n" });

    const cancelled = repository.cancel(wake.id);
    expect(cancelled?.status).toBe("cancelled");
    expect(repository.listDuePending(NOW + 60_000)).toEqual([]);
    expect(repository.cancel(wake.id)).toBeNull();
    db.close();
  });

  it("cancel no-ops once the wake has fired and leaves it fired", () => {
    const { repository, db } = createRepo();
    const wake = repository.schedule({ delaySeconds: 60, note: "n" });
    repository.markFired([wake.id], NOW + 60_000);

    expect(repository.cancel(wake.id)).toBeNull();
    expect(repository.get(wake.id)?.status).toBe("fired");
    db.close();
  });

  it("lists by status ordered by fire time", () => {
    const { repository, db } = createRepo();
    const a = repository.schedule({ delaySeconds: 120, note: "a" });
    const b = repository.schedule({ delaySeconds: 60, note: "b" });

    expect(repository.list({ status: "pending", limit: 10 }).map((w) => w.id)).toEqual([
      b.id,
      a.id,
    ]);

    repository.cancel(a.id);
    expect(repository.list({ status: "cancelled", limit: 10 }).map((w) => w.id)).toEqual([a.id]);
    expect(repository.list({ status: "pending", limit: 10 }).map((w) => w.id)).toEqual([b.id]);
    db.close();
  });
});
