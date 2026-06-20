import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEntityId, createEpisodeId, createStreamEntryId } from "../../util/ids.js";
import {
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "../common/disclosure-label.js";
import { activityMigrations } from "./migrations.js";
import { LivedExperienceDaySummaryRepository } from "./lived-experience-day-summary.js";

const tempDirs: string[] = [];

afterEach(() => {
  for (const tempDir of tempDirs.splice(0)) {
    rmSync(tempDir, { recursive: true, force: true });
  }
});

describe("LivedExperienceDaySummaryRepository", () => {
  it("idempotently upserts by self entity and UTC day, then lists overlapping windows", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-lived-day-summary-"));
    tempDirs.push(tempDir);
    const db = openDatabase(join(tempDir, "activity.db"), {
      migrations: composeMigrations(activityMigrations),
    });
    const repository = new LivedExperienceDaySummaryRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const selfEntityId = createEntityId();
    const audienceEntityId = createEntityId();
    const episodeId = createEpisodeId();
    const sourceStreamEntryId = createStreamEntryId();
    const dayStartMs = Date.UTC(2026, 5, 15);
    const dayEndMs = dayStartMs + 24 * 60 * 60_000 - 1;

    const created = repository.upsert({
      selfEntityId,
      utcDay: "2026-06-15",
      dayStartMs,
      dayEndMs,
      gist: "I held one repeated restraint across the day.",
      salience: 0.6,
      countsSnapshot: {
        activity: { conversation_turn_count: 2 },
        self_decisions: { decision_count: 4 },
      },
      sourceEpisodeIds: [episodeId],
      sourceStreamEntryIds: [sourceStreamEntryId],
      disclosureLabel: selfPrivateMemoryDisclosureLabel([audienceEntityId]),
      provenance: { kind: "offline", process: "lived-experience-day-summarizer" },
    });
    const updated = repository.upsert({
      selfEntityId,
      utcDay: "2026-06-15",
      dayStartMs,
      dayEndMs,
      gist: "I kept the same restraint and noticed one new structure.",
      countsSnapshot: {
        activity: { conversation_turn_count: 3 },
        self_decisions: { decision_count: 5 },
      },
      disclosureLabel: unknownMemoryDisclosureLabel([audienceEntityId]),
      provenance: { kind: "offline", process: "lived-experience-day-summarizer" },
      updatedAt: 12_000,
    });

    expect(updated.id).toBe(created.id);
    expect(updated.created_at).toBe(created.created_at);
    expect(updated.updated_at).toBe(12_000);
    expect(updated.gist).toBe("I kept the same restraint and noticed one new structure.");
    expect(updated.disclosure_label).toMatchObject({
      disclosureClass: "unknown",
      originAudienceEntityIds: [audienceEntityId],
    });
    expect(repository.getByDay(selfEntityId, "2026-06-15")).toEqual(updated);
    expect(
      repository.listForWindow({
        selfEntityId,
        fromMs: dayStartMs + 1,
        toMs: dayEndMs - 1,
        limit: 10,
      }),
    ).toEqual([updated]);
    expect(
      repository.listForWindow({
        selfEntityId,
        fromMs: dayEndMs + 1,
        toMs: dayEndMs + 2,
        limit: 10,
      }),
    ).toEqual([]);

    db.close();
  });
});
