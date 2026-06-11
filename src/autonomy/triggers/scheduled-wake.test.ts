import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import { StreamWatermarkRepository, streamWatermarkMigrations } from "../../stream/index.js";
import { formatAutonomyTriggerContext } from "../../cognition/autonomy-trigger.js";
import { ManualClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";
import { autonomyMigrations } from "../migrations.js";
import { ScheduledWakesRepository } from "../scheduled-wakes-repository.js";
import { createScheduledWakeTrigger } from "./scheduled-wake.js";

const NOW = 1_000_000;
const tempDirs: string[] = [];

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    rmSync(dir, { recursive: true, force: true });
  }
});

function setup() {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-scheduled-wake-trigger-"));
  tempDirs.push(tempDir);
  const clock = new ManualClock(NOW);
  const db = openDatabase(join(tempDir, "sw.db"), {
    migrations: composeMigrations(autonomyMigrations, streamWatermarkMigrations),
  });
  const scheduledWakesRepository = new ScheduledWakesRepository({ db, clock });
  const watermarkRepository = new StreamWatermarkRepository({ db, clock });
  const trigger = createScheduledWakeTrigger({
    scheduledWakesRepository,
    watermarkRepository,
    clock,
  });
  return { trigger, scheduledWakesRepository, watermarkRepository, clock, db };
}

describe("scheduled wake trigger", () => {
  it("emits a due wake once, seals it via watermark, and reconciles it to fired", async () => {
    const { trigger, scheduledWakesRepository, watermarkRepository, clock, db } = setup();
    const wake = scheduledWakesRepository.schedule({ delaySeconds: 60, note: "check in" });

    expect(await trigger.scan()).toEqual([]);

    clock.advance(60_000);
    const due = await trigger.scan();
    expect(due).toHaveLength(1);
    expect(due[0]?.id).toBe(wake.id);
    expect(due[0]?.sortTs).toBe(wake.fire_at);
    expect(due[0]?.payload).toEqual({
      note: "check in",
      scheduled_at: NOW,
      fire_at: NOW + 60_000,
      disclosure: expect.stringContaining("disclosure_class=self_private"),
      disclosure_label: {
        disclosure_class: "self_private",
        origin_audience_entity_ids: [],
        private_to_entity_ids: [],
        public_to_entity_ids: [],
      },
    });

    // The scheduler seals a fire-watermark on success.
    watermarkRepository.set(due[0]!.watermarkProcessName, DEFAULT_SESSION_ID, {
      lastTs: due[0]!.sortTs,
      lastEntryId: due[0]!.id,
    });

    // Next scan: no re-fire, and the row is reconciled to fired.
    expect(await trigger.scan()).toEqual([]);
    expect(scheduledWakesRepository.get(wake.id)?.status).toBe("fired");
    db.close();
  });

  it("builds a self-audience turn carrying the note", async () => {
    const { trigger, scheduledWakesRepository, clock, db } = setup();
    scheduledWakesRepository.schedule({ delaySeconds: 30, note: "follow up with Tom" });
    clock.advance(30_000);

    const [event] = await trigger.scan();
    const turn = trigger.buildTurn(event!);

    expect(turn.audience).toBe("self");
    expect(turn.autonomyTrigger?.source_name).toBe("scheduled_wake");
    expect(turn.autonomyTrigger?.payload).toMatchObject({ note: "follow up with Tom" });
    expect(formatAutonomyTriggerContext(turn.autonomyTrigger!)).toContain("self_private");
    db.close();
  });

  it("never emits a cancelled wake", async () => {
    const { trigger, scheduledWakesRepository, clock, db } = setup();
    const wake = scheduledWakesRepository.schedule({ delaySeconds: 30, note: "n" });
    scheduledWakesRepository.cancel(wake.id);
    clock.advance(30_000);

    expect(await trigger.scan()).toEqual([]);
    db.close();
  });

  it("onFired marks the row fired immediately (closing the cancel-after-fire window)", async () => {
    const { trigger, scheduledWakesRepository, clock, db } = setup();
    const wake = scheduledWakesRepository.schedule({ delaySeconds: 60, note: "n" });
    clock.advance(60_000);

    const [event] = await trigger.scan();
    await trigger.onFired?.(event!);

    expect(scheduledWakesRepository.get(wake.id)?.status).toBe("fired");
    expect(scheduledWakesRepository.cancel(wake.id)).toBeNull();
    db.close();
  });

  it("emits multiple due wakes in fire-time order", async () => {
    const { trigger, scheduledWakesRepository, clock, db } = setup();
    const later = scheduledWakesRepository.schedule({ delaySeconds: 120, note: "later" });
    const sooner = scheduledWakesRepository.schedule({ delaySeconds: 60, note: "sooner" });
    clock.advance(120_000);

    const due = await trigger.scan();
    expect(due.map((event) => event.id)).toEqual([sooner.id, later.id]);
    db.close();
  });

  it("describes the next pending wake without reconciling rows", async () => {
    const { trigger, scheduledWakesRepository, clock, db } = setup();
    const wake = scheduledWakesRepository.schedule({ delaySeconds: 60, note: "sooner" });
    scheduledWakesRepository.schedule({ delaySeconds: 120, note: "later" });

    await expect(trigger.nextDueAt!()).resolves.toBe(wake.fire_at);

    clock.advance(120_000);
    await expect(trigger.nextDueAt!()).resolves.toBe(clock.now());
    expect(scheduledWakesRepository.get(wake.id)?.status).toBe("pending");
    db.close();
  });
});
