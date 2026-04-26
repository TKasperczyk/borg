import { afterEach, describe, expect, it } from "vitest";

import { IdentityEventRepository } from "../../memory/identity/index.js";
import { TraitsRepository } from "../../memory/self/index.js";
import { FixedClock } from "../../util/clock.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../test-support.js";
import { CuratorProcess } from "./index.js";

const DAY_MS = 24 * 60 * 60 * 1_000;
const HOUR_MS = 60 * 60 * 1_000;

describe("curator process", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("applies promote, demote, archive, and decay policies", async () => {
    const nowMs = 100 * DAY_MS;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
    });
    cleanup.push(harness.cleanup);

    const promoteT1 = createEpisodeFixture(
      {
        title: "Warm planning note",
        created_at: nowMs - 3 * HOUR_MS,
        updated_at: nowMs - 3 * HOUR_MS,
      },
      [0, 1, 0, 0],
    );
    const promoteT2 = createEpisodeFixture(
      {
        title: "Long-lived hot note",
        created_at: nowMs - 8 * DAY_MS,
        updated_at: nowMs - 8 * DAY_MS,
      },
      [0, 1, 0, 0],
    );
    const demoteT3 = createEpisodeFixture(
      {
        title: "Cold old note",
        created_at: nowMs - 40 * DAY_MS,
        updated_at: nowMs - 40 * DAY_MS,
      },
      [0, 1, 0, 0],
    );
    const archiveEpisode = createEpisodeFixture(
      {
        title: "Archive me",
        created_at: nowMs - 50 * DAY_MS,
        updated_at: nowMs - 50 * DAY_MS,
      },
      [0, 1, 0, 0],
    );

    await harness.episodicRepository.insert(promoteT1);
    await harness.episodicRepository.insert(promoteT2);
    await harness.episodicRepository.insert(demoteT3);
    await harness.episodicRepository.insert(archiveEpisode);

    harness.episodicRepository.updateStats(promoteT1.id, {
      retrieval_count: 6,
    });
    harness.episodicRepository.updateStats(promoteT2.id, {
      tier: "T2",
      retrieval_count: 16,
      win_rate: 0.7,
    });
    harness.episodicRepository.updateStats(demoteT3.id, {
      tier: "T3",
      last_retrieved: nowMs - 31 * DAY_MS,
    });
    harness.episodicRepository.updateStats(archiveEpisode.id, {
      tier: "T1",
    });

    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: harness.traitsRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(result.errors).toEqual([]);
    expect(harness.episodicRepository.getStats(promoteT1.id)).toMatchObject({
      tier: "T2",
    });
    expect(harness.episodicRepository.getStats(promoteT2.id)).toMatchObject({
      tier: "T3",
    });
    expect(harness.episodicRepository.getStats(demoteT3.id)).toMatchObject({
      tier: "T2",
    });
    expect(harness.episodicRepository.getStats(archiveEpisode.id)).toMatchObject({
      archived: true,
    });
    expect(harness.episodicRepository.getStats(promoteT1.id)?.last_decayed_at).toBe(nowMs);
    expect(result.changes.map((change) => change.action).sort()).toEqual([
      "archive",
      "decay",
      "decay",
      "decay",
      "decay",
      "demote",
      "promote",
      "promote",
    ]);
    expect(
      harness.auditLog
        .list({ process: "curator" })
        .map((row) => row.action)
        .sort(),
    ).toEqual(["archive", "decay", "demote", "promote"]);
  });

  it("does not mutate mood_state on curator runs but still trims old mood history", async () => {
    const nowMs = 100 * DAY_MS;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
    });
    cleanup.push(harness.cleanup);

    harness.moodRepository.update("default", {
      valence: -0.6,
      arousal: 0.4,
      reason: "recent frustration",
      provenance: { kind: "system" },
    });
    harness.moodRepository.restoreHistory([
      {
        id: 10_000,
        session_id: "default",
        ts: nowMs - 120 * DAY_MS,
        valence: -0.7,
        arousal: 0.5,
        trigger_reason: "old mood",
        provenance: { kind: "system" },
      },
    ]);
    const beforeState = harness.moodRepository.listStoredStates()[0];

    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: harness.traitsRepository,
      registry: harness.registry,
    });

    const first = await process.run(harness.createContext(), {
      dryRun: false,
    });
    const second = await process.run(harness.createContext(), {
      dryRun: false,
    });
    const afterState = harness.moodRepository.listStoredStates()[0];

    expect(beforeState).toEqual(afterState);
    expect(first.changes.map((change) => change.action)).toContain("trim_mood_history");
    expect(second.changes.map((change) => change.action)).not.toContain("trim_mood_history");
    expect(harness.moodRepository.historyBefore(nowMs - 90 * DAY_MS)).toEqual([]);
  });

  it("decays stale traits through curator and records a trait identity event", async () => {
    const nowMs = 100 * DAY_MS;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
    });
    cleanup.push(harness.cleanup);

    const identityEvents = new IdentityEventRepository({
      db: harness.db,
      clock: harness.clock,
    });
    const auditedTraits = new TraitsRepository({
      db: harness.db,
      clock: harness.clock,
      identityEventRepository: identityEvents,
    });

    const staleTrait = auditedTraits.reinforce({
      label: "warm",
      delta: 0.8,
      provenance: {
        kind: "episodes",
        episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never],
      },
      timestamp: nowMs - 10 * DAY_MS,
    });
    expect(staleTrait.strength).toBe(0.8);

    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: auditedTraits,
      registry: harness.registry,
    });

    const result = await process.run(
      {
        ...harness.createContext(),
        traitsRepository: auditedTraits,
      },
      {
        dryRun: false,
      },
    );

    const decayedTrait = auditedTraits.list()[0];
    expect(result.changes.map((change) => change.action)).toContain("decay_trait");
    expect(decayedTrait?.strength ?? 1).toBeLessThan(0.8);
    expect(identityEvents.list({ recordType: "trait", recordId: decayedTrait?.id ?? "" })).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          action: "decay",
        }),
      ]),
    );

    const auditRow = harness.auditLog
      .list({ process: "curator" })
      .find((row) => row.action === "decay_trait");

    expect(auditRow).toEqual(
      expect.objectContaining({
        targets: {
          trait_id: staleTrait.id,
        },
      }),
    );

    await harness.auditLog.revert(auditRow!.id, "test");

    expect(auditedTraits.get(staleTrait.id)).toEqual(
      expect.objectContaining({
        strength: 0.8,
        last_decayed: staleTrait.last_decayed,
      }),
    );
  });
});
