import { afterEach, describe, expect, it } from "vitest";

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
});
