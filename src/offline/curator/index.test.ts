import { afterEach, describe, expect, it } from "vitest";

import { buildConsolidationCoverageHash } from "../../memory/episodic/index.js";
import { IdentityEventRepository } from "../../memory/identity/index.js";
import { TraitsRepository } from "../../memory/self/index.js";
import { FixedClock } from "../../util/clock.js";
import { createConsolidationFamilyId } from "../../util/ids.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../test-support.js";
import { CuratorProcess } from "./index.js";

const DAY_MS = 24 * 60 * 60 * 1_000;
const HOUR_MS = 60 * 60 * 1_000;

type OfflineHarness = Awaited<ReturnType<typeof createOfflineTestHarness>>;

function recordSemanticExtractionAudit(harness: OfflineHarness, episodeIds: readonly string[]) {
  harness.auditLog.record({
    run_id: harness.createContext().runId,
    process: "semantic-extractor",
    action: "extract",
    targets: {
      episode_ids: [...episodeIds],
    },
    reversal: {
      created_node_ids: [],
      updated_nodes: [],
      created_edge_ids: [],
      updated_edges: [],
    },
  });
}

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

    await harness.episodicRepository.createEpisode(promoteT1);
    await harness.episodicRepository.createEpisode(promoteT2);
    await harness.episodicRepository.createEpisode(demoteT3);
    await harness.episodicRepository.createEpisode(archiveEpisode);

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
    recordSemanticExtractionAudit(harness, [archiveEpisode.id]);

    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: harness.traitsRepository,
      moodRepository: harness.moodRepository,
      socialRepository: harness.socialRepository,
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
    ).toEqual(["archive", "archive_episode", "decay", "demote", "promote"]);
  });

  it("round-trips an archive reversal without restoring archived through stats", async () => {
    const nowMs = 100 * DAY_MS;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
    });
    cleanup.push(harness.cleanup);
    const episode = createEpisodeFixture(
      {
        title: "Explicit archive reversal",
        significance: 0.73,
        created_at: nowMs - 50 * DAY_MS,
        updated_at: nowMs - 50 * DAY_MS,
      },
      [0, 1, 0, 0],
    );

    await harness.episodicRepository.createEpisode(episode);
    harness.episodicRepository.updateStats(episode.id, {
      tier: "T1",
      retrieval_count: 2,
      use_count: 3,
      win_rate: 0.25,
      heat_multiplier: 0.01,
      last_decayed_at: nowMs,
    });
    recordSemanticExtractionAudit(harness, [episode.id]);
    const beforeArchive = harness.episodicRepository.getStats(episode.id)!;
    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: harness.traitsRepository,
      moodRepository: harness.moodRepository,
      socialRepository: harness.socialRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });
    const archiveAudit = harness.auditLog
      .list({ process: "curator" })
      .find((row) => row.action === "archive");

    expect(result.changes.map((change) => change.action)).toEqual(["archive"]);
    expect(await harness.episodicRepository.get(episode.id)).toBeNull();
    expect(harness.episodicRepository.getStats(episode.id)).toMatchObject({
      archived: true,
    });

    harness.episodicRepository.updateStats(episode.id, {
      tier: "T2",
      retrieval_count: 20,
      use_count: 30,
      heat_multiplier: 0.8,
    });
    await harness.auditLog.revert(archiveAudit!.id, "test");

    expect(harness.episodicRepository.getStats(episode.id)).toEqual(beforeArchive);
    expect(await harness.episodicRepository.get(episode.id)).toEqual(
      expect.objectContaining({
        id: episode.id,
        significance: 0.73,
      }),
    );
    await expect(harness.episodicRepository.listUnarchivedEpisodeIds()).resolves.toContain(
      episode.id,
    );
    expect(
      harness.auditLog
        .list({ process: "curator" })
        .filter((row) => row.action === "unarchive_episode"),
    ).toHaveLength(1);
  });

  it("does not heat-archive a current consolidation version even when it meets archive criteria", async () => {
    const nowMs = 100 * DAY_MS;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
    });
    cleanup.push(harness.cleanup);

    const familyId = createConsolidationFamilyId();
    const raw = createEpisodeFixture(
      {
        title: "Covered raw leaf",
        created_at: nowMs - 60 * DAY_MS,
        updated_at: nowMs - 60 * DAY_MS,
      },
      [0, 1, 0, 0],
    );
    const coverageHash = buildConsolidationCoverageHash(raw.source_stream_ids);
    // createEpisodeFixture does not carry episode_kind/consolidation_* overrides,
    // so set them on the object directly to persist a real consolidation version.
    const version = {
      ...createEpisodeFixture(
        {
          title: "Consolidation summary",
          created_at: nowMs - 50 * DAY_MS,
          updated_at: nowMs - 50 * DAY_MS,
          lineage: { derived_from: [raw.id], supersedes: [raw.id] },
        },
        [0, 1, 0, 0],
      ),
      episode_kind: "consolidation_version" as const,
      consolidation_family_id: familyId,
      consolidation_coverage_hash: coverageHash,
    };

    await harness.episodicRepository.createEpisode(raw);
    await harness.episodicRepository.createEpisode(version);
    harness.episodicRepository.createConsolidationFamily({
      familyId,
      currentVersionEpisodeId: version.id,
      coverageHash,
      policyVersion: 1,
      members: [
        {
          raw_episode_id: raw.id,
          source_stream_ids: raw.source_stream_ids,
          added_by_version_episode_id: version.id,
        },
      ],
    });
    // The current version meets every archive criterion (tier <= T2, extracted,
    // old, cold) -- only its episode_kind should keep the curator from archiving it.
    harness.episodicRepository.updateStats(version.id, { tier: "T1" });
    recordSemanticExtractionAudit(harness, [version.id]);

    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: harness.traitsRepository,
      moodRepository: harness.moodRepository,
      socialRepository: harness.socialRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });

    expect(result.errors).toEqual([]);
    expect(result.changes.some((change) => change.action === "archive")).toBe(false);
    expect(harness.episodicRepository.getStats(version.id)?.archived).toBe(false);
  });

  it("does not archive unextracted episodes but archives extracted ones", async () => {
    const nowMs = 100 * DAY_MS;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
    });
    cleanup.push(harness.cleanup);

    const unextractedEpisode = createEpisodeFixture(
      {
        title: "Unextracted old note",
        created_at: nowMs - 50 * DAY_MS,
        updated_at: nowMs - 50 * DAY_MS,
      },
      [0, 1, 0, 0],
    );
    const extractedEpisode = createEpisodeFixture(
      {
        title: "Extracted old note",
        created_at: nowMs - 49 * DAY_MS,
        updated_at: nowMs - 49 * DAY_MS,
      },
      [0, 1, 0, 0],
    );

    await harness.episodicRepository.createEpisode(unextractedEpisode);
    await harness.episodicRepository.createEpisode(extractedEpisode);
    harness.episodicRepository.updateStats(unextractedEpisode.id, {
      tier: "T1",
    });
    harness.episodicRepository.updateStats(extractedEpisode.id, {
      tier: "T1",
    });
    recordSemanticExtractionAudit(harness, [extractedEpisode.id]);

    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: harness.traitsRepository,
      moodRepository: harness.moodRepository,
      socialRepository: harness.socialRepository,
      registry: harness.registry,
    });

    const plan = await process.plan(harness.createContext(), {});
    const archiveEpisodeIds = plan.items.flatMap((item) =>
      item.action === "archive" && "episode_id" in item ? [item.episode_id] : [],
    );

    expect(archiveEpisodeIds).toEqual([extractedEpisode.id]);
  });

  it("does not mutate mood_state on curator runs but still trims old mood history", async () => {
    const nowMs = 100 * DAY_MS;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
    });
    cleanup.push(harness.cleanup);

    harness.moodRepository.update("default" as never, {
      valence: -0.6,
      arousal: 0.4,
      reason: "recent frustration",
      provenance: { kind: "system" },
    });
    harness.moodRepository.restoreHistory([
      {
        id: 10_000,
        session_id: "default" as never,
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
      moodRepository: harness.moodRepository,
      socialRepository: harness.socialRepository,
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

  it("decays episode salience and heat based on elapsed half-lives", async () => {
    const nowMs = 100 * DAY_MS;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      configOverrides: {
        offline: {
          curator: {
            episodeSalienceHalfLifeDays: 30,
            episodeHeatHalfLifeDays: 30,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);

    const episode = createEpisodeFixture(
      {
        title: "Stale but not archival",
        significance: 0.8,
        created_at: nowMs - 30 * DAY_MS,
        updated_at: nowMs - 30 * DAY_MS,
      },
      [0, 1, 0, 0],
    );
    await harness.episodicRepository.createEpisode(episode);
    harness.episodicRepository.updateStats(episode.id, {
      tier: "T2",
      retrieval_count: 10,
      last_decayed_at: nowMs - 30 * DAY_MS,
    });

    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: harness.traitsRepository,
      moodRepository: harness.moodRepository,
      socialRepository: harness.socialRepository,
      registry: harness.registry,
    });

    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });
    const decayedEpisode = await harness.episodicRepository.get(episode.id);
    const decayedStats = harness.episodicRepository.getStats(episode.id);
    const auditRow = harness.auditLog
      .list({ process: "curator" })
      .find((row) => row.action === "decay");

    expect(result.errors).toEqual([]);
    expect(result.changes.map((change) => change.action)).toEqual(["decay"]);
    expect(decayedEpisode?.significance).toBeCloseTo(0.4, 6);
    expect(decayedStats?.heat_multiplier).toBeCloseTo(0.5, 6);
    expect(decayedStats?.last_decayed_at).toBe(nowMs);
    const decayAudit = auditRow?.reversal.decay as Array<Record<string, unknown>> | undefined;
    expect(decayAudit).toHaveLength(1);
    expect(decayAudit?.[0]).toMatchObject({
      episode_id: episode.id,
      old_salience: 0.8,
      old_heat_multiplier: 1,
    });
    expect(decayAudit?.[0]?.new_salience as number).toBeCloseTo(0.4, 6);
    expect(decayAudit?.[0]?.new_heat_multiplier as number).toBeCloseTo(0.5, 6);

    await harness.auditLog.revert(auditRow!.id, "test");

    expect((await harness.episodicRepository.get(episode.id))?.significance).toBe(0.8);
    expect(harness.episodicRepository.getStats(episode.id)?.heat_multiplier).toBe(1);
  });

  it("skips repeat episode decay until the decay interval has elapsed", async () => {
    const lastDecayedAt = 100 * DAY_MS;
    const earlyHarness = await createOfflineTestHarness({
      clock: new FixedClock(lastDecayedAt + HOUR_MS),
    });
    const lateHarness = await createOfflineTestHarness({
      clock: new FixedClock(lastDecayedAt + 25 * HOUR_MS),
    });
    cleanup.push(earlyHarness.cleanup, lateHarness.cleanup);

    for (const harness of [earlyHarness, lateHarness]) {
      const episode = createEpisodeFixture(
        {
          title: "Recently decayed note",
          created_at: lastDecayedAt - DAY_MS,
          updated_at: lastDecayedAt - DAY_MS,
        },
        [0, 1, 0, 0],
      );
      await harness.episodicRepository.createEpisode(episode);
      harness.episodicRepository.updateStats(episode.id, {
        tier: "T2",
        retrieval_count: 10,
        last_decayed_at: lastDecayedAt,
      });
    }

    const earlyProcess = new CuratorProcess({
      episodicRepository: earlyHarness.episodicRepository,
      traitsRepository: earlyHarness.traitsRepository,
      moodRepository: earlyHarness.moodRepository,
      socialRepository: earlyHarness.socialRepository,
      registry: earlyHarness.registry,
    });
    const lateProcess = new CuratorProcess({
      episodicRepository: lateHarness.episodicRepository,
      traitsRepository: lateHarness.traitsRepository,
      moodRepository: lateHarness.moodRepository,
      socialRepository: lateHarness.socialRepository,
      registry: lateHarness.registry,
    });

    const early = await earlyProcess.run(earlyHarness.createContext(), {
      dryRun: true,
    });
    const late = await lateProcess.run(lateHarness.createContext(), {
      dryRun: true,
    });

    expect(early.changes.map((change) => change.action)).not.toContain("decay");
    expect(late.changes.map((change) => change.action)).toContain("decay");
  });

  it("skips repeat trait decay until the decay interval has elapsed", async () => {
    const lastDecayedAt = 100 * DAY_MS;
    const earlyHarness = await createOfflineTestHarness({
      clock: new FixedClock(lastDecayedAt + HOUR_MS),
    });
    const lateHarness = await createOfflineTestHarness({
      clock: new FixedClock(lastDecayedAt + 25 * HOUR_MS),
    });
    cleanup.push(earlyHarness.cleanup, lateHarness.cleanup);

    for (const harness of [earlyHarness, lateHarness]) {
      const trait = harness.traitsRepository.reinforce({
        label: "interval-gated",
        delta: 0.8,
        provenance: { kind: "manual" },
        timestamp: lastDecayedAt - 10 * DAY_MS,
      });
      harness.traitsRepository.decay(30 * 24, lastDecayedAt, {
        traitIds: [trait.id],
      });
    }

    const earlyProcess = new CuratorProcess({
      episodicRepository: earlyHarness.episodicRepository,
      traitsRepository: earlyHarness.traitsRepository,
      moodRepository: earlyHarness.moodRepository,
      socialRepository: earlyHarness.socialRepository,
      registry: earlyHarness.registry,
    });
    const lateProcess = new CuratorProcess({
      episodicRepository: lateHarness.episodicRepository,
      traitsRepository: lateHarness.traitsRepository,
      moodRepository: lateHarness.moodRepository,
      socialRepository: lateHarness.socialRepository,
      registry: lateHarness.registry,
    });

    const early = await earlyProcess.run(earlyHarness.createContext(), {
      dryRun: true,
    });
    const late = await lateProcess.run(lateHarness.createContext(), {
      dryRun: true,
    });

    expect(early.changes.map((change) => change.action)).not.toContain("decay_trait");
    expect(late.changes.map((change) => change.action)).toContain("decay_trait");
  });

  it("prunes retrieval log rows older than the retention window", async () => {
    const nowMs = 100 * DAY_MS;
    const retentionDays = 30;
    const cutoff = nowMs - retentionDays * DAY_MS;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
      configOverrides: {
        offline: {
          curator: {
            retrievalLogRetentionDays: retentionDays,
          },
        },
      },
    });
    cleanup.push(harness.cleanup);
    const insertLog = (episodeId: string, timestamp: number) => {
      harness.db
        .prepare("INSERT INTO retrieval_log (episode_id, timestamp, score) VALUES (?, ?, ?)")
        .run(episodeId, timestamp, 0.5);
    };

    insertLog("ep_old_retrieval", cutoff - 1);
    insertLog("ep_boundary_retrieval", cutoff);
    insertLog("ep_recent_retrieval", cutoff + 1);

    const process = new CuratorProcess({
      episodicRepository: harness.episodicRepository,
      traitsRepository: harness.traitsRepository,
      moodRepository: harness.moodRepository,
      socialRepository: harness.socialRepository,
      registry: harness.registry,
    });
    const result = await process.run(harness.createContext(), {
      dryRun: false,
    });
    const remaining = harness.db
      .prepare("SELECT episode_id FROM retrieval_log ORDER BY timestamp ASC")
      .all() as Array<{ episode_id: string }>;
    const auditRow = harness.auditLog
      .list({ process: "curator" })
      .find((row) => row.action === "prune_retrieval_log");

    expect(result.changes.map((change) => change.action)).toEqual(["prune_retrieval_log"]);
    expect(remaining.map((row) => row.episode_id)).toEqual([
      "ep_boundary_retrieval",
      "ep_recent_retrieval",
    ]);
    expect(auditRow).toEqual(
      expect.objectContaining({
        targets: {
          cutoff,
        },
        reversal: expect.objectContaining({
          no_reverser: true,
          retention_days: retentionDays,
          deleted: 1,
        }),
      }),
    );
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
      moodRepository: harness.moodRepository,
      socialRepository: harness.socialRepository,
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
