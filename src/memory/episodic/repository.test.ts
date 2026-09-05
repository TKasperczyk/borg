import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { makeArrowTable } from "@lancedb/lancedb";
import { afterEach, describe, expect, it, vi } from "vitest";

import { selfMigrations } from "../self/migrations.js";
import { offlineMigrations } from "../../offline/migrations.js";
import { LanceDbTable, LanceDbStore } from "../../storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../../storage/sqlite/index.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import {
  createConsolidationFamilyId,
  createEntityId,
  createEpisodeId,
  createStreamEntryId,
} from "../../util/ids.js";
import { episodicMigrations } from "./migrations.js";
import {
  EpisodicRepository,
  HOT_LANE_RETRIEVAL_COOLDOWN_MS,
  buildConsolidationCoverageHash,
  createEpisodesTableSchema,
  episodeLexicalSearchTokens,
} from "./repository.js";
import type { Episode } from "./types.js";
import { episodeToRawLanceRowForTest } from "./test-support.js";
import { retrievalMigrations } from "../../retrieval/migrations.js";
import { streamEntryIndexMigrations } from "../../stream/index.js";

type Harness = {
  tempDir: string;
  store: LanceDbStore;
  table: LanceDbTable;
  db: SqliteDatabase;
  repo: EpisodicRepository;
  close: () => Promise<void>;
  clock: ManualClock;
};

function createEpisode(id: string, nowMs: number, overrides: Partial<Episode> = {}): Episode {
  return {
    id: id as Episode["id"],
    title: `${id} title`,
    narrative: `${id} narrative.`,
    participants: ["user"],
    location: null,
    start_time: nowMs,
    end_time: nowMs + 1_000,
    source_stream_ids: ["strm_aaaaaaaaaaaaaaaa" as Episode["source_stream_ids"][number]],
    significance: 0.8,
    tags: ["alpha"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    embedding: Float32Array.from([1, 0, 0, 0]),
    created_at: nowMs,
    updated_at: nowMs,
    ...overrides,
    emotional_arc: overrides.emotional_arc ?? null,
  };
}

function explainQueryPlan(harness: Harness, sql: string, ...params: unknown[]): string {
  return (
    harness.db.prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...params) as Array<{ detail: string }>
  )
    .map((row) => row.detail)
    .join("\n");
}

async function createHarness(): Promise<Harness> {
  const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
  const clock = new ManualClock(1_700_000_000_000);
  const store = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: composeMigrations(
      episodicMigrations,
      selfMigrations,
      retrievalMigrations,
      offlineMigrations,
      streamEntryIndexMigrations,
    ),
  });
  const table = await store.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(4),
  });
  const repo = new EpisodicRepository({
    table,
    db,
    clock,
  });

  return {
    tempDir,
    store,
    table,
    db,
    repo,
    clock,
    close: async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    },
  };
}

describe("episodic repository", () => {
  const closers: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (closers.length > 0) {
      await closers.pop()?.();
    }
  });

  it("inserts, retrieves, updates, lists, searches, and deletes episodes", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const first = createEpisode("ep_aaaaaaaaaaaaaaaa", harness.clock.now());
    const second = createEpisode("ep_bbbbbbbbbbbbbbbb", harness.clock.now() + 5_000, {
      tags: ["beta"],
      embedding: Float32Array.from([0, 1, 0, 0]),
      source_stream_ids: ["strm_bbbbbbbbbbbbbbbb" as Episode["source_stream_ids"][number]],
    });

    await harness.repo.createEpisode(first);
    await harness.repo.createEpisode(second);
    harness.clock.advance(10_000);

    const updated = await harness.repo.update(first.id, {
      tags: ["focus"],
      confidence: 0.95,
    });
    const search = await harness.repo.searchByVector(Float32Array.from([1, 0, 0, 0]), {
      limit: 1,
      minSimilarity: 0.5,
    });
    const listed = await harness.repo.list({
      limit: 1,
    });
    const paged = await harness.repo.list({
      limit: 1,
      cursor: listed.nextCursor,
    });

    expect(await harness.repo.get(first.id)).toEqual(
      expect.objectContaining({
        id: first.id,
      }),
    );
    expect(await harness.repo.getMany([second.id, first.id])).toEqual([
      expect.objectContaining({ id: second.id }),
      expect.objectContaining({ id: first.id }),
    ]);
    expect(updated).toEqual(
      expect.objectContaining({
        confidence: 0.95,
      }),
    );
    expect(search[0]?.episode.id).toBe(first.id);
    expect(listed.items).toHaveLength(1);
    expect(listed.items[0]?.id).toBe(first.id);
    expect(paged.items).toHaveLength(1);
    expect(paged.items[0]?.id).toBe(second.id);
    expect(await harness.repo.delete(second.id)).toBe(true);
    expect(await harness.repo.get(second.id)).toBeNull();
  });

  it("filters archived episodes from get unless explicitly included", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const episode = createEpisode("ep_archivedgetxxxxx", harness.clock.now());
    await harness.repo.createEpisode(episode);
    harness.repo.archiveEpisode(episode.id, {
      caller: "repository.test",
      reason: "exercise archived-get filtering",
      process: "curator",
    });

    expect(await harness.repo.get(episode.id)).toBeNull();
    expect(await harness.repo.get(episode.id, { includeArchived: true })).toEqual(
      expect.objectContaining({
        id: episode.id,
      }),
    );
  });

  it("projects unarchived episode ids from the SQLite lifecycle index", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const active = createEpisode("ep_activeidsxxxxxxx", harness.clock.now());
    const archived = createEpisode("ep_archivedidsxxxxx", harness.clock.now() + 1_000);

    await harness.repo.createEpisode(active);
    await harness.repo.createEpisode(archived);
    harness.repo.archiveEpisode(archived.id, {
      caller: "repository.test",
      reason: "exercise active id projection",
      process: "curator",
    });

    await expect(harness.repo.listUnarchivedEpisodeIds()).resolves.toEqual([active.id]);
  });

  it("unarchives an episode with a CAS transition and restores effective visibility", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const episode = createEpisode(createEpisodeId(), harness.clock.now());

    await harness.repo.createEpisode(episode);
    harness.repo.updateStats(episode.id, {
      use_count: 4,
      heat_multiplier: 0.35,
    });
    const beforeArchive = harness.repo.getStats(episode.id)!;

    harness.repo.archiveEpisode(episode.id, {
      caller: "repository.test",
      reason: "exercise explicit archive reversal",
      process: "curator",
    });

    await expect(harness.repo.listUnarchivedEpisodeIds()).resolves.toEqual([]);
    await expect(harness.repo.listEffectivelyVisible()).resolves.toEqual([]);

    const unarchived = harness.repo.unarchiveEpisode(episode.id, {
      caller: "repository.test",
      reason: "reverse explicit archival",
      process: "curator",
    });
    const archiveFlags = harness.db
      .prepare(
        `
          SELECT stats.archived AS stats_archived, episode_index.archived AS index_archived
          FROM episode_stats AS stats
          JOIN episode_index ON episode_index.episode_id = stats.episode_id
          WHERE stats.episode_id = ?
        `,
      )
      .get(episode.id) as { stats_archived: number; index_archived: number };
    const unarchiveAuditCount = () =>
      (
        harness.db
          .prepare(
            "SELECT COUNT(*) AS count FROM maintenance_audit WHERE action = 'unarchive_episode'",
          )
          .get() as { count: number }
      ).count;

    expect(unarchived).toEqual(beforeArchive);
    expect(archiveFlags).toEqual({
      stats_archived: 0,
      index_archived: 0,
    });
    expect(await harness.repo.get(episode.id)).toEqual(
      expect.objectContaining({
        id: episode.id,
        significance: episode.significance,
      }),
    );
    await expect(harness.repo.listUnarchivedEpisodeIds()).resolves.toEqual([episode.id]);
    await expect(harness.repo.listEffectivelyVisible()).resolves.toEqual([
      expect.objectContaining({ id: episode.id }),
    ]);
    expect(unarchiveAuditCount()).toBe(1);

    const noOp = harness.repo.unarchiveEpisode(episode.id, {
      caller: "repository.test",
      reason: "exercise non-archived no-op",
      process: "curator",
    });

    expect(noOp).toEqual(unarchived);
    expect(unarchiveAuditCount()).toBe(1);
  });

  it("does not inject stats defaults when applying partial stat patches", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const episode = createEpisode("ep_statsdefaults001", harness.clock.now());
    await harness.repo.createEpisode(episode);
    harness.repo.updateStats(episode.id, {
      heat_multiplier: 0.5,
      valence_mean: 0.4,
    });
    harness.repo.archiveEpisode(episode.id, {
      caller: "repository.test",
      reason: "exercise patch-only stats retention",
      process: "curator",
    });

    const patched = harness.repo.updateStats(episode.id, {
      use_count: 3,
    });

    expect(patched).toMatchObject({
      use_count: 3,
      heat_multiplier: 0.5,
      valence_mean: 0.4,
      archived: true,
    });
    expect(harness.repo.getStats(episode.id)).toMatchObject({
      use_count: 3,
      heat_multiplier: 0.5,
      valence_mean: 0.4,
      archived: true,
    });
  });

  it("applies full stats defaults when inserting fresh episodes", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const episode = createEpisode(createEpisodeId(), harness.clock.now(), {
      source_stream_ids: [createStreamEntryId()],
    });

    await harness.repo.createEpisode(episode);

    expect(harness.repo.getStats(episode.id)).toMatchObject({
      heat_multiplier: 1,
      retrieval_count: 0,
      win_rate: 0,
      archived: false,
      valence_mean: 0,
    });
  });

  it("defaults vector search to public-only visibility unless cross-audience is explicit", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const publicEpisode = createEpisode("ep_publicpublicpub1", harness.clock.now(), {
      source_stream_ids: ["strm_publicpublic0001" as Episode["source_stream_ids"][number]],
    });
    const scopedEpisode = createEpisode("ep_scopedscopedsc12", harness.clock.now() + 1_000, {
      source_stream_ids: ["strm_scopedscoped0000" as Episode["source_stream_ids"][number]],
      audience_entity_id: "ent_aaaaaaaaaaaaaaaa" as never,
      shared: false,
    });

    await harness.repo.createEpisode(publicEpisode);
    await harness.repo.createEpisode(scopedEpisode);

    const defaultSearch = await harness.repo.searchByVector(Float32Array.from([1, 0, 0, 0]), {
      limit: 5,
    });
    const crossAudienceSearch = await harness.repo.searchByVector(Float32Array.from([1, 0, 0, 0]), {
      limit: 5,
      crossAudience: true,
    });

    expect(defaultSearch.map((item) => item.episode.id)).toEqual([publicEpisode.id]);
    expect(crossAudienceSearch).toHaveLength(2);
    expect(crossAudienceSearch.map((item) => item.episode.id)).toEqual(
      expect.arrayContaining([publicEpisode.id, scopedEpisode.id]),
    );
  });

  it("does not classify multi-origin private vector rows as public in Lance visibility filters", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const sam = "ent_bbbbbbbbbbbbbbbb" as NonNullable<Episode["audience_entity_id"]>;
    const alex = "ent_cccccccccccccccc" as NonNullable<Episode["audience_entity_id"]>;
    const publicEpisode = createEpisode("ep_publicorigin0001", harness.clock.now(), {
      source_stream_ids: [createStreamEntryId()],
    });
    const multiOrigin = createEpisode("ep_privateorigin001", harness.clock.now() + 1_000, {
      source_stream_ids: [createStreamEntryId()],
      audience_entity_id: null,
      origin_audience_entity_ids: [sam, alex],
      shared: false,
    });

    await harness.repo.createEpisode(publicEpisode);
    await harness.repo.createEpisode(multiOrigin);

    const searchSpy = vi.spyOn(harness.table, "search");
    const defaultSearch = await harness.repo.searchByVector(Float32Array.from([1, 0, 0, 0]), {
      limit: 5,
    });

    expect(defaultSearch.map((item) => item.episode.id)).toEqual([publicEpisode.id]);
    expect(searchSpy.mock.calls[0]?.[1]?.where).toBe(
      "((origin_audience_entity_ids IS NULL OR origin_audience_entity_ids = '[]') AND audience_entity_id IS NULL AND (shared IS NULL OR shared = true))",
    );
  });

  it("excludes unknown-origin shared-false rows from vector disclosure at the Lance predicate", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const audience = "ent_bbbbbbbbbbbbbbbb" as NonNullable<Episode["audience_entity_id"]>;
    const publicShared = createEpisode("ep_vecpublictrue001", harness.clock.now(), {
      source_stream_ids: [createStreamEntryId()],
      audience_entity_id: null,
      origin_audience_entity_ids: [],
      shared: true,
    });
    const publicLegacyNull = createEpisode("ep_vecpublicnull001", harness.clock.now() + 1_000, {
      source_stream_ids: [createStreamEntryId()],
      audience_entity_id: null,
      origin_audience_entity_ids: [],
      shared: true,
    });
    const unknownOrigin = createEpisode("ep_vecunknownfalse1", harness.clock.now() + 2_000, {
      source_stream_ids: [createStreamEntryId()],
      audience_entity_id: null,
      origin_audience_entity_ids: [],
      shared: false,
    });

    const rawTable = (
      harness.table as unknown as {
        table: {
          add(rows: unknown): Promise<void>;
          schema(): Promise<unknown>;
        };
      }
    ).table;
    await rawTable.add(
      makeArrowTable(
        [
          episodeToRawLanceRowForTest(publicShared),
          { ...episodeToRawLanceRowForTest(publicLegacyNull), shared: null },
        ],
        {
          schema: (await rawTable.schema()) as never,
        },
      ),
    );
    await harness.table.checkoutLatest();
    await harness.repo.createEpisode(unknownOrigin);

    const searchSpy = vi.spyOn(harness.table, "search");
    const nullAudienceResults = await harness.repo.searchByVectorForDisclosure(
      Float32Array.from([1, 0, 0, 0]),
      {
        limit: 10,
        audienceEntityId: null,
      },
    );
    const namedAudienceResults = await harness.repo.searchByVectorForDisclosure(
      Float32Array.from([1, 0, 0, 0]),
      {
        limit: 10,
        audienceEntityId: audience,
      },
    );

    for (const results of [nullAudienceResults, namedAudienceResults]) {
      const ids = results.map((item) => item.episode.id);
      expect(ids).toContain(publicShared.id);
      expect(ids).toContain(publicLegacyNull.id);
      expect(ids).not.toContain(unknownOrigin.id);
    }
    expect(searchSpy.mock.calls[0]?.[1]?.where).toBe(
      "((origin_audience_entity_ids IS NULL OR origin_audience_entity_ids = '[]') AND audience_entity_id IS NULL AND (shared IS NULL OR shared = true))",
    );
    expect(searchSpy.mock.calls[1]?.[1]?.where).toContain("(shared IS NULL OR shared = true)");
  });

  it("rejects inserts without citation anchors", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    await expect(
      harness.repo.createEpisode(
        createEpisode("ep_aaaaaaaaaaaaaaaa", harness.clock.now(), {
          source_stream_ids: [],
        }),
      ),
    ).rejects.toBeInstanceOf(StorageError);
  });

  it("refuses duplicate creates without resetting existing stats", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const episode = createEpisode(createEpisodeId(), harness.clock.now());

    await harness.repo.createEpisode(episode);
    harness.repo.updateStats(episode.id, {
      use_count: 3,
    });
    harness.repo.archiveEpisode(episode.id, {
      caller: "repository.test",
      reason: "exercise duplicate-create stats retention",
      process: "curator",
    });

    await expect(
      harness.repo.createEpisode({
        ...episode,
        narrative: "replacement body should not be written",
      }),
    ).rejects.toMatchObject({
      code: "EPISODE_ALREADY_EXISTS",
    });

    expect(harness.repo.getStats(episode.id)).toEqual(
      expect.objectContaining({
        archived: true,
        use_count: 3,
      }),
    );
    expect((await harness.repo.get(episode.id, { includeArchived: true }))?.narrative).toBe(
      episode.narrative,
    );
  });

  it("requires the audited lifecycle API for archived transitions in both directions", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const episode = createEpisode(createEpisodeId(), harness.clock.now());

    await harness.repo.createEpisode(episode);

    expect(() =>
      harness.repo.updateStats(episode.id, {
        archived: true,
      }),
    ).toThrow(/archive state must change via archiveEpisode/);

    harness.repo.archiveEpisode(episode.id, {
      caller: "repository.test",
      reason: "exercise audited archival",
      process: "curator",
    });

    expect(() =>
      harness.repo.updateStats(episode.id, {
        archived: false,
      }),
    ).toThrow(/archive state must change via archiveEpisode/);

    const reactivated = harness.repo.reactivateEpisode(episode.id, {
      caller: "repository.test",
      reason: "exercise audited reactivation",
      process: "curator",
    });
    const auditRows = harness.db
      .prepare(
        `
          SELECT process, action, targets
          FROM maintenance_audit
          WHERE action = 'reactivate_episode'
        `,
      )
      .all() as Array<{ process: string; action: string; targets: string }>;

    expect(reactivated.archived).toBe(false);
    expect(auditRows).toHaveLength(1);
    expect(auditRows[0]).toMatchObject({
      process: "curator",
      action: "reactivate_episode",
    });
    expect(JSON.parse(auditRows[0]!.targets)).toMatchObject({
      episode_id: episode.id,
      caller: "repository.test",
      reason: "exercise audited reactivation",
      initiating_process: "curator",
      lifecycle_owner: "episodic-repository",
      previous_archived: true,
      next_archived: false,
    });
  });

  it("does not restore archived from a stale stats snapshot when patch omits it", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const episode = createEpisode(createEpisodeId(), harness.clock.now());

    await harness.repo.createEpisode(episode);
    const staleStats = harness.repo.getStats(episode.id);

    expect(staleStats).toMatchObject({
      archived: false,
    });

    harness.repo.archiveEpisode(episode.id, {
      caller: "repository.test",
      reason: "exercise stale patch-only stats update",
      process: "curator",
    });

    const getStats = harness.repo.getStats.bind(harness.repo);
    const getStatsSpy = vi
      .spyOn(harness.repo, "getStats")
      .mockImplementationOnce(() => staleStats)
      .mockImplementation((episodeId) => getStats(episodeId));

    const updated = harness.repo.updateStats(episode.id, {
      use_count: 7,
    });

    expect(getStatsSpy).toHaveBeenCalled();
    expect(updated).toMatchObject({
      archived: true,
      use_count: 7,
    });
    expect(harness.repo.getStats(episode.id)).toMatchObject({
      archived: true,
      use_count: 7,
    });
    expect(await harness.repo.get(episode.id)).toBeNull();
  });

  it("applies effective visibility from consolidation family pointers", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const nowMs = harness.clock.now();
    const familyId = createConsolidationFamilyId();
    const raw = createEpisode(createEpisodeId(), nowMs, {
      source_stream_ids: [createStreamEntryId()],
    });
    const currentVersion = createEpisode(createEpisodeId(), nowMs + 1_000, {
      episode_kind: "consolidation_version",
      consolidation_family_id: familyId,
      consolidation_coverage_hash: buildConsolidationCoverageHash(raw.source_stream_ids),
      source_stream_ids: [createStreamEntryId()],
    });
    const oldVersion = createEpisode(createEpisodeId(), nowMs + 2_000, {
      episode_kind: "consolidation_version",
      consolidation_family_id: familyId,
      consolidation_coverage_hash: buildConsolidationCoverageHash(raw.source_stream_ids),
      source_stream_ids: [createStreamEntryId()],
    });

    await harness.repo.createEpisode(raw);
    await harness.repo.createEpisode(currentVersion);
    await harness.repo.createEpisode(oldVersion);
    harness.db
      .prepare(
        `
          INSERT INTO consolidation_families (
            family_id, current_version_episode_id, coverage_hash, policy_version, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        familyId,
        currentVersion.id,
        buildConsolidationCoverageHash(raw.source_stream_ids),
        1,
        nowMs,
        nowMs,
      );
    harness.db
      .prepare(
        `
          INSERT INTO consolidation_members (
            family_id, raw_episode_id, source_stream_ids_json, added_by_version_episode_id
          ) VALUES (?, ?, ?, ?)
        `,
      )
      .run(familyId, raw.id, JSON.stringify(raw.source_stream_ids), currentVersion.id);

    expect(await harness.repo.get(raw.id)).toBeNull();
    expect((await harness.repo.get(currentVersion.id))?.id).toBe(currentVersion.id);
    expect(await harness.repo.get(oldVersion.id)).toBeNull();
    expect((await harness.repo.listEffectivelyVisible()).map((episode) => episode.id)).toEqual([
      currentVersion.id,
    ]);

    expect(() =>
      harness.repo.archiveEpisode(currentVersion.id, {
        caller: "repository.test",
        reason: "must not heat-archive a current consolidation version",
        process: "curator",
      }),
    ).toThrow(/current version of consolidation family/);

    // The current version cannot be archived out from under its family, so
    // coverage stays intact and the raw leaves remain hidden behind it.
    expect(await harness.repo.get(raw.id)).toBeNull();
    expect((await harness.repo.get(currentVersion.id))?.id).toBe(currentVersion.id);
  });

  it("hides effectively visible rows when stats and index archived flags diverge", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const episode = createEpisode(createEpisodeId(), harness.clock.now());

    await harness.repo.createEpisode(episode);
    harness.db
      .prepare(
        `
          UPDATE episode_stats
          SET archived = 1
          WHERE episode_id = ?
        `,
      )
      .run(episode.id);

    expect(harness.repo.isEpisodeEffectivelyVisible(episode.id)).toBe(false);
    expect((await harness.repo.listEffectivelyVisible()).map((item) => item.id)).toEqual([]);
  });

  it("keeps missing emotional arcs unknown on read", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const episode = createEpisode("ep_unknownarc000000", harness.clock.now(), {
      emotional_arc: null,
    });

    await harness.repo.createEpisode(episode);

    expect((await harness.repo.get(episode.id))?.emotional_arc).toBeNull();
  });

  it("finds an episode whose source stream ids contain the requested ids", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const userStreamId = "strm_aaaaaaaaaaaaaaaa" as Episode["source_stream_ids"][number];
    const agentStreamId = "strm_bbbbbbbbbbbbbbbb" as Episode["source_stream_ids"][number];
    const toolCallStreamId = "strm_cccccccccccccccc" as Episode["source_stream_ids"][number];
    const episode = createEpisode(createEpisodeId(), harness.clock.now(), {
      source_stream_ids: [userStreamId, agentStreamId, toolCallStreamId],
    });
    await harness.repo.createEpisode(episode);

    const matched = await harness.repo.findBySourceStreamIdsContaining([
      userStreamId,
      agentStreamId,
    ]);

    expect(matched?.id).toBe(episode.id);
  });

  it("preserves emotional_arc when a patch omits it", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const episode = createEpisode("ep_emotionalarcxxxx", harness.clock.now(), {
      emotional_arc: {
        start: { valence: -0.4, arousal: 0.2 },
        peak: { valence: 0.1, arousal: 0.5 },
        end: { valence: 0.3, arousal: 0.2 },
        dominant_emotion: "curiosity",
      },
    });
    await harness.repo.createEpisode(episode);
    harness.clock.advance(1_000);

    const updated = await harness.repo.update(episode.id, {
      tags: ["merged"],
    });

    expect(updated?.tags).toEqual(["merged"]);
    expect(updated?.emotional_arc).toEqual(episode.emotional_arc);
  });

  it("removes the Lance row if stats insertion fails after episode upsert", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const episode = createEpisode("ep_atomicinsertfail", harness.clock.now());
    const statsSpy = vi
      .spyOn(harness.repo as unknown as { upsertStats(stats: unknown): void }, "upsertStats")
      .mockImplementationOnce(() => {
        throw new Error("sqlite failed");
      });

    await expect(harness.repo.createEpisode(episode)).rejects.toMatchObject({
      code: "EPISODE_INSERT_FAILED",
    });
    expect(await harness.repo.get(episode.id)).toBeNull();
    expect(harness.repo.getStats(episode.id)).toBeNull();

    statsSpy.mockRestore();
  });

  it("reconciles Lance episodes that are missing stats rows", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const episode = createEpisode(createEpisodeId(), harness.clock.now());

    await harness.table.upsert(
      [
        {
          id: episode.id,
          title: episode.title,
          narrative: episode.narrative,
          participants: JSON.stringify(episode.participants),
          location: episode.location,
          start_time: episode.start_time,
          end_time: episode.end_time,
          source_stream_ids: JSON.stringify(episode.source_stream_ids),
          significance: episode.significance,
          tags: JSON.stringify(episode.tags),
          confidence: episode.confidence,
          lineage_derived_from: JSON.stringify(episode.lineage.derived_from),
          lineage_supersedes: JSON.stringify(episode.lineage.supersedes),
          source_fingerprint: episode.source_stream_ids.join("\n"),
          audience_entity_id: episode.audience_entity_id ?? null,
          shared: episode.shared ?? true,
          emotional_arc: null,
          embedding: Array.from(episode.embedding),
          created_at: episode.created_at,
          updated_at: episode.updated_at,
        },
      ],
      { on: "id" },
    );

    const report = await harness.repo.reconcileCrossStoreState();
    const stats = harness.repo.getStats(episode.id);

    expect(report).toEqual({
      createdMissingStats: 1,
      deletedOrphanStats: 0,
      deletedOrphanRetrievalLogs: 0,
      deletedOrphanValueSources: 0,
    });
    expect(stats).toEqual(
      expect.objectContaining({
        episode_id: episode.id,
        retrieval_count: 0,
      }),
    );
  });

  it("removes orphaned sqlite rows during reconciliation", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const orphanEpisodeId = createEpisodeId();
    const logOnlyOrphanEpisodeId = createEpisodeId();

    harness.db
      .prepare(
        `
          INSERT INTO episode_stats (
            episode_id, retrieval_count, use_count, last_retrieved, win_rate, tier,
            promoted_at, promoted_from, gist, gist_generated_at, last_decayed_at, valence_mean, archived
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
      )
      .run(orphanEpisodeId, 1, 0, null, 0, "T1", harness.clock.now(), null, null, null, null, 0, 0);
    harness.db
      .prepare("INSERT INTO retrieval_log (episode_id, timestamp, score) VALUES (?, ?, ?)")
      .run(orphanEpisodeId, harness.clock.now(), 0.2);
    harness.db
      .prepare(
        `INSERT INTO "values" (id, label, description, priority, created_at, last_affirmed)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run("val_orphan", "Orphan value", "orphan", 0.5, harness.clock.now(), null);
    harness.db
      .prepare("INSERT INTO value_sources (value_id, episode_id) VALUES (?, ?)")
      .run("val_orphan", orphanEpisodeId);
    harness.db
      .prepare("INSERT INTO retrieval_log (episode_id, timestamp, score) VALUES (?, ?, ?)")
      .run(logOnlyOrphanEpisodeId, harness.clock.now(), 0.4);
    harness.db
      .prepare(
        `INSERT INTO "values" (id, label, description, priority, created_at, last_affirmed)
         VALUES (?, ?, ?, ?, ?, ?)`,
      )
      .run(
        "val_orphan_log_only",
        "Orphan log-only value",
        "orphan",
        0.5,
        harness.clock.now(),
        null,
      );
    harness.db
      .prepare("INSERT INTO value_sources (value_id, episode_id) VALUES (?, ?)")
      .run("val_orphan_log_only", logOnlyOrphanEpisodeId);

    const report = await harness.repo.reconcileCrossStoreState();

    expect(report).toEqual({
      createdMissingStats: 0,
      deletedOrphanStats: 1,
      deletedOrphanRetrievalLogs: 2,
      deletedOrphanValueSources: 2,
    });
    expect(harness.repo.getStats(orphanEpisodeId)).toBeNull();
    expect(
      (
        harness.db
          .prepare("SELECT COUNT(*) AS count FROM retrieval_log WHERE episode_id = ?")
          .get(orphanEpisodeId) as { count: number }
      ).count,
    ).toBe(0);
    expect(
      (
        harness.db
          .prepare("SELECT COUNT(*) AS count FROM value_sources WHERE episode_id = ?")
          .get(orphanEpisodeId) as { count: number }
      ).count,
    ).toBe(0);
    expect(
      (
        harness.db
          .prepare("SELECT COUNT(*) AS count FROM retrieval_log WHERE episode_id = ?")
          .get(logOnlyOrphanEpisodeId) as { count: number }
      ).count,
    ).toBe(0);
    expect(
      (
        harness.db
          .prepare("SELECT COUNT(*) AS count FROM value_sources WHERE episode_id = ?")
          .get(logOnlyOrphanEpisodeId) as { count: number }
      ).count,
    ).toBe(0);
  });

  it("creates SQL indexes for hot-path episodic retrieval lanes", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const audienceEntityId = "ent_aaaaaaaaaaaaaaaa" as Episode["audience_entity_id"];

    const recentPlan = explainQueryPlan(
      harness,
      `
        SELECT episode_id
        FROM episode_index INDEXED BY idx_episode_index_recent
        WHERE archived = 0
        ORDER BY updated_at DESC, episode_id DESC
        LIMIT 5
      `,
    );
    const audiencePlan = explainQueryPlan(
      harness,
      `
        SELECT episode_id
        FROM episode_index INDEXED BY idx_episode_index_audience_recent
        WHERE archived = 0 AND audience_entity_id = ?
        ORDER BY updated_at DESC, episode_id DESC
        LIMIT 5
      `,
      audienceEntityId,
    );
    const heatPlan = explainQueryPlan(
      harness,
      `
        SELECT episode_id
        FROM episode_index INDEXED BY idx_episode_index_heat
        WHERE archived = 0
        ORDER BY heat_score DESC, updated_at DESC, episode_id DESC
        LIMIT 5
      `,
    );
    const participantPlan = explainQueryPlan(
      harness,
      `
        SELECT ei.episode_id
        FROM episode_participants AS ep INDEXED BY idx_episode_participants_term
        JOIN episode_index AS ei ON ei.episode_id = ep.episode_id
        WHERE ep.term = ? AND ei.archived = 0
      `,
      "sam",
    );
    const tagPlan = explainQueryPlan(
      harness,
      `
        SELECT ei.episode_id
        FROM episode_tags AS et INDEXED BY idx_episode_tags_term
        JOIN episode_index AS ei ON ei.episode_id = et.episode_id
        WHERE et.term = ? AND ei.archived = 0
      `,
      "atlas",
    );

    expect(recentPlan).toContain("idx_episode_index_recent");
    expect(audiencePlan).toContain("idx_episode_index_audience_recent");
    expect(heatPlan).toContain("idx_episode_index_heat");
    expect(participantPlan).toContain("idx_episode_participants_term");
    expect(tagPlan).toContain("idx_episode_tags_term");
  });

  it("serves indexed retrieval lanes without scanning visible episodes", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const sam = "ent_bbbbbbbbbbbbbbbb" as NonNullable<Episode["audience_entity_id"]>;
    const older = createEpisode("ep_indexolder000001", harness.clock.now() - 10_000, {
      source_stream_ids: ["strm_indexolder000001" as Episode["source_stream_ids"][number]],
      tags: ["archive"],
      participants: ["team"],
    });
    const scoped = createEpisode("ep_indexscoped00001", harness.clock.now() - 5_000, {
      source_stream_ids: ["strm_indexscoped00001" as Episode["source_stream_ids"][number]],
      audience_entity_id: sam,
      shared: false,
      participants: ["Sam"],
    });
    const entity = createEpisode("ep_indexentity00001", harness.clock.now() - 3_000, {
      source_stream_ids: ["strm_indexentity00001" as Episode["source_stream_ids"][number]],
      tags: ["Atlas"],
      participants: ["Jordan"],
    });
    const hot = createEpisode("ep_indexhot00000001", harness.clock.now() - 8_000, {
      source_stream_ids: ["strm_indexhot00000001" as Episode["source_stream_ids"][number]],
      tags: ["heat"],
      participants: ["ops"],
    });

    await harness.repo.createEpisode(older);
    await harness.repo.createEpisode(scoped);
    await harness.repo.createEpisode(entity);
    await harness.repo.createEpisode(hot);
    harness.repo.updateStats(hot.id, {
      retrieval_count: 20,
      win_rate: 1,
      last_retrieved: harness.clock.now(),
    });

    const visibleSpy = vi.spyOn(harness.repo, "listVisibleEpisodes");
    const recent = await harness.repo.listRecent({
      limit: 1,
      crossAudience: true,
    });
    const audience = await harness.repo.listByAudience(sam, {
      limit: 1,
      orderBy: "recent",
    });
    const participantOrTag = await harness.repo.searchByParticipantsOrTags(["atlas"], {
      limit: 1,
      crossAudience: true,
    });
    const hottest = await harness.repo.listHottest({
      limit: 1,
      crossAudience: true,
    });

    expect(recent[0]?.episode.id).toBe(entity.id);
    expect(audience[0]?.episode.id).toBe(scoped.id);
    expect(participantOrTag[0]?.episode.id).toBe(entity.id);
    expect(hottest[0]?.episode.id).toBe(hot.id);
    expect(visibleSpy).not.toHaveBeenCalled();
  });

  it("tokenizes lexical terms without LIKE wildcards and requires both short-token boundaries", async () => {
    expect(episodeLexicalSearchTokens("  Marcin_Oryl% Żółć  ")).toEqual(["marcin", "oryl", "żółć"]);
    expect(episodeLexicalSearchTokens("%_%")).toEqual([]);

    const harness = await createHarness();
    closers.push(harness.close);
    const haryPrefix = createEpisode(createEpisodeId(), harness.clock.now(), {
      title: "Rozmowa z Harym o projekcie",
      narrative: "Omówiono dalsze kroki.",
      participants: ["zespół"],
      source_stream_ids: [createStreamEntryId()],
    });
    const haryParticipant = createEpisode(createEpisodeId(), harness.clock.now() + 1_000, {
      title: "Spotkanie projektowe",
      narrative: "Omówiono dalsze kroki.",
      participants: ["Hary"],
      source_stream_ids: [createStreamEntryId()],
    });
    const polishFalsePositive = createEpisode(createEpisodeId(), harness.clock.now() + 2_000, {
      title: "Zbiórka charytatywna",
      narrative: "To był plan charytatywny lokalnej grupy.",
      participants: ["zespół"],
      source_stream_ids: [createStreamEntryId()],
    });

    await harness.repo.createEpisode(haryPrefix);
    await harness.repo.createEpisode(haryParticipant);
    await harness.repo.createEpisode(polishFalsePositive);

    const matches = await harness.repo.searchByLexicalTermsForDisclosure(["Hary%_"], {
      crossAudience: true,
      limit: 10,
    });
    const ids = matches.map((candidate) => candidate.episode.id);

    expect(ids).toContain(haryParticipant.id);
    expect(ids).not.toContain(haryPrefix.id);
    expect(ids).not.toContain(polishFalsePositive.id);
  });

  it("applies audience and effective visibility to lexical candidates", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const nowMs = harness.clock.now();
    const audienceEntityId = "ent_lexicalaudience1" as NonNullable<Episode["audience_entity_id"]>;
    const familyId = createConsolidationFamilyId();
    const publicEpisode = createEpisode(createEpisodeId(), nowMs, {
      title: "Quasar public notes",
      source_stream_ids: [createStreamEntryId()],
    });
    const privateEpisode = createEpisode(createEpisodeId(), nowMs + 1_000, {
      title: "Quasar private notes",
      audience_entity_id: audienceEntityId,
      origin_audience_entity_ids: [audienceEntityId],
      shared: false,
      source_stream_ids: [createStreamEntryId()],
    });
    const hiddenRaw = createEpisode(createEpisodeId(), nowMs + 2_000, {
      title: "Quasar superseded raw notes",
      source_stream_ids: [createStreamEntryId()],
    });
    const currentVersion = createEpisode(createEpisodeId(), nowMs + 3_000, {
      title: "Current consolidated notes",
      narrative: "The current version intentionally omits the lexical handle.",
      episode_kind: "consolidation_version",
      consolidation_family_id: familyId,
      consolidation_coverage_hash: buildConsolidationCoverageHash(hiddenRaw.source_stream_ids),
      source_stream_ids: [createStreamEntryId()],
    });

    await harness.repo.createEpisode(publicEpisode);
    await harness.repo.createEpisode(privateEpisode);
    await harness.repo.createEpisode(hiddenRaw);
    await harness.repo.createEpisode(currentVersion);
    harness.db
      .prepare(
        `
          INSERT INTO consolidation_families (
            family_id, current_version_episode_id, coverage_hash, policy_version, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?)
        `,
      )
      .run(
        familyId,
        currentVersion.id,
        buildConsolidationCoverageHash(hiddenRaw.source_stream_ids),
        1,
        nowMs,
        nowMs,
      );
    harness.db
      .prepare(
        `
          INSERT INTO consolidation_members (
            family_id, raw_episode_id, source_stream_ids_json, added_by_version_episode_id
          ) VALUES (?, ?, ?, ?)
        `,
      )
      .run(familyId, hiddenRaw.id, JSON.stringify(hiddenRaw.source_stream_ids), currentVersion.id);

    const publicMatches = await harness.repo.searchByLexicalTermsForDisclosure(["Quasar"], {
      limit: 10,
    });
    const audienceMatches = await harness.repo.searchByLexicalTermsForDisclosure(["Quasar"], {
      audienceEntityId,
      limit: 10,
    });
    const cognitionMatches = await harness.repo.recallByLexicalTermsForCognition(["Quasar"], {
      limit: 10,
    });

    expect(publicMatches.map((candidate) => candidate.episode.id)).toEqual([publicEpisode.id]);
    expect(audienceMatches.map((candidate) => candidate.episode.id)).toEqual(
      expect.arrayContaining([publicEpisode.id, privateEpisode.id]),
    );
    expect(cognitionMatches.map((candidate) => candidate.episode.id)).toEqual(
      expect.arrayContaining([publicEpisode.id, privateEpisode.id]),
    );
    expect(cognitionMatches.some((candidate) => candidate.episode.id === hiddenRaw.id)).toBe(false);
  });

  it("pages lexical scans in recency order and applies one global limit after visibility", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const nowMs = harness.clock.now();
    const recentNonmatches = Array.from({ length: 65 }, (_, index) =>
      createEpisode(createEpisodeId(), nowMs + 20_000 + index, {
        title: `Recent nonmatch ${index}`,
        source_stream_ids: [createStreamEntryId()],
      }),
    );
    const archived = Array.from({ length: 65 }, (_, index) =>
      createEpisode(createEpisodeId(), nowMs + 10_000 + index, {
        title: `Quasar archived ${index}`,
        source_stream_ids: [createStreamEntryId()],
      }),
    );
    const visible = [
      createEpisode(createEpisodeId(), nowMs + 5_000, {
        title: "Quasar newest visible",
        source_stream_ids: [createStreamEntryId()],
      }),
      createEpisode(createEpisodeId(), nowMs + 4_000, {
        title: "Nebula second visible",
        source_stream_ids: [createStreamEntryId()],
      }),
      createEpisode(createEpisodeId(), nowMs + 3_000, {
        title: "Quasar third visible",
        source_stream_ids: [createStreamEntryId()],
      }),
      createEpisode(createEpisodeId(), nowMs + 2_000, {
        title: "Nebula fourth visible",
        source_stream_ids: [createStreamEntryId()],
      }),
    ];

    await harness.table.upsert(
      [...recentNonmatches, ...archived, ...visible].map(episodeToRawLanceRowForTest),
      { on: "id" },
    );
    await harness.repo.reconcileCrossStoreState();

    for (const episode of archived) {
      harness.repo.archiveEpisode(episode.id, {
        caller: "repository.test.ts",
        reason: "lexical visibility fixture",
        process: "consolidator",
      });
    }

    const matches = await harness.repo.searchByLexicalTermsForDisclosure(["Quasar", "Nebula"], {
      crossAudience: true,
      limit: 3,
    });

    expect(matches.map((candidate) => candidate.episode.id)).toEqual(
      visible.slice(0, 3).map((episode) => episode.id),
    );
  });

  it("deprioritizes recently retrieved episodes below less-hot uncooled hot-lane candidates", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const nowMs = harness.clock.now();
    const cooledHigherHeat = createEpisode(createEpisodeId(), nowMs - 5_000, {
      source_stream_ids: [createStreamEntryId()],
    });
    const uncooledLowerHeat = createEpisode(createEpisodeId(), nowMs - 10_000, {
      source_stream_ids: [createStreamEntryId()],
    });

    await harness.repo.createEpisode(cooledHigherHeat);
    await harness.repo.createEpisode(uncooledLowerHeat);
    harness.repo.updateStats(cooledHigherHeat.id, {
      retrieval_count: 40,
      last_retrieved: nowMs - 1,
    });
    harness.repo.updateStats(uncooledLowerHeat.id, {
      retrieval_count: 30,
      last_retrieved: nowMs - HOT_LANE_RETRIEVAL_COOLDOWN_MS - 1,
    });

    const first = await harness.repo.listHottestForCognition({ limit: 1 });
    const both = await harness.repo.listHottestForCognition({ limit: 2 });

    expect(first.map((item) => item.episode.id)).toEqual([uncooledLowerHeat.id]);
    expect(both.map((item) => item.episode.id)).toEqual([
      uncooledLowerHeat.id,
      cooledHigherHeat.id,
    ]);

    harness.clock.advance(HOT_LANE_RETRIEVAL_COOLDOWN_MS + 1);

    const afterCooldown = await harness.repo.listHottestForCognition({ limit: 1 });

    expect(afterCooldown.map((item) => item.episode.id)).toEqual([cooledHigherHeat.id]);
  });

  it("fills cognition hot-lane slots when all candidate episodes are cooled", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const nowMs = harness.clock.now();
    const hottest = createEpisode(createEpisodeId(), nowMs - 3_000, {
      source_stream_ids: [createStreamEntryId()],
    });
    const secondHottest = createEpisode(createEpisodeId(), nowMs - 2_000, {
      source_stream_ids: [createStreamEntryId()],
    });
    const thirdHottest = createEpisode(createEpisodeId(), nowMs - 1_000, {
      source_stream_ids: [createStreamEntryId()],
    });

    await harness.repo.createEpisode(hottest);
    await harness.repo.createEpisode(secondHottest);
    await harness.repo.createEpisode(thirdHottest);
    for (const [episode, retrievalCount] of [
      [hottest, 40],
      [secondHottest, 35],
      [thirdHottest, 30],
    ] as const) {
      harness.repo.updateStats(episode.id, {
        retrieval_count: retrievalCount,
        last_retrieved: nowMs - 1,
      });
    }

    const cooled = await harness.repo.listHottestForCognition({ limit: 2 });

    expect(cooled.map((item) => item.episode.id)).toEqual([hottest.id, secondHottest.id]);
  });

  it("keeps multi-origin private episodes visible only to origin audiences in indexed disclosure lanes", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const sam = "ent_bbbbbbbbbbbbbbbb" as NonNullable<Episode["audience_entity_id"]>;
    const alex = "ent_cccccccccccccccc" as NonNullable<Episode["audience_entity_id"]>;
    const jordan = "ent_dddddddddddddddd" as NonNullable<Episode["audience_entity_id"]>;
    const multiOrigin = createEpisode("ep_indexmulti000001", harness.clock.now(), {
      source_stream_ids: ["strm_indexmulti000001" as Episode["source_stream_ids"][number]],
      audience_entity_id: null,
      origin_audience_entity_ids: [sam, alex],
      shared: false,
      participants: ["Atlas"],
      tags: ["multi-origin"],
    });

    await harness.repo.createEpisode(multiOrigin);

    const samRecent = await harness.repo.listRecentForDisclosure({
      audienceEntityId: sam,
      limit: 5,
    });
    const alexParticipant = await harness.repo.searchByParticipantsOrTagsForDisclosure(["Atlas"], {
      audienceEntityId: alex,
      limit: 5,
    });
    const jordanRecent = await harness.repo.listRecentForDisclosure({
      audienceEntityId: jordan,
      limit: 5,
    });
    const publicRecent = await harness.repo.listRecentForDisclosure({
      audienceEntityId: null,
      limit: 5,
    });
    const scopedToSam = await harness.repo.listByAudience(sam, {
      limit: 5,
      orderBy: "recent",
    });
    const scopedToAlex = await harness.repo.listByAudience(alex, {
      limit: 5,
      orderBy: "recent",
    });

    expect(samRecent.map((item) => item.episode.id)).toContain(multiOrigin.id);
    expect(alexParticipant.map((item) => item.episode.id)).toContain(multiOrigin.id);
    expect(jordanRecent.map((item) => item.episode.id)).not.toContain(multiOrigin.id);
    expect(publicRecent.map((item) => item.episode.id)).not.toContain(multiOrigin.id);
    expect(scopedToSam.map((item) => item.episode.id)).toContain(multiOrigin.id);
    expect(scopedToAlex.map((item) => item.episode.id)).toContain(multiOrigin.id);
  });

  it("lists visible episodes sourced from one session since a lower bound in occurred-at order", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const nowMs = harness.clock.now();
    const currentSession = "sess_venuecurrent001" as Parameters<
      typeof harness.repo.listRecentForSessionForDisclosure
    >[0]["sessionId"];
    const otherSession = "sess_venueother0001" as typeof currentSession;
    const currentAudience = createEntityId();
    const otherAudience = createEntityId();
    const sourceIds = Array.from({ length: 5 }, () => createStreamEntryId());
    const insertStreamEntry = harness.db.prepare(
      `INSERT INTO stream_entry_index (entry_id, session_id, byte_offset, timestamp)
       VALUES (?, ?, ?, ?)`,
    );

    sourceIds.forEach((sourceId, index) => {
      insertStreamEntry.run(
        sourceId,
        index === sourceIds.length - 1 ? otherSession : currentSession,
        index * 100,
        nowMs + index,
      );
    });

    const oldCurrent = createEpisode(createEpisodeId(), nowMs - 1_000, {
      source_stream_ids: [sourceIds[0]!],
      start_time: nowMs - 1_000,
      end_time: nowMs - 900,
    });
    const newestCurrent = createEpisode(createEpisodeId(), nowMs + 3_000, {
      source_stream_ids: [sourceIds[1]!],
      start_time: nowMs + 3_000,
      end_time: nowMs + 3_100,
      audience_entity_id: currentAudience,
      origin_audience_entity_ids: [currentAudience],
      shared: false,
    });
    const middleCurrent = createEpisode(createEpisodeId(), nowMs + 2_000, {
      source_stream_ids: [sourceIds[2]!],
      start_time: nowMs + 2_000,
      end_time: nowMs + 2_100,
    });
    const hiddenCurrent = createEpisode(createEpisodeId(), nowMs + 4_000, {
      source_stream_ids: [sourceIds[3]!],
      start_time: nowMs + 4_000,
      end_time: nowMs + 4_100,
      audience_entity_id: otherAudience,
      origin_audience_entity_ids: [otherAudience],
      shared: false,
    });
    const otherVenue = createEpisode(createEpisodeId(), nowMs + 5_000, {
      source_stream_ids: [sourceIds[4]!],
      start_time: nowMs + 5_000,
      end_time: nowMs + 5_100,
    });

    for (const episode of [oldCurrent, newestCurrent, middleCurrent, hiddenCurrent, otherVenue]) {
      await harness.repo.createEpisode(episode);
    }

    const results = await harness.repo.listRecentForSessionForDisclosure({
      sessionId: currentSession,
      sinceMs: nowMs,
      audienceEntityId: currentAudience,
      limit: 10,
    });

    expect(results.map((candidate) => candidate.episode.id)).toEqual([
      newestCurrent.id,
      middleCurrent.id,
    ]);
    expect(
      harness.db
        .prepare(
          "SELECT COUNT(*) AS count FROM episode_index WHERE json_array_length(source_stream_ids) = 1",
        )
        .get() as {
        count: number;
      },
    ).toEqual({ count: 5 });
  });

  it("requires all consolidation provenance to be indexed in one venue session", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const nowMs = harness.clock.now();
    const currentSession = "sess_venuecurrent002" as Parameters<
      typeof harness.repo.listRecentForSessionForDisclosure
    >[0]["sessionId"];
    const otherSession = "sess_venueother0002" as typeof currentSession;
    const currentSourceA = createStreamEntryId();
    const currentSourceB = createStreamEntryId();
    const otherSource = createStreamEntryId();
    const missingSource = createStreamEntryId();
    const insertStreamEntry = harness.db.prepare(
      `INSERT INTO stream_entry_index (entry_id, session_id, byte_offset, timestamp)
       VALUES (?, ?, ?, ?)`,
    );

    insertStreamEntry.run(currentSourceA, currentSession, 0, nowMs);
    insertStreamEntry.run(currentSourceB, currentSession, 100, nowMs + 1);
    insertStreamEntry.run(otherSource, otherSession, 200, nowMs + 2);

    const mixedFamilyId = createConsolidationFamilyId();
    const singleFamilyId = createConsolidationFamilyId();
    const missingFamilyId = createConsolidationFamilyId();
    const mixed = createEpisode(createEpisodeId(), nowMs + 3_000, {
      episode_kind: "consolidation_version",
      consolidation_family_id: mixedFamilyId,
      consolidation_coverage_hash: buildConsolidationCoverageHash([currentSourceA, otherSource]),
      source_stream_ids: [currentSourceA, otherSource],
    });
    const singleVenue = createEpisode(createEpisodeId(), nowMs + 2_000, {
      episode_kind: "consolidation_version",
      consolidation_family_id: singleFamilyId,
      consolidation_coverage_hash: buildConsolidationCoverageHash([currentSourceA, currentSourceB]),
      source_stream_ids: [currentSourceA, currentSourceB],
    });
    const missingIndex = createEpisode(createEpisodeId(), nowMs + 1_000, {
      episode_kind: "consolidation_version",
      consolidation_family_id: missingFamilyId,
      consolidation_coverage_hash: buildConsolidationCoverageHash([currentSourceA, missingSource]),
      source_stream_ids: [currentSourceA, missingSource],
    });

    for (const episode of [mixed, singleVenue, missingIndex]) {
      await harness.repo.createEpisode(episode);
      harness.db
        .prepare(
          `INSERT INTO consolidation_families (
             family_id, current_version_episode_id, coverage_hash, policy_version,
             created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?)`,
        )
        .run(
          episode.consolidation_family_id,
          episode.id,
          episode.consolidation_coverage_hash,
          1,
          nowMs,
          nowMs,
        );
    }

    const currentVenue = await harness.repo.listRecentForSessionForDisclosure({
      sessionId: currentSession,
      sinceMs: nowMs,
      crossAudience: true,
      limit: 10,
    });
    const otherVenue = await harness.repo.listRecentForSessionForDisclosure({
      sessionId: otherSession,
      sinceMs: nowMs,
      crossAudience: true,
      limit: 10,
    });

    expect(currentVenue.map((candidate) => candidate.episode.id)).toEqual([singleVenue.id]);
    expect(otherVenue).toEqual([]);
  });

  it("fails closed for unknown-origin records across indexed disclosure lanes", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const jordan = "ent_eeeeeeeeeeeeeeee" as NonNullable<Episode["audience_entity_id"]>;
    const publicEpisode = createEpisode(createEpisodeId(), harness.clock.now(), {
      source_stream_ids: [createStreamEntryId()],
      audience_entity_id: null,
      origin_audience_entity_ids: [],
      shared: true,
      participants: ["Atlas"],
      tags: ["indexed-public"],
    });
    const unknownOrigin = createEpisode(createEpisodeId(), harness.clock.now() + 1_000, {
      source_stream_ids: [createStreamEntryId()],
      audience_entity_id: null,
      origin_audience_entity_ids: [],
      shared: false,
      participants: ["Atlas"],
      tags: ["indexed-public"],
    });

    await harness.repo.createEpisode(publicEpisode);
    await harness.repo.createEpisode(unknownOrigin);
    harness.repo.updateStats(publicEpisode.id, {
      retrieval_count: 10,
      win_rate: 1,
      last_retrieved: harness.clock.now(),
    });
    harness.repo.updateStats(unknownOrigin.id, {
      retrieval_count: 20,
      win_rate: 1,
      last_retrieved: harness.clock.now(),
    });

    const lanes = [
      await harness.repo.listRecentForDisclosure({ audienceEntityId: jordan, limit: 10 }),
      await harness.repo.listHottestForDisclosure({ audienceEntityId: jordan, limit: 10 }),
      await harness.repo.searchByTimeRangeForDisclosure(
        { start: harness.clock.now() - 1_000, end: harness.clock.now() + 3_000 },
        { audienceEntityId: jordan, limit: 10 },
      ),
      await harness.repo.searchByParticipantsOrTagsForDisclosure(["Atlas"], {
        audienceEntityId: jordan,
        limit: 10,
      }),
      await harness.repo.searchByParticipantsOrTagsForDisclosure(["indexed-public"], {
        audienceEntityId: jordan,
        limit: 10,
      }),
    ];

    for (const lane of lanes) {
      const ids = lane.map((item) => item.episode.id);
      expect(ids).toContain(publicEpisode.id);
      expect(ids).not.toContain(unknownOrigin.id);
    }
  });

  it("backfills normalized episode indexes from existing Lance rows", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const existing = createEpisode("ep_indexexisting001", harness.clock.now(), {
      source_stream_ids: ["strm_indexexisting001" as Episode["source_stream_ids"][number]],
      participants: ["Sam"],
      tags: ["Atlas"],
    });

    await harness.table.upsert([episodeToRawLanceRowForTest(existing)], { on: "id" });

    expect(harness.repo.getStats(existing.id)).toBeNull();

    const matches = await harness.repo.searchByParticipantsOrTags(["atlas"], {
      limit: 1,
      crossAudience: true,
    });
    const participantRows = harness.db
      .prepare("SELECT term FROM episode_participants WHERE episode_id = ?")
      .all(existing.id) as Array<{ term: string }>;
    const tagRows = harness.db
      .prepare("SELECT term FROM episode_tags WHERE episode_id = ?")
      .all(existing.id) as Array<{ term: string }>;

    expect(matches[0]?.episode.id).toBe(existing.id);
    expect(harness.repo.getStats(existing.id)).toEqual(
      expect.objectContaining({ episode_id: existing.id }),
    );
    expect(participantRows.map((row) => row.term)).toEqual(["sam"]);
    expect(tagRows.map((row) => row.term)).toEqual(["atlas"]);
  });

  it("hydrates only the requested indexed slice from a 1000-episode fixture", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const episodes = Array.from({ length: 1_000 }, (_, index) =>
      createEpisode(createEpisodeId(), harness.clock.now() + index, {
        title: `Indexed fixture ${index}`,
        source_stream_ids: [
          createEpisodeId().replace("ep_", "strm_") as Episode["source_stream_ids"][number],
        ],
        tags: [`fixture-${index}`],
        participants: ["fixture"],
        embedding: Float32Array.from([1, 0, 0, 0]),
      }),
    );

    await harness.table.upsert(
      episodes.map((episode) => episodeToRawLanceRowForTest(episode)),
      { on: "id" },
    );
    await harness.repo.reconcileCrossStoreState();

    const visibleSpy = vi.spyOn(harness.repo, "listVisibleEpisodes");
    const getManySpy = vi.spyOn(harness.repo, "getMany");
    const results = await harness.repo.listRecent({
      limit: 7,
      crossAudience: true,
    });

    expect(results).toHaveLength(7);
    expect(results[0]?.episode.updated_at).toBe(harness.clock.now() + 999);
    expect(getManySpy).toHaveBeenCalledTimes(1);
    expect(getManySpy.mock.calls[0]?.[0]).toHaveLength(7);
    expect(visibleSpy).not.toHaveBeenCalled();
  });

  it("skips stale rollback restores when a newer Lance update wins the race", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const episode = createEpisode(createEpisodeId(), harness.clock.now(), {
      tags: ["initial"],
    });
    await harness.repo.createEpisode(episode);
    harness.clock.advance(1_000);
    const originalGet = harness.repo.get.bind(harness.repo);
    const getSpy = vi
      .spyOn(harness.repo, "get")
      .mockImplementationOnce(originalGet)
      .mockImplementationOnce(async (id) => {
        const competingUpdatedAt = harness.clock.now() + 1_000;

        await harness.table.upsert(
          [
            {
              id,
              title: episode.title,
              narrative: "competing writer",
              participants: JSON.stringify(["user"]),
              location: null,
              start_time: episode.start_time,
              end_time: episode.end_time,
              source_stream_ids: JSON.stringify(episode.source_stream_ids),
              significance: episode.significance,
              tags: JSON.stringify(["competing"]),
              confidence: episode.confidence,
              lineage_derived_from: JSON.stringify([]),
              lineage_supersedes: JSON.stringify([]),
              source_fingerprint: episode.source_stream_ids.join("\n"),
              audience_entity_id: null,
              origin_audience_entity_ids: JSON.stringify([]),
              shared: true,
              emotional_arc: null,
              embedding: Array.from(episode.embedding),
              created_at: episode.created_at,
              updated_at: competingUpdatedAt,
            },
          ],
          { on: "id" },
        );

        return originalGet(id);
      });
    const statsSpy = vi
      .spyOn(
        harness.repo as unknown as {
          updateStats(episodeId: Episode["id"], patch: unknown): unknown;
        },
        "updateStats",
      )
      .mockImplementationOnce(() => {
        throw new Error("sqlite failed");
      });
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    await expect(
      harness.repo.update(episode.id, {
        tags: ["first-writer"],
      }),
    ).rejects.toMatchObject({
      code: "EPISODE_UPDATE_FAILED",
    });

    const persisted = await harness.repo.get(episode.id);

    expect(persisted?.narrative).toBe("competing writer");
    expect(persisted?.tags).toEqual(["competing"]);
    expect(warnSpy).toHaveBeenCalledWith(
      "Skipped episode rollback because newer Lance state exists.",
      expect.objectContaining({
        episodeId: episode.id,
      }),
    );

    getSpy.mockRestore();
    statsSpy.mockRestore();
  });
});
