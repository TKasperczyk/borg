import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { selfMigrations } from "../self/migrations.js";
import {
  LanceDbTable,
  LanceDbStore,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import type { SqliteDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { createEpisodeId } from "../../util/ids.js";
import { episodicMigrations } from "./migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "./repository.js";
import type { Episode } from "./types.js";
import { retrievalMigrations } from "../../retrieval/migrations.js";

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
  };
}

function toRawEpisodeRow(episode: Episode): Record<string, unknown> {
  const audienceEntityId = episode.audience_entity_id ?? null;

  return {
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
    source_fingerprint: [...new Set(episode.source_stream_ids)].sort().join("\n"),
    audience_entity_id: audienceEntityId,
    shared: episode.shared ?? audienceEntityId === null,
    emotional_arc: episode.emotional_arc === null ? null : JSON.stringify(episode.emotional_arc),
    embedding: Array.from(episode.embedding),
    created_at: episode.created_at,
    updated_at: episode.updated_at,
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
    migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
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

    await harness.repo.insert(first);
    await harness.repo.insert(second);
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
    expect(paged.items).toHaveLength(1);
    expect(await harness.repo.delete(second.id)).toBe(true);
    expect(await harness.repo.get(second.id)).toBeNull();
  });

  it("filters archived episodes from get unless explicitly included", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const episode = createEpisode("ep_archivedgetxxxxx", harness.clock.now());
    await harness.repo.insert(episode);
    harness.repo.updateStats(episode.id, {
      archived: true,
    });

    expect(await harness.repo.get(episode.id)).toBeNull();
    expect(await harness.repo.get(episode.id, { includeArchived: true })).toEqual(
      expect.objectContaining({
        id: episode.id,
      }),
    );
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

    await harness.repo.insert(publicEpisode);
    await harness.repo.insert(scopedEpisode);

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

  it("rejects inserts without citation anchors", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    await expect(
      harness.repo.insert(
        createEpisode("ep_aaaaaaaaaaaaaaaa", harness.clock.now(), {
          source_stream_ids: [],
        }),
      ),
    ).rejects.toBeInstanceOf(StorageError);
  });

  it("keeps missing emotional arcs unknown on read", async () => {
    const harness = await createHarness();
    closers.push(harness.close);

    const episode = createEpisode("ep_unknownarc000000", harness.clock.now(), {
      emotional_arc: null,
    });

    await harness.repo.insert(episode);

    expect((await harness.repo.get(episode.id))?.emotional_arc).toBeNull();
  });

  it("evolves a pre-sprint-7 LanceDB table and accepts emotional_arc on insert", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new ManualClock(1_700_000_000_000);
    const legacyStore = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });

    closers.push(async () => {
      rmSync(tempDir, { recursive: true, force: true });
    });

    const legacySchema = schema([
      utf8Field("id"),
      utf8Field("title"),
      utf8Field("narrative"),
      utf8Field("participants"),
      utf8Field("location", true),
      float64Field("start_time"),
      float64Field("end_time"),
      utf8Field("source_stream_ids"),
      float64Field("significance"),
      utf8Field("tags"),
      float64Field("confidence"),
      utf8Field("lineage_derived_from"),
      utf8Field("lineage_supersedes"),
      vectorField("embedding", 4),
      float64Field("created_at"),
      float64Field("updated_at"),
    ]);

    const legacyTable = await legacyStore.openTable({
      name: "episodes",
      schema: legacySchema,
    });
    legacyTable.close();
    await legacyStore.close();

    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    closers.push(async () => {
      db.close();
      await store.close();
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

    const emotionalArc = {
      start: { valence: -0.6, arousal: 0.5 },
      peak: { valence: -0.8, arousal: 0.7 },
      end: { valence: -0.2, arousal: 0.3 },
      dominant_emotion: "anger",
    };

    await table.upsert(
      [
        {
          id: "ep_legacyyyyyyyyyyy",
          title: "legacy arc",
          narrative: "legacy arc narrative.",
          participants: JSON.stringify(["user"]),
          location: null,
          start_time: clock.now(),
          end_time: clock.now() + 1_000,
          source_stream_ids: JSON.stringify(["strm_aaaaaaaaaaaaaaaa"]),
          significance: 0.8,
          tags: JSON.stringify(["alpha"]),
          confidence: 0.9,
          lineage_derived_from: JSON.stringify([]),
          lineage_supersedes: JSON.stringify([]),
          emotional_arc: JSON.stringify(emotionalArc),
          embedding: [1, 0, 0, 0],
          created_at: clock.now(),
          updated_at: clock.now(),
        },
      ],
      { on: "id" },
    );

    expect((await repo.get("ep_legacyyyyyyyyyyy" as Episode["id"]))?.emotional_arc).toEqual(
      emotionalArc,
    );
  });

  it("matches legacy rows by normalized source ids when source_fingerprint is missing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const clock = new ManualClock(1_700_000_000_000);
    const legacyStore = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });

    closers.push(async () => {
      rmSync(tempDir, { recursive: true, force: true });
    });

    const legacySchema = schema([
      utf8Field("id"),
      utf8Field("title"),
      utf8Field("narrative"),
      utf8Field("participants"),
      utf8Field("location", true),
      float64Field("start_time"),
      float64Field("end_time"),
      utf8Field("source_stream_ids"),
      float64Field("significance"),
      utf8Field("tags"),
      float64Field("confidence"),
      utf8Field("lineage_derived_from"),
      utf8Field("lineage_supersedes"),
      vectorField("embedding", 4),
      float64Field("created_at"),
      float64Field("updated_at"),
    ]);
    const legacyTable = await legacyStore.openTable({
      name: "episodes",
      schema: legacySchema,
    });
    await legacyTable.upsert(
      [
        {
          id: "ep_legacysourceord1",
          title: "legacy source order",
          narrative: "Legacy source ids were written in a different order.",
          participants: JSON.stringify(["user"]),
          location: null,
          start_time: clock.now(),
          end_time: clock.now() + 1_000,
          source_stream_ids: JSON.stringify(["strm_bbbbbbbbbbbbbbbb", "strm_aaaaaaaaaaaaaaaa"]),
          significance: 0.8,
          tags: JSON.stringify(["alpha"]),
          confidence: 0.9,
          lineage_derived_from: JSON.stringify([]),
          lineage_supersedes: JSON.stringify([]),
          embedding: [1, 0, 0, 0],
          created_at: clock.now(),
          updated_at: clock.now(),
        },
      ],
      { on: "id" },
    );
    legacyTable.close();
    await legacyStore.close();

    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...episodicMigrations, ...selfMigrations, ...retrievalMigrations],
    });
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    closers.push(async () => {
      db.close();
      await store.close();
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

    const matched = await repo.findBySourceStreamIds([
      "strm_aaaaaaaaaaaaaaaa" as Episode["source_stream_ids"][number],
      "strm_bbbbbbbbbbbbbbbb" as Episode["source_stream_ids"][number],
    ]);

    expect(matched?.id).toBe("ep_legacysourceord1");
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
    await harness.repo.insert(episode);

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
    await harness.repo.insert(episode);
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

    await expect(harness.repo.insert(episode)).rejects.toMatchObject({
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

    await harness.repo.insert(older);
    await harness.repo.insert(scoped);
    await harness.repo.insert(entity);
    await harness.repo.insert(hot);
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

  it("backfills normalized episode indexes from existing Lance rows", async () => {
    const harness = await createHarness();
    closers.push(harness.close);
    const legacy = createEpisode("ep_indexlegacy00001", harness.clock.now(), {
      source_stream_ids: ["strm_indexlegacy00001" as Episode["source_stream_ids"][number]],
      participants: ["Sam"],
      tags: ["Atlas"],
    });

    await harness.table.upsert([toRawEpisodeRow(legacy)], { on: "id" });

    expect(harness.repo.getStats(legacy.id)).toBeNull();

    const matches = await harness.repo.searchByParticipantsOrTags(["atlas"], {
      limit: 1,
      crossAudience: true,
    });
    const participantRows = harness.db
      .prepare("SELECT term FROM episode_participants WHERE episode_id = ?")
      .all(legacy.id) as Array<{ term: string }>;
    const tagRows = harness.db
      .prepare("SELECT term FROM episode_tags WHERE episode_id = ?")
      .all(legacy.id) as Array<{ term: string }>;

    expect(matches[0]?.episode.id).toBe(legacy.id);
    expect(harness.repo.getStats(legacy.id)).toEqual(
      expect.objectContaining({ episode_id: legacy.id }),
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
      episodes.map((episode) => toRawEpisodeRow(episode)),
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
    await harness.repo.insert(episode);
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
