import { afterEach, describe, expect, it, vi } from "vitest";

import {
  FakeLLMClient,
  EntityRepository,
  commitmentMigrations,
  episodicMigrations,
  EpisodicRepository,
  createEpisodesTableSchema,
  selfMigrations,
  retrievalMigrations,
  LanceDbStore,
  composeMigrations,
  openDatabase,
  ManualClock,
  createEpisodeId,
  createStreamEntryId,
  Borg,
  ScriptedEmbeddingClient,
  join,
  mkdtempSync,
  rmSync,
  tmpdir,
} from "./test-helpers.js";

describe("Borg", () => {
  const tempDirs: string[] = [];

  afterEach(() => {
    vi.restoreAllMocks();

    while (tempDirs.length > 0) {
      rmSync(tempDirs.pop() as string, { recursive: true, force: true });
    }
  });

  it("defaults episodic public APIs to public-only visibility unless audience access is explicit", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
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
    const entities = new EntityRepository({
      db,
      clock,
    });
    const alice = entities.resolve("Alice");

    await repo.insert({
      id: "ep_publicpublicpub1" as never,
      title: "Public planning note",
      narrative: "A public planning note.",
      participants: ["team"],
      location: null,
      start_time: 1,
      end_time: 2,
      source_stream_ids: ["strm_publicpublic0001" as never],
      significance: 0.8,
      tags: ["planning"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      audience_entity_id: null,
      shared: true,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 1,
      updated_at: 1,
    });
    await repo.insert({
      id: "ep_privateprivate01" as never,
      title: "Alice planning note",
      narrative: "A planning note only for Alice.",
      participants: ["Alice"],
      location: null,
      start_time: 3,
      end_time: 4,
      source_stream_ids: ["strm_privateprivate01" as never],
      significance: 0.8,
      tags: ["planning"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      audience_entity_id: alice,
      shared: false,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: 3,
      updated_at: 3,
    });
    db.close();
    await store.close();

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect(
        (await borg.episodic.search("planning", { limit: 5 })).map((item) => item.episode.id),
      ).toEqual(["ep_publicpublicpub1"]);
      expect((await borg.episodic.get("ep_privateprivate01" as never))?.episode.id).toBeUndefined();
      expect(
        (await borg.episodic.search("planning", { limit: 5, audience: "Alice" })).map(
          (item) => item.episode.id,
        ),
      ).toContain("ep_privateprivate01");
      expect(
        (await borg.episodic.get("ep_privateprivate01" as never, { audience: "Alice" }))?.episode
          .id,
      ).toBe("ep_privateprivate01");
      expect(
        (await borg.episodic.get("ep_privateprivate01" as never, { crossAudience: true }))?.episode
          .id,
      ).toBe("ep_privateprivate01");
    } finally {
      await borg.close();
    }
  });

  it("lets the public episodic search API rescue explicit entity matches", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(1_000_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
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

    for (let index = 0; index < 12; index += 1) {
      await repo.insert({
        id: `ep_publicentity${String(index).padStart(4, "0")}` as never,
        title: `Decoy ${index}`,
        narrative: "A strong semantic match that lacks the entity cue.",
        participants: ["team"],
        location: null,
        start_time: 1 + index,
        end_time: 2 + index,
        source_stream_ids: [`strm_publicentity${String(index).padStart(4, "0")}` as never],
        significance: 0.2,
        tags: ["decoy"],
        confidence: 0.9,
        lineage: {
          derived_from: [],
          supersedes: [],
        },
        emotional_arc: null,
        audience_entity_id: null,
        shared: true,
        embedding: Float32Array.from([1, 0, 0, 0]),
        created_at: 1 + index,
        updated_at: 1 + index,
      });
    }

    const rescuedId = "ep_entityrescue0001" as never;
    await repo.insert({
      id: rescuedId,
      title: "Atlas entity rescue",
      narrative: "A hot Atlas note that should be rescued by explicit entity terms.",
      participants: ["team"],
      location: null,
      start_time: 500,
      end_time: 600,
      source_stream_ids: ["strm_entityrescue0001" as never],
      significance: 1,
      tags: ["Atlas"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      audience_entity_id: null,
      shared: true,
      embedding: Float32Array.from([0, 1, 0, 0]),
      created_at: 500,
      updated_at: clock.now(),
    });
    repo.updateStats(rescuedId, {
      retrieval_count: 12,
      win_rate: 0.9,
      last_retrieved: clock.now() - 1_000,
    });
    db.close();
    await store.close();

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      expect(
        (await borg.episodic.search("deploy", { limit: 3, entityTerms: ["atlas"] })).map(
          (item) => item.episode.id,
        ),
      ).toContain(rescuedId);
    } finally {
      await borg.close();
    }
  });

  it("keeps explicit public API timeRange local to the time recall intent", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    tempDirs.push(tempDir);
    const clock = new ManualClock(10_000_000_000);
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: composeMigrations(
        episodicMigrations,
        selfMigrations,
        retrievalMigrations,
        commitmentMigrations,
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

    const hotOutOfRangeId = createEpisodeId();
    await repo.insert({
      id: hotOutOfRangeId,
      title: "Hot out-of-range deploy note",
      narrative: "A recent hot semantic match outside the requested time window.",
      participants: ["team"],
      location: null,
      start_time: 900_000,
      end_time: 901_000,
      source_stream_ids: [createStreamEntryId()],
      significance: 1,
      tags: ["deploy"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      audience_entity_id: null,
      shared: true,
      embedding: Float32Array.from([1, 0, 0, 0]),
      created_at: clock.now(),
      updated_at: clock.now(),
    });
    repo.updateStats(hotOutOfRangeId, {
      retrieval_count: 12,
      win_rate: 0.9,
      last_retrieved: clock.now() - 1_000,
    });

    const inRangeId = createEpisodeId();
    await repo.insert({
      id: inRangeId,
      title: "In-range deploy incident",
      narrative: "An older in-range note that should survive the strict time filter.",
      participants: ["team"],
      location: null,
      start_time: 150_000,
      end_time: 160_000,
      source_stream_ids: [createStreamEntryId()],
      significance: 1,
      tags: ["incident"],
      confidence: 0.9,
      lineage: {
        derived_from: [],
        supersedes: [],
      },
      emotional_arc: null,
      audience_entity_id: null,
      shared: true,
      embedding: Float32Array.from([0, 1, 0, 0]),
      created_at: 10,
      updated_at: 10,
    });
    db.close();
    await store.close();

    const borg = await Borg.open({
      dataDir: tempDir,
      clock,
      embeddingDimensions: 4,
      embeddingClient: new ScriptedEmbeddingClient(),
      llmClient: new FakeLLMClient(),
    });

    try {
      const results = await borg.episodic.search("deploy", {
        limit: 3,
        timeRange: {
          start: 140_000,
          end: 170_000,
        },
      });

      expect(results.map((item) => item.episode.id)).toContain(inRangeId);
      expect(results.some((item) => item.episode.id !== inRangeId)).toBe(true);
    } finally {
      await borg.close();
    }
  });
});
