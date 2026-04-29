import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { SuppressionSet, computeWeights } from "../cognition/attention/index.js";
import { summarizeRetrievedEpisodes } from "../cognition/deliberation/prompt/retrieval.js";
import type { TurnTracer } from "../cognition/tracing/tracer.js";
import type { EmbeddingClient } from "../embeddings/index.js";
import {
  StreamEntryIndexRepository,
  StreamReader,
  StreamWriter,
  streamEntryIndexMigrations,
} from "../stream/index.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../util/clock.js";
import { createEntityId } from "../util/ids.js";
import { OpenQuestionsRepository, createOpenQuestionsTableSchema } from "../memory/self/index.js";
import { selfMigrations } from "../memory/self/migrations.js";
import { semanticMigrations } from "../memory/semantic/migrations.js";
import { SemanticGraph } from "../memory/semantic/graph.js";
import {
  SemanticEdgeRepository,
  SemanticNodeRepository,
  createSemanticNodesTableSchema,
} from "../memory/semantic/repository.js";
import { episodicMigrations } from "../memory/episodic/migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "../memory/episodic/repository.js";
import { retrievalMigrations } from "./migrations.js";
import { RetrievalPipeline } from "./pipeline.js";
import type { Episode } from "../memory/episodic/types.js";

class ScriptedEmbeddingClient implements EmbeddingClient {
  async embed(text: string): Promise<Float32Array> {
    return this.embedVector(text);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.embedVector(text));
  }

  private embedVector(text: string): Float32Array {
    if (text.includes("planning") || text.includes("Atlas")) {
      return Float32Array.from([1, 0, 0, 0]);
    }

    if (text.includes("retrospective")) {
      return Float32Array.from([0, 1, 0, 0]);
    }

    return Float32Array.from([0, 0, 1, 0]);
  }
}

function createEpisode(id: string, sourceId: string, embedding: number[]): Episode {
  return {
    id: id as Episode["id"],
    title: `${id} title`,
    narrative: `${id} narrative.`,
    participants: ["user"],
    location: null,
    start_time: 1_000,
    end_time: 2_000,
    source_stream_ids: [sourceId as Episode["source_stream_ids"][number]],
    significance: 0.8,
    tags: ["planning"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: null,
    embedding: Float32Array.from(embedding),
    created_at: 1_000,
    updated_at: 1_000,
  };
}

async function openRetrievalFixture(tempDir: string) {
  const store = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: [
      ...episodicMigrations,
      ...selfMigrations,
      ...retrievalMigrations,
      ...streamEntryIndexMigrations,
    ],
  });
  const table = await store.openTable({
    name: "episodes",
    schema: createEpisodesTableSchema(4),
  });
  const episodicRepository = new EpisodicRepository({
    table,
    db,
    clock: new FixedClock(5_000),
  });
  const entryIndex = new StreamEntryIndexRepository({
    db,
    dataDir: tempDir,
  });

  return {
    store,
    db,
    episodicRepository,
    entryIndex,
  };
}

describe("retrieval pipeline", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("retrieves episodes, resolves citations, and records retrieval stats", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "retrospective note",
    });

    await repo.insert(createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]));
    await repo.insert(createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [0, 1, 0, 0]));

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const results = await pipeline.search("planning", {
      limit: 1,
    });

    expect(results).toEqual([
      expect.objectContaining({
        episode: expect.objectContaining({
          id: "ep_aaaaaaaaaaaaaaaa",
        }),
        citationChain: [
          expect.objectContaining({
            id: firstEntry.id,
          }),
        ],
      }),
    ]);
    expect(repo.getStats("ep_aaaaaaaaaaaaaaaa" as Episode["id"])?.retrieval_count).toBe(1);
  });

  it("keeps unresolved citation markers in rendered citation chains and traces them", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });
    const tracer: TurnTracer = {
      enabled: true,
      includePayloads: false,
      emit: vi.fn(),
    };

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const resolvedEntry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const missingId = "strm_cccccccccccccccc" as Episode["source_stream_ids"][number];

    await repo.insert({
      ...createEpisode("ep_aaaaaaaaaaaaaaaa", resolvedEntry.id, [1, 0, 0, 0]),
      source_stream_ids: [resolvedEntry.id, missingId],
    });

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      tracer,
    });

    const results = await pipeline.search("planning", {
      limit: 1,
      traceTurnId: "turn-citations",
    });
    const rendered = summarizeRetrievedEpisodes("Retrieved context", results);

    expect(results[0]?.citationChain.map((entry) => entry.content)).toEqual([
      "planning kickoff",
      `[citation unresolved: ${missingId}]`,
    ]);
    expect(rendered).toContain("planning kickoff");
    expect(rendered).toContain(`[citation unresolved: ${missingId}]`);
    expect(tracer.emit).toHaveBeenCalledWith("citation_unresolved", {
      turnId: "turn-citations",
      missingIds: [missingId],
      resolvedCount: 1,
    });
  });

  it("defaults search to public-only visibility unless cross-audience is explicit", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
      clock: new FixedClock(5_000),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await repo.insert(
      createEpisode("ep_publicvisible000", "strm_publicvisible000" as never, [1, 0, 0, 0]),
    );
    await repo.insert({
      ...createEpisode("ep_scopehidden00001", "strm_scopehidden00001" as never, [1, 0, 0, 0]),
      audience_entity_id: "ent_bbbbbbbbbbbbbbbb" as never,
      shared: false,
    });

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const defaultResults = await pipeline.search("planning", {
      limit: 5,
    });
    const crossAudienceResults = await pipeline.search("planning", {
      limit: 5,
      crossAudience: true,
    });

    expect(defaultResults.map((result) => result.episode.id)).toEqual(["ep_publicvisible000"]);
    expect(crossAudienceResults).toHaveLength(2);
    expect(crossAudienceResults.map((result) => result.episode.id)).toEqual(
      expect.arrayContaining(["ep_publicvisible000", "ep_scopehidden00001"]),
    );
  });

  it("batches citation resolution into a single stream scan per session", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "planning follow-up",
    });

    await repo.insert(createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]));
    await repo.insert(createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [0.9, 0.1, 0, 0]));

    const iterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const results = await pipeline.search("planning", {
      limit: 2,
    });

    expect(results).toHaveLength(2);
    expect(iterateSpy).toHaveBeenCalledTimes(1);
  });

  it("resolves citations via the entry index and matches the scan-based result", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const { store, db, episodicRepository, entryIndex } = await openRetrievalFixture(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
      entryIndex,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const entry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const episodeId = "ep_aaaaaaaaaaaaaaa1";
    await episodicRepository.insert(createEpisode(episodeId, entry.id, [1, 0, 0, 0]));

    const indexedIterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
    const indexedPipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      entryIndex,
    });
    const indexedResult = await indexedPipeline.getEpisode(episodeId as Episode["id"], {
      crossAudience: true,
    });

    expect(indexedIterateSpy).toHaveBeenCalledTimes(0);
    indexedIterateSpy.mockRestore();

    const fallbackPipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });
    const fallbackResult = await fallbackPipeline.getEpisode(episodeId as Episode["id"], {
      crossAudience: true,
    });

    expect(indexedResult).toEqual(fallbackResult);
  });

  it("falls back to a stream scan when an index row is missing", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const { store, db, episodicRepository, entryIndex } = await openRetrievalFixture(tempDir);
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
      entryIndex,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const entry = await writer.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const episodeId = "ep_bbbbbbbbbbbbbbb2";
    await episodicRepository.insert(createEpisode(episodeId, entry.id, [1, 0, 0, 0]));
    db.prepare("DELETE FROM stream_entry_index WHERE entry_id = ?").run(entry.id);

    const iterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      entryIndex,
    });

    const result = await pipeline.getEpisode(episodeId as Episode["id"], {
      crossAudience: true,
    });

    expect(result?.citationChain[0]?.id).toBe(entry.id);
    expect(iterateSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy).toHaveBeenCalledWith("Citation index miss; falling back to stream scan.", {
      entryId: entry.id,
    });
  });

  it("resolves multi-session citations from the index without scanning unrelated sessions", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const { store, db, episodicRepository, entryIndex } = await openRetrievalFixture(tempDir);
    const defaultWriter = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
      entryIndex,
    });
    const secondaryWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: "sess_aaaaaaaaaaaaaaaa" as never,
      clock: new FixedClock(2_100),
      entryIndex,
    });
    const unrelatedWriter = new StreamWriter({
      dataDir: tempDir,
      sessionId: "sess_bbbbbbbbbbbbbbbb" as never,
      clock: new FixedClock(2_200),
      entryIndex,
    });

    cleanup.push(async () => {
      defaultWriter.close();
      secondaryWriter.close();
      unrelatedWriter.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const defaultEntry = await defaultWriter.append({
      kind: "user_msg",
      content: "planning kickoff",
    });
    const secondaryEntry = await secondaryWriter.append({
      kind: "agent_msg",
      content: "planning follow-up",
    });
    await unrelatedWriter.append({
      kind: "internal_event",
      content: "not cited",
    });

    await episodicRepository.insert(
      createEpisode("ep_multisession0001", defaultEntry.id, [1, 0, 0, 0]),
    );
    await episodicRepository.insert(
      createEpisode("ep_multisession0002", secondaryEntry.id, [0.9, 0.1, 0, 0]),
    );

    const iterateSpy = vi.spyOn(StreamReader.prototype, "iterate");
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
      entryIndex,
    });

    const results = await pipeline.search("planning", {
      limit: 2,
      crossAudience: true,
    });

    expect(results).toHaveLength(2);
    expect(results.flatMap((result) => result.citationChain.map((entry) => entry.id))).toEqual(
      expect.arrayContaining([defaultEntry.id, secondaryEntry.id]),
    );
    expect(iterateSpy).toHaveBeenCalledTimes(0);
  });

  it("hides scoped episodes by id unless the caller provides audience access or cross-audience mode", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
      clock: new FixedClock(5_000),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await repo.insert({
      ...createEpisode("ep_privateepisode01", "strm_privateepisode01" as never, [1, 0, 0, 0]),
      audience_entity_id: "ent_cccccccccccccccc" as never,
      shared: false,
    });

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    expect(await pipeline.getEpisode("ep_privateepisode01" as Episode["id"])).toBeNull();
    expect(
      (
        await pipeline.getEpisode("ep_privateepisode01" as Episode["id"], {
          audienceEntityId: "ent_cccccccccccccccc" as never,
        })
      )?.episode.id,
    ).toBe("ep_privateepisode01");
    expect(
      (
        await pipeline.getEpisode("ep_privateepisode01" as Episode["id"], {
          crossAudience: true,
        })
      )?.episode.id,
    ).toBe("ep_privateepisode01");
  });

  it("filters archived episodes from direct episode lookup", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
      clock: new FixedClock(5_000),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episode = createEpisode(
      "ep_archivedlookup01",
      "strm_archivedlookup01" as never,
      [1, 0, 0, 0],
    );
    await repo.insert(episode);
    repo.updateStats(episode.id, {
      archived: true,
    });
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    expect(await pipeline.getEpisode(episode.id, { crossAudience: true })).toBeNull();
    expect(await repo.get(episode.id, { includeArchived: true })).toEqual(
      expect.objectContaining({
        id: episode.id,
      }),
    );
  });

  it("rescales results with attention weights and suppression", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
      clock: new FixedClock(5_000),
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const firstEntry = await writer.append({
      kind: "user_msg",
      content: "release planning",
    });
    const secondEntry = await writer.append({
      kind: "agent_msg",
      content: "release planning followup",
    });

    await repo.insert({
      ...createEpisode("ep_aaaaaaaaaaaaaaaa", firstEntry.id, [1, 0, 0, 0]),
      title: "release goal",
      narrative: "release goal context",
    });
    await repo.insert({
      ...createEpisode("ep_bbbbbbbbbbbbbbbb", secondEntry.id, [1, 0, 0, 0]),
      title: "generic note",
    });

    const suppression = new SuppressionSet();

    suppression.suppress("ep_bbbbbbbbbbbbbbbb", "already seen", 2);

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const results = await pipeline.search("release planning", {
      limit: 2,
      attentionWeights: computeWeights("reflective", {
        currentGoals: [
          {
            id: "goal_aaaaaaaaaaaaaaaa" as never,
            description: "release goal",
            priority: 1,
            parent_goal_id: null,
            status: "active",
            progress_notes: null,
            last_progress_ts: null,
            created_at: 0,
            target_at: null,
            provenance: { kind: "system" },
          },
        ],
        hasActiveValues: false,
        hasTemporalCue: false,
      }),
      goalDescriptions: ["release goal"],
      suppressionSet: suppression,
    });

    expect(results[0]?.episode.id).toBe("ep_aaaaaaaaaaaaaaaa");
    expect(results[1]?.scoreBreakdown.suppressionPenalty).toBe(1);
  });

  it("attaches semantic graph context and surfaces contradiction presence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        ...episodicMigrations,
        ...selfMigrations,
        ...retrievalMigrations,
        ...semanticMigrations,
      ],
    });
    const episodeTable = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const semanticTable = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table: episodeTable,
      db,
      clock: new FixedClock(5_000),
    });
    const semanticNodeRepository = new SemanticNodeRepository({
      table: semanticTable,
      db,
      clock: new FixedClock(5_000),
    });
    const semanticEdgeRepository = new SemanticEdgeRepository({
      db,
      clock: new FixedClock(5_000),
    });
    const semanticGraph = new SemanticGraph({
      nodeRepository: semanticNodeRepository,
      edgeRepository: semanticEdgeRepository,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock: new FixedClock(2_000),
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const entry = await writer.append({
      kind: "user_msg",
      content: "Atlas deploy failure",
    });

    await repo.insert(createEpisode("ep_aaaaaaaaaaaaaaaa", entry.id, [1, 0, 0, 0]));
    const atlas = await semanticNodeRepository.insert({
      id: "semn_aaaaaaaaaaaaaaaa" as never,
      kind: "entity",
      label: "Atlas",
      description: "Atlas entity",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const support = await semanticNodeRepository.insert({
      id: "semn_bbbbbbbbbbbbbbbb" as never,
      kind: "proposition",
      label: "Rerun install",
      description: "Rerun pnpm install",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const contradiction = await semanticNodeRepository.insert({
      id: "semn_cccccccccccccccc" as never,
      kind: "proposition",
      label: "Do nothing",
      description: "Do nothing and wait",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 0, 1, 0]),
      archived: false,
      superseded_by: null,
    });
    const category = await semanticNodeRepository.insert({
      id: "semn_dddddddddddddddd" as never,
      kind: "concept",
      label: "Service",
      description: "Service category",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      updated_at: 1,
      last_verified_at: 1,
      embedding: Float32Array.from([0, 0, 0, 1]),
      archived: false,
      superseded_by: null,
    });

    semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: support.id,
      relation: "supports",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      last_verified_at: 1,
    });
    semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: contradiction.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      last_verified_at: 1,
    });
    semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: category.id,
      relation: "is_a",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1,
      last_verified_at: 1,
    });

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      semanticNodeRepository,
      semanticGraph,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const result = await pipeline.searchWithContext("Atlas", {
      limit: 1,
      graphWalkDepth: 1,
      maxGraphNodes: 8,
    });

    expect(result.contradiction_present).toBe(true);
    // Phase C: semantic moved from per-episode (where it duplicated) to a
    // top-level RetrievedContext.semantic lane. Each band -- episodes,
    // semantic, open questions -- now has an independent section that can
    // contribute regardless of what the other bands returned.
    expect(result.semantic).toMatchObject({
      supports: [expect.objectContaining({ id: support.id })],
      contradicts: [expect.objectContaining({ id: contradiction.id })],
      categories: [expect.objectContaining({ id: category.id })],
      matched_node_ids: [atlas.id],
      matched_nodes: [expect.objectContaining({ id: atlas.id })],
      support_hits: [
        expect.objectContaining({
          root_node_id: atlas.id,
          node: expect.objectContaining({ id: support.id }),
        }),
      ],
      contradiction_hits: [
        expect.objectContaining({
          root_node_id: atlas.id,
          node: expect.objectContaining({ id: contradiction.id }),
        }),
      ],
      category_hits: [
        expect.objectContaining({
          root_node_id: atlas.id,
          node: expect.objectContaining({ id: category.id }),
        }),
      ],
    });
  });

  it("assigns nonzero confidence when search finds semantic evidence without episodes", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        ...episodicMigrations,
        ...selfMigrations,
        ...retrievalMigrations,
        ...semanticMigrations,
      ],
    });
    const episodeTable = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const semanticTable = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new FixedClock(10_000);
    const repo = new EpisodicRepository({
      table: episodeTable,
      db,
      clock,
    });
    const semanticNodeRepository = new SemanticNodeRepository({
      table: semanticTable,
      db,
      clock,
    });
    const semanticEdgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const semanticGraph = new SemanticGraph({
      nodeRepository: semanticNodeRepository,
      edgeRepository: semanticEdgeRepository,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const episode = createEpisode("ep_aaaaaaaaaaaaaaaa", "strm_aaaaaaaaaaaaaaaa", [1, 0, 0, 0]);
    await repo.insert(episode);
    repo.updateStats(episode.id, {
      archived: true,
    });
    const atlas = await semanticNodeRepository.insert({
      id: "semn_aaaaaaaaaaaaaaaa" as never,
      kind: "entity",
      label: "Atlas",
      description: "Atlas entity",
      aliases: [],
      confidence: 0.9,
      source_episode_ids: [episode.id],
      created_at: 10_000,
      updated_at: 10_000,
      last_verified_at: 10_000,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const support = await semanticNodeRepository.insert({
      id: "semn_bbbbbbbbbbbbbbbb" as never,
      kind: "proposition",
      label: "Atlas deploys are steadier with rollback plans",
      description: "Rollback plans support steadier Atlas deploys.",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: [episode.id],
      created_at: 10_000,
      updated_at: 10_000,
      last_verified_at: 10_000,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: support.id,
      relation: "supports",
      confidence: 0.8,
      evidence_episode_ids: [episode.id],
      created_at: 10_000,
      last_verified_at: 10_000,
    });
    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      semanticNodeRepository,
      semanticGraph,
      dataDir: tempDir,
      clock,
    });

    const result = await pipeline.searchWithContext("Atlas", {
      crossAudience: true,
      graphWalkDepth: 1,
      limit: 1,
    });

    expect(result.episodes).toEqual([]);
    expect(result.semantic.matched_nodes).toEqual(
      expect.arrayContaining([expect.objectContaining({ id: atlas.id })]),
    );
    expect(result.semantic.support_hits).toHaveLength(1);
    expect(result.confidence.overall).toBeGreaterThan(0);
  });

  it("threads semantic as-of through graph retrieval and confidence", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [
        ...episodicMigrations,
        ...selfMigrations,
        ...retrievalMigrations,
        ...semanticMigrations,
      ],
    });
    const episodeTable = await store.openTable({
      name: "episodes",
      schema: createEpisodesTableSchema(4),
    });
    const semanticTable = await store.openTable({
      name: "semantic_nodes",
      schema: createSemanticNodesTableSchema(4),
    });
    const clock = new ManualClock(1_000_000);
    const repo = new EpisodicRepository({
      table: episodeTable,
      db,
      clock,
    });
    const semanticNodeRepository = new SemanticNodeRepository({
      table: semanticTable,
      db,
      clock,
    });
    const semanticEdgeRepository = new SemanticEdgeRepository({
      db,
      clock,
    });
    const semanticGraph = new SemanticGraph({
      nodeRepository: semanticNodeRepository,
      edgeRepository: semanticEdgeRepository,
    });
    const writer = new StreamWriter({
      dataDir: tempDir,
      clock,
    });

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const entry = await writer.append({
      kind: "user_msg",
      content: "Atlas deploy note",
    });

    await repo.insert(createEpisode("ep_aaaaaaaaaaaaaaaa", entry.id, [1, 0, 0, 0]));
    const atlas = await semanticNodeRepository.insert({
      id: "semn_aaaaaaaaaaaaaaaa" as never,
      kind: "entity",
      label: "Atlas",
      description: "Atlas entity",
      aliases: [],
      confidence: 0.8,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1_000_000,
      updated_at: 1_000_000,
      last_verified_at: 1_000_000,
      embedding: Float32Array.from([1, 0, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const contradiction = await semanticNodeRepository.insert({
      id: "semn_bbbbbbbbbbbbbbbb" as never,
      kind: "proposition",
      label: "Atlas needs no deployment work",
      description: "A stale claim that Atlas deployment needs no action.",
      aliases: [],
      confidence: 0.7,
      source_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1_000_000,
      updated_at: 1_000_000,
      last_verified_at: 1_000_000,
      embedding: Float32Array.from([0, 1, 0, 0]),
      archived: false,
      superseded_by: null,
    });
    const edge = semanticEdgeRepository.addEdge({
      from_node_id: atlas.id,
      to_node_id: contradiction.id,
      relation: "contradicts",
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as Episode["id"]],
      created_at: 1_000_000,
      last_verified_at: 1_000_000,
    });
    semanticEdgeRepository.invalidateEdge(edge.id, {
      at: 1_000_500,
      by_process: "manual",
    });
    clock.set(1_001_000);

    const pipeline = new RetrievalPipeline({
      embeddingClient: new ScriptedEmbeddingClient(),
      episodicRepository: repo,
      semanticNodeRepository,
      semanticGraph,
      dataDir: tempDir,
      clock,
    });

    const current = await pipeline.searchWithContext("Atlas", {
      limit: 1,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
    });
    const historical = await pipeline.searchWithContext("Atlas", {
      limit: 1,
      graphWalkDepth: 1,
      maxGraphNodes: 4,
      asOf: 1_000_250,
    });

    expect(current.contradiction_present).toBe(false);
    expect(current.semantic.contradiction_hits).toEqual([]);
    expect(current.confidence.contradictionPresent).toBe(false);
    expect(historical.semantic.as_of).toBe(1_000_250);
    expect(historical.contradiction_present).toBe(true);
    expect(historical.semantic.contradiction_hits[0]?.edgePath[0]?.id).toBe(edge.id);
    expect(historical.confidence.contradictionPresent).toBe(true);
  });

  it("attaches relevant open questions when requested", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
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
    const openQuestionsTable = await store.openTable({
      name: "open_questions",
      schema: createOpenQuestionsTableSchema(4),
    });
    const repo = new EpisodicRepository({
      table,
      db,
      clock: new FixedClock(5_000),
    });
    const embeddingClient = new ScriptedEmbeddingClient();
    const openQuestionsRepository = new OpenQuestionsRepository({
      db,
      table: openQuestionsTable,
      embeddingClient,
      clock: new FixedClock(5_000),
    });
    const alice = createEntityId();
    const bob = createEntityId();

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await repo.insert({
      ...createEpisode("ep_aaaaaaaaaaaaaaaa", "strm_aaaaaaaaaaaaaaaa", [1, 0, 0, 0]),
      title: "Atlas deployment note",
    });
    openQuestionsRepository.add({
      question: "Why does Atlas deployment keep failing?",
      urgency: 0.8,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    openQuestionsRepository.add({
      question: "What snacks should we order?",
      urgency: 0.9,
      source: "user",
      provenance: { kind: "manual" },
    });
    openQuestionsRepository.add({
      question: "Why does Atlas deployment keep failing for Alice?",
      urgency: 1,
      audience_entity_id: alice,
      source: "reflection",
      provenance: { kind: "manual" },
    });
    await openQuestionsRepository.waitForPendingEmbeddings();

    const pipeline = new RetrievalPipeline({
      embeddingClient,
      episodicRepository: repo,
      openQuestionsRepository,
      dataDir: tempDir,
      clock: new FixedClock(10_000),
    });

    const reflective = await pipeline.searchWithContext("Atlas deployment", {
      limit: 1,
      includeOpenQuestions: true,
      audienceEntityId: bob,
    });
    const defaultResult = await pipeline.searchWithContext("Atlas deployment", {
      limit: 1,
    });

    expect(reflective.open_questions).toEqual([
      expect.objectContaining({
        question: "Why does Atlas deployment keep failing?",
      }),
    ]);
    expect(defaultResult.open_questions).toEqual([]);
  });
});
