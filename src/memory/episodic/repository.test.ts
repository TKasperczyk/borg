import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it } from "vitest";

import { selfMigrations } from "../self/migrations.js";
import {
  LanceDbStore,
  float64Field,
  schema,
  utf8Field,
  vectorField,
} from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { ManualClock } from "../../util/clock.js";
import { StorageError } from "../../util/errors.js";
import { episodicMigrations } from "./migrations.js";
import { EpisodicRepository, createEpisodesTableSchema } from "./repository.js";
import type { Episode } from "./types.js";
import { retrievalMigrations } from "../../retrieval/migrations.js";

type Harness = {
  tempDir: string;
  store: LanceDbStore;
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
});
