import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { episodicMigrations } from "../memory/episodic/migrations.js";
import { createEpisodesTableSchema, EpisodicRepository } from "../memory/episodic/repository.js";
import type { Episode } from "../memory/episodic/types.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { composeMigrations, openDatabase } from "../storage/sqlite/index.js";
import { StreamWriter } from "../stream/index.js";
import { FixedClock } from "../util/clock.js";
import { retrieveFactualChallengeEvidence } from "./factual-challenge-evidence.js";

function createEpisode(input: {
  id: string;
  sourceId: Episode["source_stream_ids"][number];
  participants?: string[];
  tags?: string[];
}): Episode {
  return {
    id: input.id as Episode["id"],
    title: `${input.id} title`,
    narrative: "Tom and Maya discussed the two-day broth and her birthday plan.",
    participants: input.participants ?? ["Tom", "Maya"],
    location: null,
    start_time: 1_000,
    end_time: 2_000,
    source_stream_ids: [input.sourceId],
    significance: 0.8,
    tags: input.tags ?? ["Maya"],
    confidence: 0.9,
    lineage: {
      derived_from: [],
      supersedes: [],
    },
    emotional_arc: null,
    embedding: Float32Array.from([1, 0, 0, 0]),
    created_at: 1_000,
    updated_at: 1_000,
  };
}

async function openFixture(tempDir: string) {
  const store = new LanceDbStore({
    uri: join(tempDir, "lancedb"),
  });
  const db = openDatabase(join(tempDir, "borg.db"), {
    migrations: composeMigrations(episodicMigrations),
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
  const writer = new StreamWriter({
    dataDir: tempDir,
    clock: new FixedClock(2_000),
  });

  return {
    store,
    db,
    writer,
    episodicRepository,
  };
}

describe("retrieveFactualChallengeEvidence", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("returns empty evidence when the disputed entity is absent", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const { store, db, writer, episodicRepository } = await openFixture(tempDir);

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await writer.append({
      kind: "user_msg",
      content: "We talked about the broth timer.",
    });

    const evidence = await retrieveFactualChallengeEvidence({
      challenge: {
        disputed_entity: "Maya",
        disputed_property: "is my partner",
        user_position: "Maya is not my partner.",
      },
      episodicRepository,
      dataDir: tempDir,
    });

    expect(evidence).toMatchObject({
      disputed_entity: "Maya",
      reference_count: 0,
      episodes: [],
      raw_user_messages: [],
    });
  });

  it("returns episode and raw user-message evidence when the entity is present", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const { store, db, writer, episodicRepository } = await openFixture(tempDir);

    cleanup.push(async () => {
      writer.close();
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const userEntry = await writer.append({
      kind: "user_msg",
      content: "Maya wants the two-day broth for her birthday.",
    });
    await episodicRepository.insert(
      createEpisode({
        id: "ep_aaaaaaaaaaaaaaaa",
        sourceId: userEntry.id,
        tags: ["Maya"],
      }),
    );

    const evidence = await retrieveFactualChallengeEvidence({
      challenge: {
        disputed_entity: "Maya",
        disputed_property: "is my partner",
        user_position: "Maya is not my partner.",
      },
      episodicRepository,
      dataDir: tempDir,
    });

    expect(evidence?.reference_count).toBe(2);
    expect(evidence?.episodes).toEqual([
      expect.objectContaining({
        episode_id: "ep_aaaaaaaaaaaaaaaa",
        tags: ["Maya"],
      }),
    ]);
    expect(evidence?.raw_user_messages).toEqual([
      expect.objectContaining({
        stream_entry_id: userEntry.id,
        timestamp: userEntry.timestamp,
        snippet: "Maya wants the two-day broth for her birthday.",
      }),
    ]);
  });
});
