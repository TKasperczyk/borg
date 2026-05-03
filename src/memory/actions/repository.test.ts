import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createActionId,
  createEntityId,
  createStreamEntryId,
  type ActionId,
} from "../../util/ids.js";
import { actionMigrations } from "./migrations.js";
import { ActionRepository, createActionRecordsTableSchema } from "./repository.js";
import type { ActionRecord } from "./types.js";

class MapEmbeddingClient implements EmbeddingClient {
  constructor(private readonly vectors: ReadonlyMap<string, readonly number[]>) {}

  async embed(text: string): Promise<Float32Array> {
    const vector = this.vectors.get(text);

    if (vector === undefined) {
      throw new Error(`No scripted embedding for ${text}`);
    }

    return Float32Array.from(vector);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return Promise.all(texts.map((text) => this.embed(text)));
  }
}

function makeAction(overrides: Partial<ActionRecord> = {}): ActionRecord {
  const nowMs = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createActionId(),
    description: overrides.description ?? "Review Atlas rollout",
    actor: overrides.actor ?? "borg",
    audience_entity_id: overrides.audience_entity_id ?? null,
    state: overrides.state ?? "committed_to_do",
    confidence: overrides.confidence ?? 0.8,
    provenance_episode_ids: overrides.provenance_episode_ids ?? [],
    provenance_stream_entry_ids: overrides.provenance_stream_entry_ids ?? [createStreamEntryId()],
    created_at: nowMs,
    updated_at: overrides.updated_at ?? nowMs,
    considering_at: overrides.considering_at ?? null,
    committed_at: overrides.committed_at ?? null,
    scheduled_at: overrides.scheduled_at ?? null,
    completed_at: overrides.completed_at ?? null,
    not_done_at: overrides.not_done_at ?? null,
    unknown_at: overrides.unknown_at ?? null,
  };
}

describe("ActionRepository", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  async function openFixture(embeddingClient?: EmbeddingClient) {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-actions-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const table = await store.openTable({
      name: "action_records",
      schema: createActionRecordsTableSchema(4),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: actionMigrations,
    });
    const repository = new ActionRepository({
      db,
      table,
      embeddingClient,
      clock: new FixedClock(5_000),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    return repository;
  }

  it("adds, gets, updates, and lists action records by state actor and audience", async () => {
    const repository = await openFixture();
    const audienceEntityId = createEntityId();
    const first = makeAction({
      actor: "borg",
      audience_entity_id: audienceEntityId,
      state: "committed_to_do",
    });
    const second = makeAction({
      description: "User reviewed the incident summary",
      actor: "user",
      state: "completed",
      completed_at: 2_000,
      updated_at: 2_000,
    });

    repository.add(first);
    repository.add(second);
    repository.update(first.id, {
      state: "completed",
      confidence: 0.95,
    });

    expect(repository.get(first.id)).toMatchObject({
      description: "Review Atlas rollout",
      state: "completed",
      confidence: 0.95,
      updated_at: 5_000,
      completed_at: 5_000,
    });
    expect(repository.list({ state: "completed" }).map((record) => record.id)).toEqual([
      first.id,
      second.id,
    ]);
    expect(repository.list({ actor: "user" }).map((record) => record.id)).toEqual([second.id]);
    expect(repository.list({ audienceEntityId }).map((record) => record.id as ActionId)).toEqual([
      first.id,
    ]);
    expect(repository.list({ audienceEntityId: null }).map((record) => record.id)).toEqual([
      second.id,
    ]);
  });

  it("finds action records by description vector similarity", async () => {
    const repository = await openFixture(
      new MapEmbeddingClient(
        new Map([
          ["Review Atlas rollout", [1, 0, 0, 0]],
          ["Draft billing follow-up", [0, 1, 0, 0]],
          ["Atlas rollout", [1, 0, 0, 0]],
        ]),
      ),
    );
    const atlas = makeAction({
      description: "Review Atlas rollout",
      state: "completed",
      completed_at: 2_000,
    });
    const billing = makeAction({
      description: "Draft billing follow-up",
      state: "completed",
      completed_at: 3_000,
    });

    repository.add(atlas);
    repository.add(billing);
    await repository.waitForPendingEmbeddings();

    await expect(repository.findByDescription("Atlas rollout", 1)).resolves.toEqual([atlas]);
  });
});
