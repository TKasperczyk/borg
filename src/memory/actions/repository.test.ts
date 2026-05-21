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
  createSharedStateEntryId,
  createEntityId,
  createStreamEntryId,
  type ActionId,
} from "../../util/ids.js";
import { actionMigrations } from "./migrations.js";
import { ActionRepository, createActionRecordsTableSchema } from "./repository.js";
import type { ActionRecord } from "./types.js";

class MapEmbeddingClient implements EmbeddingClient {
  readonly embedTexts: string[] = [];
  readonly embedBatchTexts: string[][] = [];

  constructor(private readonly vectors: ReadonlyMap<string, readonly number[]>) {}

  async embed(text: string): Promise<Float32Array> {
    this.embedTexts.push(text);
    const vector = this.vectors.get(text);

    if (vector === undefined) {
      throw new Error(`No scripted embedding for ${text}`);
    }

    return Float32Array.from(vector);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    this.embedBatchTexts.push([...texts]);
    return Promise.all(texts.map((text) => this.embed(text)));
  }

  clearCalls(): void {
    this.embedTexts.length = 0;
    this.embedBatchTexts.length = 0;
  }
}

function makeAction(overrides: Partial<ActionRecord> = {}): ActionRecord {
  const nowMs = overrides.created_at ?? 1_000;

  return {
    id: overrides.id ?? createActionId(),
    description: overrides.description ?? "Review Atlas rollout",
    actor: overrides.actor ?? "borg",
    audience_entity_id: overrides.audience_entity_id ?? null,
    goal_id: overrides.goal_id ?? null,
    open_question_id: overrides.open_question_id ?? null,
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
    expired_at: overrides.expired_at ?? null,
    archived_at: overrides.archived_at ?? null,
    unknown_at: overrides.unknown_at ?? null,
    canonicalized_by_artifact_entry_id: overrides.canonicalized_by_artifact_entry_id ?? null,
    session_scope: overrides.session_scope ?? null,
    session_anchor_id: overrides.session_anchor_id ?? null,
    last_referenced_at_ms: overrides.last_referenced_at_ms ?? nowMs,
    last_referenced_turn_counter: overrides.last_referenced_turn_counter ?? null,
    last_referenced_turn_global: overrides.last_referenced_turn_global ?? null,
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

  it("migrates legacy session-local reference counters to a shared migration watermark", () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-actions-migration-"));
    const dbPath = join(tempDir, "borg.db");
    cleanup.push(async () => {
      rmSync(tempDir, { recursive: true, force: true });
    });

    const legacyMigrations = [
      {
        id: 1,
        name: "legacy-action-records",
        up: `
          CREATE TABLE action_records (
            id TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            actor TEXT NOT NULL,
            audience_entity_id TEXT NULL,
            goal_id TEXT NULL,
            open_question_id TEXT NULL,
            state TEXT NOT NULL,
            confidence REAL NOT NULL,
            provenance_episode_ids TEXT NOT NULL,
            provenance_stream_entry_ids TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            considering_at INTEGER NULL,
            committed_at INTEGER NULL,
            scheduled_at INTEGER NULL,
            completed_at INTEGER NULL,
            not_done_at INTEGER NULL,
            expired_at INTEGER NULL,
            archived_at INTEGER NULL,
            unknown_at INTEGER NULL,
            canonicalized_by_artifact_entry_id TEXT NULL,
            session_scope TEXT NULL,
            session_anchor_id TEXT NULL,
            last_referenced_at_ms INTEGER NULL,
            last_referenced_turn_counter INTEGER NULL
          )
        `,
      },
      ...[2, 3, 4, 5].map((id) => ({
        id,
        name: `legacy-noop-${id}`,
        up: () => undefined,
      })),
    ];
    const oldActionId = createActionId();
    const recentActionId = createActionId();
    const missingReferenceActionId = createActionId();
    let db = openDatabase(dbPath, { migrations: legacyMigrations });

    const insertLegacy = db.prepare(`
      INSERT INTO action_records (
        id, description, actor, audience_entity_id, goal_id, open_question_id, state, confidence,
        provenance_episode_ids, provenance_stream_entry_ids, created_at, updated_at,
        considering_at, committed_at, scheduled_at, completed_at, not_done_at, expired_at,
        archived_at, unknown_at, canonicalized_by_artifact_entry_id, session_scope,
        session_anchor_id, last_referenced_at_ms, last_referenced_turn_counter
      ) VALUES (?, ?, 'user', NULL, NULL, NULL, 'committed_to_do', 0.8, '[]', ?, 1000, 1000,
        NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 1000, ?)
    `);
    insertLegacy.run(
      oldActionId,
      "Legacy early-session action",
      JSON.stringify([createStreamEntryId()]),
      30,
    );
    insertLegacy.run(
      recentActionId,
      "Legacy later-session action",
      JSON.stringify([createStreamEntryId()]),
      3,
    );
    insertLegacy.run(
      missingReferenceActionId,
      "Legacy unreferenced action",
      JSON.stringify([createStreamEntryId()]),
      null,
    );
    db.close();

    db = openDatabase(dbPath, { migrations: actionMigrations });
    try {
      const rows = db
        .prepare(
          `
            SELECT id, last_referenced_turn_counter, last_referenced_turn_global
            FROM action_records
          `,
        )
        .all() as Array<{
        id: ActionId;
        last_referenced_turn_counter: number | null;
        last_referenced_turn_global: number | null;
      }>;
      const rowById = new Map(rows.map((row) => [row.id, row]));
      const counter = db
        .prepare("SELECT value FROM action_lifecycle_turn_counter WHERE id = 'global'")
        .get() as { value: number };

      expect(rowById.get(oldActionId)).toMatchObject({
        last_referenced_turn_counter: 30,
        last_referenced_turn_global: 30,
      });
      expect(rowById.get(recentActionId)).toMatchObject({
        last_referenced_turn_counter: 3,
        last_referenced_turn_global: 30,
      });
      expect(rowById.get(missingReferenceActionId)).toMatchObject({
        last_referenced_turn_counter: null,
        last_referenced_turn_global: null,
      });
      expect(counter.value).toBe(30);
    } finally {
      db.close();
    }
  });

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

  it("counts active and canonicalized action records", async () => {
    const repository = await openFixture();

    repository.add(
      makeAction({
        state: "considering",
        considering_at: 1_000,
      }),
      { creationSource: "extractor" },
    );
    repository.add(
      makeAction({
        description: "Schedule Atlas rollout",
        state: "scheduled",
        scheduled_at: 1_000,
      }),
      { creationSource: "api" },
    );
    repository.add(
      makeAction({
        description: "Complete Atlas rollout",
        state: "completed",
        completed_at: 1_000,
        canonicalized_by_artifact_entry_id: createSharedStateEntryId(),
      }),
      { creationSource: "reflector" },
    );
    repository.add(
      makeAction({
        description: "Unknown Atlas action",
        state: "unknown",
        unknown_at: 1_000,
      }),
    );

    expect(repository.countActive()).toBe(3);
    expect(repository.countCanonicalized()).toBe(1);
    expect(repository.getCreationCountsBySource()).toEqual({
      extractor: 1,
      reflector: 1,
      api: 1,
      unknown: 1,
    });
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

  it("finds description similarity pairs above threshold with one embedding batch", async () => {
    const repository = await openFixture(
      new MapEmbeddingClient(
        new Map([
          ["Review Atlas rollout", [1, 0, 0, 0]],
          ["Check Atlas deployment", [0.9, 0.1, 0, 0]],
          ["Draft billing follow-up", [0, 1, 0, 0]],
        ]),
      ),
    );
    const review = makeAction({
      description: "Review Atlas rollout",
    });
    const check = makeAction({
      description: "Check Atlas deployment",
    });
    const billing = makeAction({
      description: "Draft billing follow-up",
    });

    await expect(
      repository.findSimilarDescriptionPairs([review, check, billing], 0.85),
    ).resolves.toEqual([
      {
        leftId: review.id,
        rightId: check.id,
        similarity: expect.any(Number),
      },
    ]);
  });

  it("uses stored action embeddings for description similarity pairs", async () => {
    const embeddingClient = new MapEmbeddingClient(
      new Map([
        ["Review Atlas rollout", [1, 0, 0, 0]],
        ["Check Atlas deployment", [0.9, 0.1, 0, 0]],
        ["Draft billing follow-up", [0, 1, 0, 0]],
      ]),
    );
    const repository = await openFixture(embeddingClient);
    const review = makeAction({
      description: "Review Atlas rollout",
    });
    const check = makeAction({
      description: "Check Atlas deployment",
    });
    const billing = makeAction({
      description: "Draft billing follow-up",
    });

    repository.add(review);
    repository.add(check);
    repository.add(billing);
    await repository.waitForPendingEmbeddings();
    embeddingClient.clearCalls();

    await expect(
      repository.findSimilarDescriptionPairs([review, check, billing], 0.85),
    ).resolves.toEqual([
      {
        leftId: review.id,
        rightId: check.id,
        similarity: expect.any(Number),
      },
    ]);
    expect(embeddingClient.embedBatchTexts).toEqual([]);
  });
});
