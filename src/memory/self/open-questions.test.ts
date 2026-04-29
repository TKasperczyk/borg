import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { ProvenanceError } from "../../util/errors.js";
import { createEntityId, createEpisodeId, createSemanticNodeId } from "../../util/ids.js";

import { selfMigrations } from "./migrations.js";
import { OpenQuestionsRepository, createOpenQuestionsTableSchema } from "./open-questions.js";

class MapEmbeddingClient implements EmbeddingClient {
  readonly calls: string[] = [];

  constructor(private readonly vectors: ReadonlyMap<string, readonly number[]>) {}

  async embed(text: string): Promise<Float32Array> {
    this.calls.push(text);
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

describe("OpenQuestionsRepository", () => {
  const manualProvenance = { kind: "manual" } as const;
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  async function openVectorFixture(embeddingClient: EmbeddingClient) {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-open-questions-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const table = await store.openTable({
      name: "open_questions",
      schema: createOpenQuestionsTableSchema(4),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      table,
      embeddingClient,
      clock: new FixedClock(10_000),
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    return {
      repository,
      table,
    };
  }

  it("dedupes by normalized full question text and related ids", () => {
    const clock = new FixedClock(10_000);
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock,
    });
    const episodeId = createEpisodeId();
    const semanticNodeId = createSemanticNodeId();
    const first = repository.add({
      question: "What is Atlas doing?",
      urgency: 0.4,
      related_episode_ids: [episodeId],
      related_semantic_node_ids: [semanticNodeId],
      source: "user",
    });
    const duplicate = repository.add({
      question: "What is   atlas doing?",
      urgency: 0.9,
      related_episode_ids: [episodeId],
      related_semantic_node_ids: [semanticNodeId],
      source: "user",
    });
    const differentPunctuation = repository.add({
      question: "What is atlas doing",
      urgency: 0.8,
      related_episode_ids: [episodeId],
      related_semantic_node_ids: [semanticNodeId],
      source: "user",
    });

    expect(duplicate.id).toBe(first.id);
    expect(differentPunctuation.id).not.toBe(first.id);
    expect(
      repository
        .list({ limit: 10 })
        .every((question) => question.id === first.id || question.id === differentPunctuation.id),
    ).toBe(true);

    const touched = repository.touch(first.id, 12_000);
    const resolved = repository.resolve(first.id, {
      resolution_episode_id: episodeId,
      resolution_note: "Atlas completed the rollout.",
    });
    const bumped = repository.bumpUrgency(first.id, -0.2);

    expect(touched.last_touched).toBe(12_000);
    expect(resolved.status).toBe("resolved");
    expect(bumped.urgency).toBeLessThanOrEqual(1);

    db.close();
  });

  it("embeds inserted questions into the vector table", async () => {
    const questionText = "Which Atlas deployment failure still needs an answer?";
    const embeddingClient = new MapEmbeddingClient(new Map([[questionText, [1, 0, 0, 0]]]));
    const { repository, table } = await openVectorFixture(embeddingClient);

    const question = repository.add({
      question: questionText,
      urgency: 0.4,
      source: "reflection",
      provenance: manualProvenance,
    });
    await repository.waitForPendingEmbeddings();

    const rows = await table.list({ limit: 10 });

    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      id: question.id,
      question: questionText,
      status: "open",
    });
    expect(Array.from(rows[0]?.embedding as ArrayLike<number>)).toEqual([1, 0, 0, 0]);
  });

  it("backfills missing embeddings idempotently", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-open-questions-"));
    const store = new LanceDbStore({
      uri: join(tempDir, "lancedb"),
    });
    const table = await store.openTable({
      name: "open_questions",
      schema: createOpenQuestionsTableSchema(4),
    });
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: selfMigrations,
    });
    const questionText = "Which backfill question needs a vector?";
    const embeddingClient = new MapEmbeddingClient(new Map([[questionText, [0, 1, 0, 0]]]));

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const sqliteOnlyRepository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(10_000),
    });
    sqliteOnlyRepository.add({
      question: questionText,
      urgency: 0.3,
      source: "reflection",
      provenance: manualProvenance,
    });

    const vectorRepository = new OpenQuestionsRepository({
      db,
      table,
      embeddingClient,
      clock: new FixedClock(10_000),
    });
    const first = await vectorRepository.backfillMissingEmbeddings();
    const second = await vectorRepository.backfillMissingEmbeddings();

    expect(first).toEqual({
      scanned: 1,
      embedded: 1,
      skipped: 0,
      failed: 0,
    });
    expect(second).toEqual({
      scanned: 1,
      embedded: 0,
      skipped: 1,
      failed: 0,
    });
    expect(embeddingClient.calls).toEqual([questionText]);
    expect(await table.list({ limit: 10 })).toHaveLength(1);
  });

  it("validates duplicate adds before dedupe short-circuiting", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const episodeId = createEpisodeId();

    try {
      repository.add({
        question: "Why is Atlas failing?",
        urgency: 0.4,
        related_episode_ids: [episodeId],
        source: "user",
      });

      expect(() =>
        repository.add({
          question: "Why is atlas failing",
          urgency: 0.9,
          related_episode_ids: [episodeId],
          provenance: {
            kind: "episodes",
            episode_ids: [],
          },
          source: "user",
        }),
      ).toThrow();
    } finally {
      db.close();
    }
  });

  it("rejects invalid resolve and abandon transitions", () => {
    const clock = new FixedClock(10_000);
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock,
    });
    const episodeId = createEpisodeId();
    const resolvedQuestion = repository.add({
      question: "How did Atlas stabilize?",
      urgency: 0.5,
      source: "user",
      provenance: manualProvenance,
    });
    const abandonedQuestion = repository.add({
      question: "Should I revisit old Borealis notes?",
      urgency: 0.3,
      source: "reflection",
      provenance: manualProvenance,
    });

    repository.resolve(resolvedQuestion.id, {
      resolution_episode_id: episodeId,
      resolution_note: "Atlas stabilized after the rollback rehearsal.",
    });
    repository.abandon(abandonedQuestion.id, "No longer relevant");

    expect(() =>
      repository.resolve(resolvedQuestion.id, {
        resolution_episode_id: episodeId,
      }),
    ).toThrow(/OPEN_QUESTION_INVALID_TRANSITION|Cannot resolve/);
    expect(() => repository.abandon(resolvedQuestion.id, "Too late")).toThrow(
      /OPEN_QUESTION_INVALID_TRANSITION|Cannot abandon/,
    );
    expect(() =>
      repository.resolve(abandonedQuestion.id, {
        resolution_episode_id: episodeId,
      }),
    ).toThrow(/OPEN_QUESTION_INVALID_TRANSITION|Cannot resolve/);
    expect(() => repository.abandon(abandonedQuestion.id, "Still stale")).toThrow(
      /OPEN_QUESTION_INVALID_TRANSITION|Cannot abandon/,
    );

    db.close();
  });

  it("uses the indexed dedupe key beyond the old in-memory scan window", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
    });
    const relatedEpisodeIds = Array.from({ length: 1_000 }, () => createEpisodeId());

    const inserted = Array.from({ length: 1_000 }, (_, index) =>
      repository.add({
        question: `How does Atlas question ${index} resolve?`,
        urgency: 0.2,
        related_episode_ids: [relatedEpisodeIds[index]!],
        source: "user",
      }),
    );
    const duplicate = repository.add({
      question: "How does Atlas question 999 resolve?",
      urgency: 0.9,
      related_episode_ids: [relatedEpisodeIds[999]!],
      source: "user",
    });

    expect(repository.list({ limit: 1_100 })).toHaveLength(1_000);
    expect(duplicate.id).toBe(inserted[999]?.id);

    db.close();
  });

  it("preserves non-ASCII question content in v2 dedupe normalization", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
    });

    try {
      const first = repository.add({
        question: "Ａｔｌａｓ 的部署为什么失败？",
        urgency: 0.4,
        source: "user",
        provenance: manualProvenance,
      });
      const duplicate = repository.add({
        question: "atlas 的部署为什么失败？",
        urgency: 0.9,
        source: "user",
        provenance: manualProvenance,
      });

      expect(duplicate.id).toBe(first.id);
      expect(repository.list({ limit: 10 })).toHaveLength(1);
    } finally {
      db.close();
    }
  });

  it("stores audience scope and dedupes private questions separately", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
    });
    const alice = createEntityId();
    const bob = createEntityId();

    try {
      const aliceQuestion = repository.add({
        question: "What should I remember about Atlas?",
        urgency: 0.4,
        audience_entity_id: alice,
        source: "reflection",
        provenance: manualProvenance,
      });
      const aliceDuplicate = repository.add({
        question: "What should I remember about atlas?",
        urgency: 0.9,
        audience_entity_id: alice,
        source: "reflection",
        provenance: manualProvenance,
      });
      const bobQuestion = repository.add({
        question: "What should I remember about atlas",
        urgency: 0.6,
        audience_entity_id: bob,
        source: "reflection",
        provenance: manualProvenance,
      });
      const publicQuestion = repository.add({
        question: "What public Atlas detail matters?",
        urgency: 0.8,
        source: "reflection",
        provenance: manualProvenance,
      });

      expect(aliceDuplicate.id).toBe(aliceQuestion.id);
      expect(bobQuestion.id).not.toBe(aliceQuestion.id);
      expect(repository.get(aliceQuestion.id)?.audience_entity_id).toBe(alice);
      expect(
        repository.list({ visibleToAudienceEntityId: bob, limit: 10 }).map((item) => item.id),
      ).toEqual([publicQuestion.id, bobQuestion.id]);
      expect(
        repository.list({ visibleToAudienceEntityId: null, limit: 10 }).map((item) => item.id),
      ).toEqual([publicQuestion.id]);
    } finally {
      db.close();
    }
  });

  it("rejects questions without evidence or explicit provenance", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
    });

    try {
      expect(() =>
        repository.add({
          question: "What do I believe here?",
          urgency: 0.5,
          source: "user",
        }),
      ).toThrow(ProvenanceError);
    } finally {
      db.close();
    }
  });
});
