import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../embeddings/index.js";
import {
  OpenQuestionsRepository,
  createOpenQuestionsTableSchema,
  selfMigrations,
} from "../memory/self/index.js";
import { LanceDbStore } from "../storage/lancedb/index.js";
import { openDatabase } from "../storage/sqlite/index.js";
import { FixedClock } from "../util/clock.js";
import { createEntityId, createSemanticNodeId } from "../util/ids.js";
import { retrieveOpenQuestionsForQuery } from "./open-questions.js";

class ScriptedEmbeddingClient implements EmbeddingClient {
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

class FailingEmbeddingClient implements EmbeddingClient {
  async embed(): Promise<Float32Array> {
    throw new Error("embedding failed");
  }

  async embedBatch(): Promise<Float32Array[]> {
    throw new Error("embedding failed");
  }
}

describe("retrieveOpenQuestionsForQuery", () => {
  const manualProvenance = { kind: "manual" } as const;
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    vi.restoreAllMocks();

    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  async function openFixture(options: {
    embeddingClient: EmbeddingClient;
    onEmbeddingFailure?: ConstructorParameters<
      typeof OpenQuestionsRepository
    >[0]["onEmbeddingFailure"];
  }) {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-open-question-retrieval-"));
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
      embeddingClient: options.embeddingClient,
      clock: new FixedClock(10_000),
      onEmbeddingFailure: options.onEmbeddingFailure,
    });

    cleanup.push(async () => {
      db.close();
      await store.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    return repository;
  }

  it("uses vector similarity as the base score", async () => {
    const query = "lantern probe";
    const relevant = "Which hidden question follows the copper gate?";
    const urgentButDifferent = "Which archive schedule needs care?";
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [query, [1, 0, 0, 0]],
        [relevant, [1, 0, 0, 0]],
        [urgentButDifferent, [0, 1, 0, 0]],
      ]),
    );
    const repository = await openFixture({ embeddingClient });

    repository.add({
      question: relevant,
      urgency: 0.1,
      source: "reflection",
      provenance: manualProvenance,
    });
    repository.add({
      question: urgentButDifferent,
      urgency: 1,
      source: "reflection",
      provenance: manualProvenance,
    });
    await repository.waitForPendingEmbeddings();

    await expect(
      retrieveOpenQuestionsForQuery(repository, embeddingClient, query, {
        limit: 1,
      }),
    ).resolves.toEqual([
      expect.objectContaining({
        question: relevant,
      }),
    ]);
  });

  it("keeps related-node bonus as a retrieval signal", async () => {
    const query = "unrelated probe";
    const semanticNodeId = createSemanticNodeId();
    const related = "Which contradiction should I reconcile?";
    const unrelated = "Which routine should I revisit?";
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [query, [1, 0, 0, 0]],
        [related, [0, 1, 0, 0]],
        [unrelated, [0, 1, 0, 0]],
      ]),
    );
    const repository = await openFixture({ embeddingClient });

    repository.add({
      question: related,
      urgency: 0.1,
      related_semantic_node_ids: [semanticNodeId],
      source: "reflection",
      provenance: manualProvenance,
    });
    repository.add({
      question: unrelated,
      urgency: 1,
      source: "reflection",
      provenance: manualProvenance,
    });
    await repository.waitForPendingEmbeddings();

    const results = await retrieveOpenQuestionsForQuery(repository, embeddingClient, query, {
      relatedSemanticNodeIds: [semanticNodeId],
      limit: 1,
    });

    expect(results[0]?.question).toBe(related);
  });

  it("applies audience visibility while searching vectors", async () => {
    const query = "audience probe";
    const publicQuestion = "Which public deployment question remains?";
    const aliceQuestion = "Which Alice-only deployment question remains?";
    const bobQuestion = "Which Bob deployment question remains?";
    const alice = createEntityId();
    const bob = createEntityId();
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [query, [1, 0, 0, 0]],
        [publicQuestion, [1, 0, 0, 0]],
        [aliceQuestion, [1, 0, 0, 0]],
        [bobQuestion, [1, 0, 0, 0]],
      ]),
    );
    const repository = await openFixture({ embeddingClient });

    const publicRecord = repository.add({
      question: publicQuestion,
      urgency: 0.3,
      source: "reflection",
      provenance: manualProvenance,
    });
    const aliceRecord = repository.add({
      question: aliceQuestion,
      urgency: 1,
      audience_entity_id: alice,
      source: "reflection",
      provenance: manualProvenance,
    });
    const bobRecord = repository.add({
      question: bobQuestion,
      urgency: 0.8,
      audience_entity_id: bob,
      source: "reflection",
      provenance: manualProvenance,
    });
    await repository.waitForPendingEmbeddings();

    const bobResults = await retrieveOpenQuestionsForQuery(repository, embeddingClient, query, {
      audienceEntityId: bob,
      limit: 5,
    });
    const publicResults = await retrieveOpenQuestionsForQuery(repository, embeddingClient, query, {
      audienceEntityId: null,
      limit: 5,
    });

    expect(bobResults.map((question) => question.id)).toContain(publicRecord.id);
    expect(bobResults.map((question) => question.id)).toContain(bobRecord.id);
    expect(bobResults.map((question) => question.id)).not.toContain(aliceRecord.id);
    expect(publicResults.map((question) => question.id)).toEqual([publicRecord.id]);
  });

  it("keeps urgency as a score tilt after similarity", async () => {
    const query = "tilt probe";
    const lowUrgency = "Which low urgency vector question remains?";
    const highUrgency = "Which high urgency vector question remains?";
    const embeddingClient = new ScriptedEmbeddingClient(
      new Map([
        [query, [1, 0, 0, 0]],
        [lowUrgency, [1, 0, 0, 0]],
        [highUrgency, [1, 0, 0, 0]],
      ]),
    );
    const repository = await openFixture({ embeddingClient });

    repository.add({
      question: lowUrgency,
      urgency: 0.1,
      source: "reflection",
      provenance: manualProvenance,
    });
    repository.add({
      question: highUrgency,
      urgency: 1,
      source: "reflection",
      provenance: manualProvenance,
    });
    await repository.waitForPendingEmbeddings();

    const results = await retrieveOpenQuestionsForQuery(repository, embeddingClient, query, {
      limit: 2,
    });

    expect(results.map((question) => question.question)).toEqual([highUrgency, lowUrgency]);
  });

  it("ignores unembedded questions unless relation evidence gives them a score", async () => {
    const query = "atlas fail";
    const questionText = "Why does Atlas fail?";
    const semanticNodeId = createSemanticNodeId();
    const onEmbeddingFailure = vi.fn();
    const repository = await openFixture({
      embeddingClient: new FailingEmbeddingClient(),
      onEmbeddingFailure,
    });
    const queryEmbeddingClient = new ScriptedEmbeddingClient(new Map([[query, [1, 0, 0, 0]]]));
    const question = repository.add({
      question: questionText,
      urgency: 0.2,
      source: "reflection",
      related_semantic_node_ids: [semanticNodeId],
      provenance: manualProvenance,
    });
    await repository.waitForPendingEmbeddings();

    const unrelatedResults = await retrieveOpenQuestionsForQuery(
      repository,
      queryEmbeddingClient,
      query,
      {
        limit: 1,
      },
    );
    const relatedResults = await retrieveOpenQuestionsForQuery(
      repository,
      queryEmbeddingClient,
      query,
      {
        relatedSemanticNodeIds: [semanticNodeId],
        limit: 1,
      },
    );

    expect(onEmbeddingFailure).toHaveBeenCalledWith(
      expect.any(Error),
      expect.objectContaining({
        operation: "insert",
        questionId: question.id,
      }),
    );
    expect(unrelatedResults).toEqual([]);
    expect(relatedResults[0]?.id).toBe(question.id);
  });
});
