import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

import type { EmbeddingClient } from "../../embeddings/index.js";
import { LanceDbStore } from "../../storage/lancedb/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock, ManualClock } from "../../util/clock.js";
import { IdentityCasMismatchError, ProvenanceError } from "../../util/errors.js";
import {
  createEntityId,
  createMaintenanceRunId,
  createSharedStateEntryId,
  createEpisodeId,
  createSemanticNodeId,
  createStreamEntryId,
} from "../../util/ids.js";
import { expectedRecordVersion } from "../common/cas.js";

import { selfMigrations } from "./migrations.js";
import {
  OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT,
  buildOpenQuestionDuplicatePresentation,
} from "./open-question-duplicates.js";
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
      resolution_evidence_episode_ids: [episodeId],
      resolution_evidence_stream_entry_ids: [],
      resolution_note: "Atlas completed the rollout.",
    });
    const bumped = repository.bumpUrgency(first.id, -0.2);

    expect(touched.last_touched).toBe(12_000);
    expect(resolved.status).toBe("resolved");
    expect(bumped.urgency).toBeLessThanOrEqual(1);

    db.close();
  });

  it("filters listed questions by source", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const contradiction = repository.add({
        question: "Which route claim contradicts the current plan?",
        urgency: 0.8,
        source: "contradiction",
        provenance: manualProvenance,
      });
      repository.add({
        question: "What should be reflected later?",
        urgency: 0.9,
        source: "reflection",
        provenance: manualProvenance,
      });

      expect(repository.list({ status: "open", source: "contradiction" })).toEqual([contradiction]);
    } finally {
      db.close();
    }
  });

  it("appends and lists recent rumination notes", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const evidenceEpisodeId = createEpisodeId();
      const connected = repository.add({
        question: "Which related uncertainty moved?",
        urgency: 0.3,
        source: "reflection",
        provenance: manualProvenance,
      });
      const question = repository.add({
        question: "What remains unsettled?",
        urgency: 0.6,
        source: "reflection",
        provenance: manualProvenance,
      });
      repository.recordRumination({
        open_question_id: question.id,
        note: "The first pass kept the question open.",
        tensions: ["One line of evidence moved; another resisted."],
        connected_open_question_ids: [connected.id],
        evidence_episode_ids: [evidenceEpisodeId],
        source_process: "test",
        provenance: manualProvenance,
        created_at: 11_000,
      });
      repository.recordRumination({
        open_question_id: question.id,
        note: "The second pass narrowed the live tension.",
        tensions: ["The remaining tension is narrower."],
        connected_open_question_ids: [connected.id],
        source_process: "test",
        provenance: manualProvenance,
        created_at: 12_000,
      });

      expect(repository.listRecentRuminations(question.id, { limit: 1 })).toEqual([
        expect.objectContaining({
          note: "The second pass narrowed the live tension.",
          tensions: ["The remaining tension is narrower."],
          connected_open_question_ids: [connected.id],
          evidence_episode_ids: [],
        }),
      ]);
      expect(
        repository.listRecentRuminations(question.id, { limit: 5 }).map((item) => item.note),
      ).toEqual([
        "The second pass narrowed the live tension.",
        "The first pass kept the question open.",
      ]);
    } finally {
      db.close();
    }
  });

  // Four consecutive resolutions read back with last_ruminated_at null, which is
  // readable as "nothing ever ruminated on these". It is not: both terminal writes
  // clear the active-open lifecycle in the same statement that sets the status, so
  // the null is that write's own doing and carries no history. The notes are the
  // record that survives it, so pin both halves in one place.
  it("clears the rumination lifecycle on terminal writes while the notes survive", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const resolved = repository.add({
        question: "Which pass settled this?",
        urgency: 0.5,
        source: "ruminator",
        provenance: manualProvenance,
      });
      const abandoned = repository.add({
        question: "Which pass gave up on this?",
        urgency: 0.5,
        source: "ruminator",
        provenance: manualProvenance,
      });

      for (const question of [resolved, abandoned]) {
        repository.recordRumination({
          open_question_id: question.id,
          note: "A pass ran against this question.",
          tensions: [],
          connected_open_question_ids: [],
          source_process: "test",
          provenance: manualProvenance,
          created_at: 11_000,
        });
        expect(repository.markRuminated(question.id, 3)).toMatchObject({
          unresolved_rumination_ticks: 3,
          last_ruminated_at: 10_000,
        });
      }

      const afterResolve = repository.resolve(resolved.id, {
        resolution_note: "The evidence settled it.",
        resolution_evidence_episode_ids: [createEpisodeId()],
      });
      const afterAbandon = repository.abandon(abandoned.id, "No traction.");

      for (const record of [afterResolve, afterAbandon]) {
        expect(record.unresolved_rumination_ticks).toBe(0);
        expect(record.last_ruminated_at).toBeNull();
      }

      for (const question of [resolved, abandoned]) {
        expect(
          repository.listRecentRuminations(question.id, { limit: 5 }).map((item) => item.note),
        ).toEqual(["A pass ran against this question."]);
      }
    } finally {
      db.close();
    }
  });

  it("CAS-protects open question deletion", async () => {
    const clock = new FixedClock(10_000);
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock,
    });

    try {
      const question = repository.add({
        question: "Which duplicate should survive?",
        urgency: 0.4,
        source: "ruminator",
        provenance: manualProvenance,
      });
      const staleVersion = expectedRecordVersion(question);
      const concurrent = repository.update(
        question.id,
        {
          urgency: 0.8,
        },
        {
          expectedVersion: staleVersion,
        },
      );

      await expect(
        repository.delete(question.id, {
          expectedVersion: staleVersion,
        }),
      ).rejects.toThrow(IdentityCasMismatchError);
      expect(repository.get(question.id)).toMatchObject({
        urgency: 0.8,
        record_version: concurrent.record_version,
      });
    } finally {
      db.close();
    }
  });

  it("clears artifact resolution back-reference when reopening for reversal", () => {
    const clock = new FixedClock(10_000);
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock,
    });

    try {
      const artifactEntryId = createSharedStateEntryId();
      const streamEntryId = createStreamEntryId();
      const question = repository.add({
        question: "Is Granada locked?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });
      repository.resolve(
        question.id,
        {
          resolution_evidence_stream_entry_ids: [streamEntryId],
          resolution_note: "Resolved by artifact.",
        },
        {
          resolvedByArtifactEntryId: artifactEntryId,
        },
      );

      const reopened = repository.reopenForReversal(question.id, 0.8);

      expect(reopened).toMatchObject({
        status: "open",
        urgency: 0.8,
        resolved_by_artifact_entry_id: null,
        resolution_evidence_stream_entry_ids: [],
        resolution_note: null,
        resolved_at: null,
      });
      expect(repository.get(question.id)?.resolved_by_artifact_entry_id).toBeNull();
    } finally {
      db.close();
    }
  });

  it("CAS-protects rumination marker updates", () => {
    const clock = new FixedClock(10_000);
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock,
    });

    try {
      const question = repository.add({
        question: "Which rumination update is stale?",
        urgency: 0.4,
        source: "ruminator",
        provenance: manualProvenance,
      });
      const staleVersion = expectedRecordVersion(question);
      const concurrent = repository.bumpUrgency(question.id, 0.1, {
        expectedVersion: staleVersion,
      });

      expect(() =>
        repository.markRuminated(question.id, 1, {
          expectedVersion: staleVersion,
        }),
      ).toThrow(IdentityCasMismatchError);
      expect(repository.get(question.id)).toMatchObject({
        record_version: concurrent.record_version,
        unresolved_rumination_ticks: 0,
      });
    } finally {
      db.close();
    }
  });

  it("stamps reached runs but only increments ticks when it records a note", () => {
    const clock = new ManualClock(10_000);
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock,
    });

    try {
      const question = repository.add({
        question: "Which stamped rumination produced a durable note?",
        urgency: 0.4,
        source: "ruminator",
        provenance: manualProvenance,
      });
      const noteLessRunId = createMaintenanceRunId();
      clock.advance(1_000);

      const noteLess = repository.stampRuminationForRun({
        open_question_id: question.id,
        source_run_id: noteLessRunId,
        next_unresolved_rumination_ticks: 1,
        rumination: null,
      });

      expect(noteLess).toMatchObject({
        stamped: true,
        rumination: null,
        question: {
          unresolved_rumination_ticks: 0,
          last_ruminated_at: 11_000,
        },
      });
      expect(
        db
          .prepare(
            `
              SELECT stamped_at
              FROM open_question_rumination_stamps
              WHERE open_question_id = ? AND source_run_id = ?
            `,
          )
          .get(question.id, noteLessRunId),
      ).toEqual({ stamped_at: 11_000 });

      const notedRunId = createMaintenanceRunId();
      clock.advance(1_000);
      const noted = repository.stampRuminationForRun({
        open_question_id: question.id,
        source_run_id: notedRunId,
        next_unresolved_rumination_ticks: 1,
        rumination: {
          note: "This run produced a durable rumination note.",
          source_process: "test",
          provenance: manualProvenance,
        },
      });

      expect(noted).toMatchObject({
        stamped: true,
        rumination: {
          note: "This run produced a durable rumination note.",
          source_run_id: notedRunId,
        },
        question: {
          unresolved_rumination_ticks: 1,
          last_ruminated_at: 12_000,
        },
      });

      clock.advance(1_000);
      const replay = repository.stampRuminationForRun({
        open_question_id: question.id,
        source_run_id: notedRunId,
        next_unresolved_rumination_ticks: 2,
        rumination: {
          note: "This replay must not produce a second note.",
          source_process: "test",
          provenance: manualProvenance,
        },
      });

      expect(replay).toMatchObject({
        stamped: false,
        rumination: null,
        question: {
          unresolved_rumination_ticks: 1,
          last_ruminated_at: 12_000,
        },
      });
      expect(repository.listRecentRuminations(question.id)).toEqual([
        expect.objectContaining({
          note: "This run produced a durable rumination note.",
          source_run_id: notedRunId,
        }),
      ]);
      expect(
        db
          .prepare(
            "SELECT COUNT(*) AS count FROM open_question_rumination_stamps WHERE open_question_id = ?",
          )
          .get(question.id),
      ).toEqual({ count: 2 });
    } finally {
      db.close();
    }
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
      resolution_evidence_episode_ids: [episodeId],
      resolution_evidence_stream_entry_ids: [],
      resolution_note: "Atlas stabilized after the rollback rehearsal.",
    });
    repository.abandon(abandonedQuestion.id, "No longer relevant");

    expect(() =>
      repository.resolve(resolvedQuestion.id, {
        resolution_evidence_episode_ids: [episodeId],
        resolution_evidence_stream_entry_ids: [],
        resolution_note: "Second resolution.",
      }),
    ).toThrow(/OPEN_QUESTION_INVALID_TRANSITION|Cannot resolve/);
    expect(() => repository.abandon(resolvedQuestion.id, "Too late")).toThrow(
      /OPEN_QUESTION_INVALID_TRANSITION|Cannot abandon/,
    );
    expect(() =>
      repository.resolve(abandonedQuestion.id, {
        resolution_evidence_episode_ids: [episodeId],
        resolution_evidence_stream_entry_ids: [],
        resolution_note: "Too late.",
      }),
    ).toThrow(/OPEN_QUESTION_INVALID_TRANSITION|Cannot resolve/);
    expect(() => repository.abandon(abandonedQuestion.id, "Still stale")).toThrow(
      /OPEN_QUESTION_INVALID_TRANSITION|Cannot abandon/,
    );

    db.close();
  });

  it("stores stream-evidence-only resolutions and rejects source-less resolutions", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const streamEntryId = createStreamEntryId();

    try {
      const question = repository.add({
        question: "What did the current turn settle?",
        urgency: 0.5,
        source: "reflection",
        provenance: manualProvenance,
      });
      const sourceLessQuestion = repository.add({
        question: "What still has no evidence?",
        urgency: 0.4,
        source: "reflection",
        provenance: manualProvenance,
      });
      const resolved = repository.resolve(question.id, {
        resolution_evidence_episode_ids: [],
        resolution_evidence_stream_entry_ids: [streamEntryId],
        resolution_note: "The current turn supplied the answer.",
      });

      expect(resolved).toMatchObject({
        status: "resolved",
        resolution_evidence_episode_ids: [],
        resolution_evidence_stream_entry_ids: [streamEntryId],
        resolution_note: "The current turn supplied the answer.",
      });
      expect(repository.get(question.id)?.resolution_evidence_stream_entry_ids).toEqual([
        streamEntryId,
      ]);
      expect(() =>
        repository.resolve(sourceLessQuestion.id, {
          resolution_evidence_episode_ids: [],
          resolution_evidence_stream_entry_ids: [],
          resolution_note: "No evidence.",
        }),
      ).toThrow(/OPEN_QUESTION_RESOLUTION_EVIDENCE_REQUIRED|requires episode or stream evidence/);
    } finally {
      db.close();
    }
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

  // Load-bearing downstream: the evidence ledger feeds these rows into the open_questions
  // section in the order they arrive, and compaction labels whatever it drops
  // "lower-priority" (evidence-ledger/compaction.ts). That label is only true because the
  // handle lookup sorts by urgency here. If this ordering changes, the breadcrumb the model
  // reads becomes false without anything else failing -- so pin it.
  it("returns handle matches ordered by urgency, then recency of contact", () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(10_000),
    });
    const episodeId = createEpisodeId();
    const add = (question: string, urgency: number) =>
      repository.add({
        question,
        urgency,
        related_episode_ids: [episodeId],
        related_semantic_node_ids: [createSemanticNodeId()],
        source: "user",
      });

    const low = add("What did the quiet thread settle?", 0.2);
    const high = add("What did the loud thread settle?", 0.9);
    const touchedTie = add("What did the first even thread settle?", 0.5);
    const untouchedTie = add("What did the second even thread settle?", 0.5);

    repository.touch(touchedTie.id, 12_000);

    try {
      expect(
        repository
          .findByHandles({ streamEntryIds: [], episodeIds: [episodeId] })
          .map((question) => question.id),
      ).toEqual([high.id, touchedTie.id, untouchedTie.id, low.id]);
    } finally {
      db.close();
    }
  });

  it("retrieves cosine candidates globally across audience scopes", async () => {
    const firstText = "Which private uncertainty belongs to the first audience?";
    const secondText = "¿Qué incertidumbre equivalente pertenece a la segunda audiencia?";
    const { repository } = await openVectorFixture(
      new MapEmbeddingClient(
        new Map([
          [firstText, [1, 0, 0, 0]],
          [secondText, [1, 0, 0, 0]],
        ]),
      ),
    );
    const first = repository.add({
      question: firstText,
      urgency: 0.4,
      audience_entity_id: createEntityId(),
      source: "reflection",
      provenance: manualProvenance,
    });
    const second = repository.add({
      question: secondText,
      urgency: 0.5,
      audience_entity_id: createEntityId(),
      source: "reflection",
      provenance: manualProvenance,
    });
    await repository.waitForPendingEmbeddings();

    expect((await repository.searchSimilar(first, { minSimilarity: 0.9 }))[0]).toMatchObject({
      question: { id: second.id },
      similarity: 1,
    });
  });

  it("uses the complete open set until the presentation limit, then marks proxy fallback", async () => {
    const db = openDatabase(":memory:", {
      migrations: selfMigrations,
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(10_000),
    });

    try {
      const questions = Array.from(
        { length: OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT },
        (_, index) =>
          repository.add({
            question: `Open uncertainty ${index}?`,
            urgency: 0.5,
            source: "reflection",
            provenance: manualProvenance,
          }),
      );
      const complete = await buildOpenQuestionDuplicatePresentation({
        repository,
        sourceTextProxy: "complete-set proxy",
      });

      expect(complete).toMatchObject({
        complete: true,
        total_open_questions: OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT,
        presented_count: OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT,
        omitted_count: 0,
      });

      const overflow = repository.add({
        question: "Overflow uncertainty?",
        urgency: 0.5,
        source: "reflection",
        provenance: manualProvenance,
      });
      const nearest = [...questions.slice(1), overflow];
      const searchByText = vi.spyOn(repository, "searchByText").mockResolvedValue(
        nearest.map((question, index) => ({
          question,
          similarity: 1 - index / 1_000,
        })),
      );
      const incomplete = await buildOpenQuestionDuplicatePresentation({
        repository,
        sourceTextProxy: "overflow proxy",
      });

      expect(searchByText).toHaveBeenCalledWith("overflow proxy", {
        status: "open",
        limit: OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT,
      });
      expect(incomplete).toMatchObject({
        complete: false,
        total_open_questions: OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT + 1,
        presented_count: OPEN_QUESTION_DUPLICATE_PRESENTATION_LIMIT,
        omitted_count: 1,
      });
      expect(incomplete.rows.map((row) => row.id)).toEqual(nearest.map((question) => question.id));
    } finally {
      db.close();
    }
  });
});
