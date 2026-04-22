import { describe, expect, it } from "vitest";

import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { ProvenanceError } from "../../util/errors.js";
import { createEpisodeId, createSemanticNodeId } from "../../util/ids.js";

import { selfMigrations } from "./migrations.js";
import { OpenQuestionsRepository } from "./open-questions.js";

describe("OpenQuestionsRepository", () => {
  const manualProvenance = { kind: "manual" } as const;

  it("dedupes by normalized question and related ids", () => {
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
      question: "What is   atlas doing",
      urgency: 0.9,
      related_episode_ids: [episodeId],
      related_semantic_node_ids: [semanticNodeId],
      source: "user",
    });

    expect(duplicate.id).toBe(first.id);

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
