import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StreamWriter } from "../../stream/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { createSemanticNodeId, createSemanticEdgeId } from "../../util/ids.js";
import { REVIEW_QUEUE_BASELINE_SQL } from "../review-queue/migrations.js";
import { ReviewQueueRepository } from "../review-queue/review-queue.js";
import { OpenQuestionsRepository } from "./open-questions.js";
import { selfMigrations } from "./migrations.js";
import {
  enqueueOpenQuestionForReview,
  appendInternalFailureEvent,
  appendOpenQuestionHookFailureEvent,
} from "./review-open-question-hook.js";

it.each([
  "contradiction",
  "misattribution",
  "identity_inconsistency",
  "commitment_reconciliation",
] as const)(
  "stores the integer review id for a new %s question, independently of filtered refs",
  async (kind) => {
    const db = openDatabase(":memory:", { migrations: selfMigrations });
    db.exec(REVIEW_QUEUE_BASELINE_SQL);
    try {
      const repository = new OpenQuestionsRepository({ db });
      const nodes = [createSemanticNodeId(), createSemanticNodeId()];
      const review = new ReviewQueueRepository({ db }).enqueue({
        kind,
        reason: "Synthetic review",
        refs: { node_ids: nodes, edge_id: createSemanticEdgeId() },
      });
      const extractor = {
        extract: async () => ({
          question: "Synthetic review question?",
          urgency: 0.5,
          related_episode_ids: [],
          related_semantic_node_ids: [...nodes, createSemanticNodeId()],
          duplicate_of_open_question_id: null,
        }),
      };
      await enqueueOpenQuestionForReview(repository, review, { extractor });
      const [question] = repository.list({ status: "open" });
      expect(question).toMatchObject({
        review_id: review.id,
        related_semantic_node_ids: nodes,
        review: { id: review.id, kind, status: "open" },
      });
      expect(
        db.prepare("SELECT review_id, typeof(review_id) AS storage_type FROM open_questions").get(),
      ).toEqual({ review_id: review.id, storage_type: "integer" });
      // Reinforcement did not open the existing question; keep its original review link.
      await enqueueOpenQuestionForReview(
        repository,
        { ...review, id: review.id + 1 },
        { extractor },
      );
      expect(repository.list({ status: "open" })).toHaveLength(1);
      expect(repository.get(question!.id)?.review_id).toBe(review.id);
    } finally {
      db.close();
    }
  },
);

describe("failure event reporting", () => {
  const dirs: string[] = [];
  afterEach(() => {
    vi.restoreAllMocks();
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  it.each(["synchronous", "asynchronous"])(
    "logs and retains a %s append failure without rejecting the live caller",
    async (mode) => {
      const dataDir = mkdtempSync(join(tmpdir(), "borg-failure-report-"));
      dirs.push(dataDir);
      const writer = new StreamWriter({ dataDir });
      const failure = new Error("diagnostic append failed");
      vi.spyOn(writer, "append").mockImplementation(() => {
        if (mode === "synchronous") throw failure;
        return Promise.reject(failure);
      });
      const log = vi.spyOn(console, "error").mockImplementation(() => {});
      await expect(
        appendInternalFailureEvent(writer, "test", new Error("original failure")),
      ).resolves.toBeUndefined();
      await expect(
        appendOpenQuestionHookFailureEvent(writer, "test", new Error("original failure")),
      ).resolves.toBeUndefined();
      expect(log).toHaveBeenCalledTimes(2);
      await expect(writer.failureReports.flush()).rejects.toMatchObject({
        errors: [failure, failure],
      });
      await expect(writer.failureReports.flush()).rejects.toMatchObject({
        errors: [failure, failure],
      });
      writer.close();
    },
  );
});
