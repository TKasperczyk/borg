import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { type LLMCompleteResult } from "../../llm/index.js";
import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import {
  createEntityId,
  createOpenQuestionId,
  parseEpisodeId,
  parseOpenQuestionId,
  parseSemanticNodeId,
} from "../../util/ids.js";
import { relationshipPrivateMemoryDisclosureLabel } from "../common/disclosure-label.js";
import type { ReviewQueueItem } from "../review-queue/index.js";
import { OpenQuestionsRepository, selfMigrations } from "./index.js";
import {
  REVIEW_OPEN_QUESTION_TOOL,
  ReviewOpenQuestionExtractor,
  type ReviewOpenQuestionContext,
} from "./review-open-question-extractor.js";
import { enqueueOpenQuestionForReview } from "./review-open-question-hook.js";

const TOOL_NAME = REVIEW_OPEN_QUESTION_TOOL.name;

function createReviewItem(overrides: Partial<ReviewQueueItem> = {}): ReviewQueueItem {
  return {
    id: 1,
    kind: "misattribution",
    refs: {
      target_type: "episode",
      target_id: "ep_aaaaaaaaaaaaaaaa",
    },
    reason: "La memoria mezcla dos atribuciones.",
    created_at: 1_000,
    resolved_at: null,
    resolution: null,
    ...overrides,
  };
}

function createContext(
  overrides: Partial<ReviewOpenQuestionContext> = {},
): ReviewOpenQuestionContext {
  return {
    audience_entity_id: null,
    allowed_episode_ids: [parseEpisodeId("ep_aaaaaaaaaaaaaaaa")],
    allowed_semantic_node_ids: [parseSemanticNodeId("semn_aaaaaaaaaaaaaaaa")],
    ...overrides,
  };
}

function createToolResponse(input: unknown): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 10,
    output_tokens: 8,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_review_question",
        name: TOOL_NAME,
        input,
      },
    ],
  };
}

describe("review open-question extractor", () => {
  const cleanup: Array<() => void | Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("extracts a structured LLM proposal with the configured model", async () => {
    const candidateId = parseOpenQuestionId("oq_aaaaaaaaaaaaaaaa");
    const llm = new FakeLLMClient({
      responses: [
        createToolResponse({
          question: "¿Qué atribución debería conservar esta memoria?",
          urgency: 0.64,
          related_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
          related_semantic_node_ids: ["semn_aaaaaaaaaaaaaaaa"],
        }),
      ],
    });
    const extractor = new ReviewOpenQuestionExtractor({
      llmClient: llm,
      model: "bg-model",
    });

    const proposal = await extractor.extract(
      createReviewItem(),
      createContext({
        open_question_duplicate_candidates: {
          complete: true,
          total_open_questions: 1,
          presented_count: 1,
          omitted_count: 0,
          rows: [
            {
              id: candidateId,
              text_excerpt: "¿Qué atribución sigue sin resolverse?",
              urgency: 0.4,
              source: "reflection",
              disclosure_label: relationshipPrivateMemoryDisclosureLabel([createEntityId()]),
            },
          ],
        },
      }),
    );

    expect(proposal).toEqual({
      question: "¿Qué atribución debería conservar esta memoria?",
      urgency: 0.64,
      related_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      related_semantic_node_ids: ["semn_aaaaaaaaaaaaaaaa"],
      duplicate_of_open_question_id: null,
    });
    expect(llm.requests[0]?.model).toBe("bg-model");
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: TOOL_NAME,
    });
    expect(String(llm.requests[0]?.messages[0]?.content)).toContain(candidateId);
    expect(String(llm.requests[0]?.messages[0]?.content)).toContain('"complete":true');
  });

  it("fails closed and emits degraded observability when no LLM is configured", async () => {
    const events: unknown[] = [];
    const extractor = new ReviewOpenQuestionExtractor({
      onDegraded: (event) => {
        events.push(event);
      },
    });

    await expect(extractor.extract(createReviewItem(), createContext())).resolves.toBeNull();
    expect(events).toEqual([
      expect.objectContaining({
        reason: "llm_unavailable",
        review_item_id: 1,
        review_kind: "misattribution",
      }),
    ]);
  });

  it("fails closed when an LLM client is configured without a model", async () => {
    const events: unknown[] = [];
    const llm = new FakeLLMClient({
      responses: [
        createToolResponse({
          question: "¿Qué atribución debería conservar esta memoria?",
          urgency: 0.64,
          related_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
          related_semantic_node_ids: [],
        }),
      ],
    });
    const extractor = new ReviewOpenQuestionExtractor({
      llmClient: llm,
      onDegraded: (event) => {
        events.push(event);
      },
    });

    await expect(extractor.extract(createReviewItem(), createContext())).resolves.toBeNull();
    expect(llm.requests).toEqual([]);
    expect(events).toEqual([
      expect.objectContaining({
        reason: "llm_unavailable",
        review_item_id: 1,
        review_kind: "misattribution",
      }),
    ]);
  });

  it("fails closed and emits degraded observability when the LLM call fails", async () => {
    const events: unknown[] = [];
    const extractor = new ReviewOpenQuestionExtractor({
      llmClient: new FakeLLMClient(),
      model: "bg-model",
      onDegraded: (event) => {
        events.push(event);
      },
    });

    await expect(extractor.extract(createReviewItem(), createContext())).resolves.toBeNull();
    expect(events).toEqual([
      expect.objectContaining({
        reason: "llm_call_failed",
        review_item_id: 1,
        review_kind: "misattribution",
      }),
    ]);
  });

  it("fails closed and emits degraded observability when the LLM omits the tool call", async () => {
    const events: unknown[] = [];
    const extractor = new ReviewOpenQuestionExtractor({
      llmClient: new FakeLLMClient({
        responses: [
          {
            text: "",
            input_tokens: 4,
            output_tokens: 2,
            stop_reason: "end_turn",
            tool_calls: [],
          },
        ],
      }),
      model: "bg-model",
      onDegraded: (event) => {
        events.push(event);
      },
    });

    await expect(extractor.extract(createReviewItem(), createContext())).resolves.toBeNull();
    expect(events).toEqual([
      expect.objectContaining({
        reason: "missing_tool",
        review_item_id: 1,
        review_kind: "misattribution",
      }),
    ]);
  });

  it("fails closed and emits degraded observability when the tool payload is invalid", async () => {
    const events: unknown[] = [];
    const invalidResponse = createToolResponse({
      urgency: -0.1,
      related_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      related_semantic_node_ids: [],
    });
    const extractor = new ReviewOpenQuestionExtractor({
      llmClient: new FakeLLMClient({
        responses: [invalidResponse, invalidResponse],
      }),
      model: "bg-model",
      onDegraded: (event) => {
        events.push(event);
      },
    });

    await expect(extractor.extract(createReviewItem(), createContext())).resolves.toBeNull();
    expect(events).toEqual([
      expect.objectContaining({
        reason: "invalid_payload",
        review_item_id: 1,
        review_kind: "misattribution",
      }),
    ]);
  });

  it("filters proposal IDs to the review item's referenced IDs before persisting", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...selfMigrations],
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(1_000),
    });
    cleanup.push(() => {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = {
      extract: vi.fn(async () => ({
        question: "¿Qué relación debe quedar vinculada?",
        urgency: 0.73,
        related_episode_ids: [
          parseEpisodeId("ep_aaaaaaaaaaaaaaaa"),
          parseEpisodeId("ep_bbbbbbbbbbbbbbbb"),
        ],
        related_semantic_node_ids: [
          parseSemanticNodeId("semn_aaaaaaaaaaaaaaaa"),
          parseSemanticNodeId("semn_bbbbbbbbbbbbbbbb"),
        ],
      })),
    };

    await enqueueOpenQuestionForReview(
      repository,
      createReviewItem({
        kind: "identity_inconsistency",
        refs: {
          target_type: "episode",
          target_id: "ep_aaaaaaaaaaaaaaaa",
          patch: {
            related_node_id: "semn_aaaaaaaaaaaaaaaa",
          },
        },
      }),
      { extractor },
    );

    expect(repository.list({ status: "open" })).toEqual([
      expect.objectContaining({
        question: "¿Qué relación debe quedar vinculada?",
        urgency: 0.73,
        related_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
        related_semantic_node_ids: ["semn_aaaaaaaaaaaaaaaa"],
      }),
    ]);
  });

  it("persists all-filtered proposals with offline provenance", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...selfMigrations],
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(1_000),
    });
    cleanup.push(() => {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const extractor = {
      extract: vi.fn(async () => ({
        question: "¿Qué debería quedar pendiente?",
        urgency: 0.58,
        related_episode_ids: [
          parseEpisodeId("ep_bbbbbbbbbbbbbbbb"),
          parseEpisodeId("ep_cccccccccccccccc"),
        ],
        related_semantic_node_ids: [parseSemanticNodeId("semn_bbbbbbbbbbbbbbbb")],
      })),
    };

    await enqueueOpenQuestionForReview(
      repository,
      createReviewItem({
        kind: "identity_inconsistency",
        refs: {
          target_type: "episode",
          target_id: "ep_aaaaaaaaaaaaaaaa",
        },
      }),
      { extractor },
    );

    expect(repository.list({ status: "open" })).toEqual([
      expect.objectContaining({
        question: "¿Qué debería quedar pendiente?",
        urgency: 0.58,
        related_episode_ids: [],
        related_semantic_node_ids: [],
        provenance: {
          kind: "offline",
          process: "overseer",
        },
      }),
    ]);
  });

  it("reinforces an existing open question when the proposal matches its normalized text", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...selfMigrations],
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(1_000),
    });
    cleanup.push(() => {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const existing = repository.add({
      question: "Is €2,800 a hard ceiling for the trip?",
      urgency: 0.61,
      source: "overseer",
      provenance: { kind: "offline", process: "overseer" },
    });
    const extractor = {
      extract: vi.fn(async () => ({
        // Same question, different whitespace/case -- normalization should collapse it.
        question: "  is €2,800 A HARD ceiling for the trip?  ",
        urgency: 0.8,
        related_episode_ids: [parseEpisodeId("ep_aaaaaaaaaaaaaaaa")],
        related_semantic_node_ids: [],
      })),
    };

    await enqueueOpenQuestionForReview(repository, createReviewItem(), { extractor });

    const openQuestions = repository.list({ status: "open" });

    expect(openQuestions).toHaveLength(1);
    expect(openQuestions[0]).toMatchObject({ id: existing.id });
    expect(openQuestions[0]?.urgency).toBeCloseTo(0.63);
  });

  it("honors a presented model duplicate across audiences and folds evidence fail-closed", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...selfMigrations],
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(1_000),
    });
    cleanup.push(() => {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    });
    const firstAudience = createEntityId();
    const secondAudience = createEntityId();
    const firstEpisodeId = parseEpisodeId("ep_bbbbbbbbbbbbbbbb");
    const secondEpisodeId = parseEpisodeId("ep_aaaaaaaaaaaaaaaa");
    const existing = repository.add({
      question: "Which attribution remains unsettled?",
      urgency: 0.41,
      audience_entity_id: firstAudience,
      disclosure_label: relationshipPrivateMemoryDisclosureLabel([firstAudience]),
      related_episode_ids: [firstEpisodeId],
      source: "overseer",
      provenance: { kind: "offline", process: "overseer" },
    });
    const extractor = {
      extract: vi.fn(async (_item: ReviewQueueItem, context: ReviewOpenQuestionContext) => {
        expect(context.open_question_duplicate_candidates).toMatchObject({
          complete: true,
          total_open_questions: 1,
          rows: [expect.objectContaining({ id: existing.id })],
        });

        return {
          question: "¿Qué autoría sigue sin aclararse?",
          urgency: 0.72,
          related_episode_ids: [secondEpisodeId],
          related_semantic_node_ids: [],
          duplicate_of_open_question_id: existing.id,
        };
      }),
    };

    await enqueueOpenQuestionForReview(
      repository,
      createReviewItem({
        refs: {
          target_type: "episode",
          target_id: secondEpisodeId,
          audience_entity_id: secondAudience,
          disclosure_label: relationshipPrivateMemoryDisclosureLabel([secondAudience]),
        },
      }),
      { extractor },
    );

    const openQuestions = repository.listAllOpen();
    expect(openQuestions).toHaveLength(1);
    expect(openQuestions[0]).toMatchObject({
      id: existing.id,
      urgency: 0.43,
      related_episode_ids: [firstEpisodeId, secondEpisodeId],
      disclosure_label: {
        disclosureClass: "relationship_private",
        originAudienceEntityIds: [firstAudience, secondAudience].sort(),
        privateToEntityIds: [firstAudience, secondAudience].sort(),
      },
    });
  });

  it("ignores a duplicate advisory id that was not presented", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...selfMigrations],
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(1_000),
    });
    cleanup.push(() => {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    });
    const extractor = {
      extract: vi.fn(async () => ({
        question: "Which unrepresented uncertainty should remain separate?",
        urgency: 0.5,
        related_episode_ids: [parseEpisodeId("ep_aaaaaaaaaaaaaaaa")],
        related_semantic_node_ids: [],
        duplicate_of_open_question_id: createOpenQuestionId(),
      })),
    };

    await enqueueOpenQuestionForReview(repository, createReviewItem(), { extractor });

    expect(repository.listAllOpen()).toEqual([
      expect.objectContaining({
        question: "Which unrepresented uncertainty should remain separate?",
      }),
    ]);
  });

  it("does not collapse distinct review questions that share wording but differ on a specific", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...selfMigrations],
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(1_000),
    });
    cleanup.push(() => {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    repository.add({
      question: "Is €2,800 a hard ceiling for the trip?",
      urgency: 0.5,
      source: "overseer",
      provenance: { kind: "offline", process: "overseer" },
    });
    const extractor = {
      extract: vi.fn(async () => ({
        question: "Is €5,000 a hard ceiling for the trip?",
        urgency: 0.55,
        related_episode_ids: [parseEpisodeId("ep_aaaaaaaaaaaaaaaa")],
        related_semantic_node_ids: [],
      })),
    };

    await enqueueOpenQuestionForReview(repository, createReviewItem(), { extractor });

    const openQuestions = repository.list({ status: "open" });

    expect(openQuestions).toHaveLength(2);
    expect(openQuestions.map((question) => question.question).sort()).toEqual([
      "Is €2,800 a hard ceiling for the trip?",
      "Is €5,000 a hard ceiling for the trip?",
    ]);
  });

  it("finds an exact-normalized match outside the urgency top window", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...selfMigrations],
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(1_000),
    });
    cleanup.push(() => {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    const lowUrgencyTarget = repository.add({
      question: "Should the participant roster collapse Borg and assistant?",
      urgency: 0.01,
      source: "overseer",
      provenance: { kind: "offline", process: "overseer" },
    });
    // Add 60 high-urgency unrelated questions so the target falls below the
    // pre-fix top-50 window.
    for (let i = 0; i < 60; i += 1) {
      repository.add({
        question: `Padding question ${i}?`,
        urgency: 0.9,
        source: "overseer",
        provenance: { kind: "offline", process: "overseer" },
      });
    }

    const extractor = {
      extract: vi.fn(async (_item: ReviewQueueItem, context: ReviewOpenQuestionContext) => {
        expect(context.open_question_duplicate_candidates).toMatchObject({
          complete: true,
          total_open_questions: 61,
          presented_count: 61,
          omitted_count: 0,
        });
        expect(context.open_question_duplicate_candidates?.rows.map((row) => row.id)).toContain(
          lowUrgencyTarget.id,
        );

        return {
          question: "Should the participant roster collapse Borg and assistant?",
          urgency: 0.7,
          related_episode_ids: [parseEpisodeId("ep_aaaaaaaaaaaaaaaa")],
          related_semantic_node_ids: [],
        };
      }),
    };

    await enqueueOpenQuestionForReview(repository, createReviewItem(), { extractor });

    const matching = repository
      .list({ status: "open", limit: 200 })
      .filter(
        (question) =>
          question.question === "Should the participant roster collapse Borg and assistant?",
      );

    expect(matching).toHaveLength(1);
    expect(matching[0]?.id).toBe(lowUrgencyTarget.id);
  });

  it("does not write an open question when no extractor is supplied", async () => {
    const tempDir = mkdtempSync(join(tmpdir(), "borg-"));
    const db = openDatabase(join(tempDir, "borg.db"), {
      migrations: [...selfMigrations],
    });
    const repository = new OpenQuestionsRepository({
      db,
      clock: new FixedClock(1_000),
    });
    cleanup.push(() => {
      db.close();
      rmSync(tempDir, { recursive: true, force: true });
    });

    await enqueueOpenQuestionForReview(repository, createReviewItem(), {
      extractor: null,
    });

    expect(repository.list({ status: "open" })).toEqual([]);
  });
});
