import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { afterEach, describe, expect, it, vi } from "vitest";

import { FakeLLMClient, type LLMCompleteResult } from "../../llm/index.js";
import { openDatabase } from "../../storage/sqlite/index.js";
import { FixedClock } from "../../util/clock.js";
import { parseEpisodeId, parseSemanticNodeId } from "../../util/ids.js";
import type { ReviewQueueItem } from "../semantic/index.js";
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
  const cleanup: Array<() => void> = [];

  afterEach(() => {
    while (cleanup.length > 0) {
      cleanup.pop()?.();
    }
  });

  it("extracts a structured LLM proposal with the configured model", async () => {
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

    const proposal = await extractor.extract(createReviewItem(), createContext());

    expect(proposal).toEqual({
      question: "¿Qué atribución debería conservar esta memoria?",
      urgency: 0.64,
      related_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
      related_semantic_node_ids: ["semn_aaaaaaaaaaaaaaaa"],
    });
    expect(llm.requests[0]?.model).toBe("bg-model");
    expect(llm.requests[0]?.tool_choice).toEqual({
      type: "tool",
      name: TOOL_NAME,
    });
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
