import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";
import { FixedClock } from "../../util/clock.js";
import { createEpisodeId, createSemanticNodeId } from "../../util/ids.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../test-support.js";
import { RuminatorProcess } from "./index.js";

const RUMINATOR_TOOL_NAME = "EmitRuminatorDecisions";

function createRuminatorResponse(input: {
  resolution_note: string;
  growth_marker: null | {
    category: string;
    what_changed: string;
    before_description: string | null;
    after_description: string | null;
    confidence: number;
  };
}) {
  return {
    text: "",
    input_tokens: 50,
    output_tokens: 40,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: RUMINATOR_TOOL_NAME,
        input,
      },
    ],
  };
}

describe("RuminatorProcess", () => {
  it("plans and applies a resolution with capped growth confidence, and apply is idempotent", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createRuminatorResponse({
          resolution_note: "Atlas now succeeds after the rollback rehearsal.",
          growth_marker: {
            category: "understanding",
            what_changed: "I understand Atlas rollback sequencing better.",
            before_description: "The deployment order was unclear.",
            after_description: "The rollback rehearsal clarified the order.",
            confidence: 0.95,
          },
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const episode = createEpisodeFixture(
        {
          title: "Atlas rollback rehearsal",
          narrative: "Atlas stabilized after a rollback rehearsal.",
          tags: ["atlas", "deploy"],
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [1, 0, 0, 0],
      );
      await harness.episodicRepository.insert(episode);
      const question = harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail?",
        urgency: 0.7,
        related_episode_ids: [createEpisodeId()],
        related_semantic_node_ids: [createSemanticNodeId()],
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
      });

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toHaveLength(1);
      expect(llm.requests[0]?.tool_choice).toEqual({
        type: "tool",
        name: RUMINATOR_TOOL_NAME,
      });
      expect(plan.items[0]).toMatchObject({
        action: "resolve",
        question_id: question.id,
        resolution_episode_id: episode.id,
      });
      expect(
        plan.items[0]?.action === "resolve" ? plan.items[0].growth_marker?.confidence : 0,
      ).toBe(0.6);

      await process.apply(harness.createContext(), plan);
      await process.apply(harness.createContext(), plan);

      expect(harness.openQuestionsRepository.get(question.id)?.status).toBe("resolved");
      expect(harness.growthMarkersRepository.list()).toHaveLength(1);

      const audits = harness.auditLog.list({ process: "ruminator" });
      const growthAudit = audits.find((item) => item.action === "add_growth_marker");
      const resolveAudit = audits.find((item) => item.action === "resolve");

      expect(growthAudit).toBeDefined();
      expect(resolveAudit).toBeDefined();

      if (growthAudit !== undefined) {
        await harness.auditLog.revert(growthAudit.id, "test");
      }

      if (resolveAudit !== undefined) {
        await harness.auditLog.revert(resolveAudit.id, "test");
      }

      expect(harness.growthMarkersRepository.list()).toHaveLength(0);
      expect(harness.openQuestionsRepository.get(question.id)?.status).toBe("open");
    } finally {
      await harness.cleanup();
    }
  });

  it("plans urgency bumps and abandonments without LLM calls when evidence is weak", async () => {
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(40 * 24 * 60 * 60 * 1_000),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const staleQuestion = harness.openQuestionsRepository.add({
        question: "What was the exact rollback order?",
        urgency: 0.1,
        source: "user",
        created_at: 0,
        last_touched: 0,
      });
      const agingQuestion = harness.openQuestionsRepository.add({
        question: "Should I revisit Atlas logging?",
        urgency: 0.4,
        source: "reflection",
        created_at: 0,
        last_touched: 0,
      });

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "abandon",
            question_id: staleQuestion.id,
          }),
          expect.objectContaining({
            action: "bump_urgency",
            question_id: agingQuestion.id,
          }),
        ]),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("halts on budget exhaustion without making further LLM calls", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          ...createRuminatorResponse({
            resolution_note: "First answer",
            growth_marker: null,
          }),
          input_tokens: 20,
          output_tokens: 20,
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock: new FixedClock(40 * 24 * 60 * 60 * 1_000),
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const firstEpisode = createEpisodeFixture(
        {
          title: "Atlas deploy fix",
          narrative: "Atlas deploy fix landed.",
          tags: ["atlas", "deploy"],
          created_at: 2_000_000,
          updated_at: 2_000_000,
        },
        [1, 0, 0, 0],
      );
      const secondEpisode = createEpisodeFixture(
        {
          title: "Atlas retry plan",
          narrative: "Atlas retry plan landed.",
          tags: ["atlas", "deploy"],
          created_at: 2_100_000,
          updated_at: 2_100_000,
        },
        [1, 0, 0, 0],
      );
      await harness.episodicRepository.insert(firstEpisode);
      await harness.episodicRepository.insert(secondEpisode);
      harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail?",
        urgency: 0.7,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
      });
      harness.openQuestionsRepository.add({
        question: "Why does Atlas deploy fail again?",
        urgency: 0.65,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
      });

      const plan = await process.plan(harness.createContext(), {
        budget: 10,
      });

      expect(plan.budget_exhausted).toBe(true);
      expect(llm.requests).toHaveLength(1);
    } finally {
      await harness.cleanup();
    }
  });
});
