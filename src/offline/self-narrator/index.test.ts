import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../test-support.js";
import { SelfNarratorProcess } from "./index.js";

describe("SelfNarratorProcess", () => {
  it("opens a period, updates the narrative through the plan, and records growth markers", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: JSON.stringify({
            observation: {
              category: "understanding",
              what_changed: "Deployment review became more disciplined.",
              before_description: "The team improvised during deploys.",
              after_description: "The team now rehearses rollback steps.",
              confidence: 0.9,
              evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
            },
          }),
          input_tokens: 40,
          output_tokens: 30,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock: {
        now: () => Date.UTC(2026, 3, 1),
      },
    });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Deploy rehearsal",
          narrative: "The team rehearsed rollback steps.",
          tags: ["deploy"],
        }),
      );
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Deploy review",
          narrative: "The team reviewed deployment risks.",
          tags: ["deploy"],
          created_at: Date.UTC(2026, 3, 2),
          updated_at: Date.UTC(2026, 3, 2),
        }),
      );

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "open_period",
          }),
          expect.objectContaining({
            action: "add_growth_marker",
          }),
        ]),
      );

      await process.apply(harness.createContext(), plan);

      expect(harness.autobiographicalRepository.currentPeriod()).toEqual(
        expect.objectContaining({
          label: "2026-Q2",
        }),
      );
      expect(harness.growthMarkersRepository.list()).toEqual([
        expect.objectContaining({
          what_changed: "Deployment review became more disciplined.",
          confidence: 0.6,
        }),
      ]);

      const audits = harness.auditLog.list({ process: "self-narrator" });
      const markerAudit = audits.find((item) => item.action === "add_growth_marker");
      const periodAudit = audits.find((item) => item.action === "open_period");

      if (markerAudit !== undefined) {
        await harness.auditLog.revert(markerAudit.id, "test");
      }

      if (periodAudit !== undefined) {
        await harness.auditLog.revert(periodAudit.id, "test");
      }

      expect(harness.growthMarkersRepository.list()).toEqual([]);
      expect(harness.autobiographicalRepository.currentPeriod()).toBeNull();
    } finally {
      await harness.cleanup();
    }
  });

  it("closes the current period and opens a new one atomically without losing same-label history", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: JSON.stringify({
            observation: {
              category: "skill",
              what_changed: "Atlas operations became the main focus.",
              before_description: "Planning dominated the period.",
              after_description: "Operational debugging dominates now.",
              confidence: 0.8,
              evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
            },
          }),
          input_tokens: 40,
          output_tokens: 30,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock: {
        now: () => Date.UTC(2026, 6, 15),
      },
    });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q1",
        start_ts: Date.UTC(2026, 0, 1),
        narrative: "Planning quarter.",
        themes: ["planning"],
      });
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Atlas incident",
          narrative: "Atlas incident required live debugging.",
          tags: ["atlas"],
          created_at: Date.UTC(2026, 6, 10),
          updated_at: Date.UTC(2026, 6, 10),
        }),
      );
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Atlas follow-up",
          narrative: "Atlas debugging continued.",
          tags: ["atlas"],
          created_at: Date.UTC(2026, 6, 11),
          updated_at: Date.UTC(2026, 6, 11),
        }),
      );

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({ action: "close_period" }),
          expect.objectContaining({ action: "open_period" }),
        ]),
      );

      await process.apply(harness.createContext(), plan);

      const periods = harness.autobiographicalRepository.listPeriods({
        limit: 10,
      });
      const matchingLabels = periods.filter((period) => period.label === "2026-Q1");

      expect(matchingLabels).toHaveLength(2);
      expect(matchingLabels[0]?.end_ts).toBeNull();
      expect(matchingLabels[1]?.end_ts).not.toBeNull();
    } finally {
      await harness.cleanup();
    }
  });

  it("skips invalid observations that do not cite enough supporting episodes", async () => {
    const llm = new FakeLLMClient({
      responses: [
        {
          text: JSON.stringify({
            observation: {
              category: "understanding",
              what_changed: "Weak observation",
              before_description: null,
              after_description: "Only one episode cited.",
              confidence: 0.9,
              evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
            },
          }),
          input_tokens: 40,
          output_tokens: 30,
          stop_reason: "end_turn",
          tool_calls: [],
        },
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
    });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          tags: ["atlas"],
        }),
      );
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          tags: ["atlas"],
          created_at: 2_000_000,
          updated_at: 2_000_000,
        }),
      );

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.errors).toHaveLength(1);
      expect(plan.items.some((item) => item.action === "add_growth_marker")).toBe(false);
    } finally {
      await harness.cleanup();
    }
  });
});
