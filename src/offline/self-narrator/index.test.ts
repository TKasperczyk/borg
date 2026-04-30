import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/index.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../test-support.js";
import { SelfNarratorProcess } from "./index.js";

const SELF_NARRATOR_TOOL_NAME = "EmitSelfNarratorObservations";

function createSelfNarratorResponse(input: {
  observation: {
    theme?: string;
    category: string;
    what_changed: string;
    before_description: string | null;
    after_description: string | null;
    confidence: number;
    evidence_episode_ids: string[];
  } | null;
  period_decision?: "continue_current" | "open_new";
}) {
  const observations =
    input.observation === null
      ? []
      : [
          {
            theme: input.observation.theme ?? input.observation.category,
            category: input.observation.category,
            what_changed: input.observation.what_changed,
            before_description: input.observation.before_description,
            after_description: input.observation.after_description,
            confidence: input.observation.confidence,
            evidence_episode_ids: input.observation.evidence_episode_ids,
          },
        ];

  return {
    text: "",
    input_tokens: 40,
    output_tokens: 30,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_1",
        name: SELF_NARRATOR_TOOL_NAME,
        input: {
          observations,
          period_decision: input.period_decision ?? "open_new",
          period_decision_confidence: 0.8,
        },
      },
    ],
  };
}

describe("SelfNarratorProcess", () => {
  it("opens a period, updates the narrative through the plan, and records growth markers", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          observation: {
            category: "understanding",
            what_changed: "Deployment review became more disciplined.",
            before_description: "The team improvised during deploys.",
            after_description: "The team now rehearses rollback steps.",
            confidence: 0.9,
            evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
          },
        }),
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

      expect(llm.requests[0]?.tool_choice).toEqual({
        type: "tool",
        name: SELF_NARRATOR_TOOL_NAME,
      });
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
      expect(
        harness.identityEventRepository.list({ recordType: "autobiographical_period" }),
      ).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "create",
            record_type: "autobiographical_period",
            provenance: {
              kind: "offline",
              process: "self-narrator",
            },
          }),
        ]),
      );
      expect(harness.identityEventRepository.list({ recordType: "growth_marker" })).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "create",
            record_type: "growth_marker",
            provenance: {
              kind: "offline",
              process: "self-narrator",
            },
          }),
        ]),
      );

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

  it("honors continue_current for theme updates without opening a new period", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          period_decision: "continue_current",
          observation: {
            category: "understanding",
            theme: "deployment",
            what_changed: "Deployment review stayed within the current period.",
            before_description: null,
            after_description: "The current period now includes deployment review.",
            confidence: 0.8,
            evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
          },
        }),
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
      const current = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q1",
        start_ts: Date.UTC(2026, 0, 1),
        narrative: "Planning quarter.",
        themes: ["planning"],
        provenance: { kind: "offline", process: "self-narrator" },
      });
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Deploy rehearsal",
          narrative: "Deployment review became concrete.",
          tags: ["deploy"],
          created_at: Date.UTC(2026, 6, 10),
          updated_at: Date.UTC(2026, 6, 10),
        }),
      );
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Deploy follow-up",
          narrative: "Deployment review stayed active.",
          tags: ["deploy"],
          created_at: Date.UTC(2026, 6, 11),
          updated_at: Date.UTC(2026, 6, 11),
        }),
      );

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items.some((item) => item.action === "close_period")).toBe(false);
      expect(plan.items.some((item) => item.action === "open_period")).toBe(false);
      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "update_period_narrative",
            period_id: current.id,
          }),
        ]),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("ignores open_new when cadence has not elapsed", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          period_decision: "open_new",
          observation: {
            category: "skill",
            theme: "operations",
            what_changed: "Operations became more visible.",
            before_description: null,
            after_description: "Operations evidence is accumulating.",
            confidence: 0.8,
            evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
          },
        }),
      ],
    });
    const now = Date.UTC(2026, 0, 3);
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock: {
        now: () => now,
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
        narrative: "Fresh period.",
        themes: ["planning"],
        provenance: { kind: "offline", process: "self-narrator" },
      });
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          tags: ["ops"],
          created_at: now,
          updated_at: now,
        }),
      );
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          tags: ["ops"],
          created_at: now + 1,
          updated_at: now + 1,
        }),
      );

      const plan = await process.plan(harness.createContext(), {});

      expect(plan.items.some((item) => item.action === "close_period")).toBe(false);
      expect(plan.items.some((item) => item.action === "open_period")).toBe(false);
      expect(plan.items.some((item) => item.action === "update_period_narrative")).toBe(true);
    } finally {
      await harness.cleanup();
    }
  });

  it("queues period rollover for review without mutating established period history", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          observation: {
            category: "skill",
            what_changed: "Atlas operations became the main focus.",
            before_description: "Planning dominated the period.",
            after_description: "Operational debugging dominates now.",
            confidence: 0.8,
            evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
          },
        }),
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
        provenance: { kind: "offline", process: "self-narrator" },
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

      const nextPeriod = plan.items.find((item) => item.action === "open_period")?.period;

      await process.apply(harness.createContext(), plan);

      const reviewItem = harness.reviewQueueRepository.getOpen()[0];
      const periods = harness.autobiographicalRepository.listPeriods({
        limit: 10,
      });
      const matchingLabels = periods.filter((period) => period.label === "2026-Q1");

      expect(reviewItem).toEqual(
        expect.objectContaining({
          kind: "identity_inconsistency",
          refs: expect.objectContaining({
            target_type: "autobiographical_period",
            next_period_open_payload: expect.objectContaining({
              id: nextPeriod?.id,
            }),
          }),
        }),
      );
      expect(matchingLabels).toHaveLength(1);
      expect(matchingLabels[0]?.end_ts).toBeNull();
    } finally {
      await harness.cleanup();
    }
  });

  it("replays the planned successor period when a reviewed rollover is accepted", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          observation: {
            category: "skill",
            what_changed: "Atlas operations became the main focus.",
            before_description: "Planning dominated the period.",
            after_description: "Operational debugging dominates now.",
            confidence: 0.8,
            evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
          },
        }),
      ],
    });
    const rolloverTs = Date.UTC(2026, 6, 15);
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock: {
        now: () => rolloverTs,
      },
    });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const currentPeriod = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q1",
        start_ts: Date.UTC(2026, 0, 1),
        narrative: "Planning quarter.",
        key_episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never, "ep_bbbbbbbbbbbbbbbb" as never],
        themes: ["planning"],
        provenance: {
          kind: "episodes",
          episode_ids: ["ep_aaaaaaaaaaaaaaaa" as never, "ep_bbbbbbbbbbbbbbbb" as never],
        },
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
      const nextPeriod = plan.items.find((item) => item.action === "open_period")?.period;

      expect(nextPeriod).toBeDefined();

      await process.apply(harness.createContext(), plan);

      const reviewItem = harness.reviewQueueRepository.getOpen()[0];

      expect(reviewItem).toEqual(
        expect.objectContaining({
          kind: "identity_inconsistency",
          refs: expect.objectContaining({
            target_type: "autobiographical_period",
            target_id: currentPeriod.id,
            next_period_open_payload: expect.objectContaining({
              id: nextPeriod?.id,
            }),
          }),
        }),
      );
      expect(harness.autobiographicalRepository.currentPeriod()?.id).toBe(currentPeriod.id);

      await harness.reviewQueueRepository.resolve(reviewItem!.id, "accept");

      expect(harness.autobiographicalRepository.currentPeriod()?.id).toBe(nextPeriod?.id);
      expect(harness.autobiographicalRepository.getPeriod(currentPeriod.id)?.end_ts).toBe(
        rolloverTs,
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("does not use audience-scoped episodes as global growth-marker evidence", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          observation: {
            category: "understanding",
            what_changed: "Public planning became more explicit.",
            before_description: "Planning evidence was scattered.",
            after_description: "Public evidence now supports the planning pattern.",
            confidence: 0.9,
            evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
          },
        }),
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
      const sam = harness.entityRepository.resolve("Sam");
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Public planning start",
          narrative: "Public planning started with clearer goals.",
          tags: ["planning"],
        }),
      );
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Public planning follow-up",
          narrative: "Public planning follow-up reinforced the goals.",
          tags: ["planning"],
          created_at: 2_000_000,
          updated_at: 2_000_000,
        }),
      );
      await harness.episodicRepository.insert(
        createEpisodeFixture({
          id: "ep_cccccccccccccccc" as never,
          title: "Sam private planning",
          narrative: "Sam shared a private planning detail.",
          tags: ["planning"],
          audience_entity_id: sam,
          shared: false,
          created_at: 3_000_000,
          updated_at: 3_000_000,
        }),
      );

      const plan = await process.plan(harness.createContext(), {});
      const marker = plan.items.find((item) => item.action === "add_growth_marker")?.marker;

      expect(llm.requests[0]?.messages[0]?.content).not.toContain("Sam private planning");
      expect(marker?.evidence_episode_ids).toEqual(["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"]);
      expect(marker?.evidence_episode_ids).not.toContain("ep_cccccccccccccccc");
    } finally {
      await harness.cleanup();
    }
  });

  it("does not synthesize a fallback narrative when no observations are emitted", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          observation: null,
          period_decision: "open_new",
        }),
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

      expect(plan.items.some((item) => item.action === "open_period")).toBe(false);
      expect(plan.errors).toEqual([
        expect.objectContaining({
          code: "SELF_NARRATOR_EMPTY_NARRATIVE",
        }),
      ]);
    } finally {
      await harness.cleanup();
    }
  });

  it("skips invalid observations that do not cite enough supporting episodes", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          observation: {
            category: "understanding",
            what_changed: "Weak observation",
            before_description: null,
            after_description: "Only one episode cited.",
            confidence: 0.9,
            evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa"],
          },
        }),
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
