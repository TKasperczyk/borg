import { describe, expect, it, vi } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import {
  publicMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import { createEpisodeId, createMaintenanceRunId, createStreamEntryId } from "../../util/ids.js";
import { SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE } from "../../util/self-memory-voice.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";

import { episodeEvidencePromptRow } from "../evidence-labels.js";
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
  period_narrative?: string | null;
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
          period_narrative: input.period_narrative ?? null,
          period_decision: input.period_decision ?? "open_new",
          period_decision_confidence: 0.8,
        },
      },
    ],
  };
}

describe("SelfNarratorProcess", () => {
  it("bounds the prompt by newest episodes while retaining all prior context and scoped citations", async () => {
    const episodes = Array.from({ length: 6 }, (_, index) =>
      createEpisodeFixture({
        start_time: 1_000 + index * 1_000,
        end_time: 1_500 + index * 1_000,
        narrative: `${index}: ${"経験".repeat(4_000)}`,
      }),
    );
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({ observation: null, period_decision: "continue_current" }),
        createSelfNarratorResponse({ observation: null, period_decision: "continue_current" }),
        createSelfNarratorResponse({
          observation: {
            category: "understanding",
            what_changed: "A proposed change citing omitted prior evidence.",
            before_description: null,
            after_description: null,
            confidence: 0.5,
            evidence_episode_ids: [episodes[0]!.id, episodes[5]!.id],
          },
          period_decision: "continue_current",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({ llmClient: llm });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });
    try {
      harness.autobiographicalRepository.upsertPeriod({
        label: "Established period",
        start_ts: 1_000,
        narrative: "I already have a story of the early part of this period.",
        key_episode_ids: [episodes[0]!.id],
        provenance: { kind: "offline", process: "self-narrator" },
      });
      // More than list()'s default page, including a marker at the period boundary.
      const markers = Array.from({ length: 51 }, (_, index) =>
        harness.growthMarkersRepository.add({
          ts: 1_000 + index,
          category: "understanding",
          what_changed: `Prior development ${index}`,
          evidence_episode_ids: [episodes[0]!.id],
          confidence: 0.5,
          source_process: "self-narrator",
          provenance: { kind: "offline", process: "self-narrator" },
        }),
      );
      // Deliberately out of order: selection must use timestamps, not repository order.
      vi.spyOn(harness.episodicRepository, "listEffectivelyVisible").mockResolvedValue(
        [...episodes].reverse(),
      );
      await process.plan(harness.createContext());
      const fullPrompt = String(llm.requests[0]!.messages[0]!.content);
      const overhead =
        estimatePromptTokens(String(llm.requests[0]!.system)) +
        estimatePromptTokens(JSON.stringify(llm.requests[0]!.tools));
      const omittedRows = episodes
        .slice(0, 4)
        .map((episode) =>
          JSON.stringify(episodeEvidencePromptRow(episode, { start_time: episode.start_time })),
        );
      // Leave space for the coverage notice, but nowhere near another episode.
      const cap =
        overhead +
        estimatePromptTokens(fullPrompt) -
        estimatePromptTokens(omittedRows.join("\n")) +
        100;
      harness.config.offline.selfNarrator.maxPromptTokens = cap;
      await process.plan(harness.createContext());
      const prompt = String(llm.requests[1]!.messages[0]!.content);
      const request = llm.requests[1]!;
      expect(
        estimatePromptTokens(String(request.system)) +
          estimatePromptTokens(JSON.stringify(request.tools)) +
          request.messages.reduce(
            (sum, message) => sum + estimatePromptTokens(String(message.content)),
            0,
          ),
      ).toBeLessThanOrEqual(cap);
      expect(prompt.split("\nEpisodes:\n")[1]).toBe(
        episodes
          .slice(4)
          .map((episode) =>
            JSON.stringify(episodeEvidencePromptRow(episode, { start_time: episode.start_time })),
          )
          .join("\n"),
      );
      expect(prompt).toContain("4 earlier episodes of this period are not shown");
      expect(prompt).toContain("1970-01-01T00:00:01.000Z to 1970-01-01T00:00:04.500Z");
      expect(prompt).toContain("I already have a story of the early part of this period.");
      for (const marker of markers) expect(prompt).toContain(marker.what_changed);
      expect(prompt).toContain("I revise and extend its existing story");

      // IDs carried by the prior narrative/markers do not enlarge allowedIds.
      const invalidPlan = await process.plan(harness.createContext());
      expect(invalidPlan.errors).toEqual([
        expect.objectContaining({ code: "SELF_NARRATOR_INVALID_REF" }),
      ]);
      expect(invalidPlan.items.some((item) => item.action === "add_growth_marker")).toBe(false);
    } finally {
      await harness.cleanup();
    }
  });

  it("keeps every episode unchanged when the complete prompt fits exactly at the cap", async () => {
    const episodes = [
      createEpisodeFixture({ start_time: 1_000 }),
      createEpisodeFixture({ start_time: 2_000 }),
    ];
    const response = createSelfNarratorResponse({ observation: null });
    const llm = new FakeLLMClient({ responses: [response, response] });
    const harness = await createOfflineTestHarness({ llmClient: llm });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });
    try {
      harness.autobiographicalRepository.upsertPeriod({
        label: "Current period",
        start_ts: 1_000,
        narrative: "My existing narrative.",
        provenance: { kind: "offline", process: "self-narrator" },
      });
      vi.spyOn(harness.episodicRepository, "listEffectivelyVisible").mockResolvedValue(episodes);
      await process.plan(harness.createContext());
      const prompt = String(llm.requests[0]!.messages[0]!.content);
      const cap =
        estimatePromptTokens(String(llm.requests[0]!.system)) +
        estimatePromptTokens(JSON.stringify(llm.requests[0]!.tools)) +
        estimatePromptTokens(prompt);
      harness.config.offline.selfNarrator.maxPromptTokens = cap;
      await process.plan(harness.createContext());
      expect(llm.requests[1]!.messages[0]!.content).toBe(prompt);
      const request = llm.requests[1]!;
      expect(
        estimatePromptTokens(String(request.system)) +
          estimatePromptTokens(JSON.stringify(request.tools)) +
          request.messages.reduce(
            (sum, message) => sum + estimatePromptTokens(String(message.content)),
            0,
          ),
      ).toBe(cap);
      expect(prompt).toContain("All 2 candidate episodes are shown; 0 earlier episodes omitted.");
      expect(prompt.split("\nEpisodes:\n")[1]).toBe(
        episodes
          .map((episode) =>
            JSON.stringify(episodeEvidencePromptRow(episode, { start_time: episode.start_time })),
          )
          .join("\n"),
      );
      // The user message alone fitting is insufficient: reserve system and tools.
      harness.config.offline.selfNarrator.maxPromptTokens = estimatePromptTokens(prompt);
      const tooSmall = await process.plan(harness.createContext());
      expect(tooSmall.errors).toEqual([
        expect.objectContaining({ code: "SELF_NARRATOR_PROMPT_BUDGET_TOO_SMALL" }),
      ]);
      expect(llm.requests).toHaveLength(2);
    } finally {
      await harness.cleanup();
    }
  });

  it("reports a cap too small for prior context or the newest evidence without an oversized call", async () => {
    const llm = new FakeLLMClient();
    const harness = await createOfflineTestHarness({ llmClient: llm });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });
    try {
      harness.config.offline.selfNarrator.maxPromptTokens = 1_000;
      const episodes = [
        createEpisodeFixture(),
        createEpisodeFixture({ start_time: 2_000_000, narrative: "x".repeat(8_000) }),
      ];
      vi.spyOn(harness.episodicRepository, "listEffectivelyVisible").mockResolvedValue(episodes);
      const plan = await process.plan(harness.createContext());
      expect(plan.errors).toEqual([
        expect.objectContaining({ code: "SELF_NARRATOR_PROMPT_BUDGET_TOO_SMALL" }),
      ]);
      expect(plan.items).toEqual([]);
      expect(llm.requests).toEqual([]);
    } finally {
      await harness.cleanup();
    }
  });

  it.each(["row", "count", "oversized_newest"] as const)(
    "combines row/count/full bounds without forcing oversized evidence (%s)",
    async (bound) => {
      const episodes = Array.from({ length: 5 }, (_, index) =>
        createEpisodeFixture({
          start_time: 1000 + index * 1000,
          end_time: 1500 + index * 1000,
          narrative: `${index}: ${"経験".repeat(bound === "oversized_newest" && index === 4 ? 10_000 : 100)}`,
        }),
      );
      const llm = new FakeLLMClient({
        responses: [createSelfNarratorResponse({ observation: null })],
      });
      const harness = await createOfflineTestHarness({ llmClient: llm });
      const process = new SelfNarratorProcess({
        autobiographicalRepository: harness.autobiographicalRepository,
        growthMarkersRepository: harness.growthMarkersRepository,
        registry: harness.registry,
      });
      try {
        vi.spyOn(harness.episodicRepository, "listEffectivelyVisible").mockResolvedValue(episodes);
        const newestTwoTokens = episodes
          .slice(-2)
          .reduce(
            (sum, episode) =>
              sum +
              estimatePromptTokens(
                JSON.stringify(
                  episodeEvidencePromptRow(episode, { start_time: episode.start_time }),
                ),
              ),
            0,
          );
        const config = harness.config.offline.selfNarrator;
        config.maxEpisodesPerRun = bound === "count" ? 1 : 3;
        config.maxInputTokensPerRun = bound === "oversized_newest" ? 1000 : newestTwoTokens;
        const plan = await process.plan(harness.createContext());
        if (bound === "row") {
          expect(llm.requests).toHaveLength(1);
          const request = llm.requests[0]!;
          const prompt = String(request.messages[0]!.content);
          expect(prompt).toContain("due to maxInputTokensPerRun");
          expect(prompt).toContain("3 earlier episodes");
          expect(prompt).toContain("1970-01-01T00:00:01.000Z to 1970-01-01T00:00:03.500Z");
          expect(prompt.split("\nEpisodes:\n")[1]).toBe(
            episodes
              .slice(-2)
              .map((episode) =>
                JSON.stringify(
                  episodeEvidencePromptRow(episode, { start_time: episode.start_time }),
                ),
              )
              .join("\n"),
          );
          expect(
            estimatePromptTokens(String(request.system)) +
              estimatePromptTokens(JSON.stringify(request.tools)) +
              estimatePromptTokens(prompt),
          ).toBeLessThanOrEqual(config.maxPromptTokens);
        } else {
          expect(llm.requests).toEqual([]);
          expect(plan.errors).toEqual([
            expect.objectContaining({
              code: "SELF_NARRATOR_PROMPT_BUDGET_TOO_SMALL",
              message: expect.stringContaining(
                bound === "count" ? "maxEpisodesPerRun" : "maxInputTokensPerRun",
              ),
            }),
          ]);
          expect(plan.items).toEqual([]);
        }
      } finally {
        await harness.cleanup();
      }
    },
  );

  it.each(["episodes", "online_reflector"] as const)(
    "keeps prior growth context without uncitable ids in %s provenance",
    async (kind) => {
      const episodes = [createEpisodeFixture(), createEpisodeFixture()];
      const priorEpisodeId = createEpisodeId();
      const priorStreamId = createStreamEntryId();
      const llm = new FakeLLMClient({
        responses: [
          createSelfNarratorResponse({ observation: null, period_decision: "continue_current" }),
        ],
      });
      const harness = await createOfflineTestHarness({ llmClient: llm });
      const process = new SelfNarratorProcess({
        autobiographicalRepository: harness.autobiographicalRepository,
        growthMarkersRepository: harness.growthMarkersRepository,
        registry: harness.registry,
      });
      try {
        vi.spyOn(harness.episodicRepository, "listEffectivelyVisible").mockResolvedValue(episodes);
        harness.autobiographicalRepository.upsertPeriod({
          label: "Current period",
          start_ts: 1_000,
          narrative: "My earlier story.",
          provenance: { kind: "offline", process: "self-narrator" },
        });
        const disclosure = selfPrivateMemoryDisclosureLabel();
        const marker = harness.growthMarkersRepository.add({
          ts: 2_000,
          category: "understanding",
          what_changed: "以前の経験から学んだこと。",
          before_description: "I had not noticed the pattern.",
          after_description: "I learned to ask earlier.",
          evidence_episode_ids: [priorEpisodeId],
          confidence: 0.5,
          source_process: "reflector",
          disclosure_label: disclosure,
          provenance:
            kind === "episodes"
              ? { kind, episode_ids: [priorEpisodeId] }
              : {
                  kind,
                  evidence_episode_ids: [priorEpisodeId],
                  evidence_stream_entry_ids: [priorStreamId],
                },
        });
        await process.plan(harness.createContext());
        const prompt = String(llm.requests[0]!.messages[0]!.content);
        const row = prompt.split("\n").find((line) => line.includes(marker.what_changed))!;
        expect(JSON.parse(row)).toMatchObject({
          ts: new Date(marker.ts).toISOString(),
          category: marker.category,
          what_changed: marker.what_changed,
          before_description: marker.before_description,
          after_description: marker.after_description,
          confidence: marker.confidence,
          source_process: marker.source_process,
          disclosure_label: { disclosure_class: "self_private" },
        });
        for (const id of [priorEpisodeId, priorStreamId, marker.id])
          expect(prompt).not.toContain(id);
        for (const episode of episodes) expect(prompt).toContain(episode.id);
        expect(prompt).toContain("I only cite evidence_episode_ids from the provided episodes.");
        expect(Buffer.byteLength(row)).toBeLessThan(
          Buffer.byteLength(
            JSON.stringify({
              ...marker,
              ...memoryDisclosurePayloadFields(disclosure),
            }),
          ),
        );
      } finally {
        await harness.cleanup();
      }
    },
  );

  it("carries prior growth-marker disclosure into a revised narrative", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          observation: null,
          period_decision: "continue_current",
          period_narrative: "I have revised my story with earlier growth.",
        }),
      ],
    });
    const harness = await createOfflineTestHarness({ llmClient: llm });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });
    try {
      const episodes = [createEpisodeFixture(), createEpisodeFixture()];
      vi.spyOn(harness.episodicRepository, "listEffectivelyVisible").mockResolvedValue(episodes);
      harness.autobiographicalRepository.upsertPeriod({
        label: "Current period",
        start_ts: 1_000,
        narrative: "My earlier public narrative.",
        disclosure_label: publicMemoryDisclosureLabel(),
        provenance: { kind: "offline", process: "self-narrator" },
      });
      harness.growthMarkersRepository.add({
        ts: 2_000,
        category: "understanding",
        what_changed: "An earlier private development.",
        evidence_episode_ids: [episodes[0]!.id],
        confidence: 0.5,
        source_process: "ruminator",
        disclosure_label: selfPrivateMemoryDisclosureLabel(),
        provenance: { kind: "offline", process: "ruminator" },
      });
      const plan = await process.plan(harness.createContext());
      expect(plan.items).toEqual([
        expect.objectContaining({
          action: "update_period_narrative",
          disclosure_label: expect.objectContaining({ disclosureClass: "self_private" }),
        }),
      ]);
      expect(String(llm.requests[0]!.messages[0]!.content)).toContain(
        '"disclosure_class":"self_private"',
      );
    } finally {
      await harness.cleanup();
    }
  });

  it.each(["continue_current", "open_new"] as const)(
    "labels public-episode observations and fallback narratives from private prior context (%s)",
    async (periodDecision) => {
      const episodes = [
        createEpisodeFixture({ shared: true, created_at: 2_000 }),
        createEpisodeFixture({ shared: true, created_at: 3_000 }),
      ];
      const whatChanged = "I carried an earlier private development into my new understanding.";
      const llm = new FakeLLMClient({
        responses: [
          createSelfNarratorResponse({
            observation: {
              category: "understanding",
              what_changed: whatChanged,
              before_description: null,
              after_description: null,
              confidence: 0.5,
              evidence_episode_ids: episodes.map((episode) => episode.id),
            },
            period_decision: periodDecision,
            period_narrative: null,
          }),
        ],
      });
      const harness = await createOfflineTestHarness({
        llmClient: llm,
        clock: { now: () => 10 * 86_400_000 },
      });
      const process = new SelfNarratorProcess({
        autobiographicalRepository: harness.autobiographicalRepository,
        growthMarkersRepository: harness.growthMarkersRepository,
        registry: harness.registry,
      });
      try {
        for (const episode of episodes) await harness.episodicRepository.createEpisode(episode);
        harness.autobiographicalRepository.upsertPeriod({
          label: "Current period",
          start_ts: 1_000,
          narrative: "My earlier public narrative.",
          disclosure_label: publicMemoryDisclosureLabel(),
          provenance: { kind: "offline", process: "self-narrator" },
        });
        const priorMarker = harness.growthMarkersRepository.add({
          ts: 1_500,
          category: "understanding",
          what_changed: "An earlier private development.",
          evidence_episode_ids: [episodes[0]!.id],
          confidence: 0.5,
          source_process: "ruminator",
          disclosure_label: selfPrivateMemoryDisclosureLabel(),
          provenance: { kind: "offline", process: "ruminator" },
        });
        const plan = await process.plan(harness.createContext());
        expect(plan.errors).toEqual([]);
        const marker = plan.items.find((item) => item.action === "add_growth_marker")?.marker;
        expect(marker).toMatchObject({
          what_changed: whatChanged,
          evidence_episode_ids: episodes.map((episode) => episode.id),
          disclosure_label: expect.objectContaining({ disclosureClass: "self_private" }),
        });
        const periodItem = plan.items.find(
          (item) =>
            item.action ===
            (periodDecision === "open_new" ? "open_period" : "update_period_narrative"),
        );
        expect(periodItem).toBeDefined();
        const period = periodItem?.action === "open_period" ? periodItem.period : periodItem;
        expect(period).toMatchObject({
          narrative: expect.stringContaining(whatChanged),
          disclosure_label: expect.objectContaining({ disclosureClass: "self_private" }),
        });
        const prompt = String(llm.requests[0]!.messages[0]!.content);
        expect(prompt).toContain(priorMarker.what_changed);
        expect(prompt).toContain('"disclosure_class":"self_private"');
        expect(prompt.split("\nEpisodes:\n")[1]).toBe(
          episodes
            .map((episode) =>
              JSON.stringify(episodeEvidencePromptRow(episode, { start_time: episode.start_time })),
            )
            .join("\n"),
        );
        await process.apply(harness.createContext(), plan);
        expect(
          harness.growthMarkersRepository.get(marker!.id)?.disclosure_label?.disclosureClass,
        ).toBe("self_private");
      } finally {
        await harness.cleanup();
      }
    },
  );

  it("rejects malformed reversal ids before repository deletes", async () => {
    const harness = await createOfflineTestHarness();
    new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      const deletePeriodSpy = vi.spyOn(harness.autobiographicalRepository, "deletePeriod");
      const deleteMarkerSpy = vi.spyOn(harness.growthMarkersRepository, "delete");
      const periodAudit = harness.auditLog.record({
        run_id: createMaintenanceRunId(),
        process: "self-narrator",
        action: "open_period",
        targets: {},
        reversal: {
          period_id: "not-a-period-id",
        },
      });
      const markerAudit = harness.auditLog.record({
        run_id: createMaintenanceRunId(),
        process: "self-narrator",
        action: "add_growth_marker",
        targets: {},
        reversal: {
          marker_id: "not-a-marker-id",
        },
      });

      await expect(harness.auditLog.revert(periodAudit.id, "test")).rejects.toMatchObject({
        code: "SELF_NARRATOR_REVERSAL_INVALID",
      });
      await expect(harness.auditLog.revert(markerAudit.id, "test")).rejects.toMatchObject({
        code: "SELF_NARRATOR_REVERSAL_INVALID",
      });
      expect(deletePeriodSpy).not.toHaveBeenCalled();
      expect(deleteMarkerSpy).not.toHaveBeenCalled();
    } finally {
      await harness.cleanup();
    }
  });

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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Deploy rehearsal",
          narrative: "The team rehearsed rollback steps.",
          tags: ["deploy"],
        }),
      );
      await harness.episodicRepository.createEpisode(
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
      expect(llm.requests[0]?.system).toContain(SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE);
      expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
        SELF_REFERENTIAL_MEMORY_VOICE_GUIDANCE,
      );
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Deploy rehearsal",
          narrative: "Deployment review became concrete.",
          tags: ["deploy"],
          created_at: Date.UTC(2026, 6, 10),
          updated_at: Date.UTC(2026, 6, 10),
        }),
      );
      await harness.episodicRepository.createEpisode(
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          tags: ["ops"],
          created_at: now,
          updated_at: now,
        }),
      );
      await harness.episodicRepository.createEpisode(
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Atlas incident",
          narrative: "Atlas incident required live debugging.",
          tags: ["atlas"],
          created_at: Date.UTC(2026, 6, 10),
          updated_at: Date.UTC(2026, 6, 10),
        }),
      );
      await harness.episodicRepository.createEpisode(
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

      expect(nextPeriod?.label).toBe("2026-Q3");

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

  it("labels a successor period from its own start, not from the period it closes", async () => {
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
        now: () => Date.UTC(2026, 9, 1),
      },
    });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q2",
        start_ts: Date.UTC(2026, 5, 5),
        narrative: "Planning quarter.",
        themes: ["planning"],
        provenance: { kind: "offline", process: "self-narrator" },
      });
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Atlas incident",
          narrative: "Atlas incident required live debugging.",
          tags: ["atlas"],
          created_at: Date.UTC(2026, 8, 10),
          updated_at: Date.UTC(2026, 8, 10),
        }),
      );
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Atlas follow-up",
          narrative: "Atlas debugging continued.",
          tags: ["atlas"],
          created_at: Date.UTC(2026, 8, 11),
          updated_at: Date.UTC(2026, 8, 11),
        }),
      );

      const plan = await process.plan(harness.createContext(), {});
      const nextPeriod = plan.items.find((item) => item.action === "open_period")?.period;

      expect(nextPeriod?.label).toBe("2026-Q3");
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
    const successorStartTs = Date.UTC(2026, 6, 11);
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Atlas incident",
          narrative: "Atlas incident required live debugging.",
          tags: ["atlas"],
          created_at: Date.UTC(2026, 6, 10),
          updated_at: Date.UTC(2026, 6, 10),
        }),
      );
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Atlas follow-up",
          start_time: successorStartTs,
          end_time: successorStartTs + 1_000,
          narrative: "Atlas debugging continued.",
          tags: ["atlas"],
          created_at: Date.UTC(2026, 6, 11),
          updated_at: Date.UTC(2026, 6, 11),
        }),
      );

      const plan = await process.plan(harness.createContext(), {});
      const nextPeriod = plan.items.find((item) => item.action === "open_period")?.period;

      expect(nextPeriod?.start_ts).toBe(successorStartTs);
      expect(plan.items.find((item) => item.action === "close_period")?.end_ts).toBe(
        successorStartTs,
      );

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
        successorStartTs,
      );
      expect(harness.autobiographicalRepository.currentPeriod()?.start_ts).toBe(successorStartTs);
    } finally {
      await harness.cleanup();
    }
  });

  it("uses audience-scoped episodes as labeled global growth-marker evidence", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          observation: {
            category: "understanding",
            what_changed: "Public planning became more explicit.",
            before_description: "Planning evidence was scattered.",
            after_description:
              "Public and labeled private evidence now support the planning pattern.",
            confidence: 0.9,
            evidence_episode_ids: [
              "ep_aaaaaaaaaaaaaaaa",
              "ep_bbbbbbbbbbbbbbbb",
              "ep_cccccccccccccccc",
            ],
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Public planning start",
          narrative: "Public planning started with clearer goals.",
          tags: ["planning"],
        }),
      );
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Public planning follow-up",
          narrative: "Public planning follow-up reinforced the goals.",
          tags: ["planning"],
          created_at: 2_000_000,
          updated_at: 2_000_000,
        }),
      );
      await harness.episodicRepository.createEpisode(
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

      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

      expect(prompt).toContain("Sam private planning");
      expect(prompt).toContain("relationship_private");
      expect(prompt).toContain(sam);
      expect(marker?.evidence_episode_ids).toEqual([
        "ep_aaaaaaaaaaaaaaaa",
        "ep_bbbbbbbbbbbbbbbb",
        "ep_cccccccccccccccc",
      ]);
      expect(marker?.disclosure_label).toMatchObject({
        disclosureClass: "relationship_private",
        privateToEntityIds: [sam],
      });
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          tags: ["atlas"],
        }),
      );
      await harness.episodicRepository.createEpisode(
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

  it("uses composed period narrative when the model emits one", async () => {
    const composedNarrative =
      "I have been learning to treat deployment review as an ongoing practice rather than a one-off checklist, carrying the earlier planning period into more concrete operational judgment.";
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          period_decision: "continue_current",
          period_narrative: composedNarrative,
          observation: {
            category: "understanding",
            theme: "deployment review",
            what_changed: "Deployment review became a continuous practice.",
            before_description: "Deployment review felt episodic.",
            after_description: "Deployment review now connects planning and operations.",
            confidence: 0.8,
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
      const current = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q1",
        start_ts: Date.UTC(2026, 0, 1),
        narrative: "Planning quarter.",
        themes: ["planning"],
        provenance: { kind: "offline", process: "self-narrator" },
      });
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Deploy rehearsal",
          narrative: "Deployment review became concrete.",
          tags: ["deploy"],
          created_at: Date.UTC(2026, 6, 10),
          updated_at: Date.UTC(2026, 6, 10),
        }),
      );
      await harness.episodicRepository.createEpisode(
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
      const update = plan.items.find((item) => item.action === "update_period_narrative");

      expect(update).toMatchObject({
        action: "update_period_narrative",
        period_id: current.id,
        narrative: composedNarrative,
      });
      expect(String(llm.requests[0]?.messages[0]?.content ?? "")).toContain(
        "I compose it in first-person prose",
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("leads a full period's revision evidence with the episodes this pass cited", async () => {
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          period_decision: "continue_current",
          observation: {
            category: "understanding",
            what_changed: "Deployment review became a continuous practice.",
            before_description: "Deployment review felt episodic.",
            after_description: "Deployment review now connects planning and operations.",
            confidence: 0.8,
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
      const priorKeyEpisodeIds = Array.from({ length: 8 }, () => createEpisodeId());
      const current = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q1",
        start_ts: Date.UTC(2026, 0, 1),
        narrative: "Planning quarter.",
        key_episode_ids: priorKeyEpisodeIds,
        themes: ["planning"],
        provenance: { kind: "offline", process: "self-narrator" },
      });
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Deploy rehearsal",
          narrative: "Deployment review became concrete.",
          tags: ["deploy"],
          created_at: Date.UTC(2026, 6, 10),
          updated_at: Date.UTC(2026, 6, 10),
        }),
      );
      await harness.episodicRepository.createEpisode(
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
      await process.apply(harness.createContext(), plan);

      expect(harness.reviewQueueRepository.getOpen()[0]?.refs).toMatchObject({
        target_id: current.id,
        patch: { key_episode_ids: priorKeyEpisodeIds },
        evidence_episode_ids: [
          "ep_aaaaaaaaaaaaaaaa",
          "ep_bbbbbbbbbbbbbbbb",
          ...priorKeyEpisodeIds.slice(0, 6),
        ],
      });
    } finally {
      await harness.cleanup();
    }
  });

  it("dismisses an older revision when its sibling applies", async () => {
    const observation = {
      category: "understanding",
      what_changed: "Deployment review became a continuous practice.",
      before_description: "Deployment review felt episodic.",
      after_description: "Deployment review now connects planning and operations.",
      confidence: 0.8,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
    };
    const olderText = "Planning quarter; deployment review started.";
    const newerText = "Planning quarter; deployment review became continuous.";
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          period_decision: "continue_current",
          observation,
          period_narrative: olderText,
        }),
        createSelfNarratorResponse({
          period_decision: "continue_current",
          observation,
          period_narrative: newerText,
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
      const period = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q1",
        start_ts: Date.UTC(2026, 0, 1),
        narrative: "Planning quarter.",
        themes: ["planning"],
        provenance: { kind: "offline", process: "self-narrator" },
      });
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          title: "Deploy rehearsal",
          narrative: "Deployment review became concrete.",
          tags: ["deploy"],
          created_at: Date.UTC(2026, 6, 10),
          updated_at: Date.UTC(2026, 6, 10),
        }),
      );
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          title: "Deploy follow-up",
          narrative: "Deployment review stayed active.",
          tags: ["deploy"],
          created_at: Date.UTC(2026, 6, 11),
          updated_at: Date.UTC(2026, 6, 11),
        }),
      );

      // Two passes compose against the same text while neither proposal is decided yet.
      for (let pass = 0; pass < 2; pass += 1) {
        const plan = await process.plan(harness.createContext(), {});
        await process.apply(harness.createContext(), plan);
      }

      const proposalFor = (text: string) =>
        harness.reviewQueueRepository
          .getOpen()
          .find((item) => (item.refs.patch as { narrative?: string }).narrative === text);
      const older = proposalFor(olderText);
      const newer = proposalFor(newerText);

      expect(older?.refs).toMatchObject({ expected_record_version: period.record_version });
      expect(newer?.refs).toMatchObject({ expected_record_version: period.record_version });

      const { expected_record_version: _fence, ...unfencedRefs } = older!.refs;
      const unfenced = harness.reviewQueueRepository.enqueue({
        kind: "identity_inconsistency",
        refs: unfencedRefs,
        reason: "independent unfenced patch",
      });
      const rollover = harness.reviewQueueRepository.enqueue({
        kind: "identity_inconsistency",
        refs: { ...older!.refs, next_period_open_payload: period },
        reason: "rollover",
      });
      const future = harness.reviewQueueRepository.enqueue({
        kind: "identity_inconsistency",
        refs: { ...older!.refs, expected_record_version: period.record_version! + 5 },
        reason: "newer version",
      });
      const unrelated = harness.reviewQueueRepository.enqueue({
        kind: "identity_inconsistency",
        refs: { ...older!.refs, target_id: "abp_aaaaaaaaaaaaaaaa" },
        reason: "other target",
      });
      const partial = harness.reviewQueueRepository.enqueue({
        kind: "identity_inconsistency",
        refs: { ...older!.refs, patch: { label: "renamed" } },
        reason: "non-narrative patch",
      });
      // A failed CAS must not dismiss any siblings.
      await expect(harness.reviewQueueRepository.resolve(future.id, "accept")).rejects.toThrow();
      expect(harness.reviewQueueRepository.get(older!.id)?.resolved_at).toBeNull();
      await harness.reviewQueueRepository.resolve(newer!.id, "accept");
      expect(await harness.reviewQueueRepository.resolve(older!.id, "accept")).toMatchObject({
        resolution: "dismiss",
      });

      expect(harness.autobiographicalRepository.getPeriod(period.id)).toMatchObject({
        narrative: newerText,
        record_version: period.record_version! + 1,
      });
      expect(harness.reviewQueueRepository.get(older!.id)).toMatchObject({
        resolution: "dismiss",
        refs: { superseded_by_review_id: newer!.id },
      });
      for (const untouched of [unfenced, rollover, future, unrelated, partial]) {
        expect(harness.reviewQueueRepository.get(untouched.id)?.resolved_at).toBeNull();
      }
    } finally {
      await harness.cleanup();
    }
  });

  it("bounds the prompt to the newest episodes and says how many were omitted", async () => {
    const llm = new FakeLLMClient({
      responses: [createSelfNarratorResponse({ observation: null })],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      clock: { now: () => Date.UTC(2026, 3, 10) },
      configOverrides: { offline: { selfNarrator: { maxEpisodesPerRun: 3 } } },
    });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      for (const day of [1, 2, 3, 4, 5]) {
        const at = Date.UTC(2026, 3, day);
        await harness.episodicRepository.createEpisode(
          createEpisodeFixture({
            id: `ep_day${day}day${day}day${day}day${day}`.slice(0, 19) as never,
            title: `Day ${day}`,
            narrative: `Something happened on day ${day}.`,
            start_time: at,
            end_time: at,
            created_at: at,
            updated_at: at,
          }),
        );
      }

      await process.plan(harness.createContext(), {});

      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
      // Newest three are present, oldest two are not.
      expect(prompt).toContain('"title":"Day 5"');
      expect(prompt).toContain('"title":"Day 4"');
      expect(prompt).toContain('"title":"Day 3"');
      expect(prompt).not.toContain('"title":"Day 2"');
      expect(prompt).not.toContain('"title":"Day 1"');
      // Kept episodes stay in chronological order.
      expect(prompt.indexOf('"title":"Day 3"')).toBeLessThan(prompt.indexOf('"title":"Day 5"'));
      expect(prompt).toContain(
        "2 earlier episodes of this period are not shown due to maxEpisodesPerRun.",
      );
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
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_aaaaaaaaaaaaaaaa" as never,
          tags: ["atlas"],
        }),
      );
      await harness.episodicRepository.createEpisode(
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

  it("keeps valid observations after an invalid one and counts each rejection", async () => {
    const valid = (what_changed: string) => ({
      theme: "understanding",
      category: "understanding",
      what_changed,
      before_description: null,
      after_description: null,
      confidence: 0.7,
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_bbbbbbbbbbbbbbbb"],
    });
    const response = createSelfNarratorResponse({ observation: valid("First valid change.") });
    const outside = (what_changed: string) => ({
      ...valid(what_changed),
      evidence_episode_ids: ["ep_aaaaaaaaaaaaaaaa", "ep_zzzzzzzzzzzzzzzz"],
    });
    response.tool_calls[0]!.input.observations.push(
      outside("First invalid change."),
      valid("Second valid change."),
      outside("Second invalid change."),
    );
    const llm = new FakeLLMClient({ responses: [response] });
    const harness = await createOfflineTestHarness({ llmClient: llm });
    const process = new SelfNarratorProcess({
      autobiographicalRepository: harness.autobiographicalRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({ id: "ep_aaaaaaaaaaaaaaaa" as never }),
      );
      await harness.episodicRepository.createEpisode(
        createEpisodeFixture({
          id: "ep_bbbbbbbbbbbbbbbb" as never,
          created_at: 2_000_000,
          updated_at: 2_000_000,
        }),
      );

      const plan = await process.plan(harness.createContext(), {});

      expect(
        plan.items.flatMap((item) =>
          item.action === "add_growth_marker" ? [item.marker.what_changed] : [],
        ),
      ).toEqual(["First valid change.", "Second valid change."]);
      const rejection = expect.objectContaining({
        code: "SELF_NARRATOR_INVALID_REF",
        message: expect.stringContaining("outside candidates"),
      });
      expect(plan.errors).toEqual([rejection, rejection]);
    } finally {
      await harness.cleanup();
    }
  });
});
