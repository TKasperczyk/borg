import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";
import {
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../test-support.js";
import { SelfNarratorProcess } from "./index.js";

const SELF_NARRATOR_TOOL_NAME = "EmitSelfNarratorObservations";

function createSelfNarratorResponse(input: {
  evidenceEpisodeIds: string[];
  periodDecision?: "continue_current" | "open_new";
  theme?: string;
  category?: "understanding" | "skill" | "relationship" | "value" | "habit";
  whatChanged?: string;
  beforeDescription?: string | null;
  afterDescription?: string | null;
  periodNarrative?: string | null;
}) {
  return {
    text: "",
    input_tokens: 40,
    output_tokens: 30,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_v101_1_self_narrator",
        name: SELF_NARRATOR_TOOL_NAME,
        input: {
          observations: [
            {
              theme: input.theme ?? "private continuity pattern",
              category: input.category ?? "understanding",
              what_changed:
                input.whatChanged ?? "Private evidence influenced the autobiographical narrative.",
              before_description: input.beforeDescription ?? null,
              after_description:
                input.afterDescription ??
                "The self-narrative incorporates labeled private evidence.",
              confidence: 0.8,
              evidence_episode_ids: input.evidenceEpisodeIds,
            },
          ],
          period_narrative: input.periodNarrative ?? null,
          period_decision: input.periodDecision ?? "open_new",
          period_decision_confidence: 0.8,
        },
      },
    ],
  };
}

describe("v101.1 offline self-narrator cognition invariants", () => {
  it("lets Alice-private evidence influence self-narration with disclosure labels", async () => {
    const aliceEpisodes = [
      createEpisodeFixture({
        id: "ep_aaaaaaaaaaaaaaaa" as never,
        title: "Alice private self-narration evidence one",
        narrative: "Alice-only evidence describes a recurring self-understanding pattern.",
        participants: ["Alice"],
        tags: ["private-continuity"],
        audience_entity_id: "ent_aaaaaaaaaaaaaaaa" as never,
        shared: false,
        created_at: 10_000,
        updated_at: 10_000,
      }),
      createEpisodeFixture({
        id: "ep_bbbbbbbbbbbbbbbb" as never,
        title: "Alice private self-narration evidence two",
        narrative: "Alice-only evidence gives a second support point for the same pattern.",
        participants: ["Alice"],
        tags: ["private-continuity"],
        audience_entity_id: "ent_aaaaaaaaaaaaaaaa" as never,
        shared: false,
        created_at: 20_000,
        updated_at: 20_000,
      }),
    ];
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          evidenceEpisodeIds: aliceEpisodes.map((episode) => episode.id),
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
      for (const episode of aliceEpisodes) {
        await harness.episodicRepository.createEpisode(episode);
      }

      const plan = await process.plan(harness.createContext(), {});
      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

      expect(llm.requests).toHaveLength(1);
      expect(prompt).toContain("Alice private self-narration evidence one");
      expect(prompt).toContain("Alice private self-narration evidence two");
      expect(prompt).toContain("relationship_private");
      expect(prompt).toContain("ent_aaaaaaaaaaaaaaaa");
      expect(plan.items).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            action: "add_growth_marker",
            marker: expect.objectContaining({
              evidence_episode_ids: aliceEpisodes.map((episode) => episode.id),
              disclosure_label: expect.objectContaining({
                disclosureClass: "relationship_private",
                privateToEntityIds: ["ent_aaaaaaaaaaaaaaaa"],
              }),
            }),
          }),
        ]),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("labels a period from all contributing markers even when private evidence is outside display keys", async () => {
    const alice = "ent_aaaaaaaaaaaaaaaa" as never;
    const publicEpisodes = Array.from({ length: 8 }, (_, index) =>
      createEpisodeFixture({
        id: `ep_publicperiod000${index + 1}` as never,
        title: `Public period evidence ${index + 1}`,
        narrative: `Public evidence ${index + 1} supports the period narrative.`,
        shared: true,
        created_at: 10_000 + index,
        updated_at: 10_000 + index,
      }),
    );
    const privateEpisode = createEpisodeFixture({
      id: "ep_privateperiod001" as never,
      title: "Alice private period evidence",
      narrative: "Alice-private evidence also contributes to the period narrative.",
      audience_entity_id: alice,
      shared: false,
      created_at: 20_000,
      updated_at: 20_000,
    });
    const allEpisodes = [...publicEpisodes, privateEpisode];
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          evidenceEpisodeIds: allEpisodes.map((episode) => episode.id),
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
      for (const episode of allEpisodes) {
        await harness.episodicRepository.createEpisode(episode);
      }

      const plan = await process.plan(harness.createContext(), {});
      const openedPeriod = plan.items.find((item) => item.action === "open_period");

      expect(openedPeriod).toMatchObject({
        action: "open_period",
        period: expect.objectContaining({
          key_episode_ids: publicEpisodes.map((episode) => episode.id),
          disclosure_label: expect.objectContaining({
            disclosureClass: "relationship_private",
            privateToEntityIds: [alice],
          }),
        }),
      });
      if (openedPeriod?.action === "open_period") {
        expect(openedPeriod.period.key_episode_ids).not.toContain(privateEpisode.id);
      }
    } finally {
      await harness.cleanup();
    }
  });

  it("does not demote an existing private period when public markers update it", async () => {
    const alice = "ent_aaaaaaaaaaaaaaaa" as never;
    const privateLabel = relationshipPrivateMemoryDisclosureLabel([alice]);
    const publicEpisodes = [
      createEpisodeFixture({
        id: "ep_publicupdate0001" as never,
        title: "Public update evidence one",
        narrative: "Public evidence adds context to the period.",
        shared: true,
        created_at: 30_000,
        updated_at: 30_000,
      }),
      createEpisodeFixture({
        id: "ep_publicupdate0002" as never,
        title: "Public update evidence two",
        narrative: "More public evidence adds context to the period.",
        shared: true,
        created_at: 31_000,
        updated_at: 31_000,
      }),
    ];
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          evidenceEpisodeIds: publicEpisodes.map((episode) => episode.id),
          periodDecision: "continue_current",
          theme: "understanding",
          category: "understanding",
          whatChanged: "Public evidence extended the period narrative.",
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
      const currentPeriod = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q1",
        start_ts: 1_000,
        narrative: "Private baseline narrative.",
        themes: ["understanding"],
        disclosure_label: privateLabel,
        provenance: { kind: "offline", process: "self-narrator" },
      });
      for (const episode of publicEpisodes) {
        await harness.episodicRepository.createEpisode(episode);
      }

      const plan = await process.plan(harness.createContext(), {});
      const update = plan.items.find((item) => item.action === "update_period_narrative");

      expect(update).toMatchObject({
        action: "update_period_narrative",
        period_id: currentPeriod.id,
        disclosure_label: expect.objectContaining({
          disclosureClass: "relationship_private",
          privateToEntityIds: [alice],
        }),
      });

      await process.apply(harness.createContext(), plan);

      const reviewItem = harness.reviewQueueRepository.getOpen()[0];
      expect(harness.autobiographicalRepository.currentPeriod()?.disclosure_label).toMatchObject({
        disclosureClass: "relationship_private",
        privateToEntityIds: [alice],
      });
      expect(reviewItem).toEqual(
        expect.objectContaining({
          kind: "identity_inconsistency",
          refs: expect.objectContaining({
            target_type: "autobiographical_period",
            target_id: currentPeriod.id,
            patch: expect.objectContaining({
              disclosure_label: expect.objectContaining({
                disclosureClass: "relationship_private",
                privateToEntityIds: [alice],
              }),
            }),
          }),
        }),
      );
    } finally {
      await harness.cleanup();
    }
  });

  it("persists a label-only private period upgrade for review", async () => {
    const alice = "ent_aaaaaaaaaaaaaaaa" as never;
    const privateEpisodes = [
      createEpisodeFixture({
        id: "ep_privateupdate001" as never,
        title: "Private label evidence one",
        narrative: "Private evidence supports the unchanged period narrative.",
        audience_entity_id: alice,
        shared: false,
        created_at: 40_000,
        updated_at: 40_000,
      }),
      createEpisodeFixture({
        id: "ep_privateupdate002" as never,
        title: "Private label evidence two",
        narrative: "More private evidence supports the unchanged period narrative.",
        audience_entity_id: alice,
        shared: false,
        created_at: 41_000,
        updated_at: 41_000,
      }),
    ];
    const llm = new FakeLLMClient({
      responses: [
        createSelfNarratorResponse({
          evidenceEpisodeIds: privateEpisodes.map((episode) => episode.id),
          periodDecision: "continue_current",
          theme: "understanding",
          category: "understanding",
          whatChanged: "Stable period narrative.",
          afterDescription: "Stable period narrative.",
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
      const currentPeriod = harness.autobiographicalRepository.upsertPeriod({
        label: "2026-Q1",
        start_ts: 1_000,
        narrative: "Stable period narrative.",
        key_episode_ids: privateEpisodes.map((episode) => episode.id),
        themes: ["understanding"],
        disclosure_label: publicMemoryDisclosureLabel(),
        provenance: { kind: "offline", process: "self-narrator" },
      });
      for (const episode of privateEpisodes) {
        await harness.episodicRepository.createEpisode(episode);
      }

      const plan = await process.plan(harness.createContext(), {});
      const update = plan.items.find((item) => item.action === "update_period_narrative");

      expect(update).toMatchObject({
        action: "update_period_narrative",
        period_id: currentPeriod.id,
        narrative: "Stable period narrative.",
        key_episode_ids: privateEpisodes.map((episode) => episode.id),
        themes: ["understanding"],
        disclosure_label: expect.objectContaining({
          disclosureClass: "relationship_private",
          privateToEntityIds: [alice],
        }),
      });

      await process.apply(harness.createContext(), plan);

      const reviewItem = harness.reviewQueueRepository.getOpen()[0];
      expect(reviewItem).toEqual(
        expect.objectContaining({
          kind: "identity_inconsistency",
          refs: expect.objectContaining({
            target_type: "autobiographical_period",
            target_id: currentPeriod.id,
            patch: expect.objectContaining({
              disclosure_label: expect.objectContaining({
                disclosureClass: "relationship_private",
                privateToEntityIds: [alice],
              }),
            }),
          }),
        }),
      );
    } finally {
      await harness.cleanup();
    }
  });
});
