import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../test-support.js";
import { SelfNarratorProcess } from "./index.js";

const SELF_NARRATOR_TOOL_NAME = "EmitSelfNarratorObservations";

function createSelfNarratorResponse(input: { evidenceEpisodeIds: string[] }) {
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
              theme: "private continuity pattern",
              category: "understanding",
              what_changed: "Private evidence influenced the autobiographical narrative.",
              before_description: null,
              after_description: "The self-narrative incorporates labeled private evidence.",
              confidence: 0.8,
              evidence_episode_ids: input.evidenceEpisodeIds,
            },
          ],
          period_decision: "open_new",
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
        await harness.episodicRepository.insert(episode);
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
});
