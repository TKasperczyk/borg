import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
} from "../test-support.js";
import { ReflectorProcess } from "./index.js";

const REFLECTOR_TOOL_NAME = "EmitReflectorInsights";

function createReflectorResponse(input: {
  sourceEpisodeIds: string[];
}) {
  return {
    text: "",
    input_tokens: 18,
    output_tokens: 12,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_v101_1_reflector",
        name: REFLECTOR_TOOL_NAME,
        input: {
          label: "One cross-audience pattern insight",
          description:
            "Public, Alice-private, and Bob-private evidence jointly support one pattern.",
          confidence: 0.7,
          source_episode_ids: input.sourceEpisodeIds,
        },
      },
    ],
  };
}

describe("v101.1 cross-audience synthesis invariants", () => {
  it("synthesizes public, Alice-private, and Bob-private episodes into one labeled insight", async () => {
    const alice = "ent_aaaaaaaaaaaaaaaa" as never;
    const bob = "ent_bbbbbbbbbbbbbbbb" as never;
    const episodes = [
      createEpisodeFixture(
        {
          id: "ep_ffffffffffffffff" as never,
          title: "Public pattern evidence",
          narrative: "Public evidence shows the recurring pattern.",
          tags: ["cross-audience-pattern"],
          audience_entity_id: null,
          shared: true,
          created_at: 10_000,
          updated_at: 10_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_1111111111111111" as never,
          title: "Alice private pattern evidence",
          narrative: "Alice-private evidence supports the same recurring pattern.",
          tags: ["cross-audience-pattern"],
          audience_entity_id: alice,
          shared: false,
          created_at: 20_000,
          updated_at: 20_000,
        },
        [1, 0, 0, 0],
      ),
      createEpisodeFixture(
        {
          id: "ep_2222222222222222" as never,
          title: "Bob private pattern evidence",
          narrative: "Bob-private evidence supports the same recurring pattern.",
          tags: ["cross-audience-pattern"],
          audience_entity_id: bob,
          shared: false,
          created_at: 30_000,
          updated_at: 30_000,
        },
        [1, 0, 0, 0],
      ),
    ];
    const llm = new FakeLLMClient({
      responses: [
        createReflectorResponse({
          sourceEpisodeIds: episodes.map((episode) => episode.id),
        }),
      ],
    });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(
        new Map([
          ["cross-audience-pattern", [1, 0, 0, 0]],
          ["One cross-audience pattern insight", [1, 0, 0, 0]],
          [
            "One cross-audience pattern insight\nPublic, Alice-private, and Bob-private evidence jointly support one pattern.",
            [1, 0, 0, 0],
          ],
        ]),
      ),
    });
    const process = new ReflectorProcess({
      semanticNodeRepository: harness.semanticNodeRepository,
      semanticEdgeRepository: harness.semanticEdgeRepository,
      reviewQueueRepository: harness.reviewQueueRepository,
      registry: harness.registry,
      clock: harness.clock,
    });

    try {
      for (const episode of episodes) {
        await harness.episodicRepository.insert(episode);
      }

      const plan = await process.plan(harness.createContext(), {});
      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");
      const itemJson = JSON.stringify(plan.items[0] ?? {});

      expect(llm.requests).toHaveLength(1);
      expect(prompt).toContain("Public pattern evidence");
      expect(prompt).toContain("Alice private pattern evidence");
      expect(prompt).toContain("Bob private pattern evidence");
      expect(prompt).toContain("relationship_private");
      expect(prompt).toContain(alice);
      expect(prompt).toContain(bob);
      expect(plan.items).toHaveLength(1);
      expect(plan.items[0]?.episode_ids).toHaveLength(episodes.length);
      expect(plan.items[0]?.episode_ids).toEqual(
        expect.arrayContaining(episodes.map((episode) => episode.id)),
      );
      expect(itemJson).toContain("relationship_private");
      expect(itemJson).toContain(alice);
      expect(itemJson).toContain(bob);
    } finally {
      await harness.cleanup();
    }
  });
});
