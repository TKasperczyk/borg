import { describe, expect, it } from "vitest";

import { FakeLLMClient } from "../../llm/test-support/fake-client.js";

import {
  createEpisodeFixture,
  createOfflineTestHarness,
  TestEmbeddingClient,
} from "../test-support.js";
import { RuminatorProcess } from "./index.js";

const RUMINATOR_TOOL_NAME = "EmitRuminatorDecisions";

function createRuminatorResponse() {
  return {
    text: "",
    input_tokens: 50,
    output_tokens: 40,
    stop_reason: "tool_use" as const,
    tool_calls: [
      {
        id: "toolu_v101_1_ruminator",
        name: RUMINATOR_TOOL_NAME,
        input: {
          outcome: "resolved",
          resolution_note: "Bob-private evidence resolved the self/global open question.",
          growth_marker: null,
        },
      },
    ],
  };
}

describe("v101.1 ruminator cognition invariants", () => {
  it("uses Bob-private evidence to resolve a global open question with disclosure labels", async () => {
    const questionText = "What resolved the recurring private pattern?";
    const bob = "ent_bbbbbbbbbbbbbbbb" as never;
    const bobEpisode = createEpisodeFixture(
      {
        id: "ep_cccccccccccccccc" as never,
        title: "Bob private resolution evidence",
        narrative: "Bob-only evidence gives the missing answer to the global self question.",
        participants: ["Bob"],
        tags: ["private-resolution"],
        audience_entity_id: bob,
        shared: false,
        significance: 0.95,
        created_at: 2_000_000,
        updated_at: 2_000_000,
      },
      [1, 0, 0, 0],
    );
    const llm = new FakeLLMClient({ responses: [createRuminatorResponse()] });
    const harness = await createOfflineTestHarness({
      llmClient: llm,
      embeddingClient: new TestEmbeddingClient(new Map([[questionText, [1, 0, 0, 0]]])),
      configOverrides: {
        offline: {
          ruminator: {
            resolveConfidenceThreshold: 0.01,
          },
        },
      },
    });
    const process = new RuminatorProcess({
      openQuestionsRepository: harness.openQuestionsRepository,
      growthMarkersRepository: harness.growthMarkersRepository,
      registry: harness.registry,
    });

    try {
      await harness.episodicRepository.createEpisode(bobEpisode);
      const question = harness.openQuestionsRepository.add({
        question: questionText,
        urgency: 0.8,
        source: "reflection",
        created_at: 1_000_000,
        last_touched: 1_000_000,
        provenance: { kind: "manual" },
      });

      const plan = await process.plan(harness.createContext(), {});
      const prompt = String(llm.requests[0]?.messages[0]?.content ?? "");

      expect(llm.requests).toHaveLength(1);
      expect(prompt).toContain("Bob private resolution evidence");
      expect(prompt).toContain("relationship_private");
      expect(prompt).toContain(bob);
      expect(plan.items).toEqual([
        expect.objectContaining({
          action: "resolve",
          question_id: question.id,
          resolution_evidence_episode_ids: [bobEpisode.id],
          resolution_disclosure_label: expect.objectContaining({
            disclosureClass: "relationship_private",
            privateToEntityIds: [bob],
          }),
        }),
      ]);
    } finally {
      await harness.cleanup();
    }
  });
});
