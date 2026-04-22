import { afterEach, describe, expect, it } from "vitest";

import { FixedClock } from "../util/clock.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  type OfflineTestHarness,
} from "../offline/test-support.js";

const NOW_MS = 10_000_000_000;

function preferenceWeights() {
  return {
    semantic: 0.2,
    goal_relevance: 0,
    value_alignment: 0.35,
    mood: 0,
    time: 0,
    social: 0,
    entity: 0,
    heat: 0.1,
    suppression_penalty: 0.5,
  };
}

async function createHarness(): Promise<OfflineTestHarness> {
  return createOfflineTestHarness({
    clock: new FixedClock(NOW_MS),
  });
}

describe("RetrievalPipeline Sprint 8 preference formation", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("rescues value-aligned episodes over vector decoys when active values are present", async () => {
    const harness = await createHarness();
    cleanup.push(harness.cleanup);

    const aligned = createEpisodeFixture(
      {
        title: "Clarity-first postmortem",
        narrative: "Explicit state and careful handoff notes kept the team aligned.",
        tags: ["clarity", "handoff"],
        participants: ["team"],
        significance: 0.3,
        created_at: NOW_MS - 10,
        updated_at: NOW_MS - 10,
      },
      [0, 0, 0, 1],
    );
    await harness.episodicRepository.insert(aligned);

    for (let index = 0; index < 4; index += 1) {
      await harness.episodicRepository.insert(
        createEpisodeFixture(
          {
            title: `Architecture decoy ${index}`,
            narrative: `A strong architecture match ${index} with no preference alignment.`,
            tags: ["architecture"],
            participants: ["team"],
            significance: 0.3,
            created_at: NOW_MS - 100 - index,
            updated_at: NOW_MS - 100 - index,
          },
          [1, 0, 0, 0],
        ),
      );
    }

    const heldValue = harness.valuesRepository.add({
      label: "clarity",
      description: "Prefer explicit state and careful handoffs.",
      priority: 9,
      provenance: { kind: "manual" },
      createdAt: NOW_MS - 50,
    });

    const vectorOnly = await harness.retrievalPipeline.search("architecture", {
      limit: 3,
      attentionWeights: preferenceWeights(),
    });
    const withValues = await harness.retrievalPipeline.search("architecture", {
      limit: 3,
      attentionWeights: preferenceWeights(),
      activeValues: [heldValue],
    });

    expect(vectorOnly[0]?.episode.id).not.toBe(aligned.id);
    expect(withValues[0]?.episode.id).toBe(aligned.id);
    expect(withValues[0]?.scoreBreakdown.valueAlignment ?? 0).toBeGreaterThan(0);
  });
});
