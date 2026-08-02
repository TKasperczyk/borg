import { describe, expect, it, vi } from "vitest";

import { createOfflineTestHarness, TestEmbeddingClient } from "../offline/test-support.js";
import { FixedClock } from "../util/clock.js";

const QUERY = "architecture";
const NOW_MS = 10_000_000_000;

// Pins the RetrievalProjection contract (see pipeline.ts): the episodes-only
// entry must never pay for the context lanes it provably cannot surface, and
// the full-context entries must keep running them. If a lane ever starts
// feeding projectEpisodes, this test is the tripwire that forces the skip to
// be re-justified.
describe("RetrievalProjection episodes-only", () => {
  async function createHarness() {
    return createOfflineTestHarness({
      clock: new FixedClock(NOW_MS),
      embeddingClient: new TestEmbeddingClient(new Map([[QUERY, [1, 0, 0, 0]]])),
    });
  }

  function laneSpies(pipeline: object) {
    const target = pipeline as Record<string, (...args: never[]) => unknown>;
    return {
      semantic: vi.spyOn(target, "collectSemanticRetrievals"),
      openQuestions: vi.spyOn(target, "collectOpenQuestions"),
      imagePerception: vi.spyOn(target, "collectImagePerceptionEvidenceWithDisclosureMode"),
      commitments: vi.spyOn(target, "collectCommitmentEvidence"),
    };
  }

  it("searchEpisodesForDisclosure skips every context lane", async () => {
    const harness = await createHarness();
    const spies = laneSpies(harness.retrievalPipeline);

    await harness.retrievalPipeline.searchEpisodesForDisclosure(QUERY, { limit: 3 });

    expect(spies.semantic).not.toHaveBeenCalled();
    expect(spies.openQuestions).not.toHaveBeenCalled();
    expect(spies.imagePerception).not.toHaveBeenCalled();
    expect(spies.commitments).not.toHaveBeenCalled();
  });

  it("searchWithContextForDisclosure still runs the full pipeline", async () => {
    const harness = await createHarness();
    const spies = laneSpies(harness.retrievalPipeline);

    await harness.retrievalPipeline.searchWithContextForDisclosure(QUERY, { limit: 3 });

    expect(spies.semantic).toHaveBeenCalled();
    expect(spies.openQuestions).toHaveBeenCalled();
    expect(spies.imagePerception).toHaveBeenCalled();
    expect(spies.commitments).toHaveBeenCalled();
  });
});
