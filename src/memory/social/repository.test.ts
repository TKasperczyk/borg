import { afterEach, describe, expect, it } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { ProvenanceError } from "../../util/errors.js";

describe("SocialRepository", () => {
  const manualProvenance = { kind: "manual" } as const;

  let harness: Awaited<ReturnType<typeof createOfflineTestHarness>> | undefined;

  afterEach(async () => {
    await harness?.cleanup();
    harness = undefined;
  });

  it("upserts idempotently, records bounded sentiment history, and clamps trust", async () => {
    harness = await createOfflineTestHarness();
    const entityId = harness.entityRepository.resolve("Sam");

    const first = harness.socialRepository.upsertProfile(entityId);
    const second = harness.socialRepository.upsertProfile(entityId);

    expect(first.entity_id).toBe(second.entity_id);
    expect(second.interaction_count).toBe(0);

    for (let index = 0; index < 60; index += 1) {
      harness.socialRepository.recordInteraction(entityId, {
        valence: index % 2 === 0 ? 0.4 : -0.2,
        now: 1_000_000 + index,
        provenance: manualProvenance,
      });
    }

    const recorded = harness.socialRepository.getProfile(entityId);
    expect(recorded?.interaction_count).toBe(60);
    expect(recorded?.sentiment_history).toHaveLength(50);

    const trusted = harness.socialRepository.adjustTrust(entityId, 1, manualProvenance);
    const distrusted = harness.socialRepository.adjustTrust(entityId, -2, manualProvenance);

    expect(trusted.trust).toBe(1);
    expect(distrusted.trust).toBe(0);
    expect(harness.socialRepository.listEvents(entityId)).toHaveLength(62);
  });

  it("rejects provenance-less social mutations", async () => {
    harness = await createOfflineTestHarness();
    const entityId = harness.entityRepository.resolve("Sam");

    expect(() =>
      harness.socialRepository.recordInteraction(entityId, {
        provenance: undefined as never,
      }),
    ).toThrow(ProvenanceError);
    expect(() =>
      harness.socialRepository.adjustTrust(entityId, 0.2, undefined as never),
    ).toThrow(ProvenanceError);
  });
});
