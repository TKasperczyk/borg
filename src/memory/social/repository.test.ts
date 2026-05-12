import { afterEach, describe, expect, it, vi } from "vitest";

import { createOfflineTestHarness } from "../../offline/test-support.js";
import { IdentityCasMismatchError, ProvenanceError } from "../../util/errors.js";

describe("SocialRepository", () => {
  const manualProvenance = { kind: "manual" } as const;

  let harness: Awaited<ReturnType<typeof createOfflineTestHarness>> | undefined;

  afterEach(async () => {
    vi.restoreAllMocks();
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
      harness!.socialRepository.recordInteraction(entityId, {
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
      harness!.socialRepository.recordInteraction(entityId, {
        provenance: undefined as never,
      }),
    ).toThrow(ProvenanceError);
    expect(() => harness!.socialRepository.adjustTrust(entityId, 0.2, undefined as never)).toThrow(
      ProvenanceError,
    );
  });

  it("attaches lagged sentiment without incrementing interaction count", async () => {
    harness = await createOfflineTestHarness();
    const entityId = harness.entityRepository.resolve("Sam");

    const recorded = harness.socialRepository.recordInteractionWithId(entityId, {
      now: 1_000,
      provenance: manualProvenance,
    });
    const attached = harness.socialRepository.attachSentiment(recorded.interaction_id, {
      valence: -0.6,
      now: 2_000,
    });

    expect(attached.interaction_count).toBe(1);
    expect(attached.last_interaction_at).toBe(1_000);
    expect(attached.sentiment_history).toEqual([
      {
        ts: 1_000,
        valence: -0.6,
      },
    ]);
    expect(harness.socialRepository.listEvents(entityId)).toEqual([
      expect.objectContaining({
        id: recorded.interaction_id,
        interaction_delta: 1,
        valence: -0.6,
        ts: 1_000,
      }),
    ]);
  });

  it("uses atomic SQL for interaction increments and event-derived sentiment", async () => {
    harness = await createOfflineTestHarness();
    const entityId = harness.entityRepository.resolve("Sam");
    const preparedSql: string[] = [];
    const prepare = harness.db.prepare.bind(harness.db);
    vi.spyOn(harness.db, "prepare").mockImplementation((statement: string) => {
      preparedSql.push(statement);
      return prepare(statement);
    });

    const first = harness.socialRepository.recordInteractionWithId(entityId, {
      valence: 0.4,
      now: 1_000,
      provenance: manualProvenance,
    });
    const second = harness.socialRepository.recordInteractionWithId(entityId, {
      valence: -0.6,
      now: 1_001,
      provenance: manualProvenance,
    });

    expect(second.profile.interaction_count).toBe(2);
    expect(second.profile.last_interaction_at).toBe(1_001);
    expect(second.profile.sentiment_history).toEqual([
      {
        ts: 1_000,
        valence: 0.4,
      },
      {
        ts: 1_001,
        valence: -0.6,
      },
    ]);
    expect(harness.socialRepository.listEvents(entityId)).toEqual([
      expect.objectContaining({
        id: second.interaction_id,
        valence: -0.6,
        interaction_delta: 1,
      }),
      expect.objectContaining({
        id: first.interaction_id,
        valence: 0.4,
        interaction_delta: 1,
      }),
    ]);

    const normalizedSql = preparedSql.map((statement) => statement.replace(/\s+/g, " ").trim());
    expect(
      normalizedSql.some((statement) =>
        statement.includes("interaction_count = interaction_count + 1"),
      ),
    ).toBe(true);
    expect(
      normalizedSql.some(
        (statement) =>
          statement.includes("FROM social_events") &&
          statement.includes("valence IS NOT NULL") &&
          statement.includes("LIMIT 50"),
      ),
    ).toBe(true);
  });

  it("rejects stale full-profile restores with CAS mismatch", async () => {
    harness = await createOfflineTestHarness();
    const entityId = harness.entityRepository.resolve("Sam");
    const stale = harness.socialRepository.upsertProfile(entityId);

    harness.socialRepository.recordInteractionWithId(entityId, {
      now: 1_000,
      provenance: manualProvenance,
    });

    expect(() =>
      harness!.socialRepository.restoreProfile({
        ...stale,
        notes: "stale restore should not win",
      }),
    ).toThrow(IdentityCasMismatchError);

    const current = harness.socialRepository.getProfile(entityId);
    expect(current).toMatchObject({
      interaction_count: 1,
      notes: null,
    });
  });
});
