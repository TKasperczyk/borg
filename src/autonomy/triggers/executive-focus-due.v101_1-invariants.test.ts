import { describe, expect, it } from "vitest";

import { StreamWatermarkRepository } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";

import { createEpisodeFixture, createOfflineTestHarness } from "../../offline/test-support.js";

import { createExecutiveFocusDueTrigger } from "./executive-focus-due.js";

describe("v101.1 executive focus cognition invariants", () => {
  // v101.1 Sprint C expected flip: autonomous executive focus can use private grounded goals.
  it("keeps a private-episode-grounded goal eligible for autonomous focus with labels", async () => {
    const clock = new ManualClock(2_000_000_000);
    const harness = await createOfflineTestHarness({ clock });
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const alice = "ent_aaaaaaaaaaaaaaaa" as never;
    const privateEpisode = createEpisodeFixture({
      id: "ep_dddddddddddddddd" as never,
      title: "Alice private goal evidence",
      narrative: "Alice-only evidence is the sole grounding for an autonomous goal.",
      participants: ["Alice"],
      tags: ["autonomous-focus"],
      audience_entity_id: alice,
      shared: false,
      created_at: clock.now() - 120_000_000,
      updated_at: clock.now() - 120_000_000,
    });

    try {
      await harness.episodicRepository.createEpisode(privateEpisode);
      const goal = harness.goalsRepository.add({
        description: "Follow up on the private autonomous focus pattern",
        priority: 10,
        provenance: {
          kind: "episodes",
          episode_ids: [privateEpisode.id],
        },
        createdAt: clock.now() - 90_000_000,
      });
      const step = harness.executiveStepsRepository.add({
        goalId: goal.id,
        description: "Review the Alice-private autonomous focus source",
        kind: "think",
        dueAt: null,
        provenance: { kind: "manual" },
      });
      const trigger = createExecutiveFocusDueTrigger({
        enabled: true,
        goalsRepository: harness.goalsRepository,
        executiveStepsRepository: harness.executiveStepsRepository,
        episodicRepository: harness.episodicRepository,
        embeddingClient: harness.embeddingClient,
        watermarkRepository,
        threshold: 0.45,
        stalenessMs: 86_400_000,
        dueLeadMs: 0,
        wakeCooldownMs: 3_600_000,
        wakeCooldownBackoffMultiplier: 2,
        wakeCooldownMaxMs: 86_400_000,
        deadlineLookaheadMs: 604_800_000,
        goalFollowupDue: {
          enabled: false,
          lookaheadMs: 604_800_000,
          staleMs: 1_209_600_000,
        },
        clock,
      });

      const events = await trigger.scan();
      const payloadJson = JSON.stringify(events[0]?.payload ?? {});

      expect(events).toHaveLength(1);
      expect(events[0]?.payload).toMatchObject({
        reason: "goal_stale",
        selected_goal_id: goal.id,
        selected_goal: {
          source_disclosure_label: {
            disclosure_class: "relationship_private",
            private_to_entity_ids: [alice],
          },
        },
        top_open_step: {
          id: step.id,
          disclosure_label: {
            disclosure_class: "self_private",
            private_to_entity_ids: [alice],
          },
        },
      });
      expect(payloadJson).toContain("relationship_private");
      expect(payloadJson).toContain(alice);
    } finally {
      await harness.cleanup();
    }
  });

  it("keeps unresolved source episodes fail-closed in executive focus disclosure", async () => {
    const clock = new ManualClock(2_000_000_000);
    const harness = await createOfflineTestHarness({ clock });
    const watermarkRepository = new StreamWatermarkRepository({
      db: harness.db,
      clock,
    });
    const alice = "ent_bbbbbbbbbbbbbbbb" as never;
    const privateEpisode = createEpisodeFixture({
      id: "ep_eeeeeeeeeeeeeeee" as never,
      title: "Alice private mixed-source goal evidence",
      narrative: "Resolved private source for a mixed-source autonomous goal.",
      participants: ["Alice"],
      tags: ["autonomous-focus"],
      audience_entity_id: alice,
      shared: false,
      created_at: clock.now() - 120_000_000,
      updated_at: clock.now() - 120_000_000,
    });
    const missingEpisodeId = "ep_ffffffffffffffff" as never;

    try {
      await harness.episodicRepository.createEpisode(privateEpisode);
      const goal = harness.goalsRepository.add({
        description: "Follow up on the mixed-source autonomous focus pattern",
        priority: 10,
        provenance: {
          kind: "episodes",
          episode_ids: [privateEpisode.id, missingEpisodeId],
        },
        createdAt: clock.now() - 90_000_000,
      });
      const trigger = createExecutiveFocusDueTrigger({
        enabled: true,
        goalsRepository: harness.goalsRepository,
        executiveStepsRepository: harness.executiveStepsRepository,
        episodicRepository: harness.episodicRepository,
        embeddingClient: harness.embeddingClient,
        watermarkRepository,
        threshold: 0.45,
        stalenessMs: 86_400_000,
        dueLeadMs: 0,
        wakeCooldownMs: 3_600_000,
        wakeCooldownBackoffMultiplier: 2,
        wakeCooldownMaxMs: 86_400_000,
        deadlineLookaheadMs: 604_800_000,
        goalFollowupDue: {
          enabled: false,
          lookaheadMs: 604_800_000,
          staleMs: 1_209_600_000,
        },
        clock,
      });

      const events = await trigger.scan();

      expect(events).toHaveLength(1);
      expect(events[0]?.payload).toMatchObject({
        reason: "goal_stale",
        selected_goal_id: goal.id,
        selected_goal: {
          source_disclosure: expect.stringContaining("disclosure_class=unknown"),
          source_disclosure_label: {
            disclosure_class: "unknown",
            private_to_entity_ids: [alice],
          },
        },
      });
    } finally {
      await harness.cleanup();
    }
  });
});
