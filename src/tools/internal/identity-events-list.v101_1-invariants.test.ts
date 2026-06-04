import { describe, expect, it } from "vitest";

import { ScheduledWakesRepository } from "../../autonomy/index.js";
import { buildToolDispatcher } from "../../borg/tools-setup.js";
import { SemanticGraph } from "../../memory/semantic/index.js";
import { createEpisodeFixture, createOfflineTestHarness } from "../../offline/test-support.js";
import { StreamWriter } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import { DEFAULT_SESSION_ID } from "../../util/ids.js";

function createHarnessToolDispatcher(
  harness: Awaited<ReturnType<typeof createOfflineTestHarness>>,
) {
  const clock = new ManualClock(1_000_100);
  const semanticGraph = new SemanticGraph({
    nodeRepository: harness.semanticNodeRepository,
    edgeRepository: harness.semanticEdgeRepository,
  });

  return buildToolDispatcher({
    retrievalPipeline: harness.retrievalPipeline,
    episodicRepository: harness.episodicRepository,
    semanticNodeRepository: harness.semanticNodeRepository,
    semanticGraph,
    commitmentRepository: harness.commitmentRepository,
    identityService: harness.identityService,
    skillRepository: harness.skillRepository,
    scheduledWakesRepository: new ScheduledWakesRepository({ db: harness.db, clock }),
    createStreamWriter: (sessionId) =>
      new StreamWriter({
        dataDir: harness.tempDir,
        sessionId,
        clock,
      }),
    clock,
  });
}

describe("v101.1 identity-events cognition-tool invariants", () => {
  // v101.1 Sprint D expected flip: internal model tools return global evidence with labels.
  it.fails("returns a private identity event to Sol cognition with a disclosure label", async () => {
    const harness = await createOfflineTestHarness();
    const alice = "ent_aaaaaaaaaaaaaaaa" as never;
    const carol = "ent_cccccccccccccccc" as never;
    const episode = createEpisodeFixture({
      id: "ep_eeeeeeeeeeeeeeee" as never,
      title: "Alice private identity event source",
      audience_entity_id: alice,
      origin_audience_entity_ids: [alice],
      shared: false,
    });

    try {
      harness.identityEventRepository.record({
        record_type: "episode",
        record_id: episode.id,
        action: "correction_apply",
        old_value: null,
        new_value: {
          id: episode.id,
          title: episode.title,
          audience_entity_id: episode.audience_entity_id ?? null,
          origin_audience_entity_ids: episode.origin_audience_entity_ids ?? [],
          shared: episode.shared ?? false,
        },
        reason: "v101.1 invariant fixture",
        provenance: { kind: "manual" },
      });

      const dispatcher = createHarnessToolDispatcher(harness);
      const result = await dispatcher.dispatch({
        toolName: "tool.identityEvents.list",
        input: {
          recordType: "episode",
          limit: 10,
        },
        origin: "deliberator",
        sessionId: DEFAULT_SESSION_ID,
        audienceEntityId: carol,
      });

      expect(result.ok).toBe(true);
      if (!result.ok) {
        throw new Error(result.error);
      }
      const events = (result.output as { events: Array<{ record_id: string }> }).events;
      const returnedEvent = events.find((event) => event.record_id === episode.id);
      const eventJson = JSON.stringify(returnedEvent ?? {});

      expect(returnedEvent).toBeDefined();
      expect(eventJson).toContain("relationship_private");
      expect(eventJson).toContain(alice);
    } finally {
      await harness.cleanup();
    }
  });
});
