import { afterEach, describe, expect, it } from "vitest";

import { FixedClock, ManualClock } from "../../util/clock.js";
import type { EpisodeId, StreamEntryId } from "../../util/ids.js";
import {
  createEpisodeFixture,
  createOfflineTestHarness,
  createSemanticNodeFixture,
} from "../test-support.js";
import { revalidateReviewQueue } from "./revalidate.js";

type OfflineHarness = Awaited<ReturnType<typeof createOfflineTestHarness>>;

async function appendSourceEntry(harness: OfflineHarness, content: string) {
  return harness.streamWriter.append({
    kind: "user_msg",
    content,
  });
}

function overseerPayload(input: {
  citedStreamIds: readonly StreamEntryId[];
  sourceEpisodeIds?: readonly EpisodeId[];
  sourceStreamIds?: readonly StreamEntryId[];
  quotedSpan: string;
  audienceEntities?: Array<{
    entity_id: string;
    display_name: string;
    source_episode_ids: EpisodeId[];
  }>;
}) {
  return {
    flag_kind: "misattribution",
    kind: "misattribution",
    reason: "The target memory attribution is unsupported by its cited source.",
    confidence: 0.82,
    patch: {
      description: "Corrected description.",
    },
    source_assessment: "supports_flag",
    cited_stream_ids: [...input.citedStreamIds],
    quoted_span: input.quotedSpan,
    audience_entities: input.audienceEntities ?? [],
    source_episode_ids: [...(input.sourceEpisodeIds ?? [])],
    source_stream_ids: [...(input.sourceStreamIds ?? input.citedStreamIds)],
  };
}

describe("review queue revalidation", () => {
  const cleanup: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (cleanup.length > 0) {
      await cleanup.pop()?.();
    }
  });

  it("dismisses audience-name false positives, keeps real flags, and skips legacy rows", async () => {
    const nowMs = 10 * 24 * 60 * 60 * 1_000;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
    });
    cleanup.push(harness.cleanup);

    const audienceEntityId = harness.entityRepository.resolve("Tom", {
      provenance: "transport_audience_label",
    });
    const audienceSource = await appendSourceEntry(harness, "Otto is my dog.");
    const audienceEpisode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture(
        {
          title: "Audience source",
          narrative: "The user said Otto is their dog.",
          source_stream_ids: [audienceSource.id],
          audience_entity_id: audienceEntityId,
          shared: false,
          created_at: nowMs - 3_000,
          updated_at: nowMs - 3_000,
        },
        [1, 0, 0, 0],
      ),
    );
    const audienceNode = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Otto as Tom's dog",
          description: "Otto is Tom's dog.",
          source_episode_ids: [audienceEpisode.id],
          created_at: nowMs - 2_000,
          updated_at: nowMs - 2_000,
          last_verified_at: nowMs - 2_000,
        },
        [0, 1, 0, 0],
      ),
    );

    const realSource = await appendSourceEntry(harness, "Riley led the workshop, not Tom.");
    const realEpisode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture(
        {
          title: "Real misattribution source",
          narrative: "The user said Riley led the workshop.",
          source_stream_ids: [realSource.id],
          created_at: nowMs - 1_500,
          updated_at: nowMs - 1_500,
        },
        [0, 0, 1, 0],
      ),
    );
    const realNode = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Tom led the workshop",
          description: "Tom led the workshop.",
          source_episode_ids: [realEpisode.id],
          created_at: nowMs - 1_000,
          updated_at: nowMs - 1_000,
          last_verified_at: nowMs - 1_000,
        },
        [0, 0, 0, 1],
      ),
    );

    const falsePositive = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs: {
        target_type: "semantic_node",
        target_id: audienceNode.id,
        patch: {
          description: "Corrected description.",
        },
        evidence_stream_ids: [audienceSource.id],
        overseer_flag: overseerPayload({
          citedStreamIds: [audienceSource.id],
          sourceEpisodeIds: [audienceEpisode.id],
          quotedSpan: "Tom",
          audienceEntities: [
            {
              entity_id: audienceEntityId,
              display_name: "Tom",
              source_episode_ids: [audienceEpisode.id],
            },
          ],
        }),
      },
      reason: "The source text does not literally name Tom.",
    });
    const real = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs: {
        target_type: "semantic_node",
        target_id: realNode.id,
        patch: {
          description: "Corrected description.",
        },
        evidence_stream_ids: [realSource.id],
        overseer_flag: overseerPayload({
          citedStreamIds: [realSource.id],
          sourceEpisodeIds: [realEpisode.id],
          quotedSpan: "Tom",
        }),
      },
      reason: "The target names Tom, but the source supports Riley.",
    });
    const legacy = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs: {
        target_type: "semantic_node",
        target_id: realNode.id,
        patch: {
          description: "Corrected description.",
        },
        evidence_stream_ids: [realSource.id],
      },
      reason: "Legacy row without full overseer payload.",
    });

    const result = await revalidateReviewQueue(harness.createContext(), {
      kind: "misattribution",
    });

    expect(result).toMatchObject({
      revalidated: 2,
      dismissed_as_suppressed: 1,
      skipped_legacy: 1,
      unchanged: 1,
      diagnostics: {
        "AUDIENCE-NAME-GROUNDED": 1,
      },
    });
    expect(result.warnings).toEqual([
      `review ${legacy.id} skipped: legacy item has no overseer_flag payload`,
    ]);
    expect(harness.reviewQueueRepository.get(falsePositive.id)).toMatchObject({
      resolution: "dismiss",
      refs: {
        review_resolution: {
          decision: "dismiss",
          reason: expect.stringContaining(
            "suppressed by current gate logic against persisted enqueue-time inputs",
          ),
        },
      },
    });
    expect(harness.reviewQueueRepository.get(falsePositive.id)?.refs.review_resolution).toEqual(
      expect.objectContaining({
        reason: expect.stringContaining("AUDIENCE-NAME-GROUNDED"),
      }),
    );
    expect(harness.reviewQueueRepository.get(real.id)).toMatchObject({
      resolution: null,
    });
    expect(harness.reviewQueueRepository.get(legacy.id)).toMatchObject({
      resolution: null,
    });
  });

  it("uses persisted enqueue-time audience inputs when live target sources change", async () => {
    const nowMs = 15 * 24 * 60 * 60 * 1_000;
    const harness = await createOfflineTestHarness({
      clock: new FixedClock(nowMs),
    });
    cleanup.push(harness.cleanup);

    const audienceEntityId = harness.entityRepository.resolve("Tom", {
      provenance: "transport_audience_label",
    });
    const persistedSource = await appendSourceEntry(harness, "Otto is my dog.");
    const persistedEpisode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture(
        {
          title: "Persisted enqueue source",
          narrative: "The user said Otto is their dog.",
          source_stream_ids: [persistedSource.id],
          audience_entity_id: audienceEntityId,
          shared: false,
          created_at: nowMs - 4_000,
          updated_at: nowMs - 4_000,
        },
        [1, 0, 0, 0],
      ),
    );
    const node = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Otto as Tom's dog",
          description: "Otto is Tom's dog.",
          source_episode_ids: [persistedEpisode.id],
          created_at: nowMs - 3_000,
          updated_at: nowMs - 3_000,
          last_verified_at: nowMs - 3_000,
        },
        [0, 1, 0, 0],
      ),
    );
    const liveSource = await appendSourceEntry(harness, "Riley led the workshop, not Tom.");
    const liveEpisode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture(
        {
          title: "Changed live source",
          narrative: "The user said Riley led the workshop.",
          source_stream_ids: [liveSource.id],
          created_at: nowMs - 2_000,
          updated_at: nowMs - 2_000,
        },
        [0, 0, 1, 0],
      ),
    );

    const review = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs: {
        target_type: "semantic_node",
        target_id: node.id,
        patch: {
          description: "Corrected description.",
        },
        evidence_stream_ids: [persistedSource.id],
        overseer_flag: overseerPayload({
          citedStreamIds: [persistedSource.id],
          sourceEpisodeIds: [persistedEpisode.id],
          quotedSpan: "Tom",
          audienceEntities: [
            {
              entity_id: audienceEntityId,
              display_name: "Tom",
              source_episode_ids: [persistedEpisode.id],
            },
          ],
        }),
      },
      reason: "The source text does not literally name Tom.",
    });

    await harness.semanticNodeRepository.update(node.id, {
      description: "Otto is associated with a different live source now.",
      source_episode_ids: [liveEpisode.id],
      replace_source_episode_ids: true,
    });
    harness.episodicRepository.archiveEpisode(persistedEpisode.id, {
      caller: "revalidate.test",
      reason: "seed an archived episode",
      process: "curator",
    });

    const result = await revalidateReviewQueue(harness.createContext(), {
      kind: "misattribution",
    });

    expect(result).toMatchObject({
      revalidated: 1,
      dismissed_as_suppressed: 1,
      skipped_legacy: 0,
      unchanged: 0,
      diagnostics: {
        "AUDIENCE-NAME-GROUNDED": 1,
      },
    });
    expect(result.warnings).toEqual([
      `review ${review.id}: persisted source episodes are now archived: ${persistedEpisode.id}`,
    ]);
    expect(harness.reviewQueueRepository.get(review.id)).toMatchObject({
      resolution: "dismiss",
      refs: {
        review_resolution: {
          decision: "dismiss",
          reason: expect.stringContaining(
            "suppressed by current gate logic against persisted enqueue-time inputs",
          ),
        },
      },
    });
    expect(harness.reviewQueueRepository.get(review.id)?.refs.review_resolution).toEqual(
      expect.objectContaining({
        reason: expect.stringContaining("AUDIENCE-NAME-GROUNDED"),
      }),
    );
  });

  it("only revalidates items older than maxAgeDays when provided", async () => {
    const nowMs = 20 * 24 * 60 * 60 * 1_000;
    const clock = new ManualClock(nowMs - 9 * 24 * 60 * 60 * 1_000);
    const harness = await createOfflineTestHarness({
      clock,
    });
    cleanup.push(harness.cleanup);

    const audienceEntityId = harness.entityRepository.resolve("Tom", {
      provenance: "transport_audience_label",
    });
    const source = await appendSourceEntry(harness, "Otto is my dog.");
    const episode = await harness.episodicRepository.createEpisode(
      createEpisodeFixture(
        {
          title: "Audience source",
          source_stream_ids: [source.id],
          audience_entity_id: audienceEntityId,
          shared: false,
          created_at: nowMs - 9 * 24 * 60 * 60 * 1_000,
          updated_at: nowMs - 9 * 24 * 60 * 60 * 1_000,
        },
        [1, 0, 0, 0],
      ),
    );
    const node = await harness.semanticNodeRepository.insert(
      createSemanticNodeFixture(
        {
          label: "Otto as Tom's dog",
          description: "Otto is Tom's dog.",
          source_episode_ids: [episode.id],
          created_at: nowMs - 9 * 24 * 60 * 60 * 1_000,
          updated_at: nowMs - 9 * 24 * 60 * 60 * 1_000,
          last_verified_at: nowMs - 9 * 24 * 60 * 60 * 1_000,
        },
        [0, 1, 0, 0],
      ),
    );
    const refs = {
      target_type: "semantic_node",
      target_id: node.id,
      patch: {
        description: "Corrected description.",
      },
      evidence_stream_ids: [source.id],
      overseer_flag: overseerPayload({
        citedStreamIds: [source.id],
        sourceEpisodeIds: [episode.id],
        quotedSpan: "Tom",
        audienceEntities: [
          {
            entity_id: audienceEntityId,
            display_name: "Tom",
            source_episode_ids: [episode.id],
          },
        ],
      }),
    };

    clock.set(nowMs - 8 * 24 * 60 * 60 * 1_000);
    const oldReview = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs,
      reason: "Old audience-name false positive.",
    });
    clock.set(nowMs - 1 * 24 * 60 * 60 * 1_000);
    const youngReview = harness.reviewQueueRepository.enqueue({
      kind: "misattribution",
      refs,
      reason: "Young audience-name false positive.",
    });
    clock.set(nowMs);

    const result = await revalidateReviewQueue(harness.createContext(), {
      kind: "misattribution",
      maxAgeDays: 7,
    });

    expect(result).toMatchObject({
      revalidated: 1,
      dismissed_as_suppressed: 1,
      skipped_legacy: 0,
      unchanged: 0,
    });
    expect(harness.reviewQueueRepository.get(oldReview.id)).toMatchObject({
      resolution: "dismiss",
    });
    expect(harness.reviewQueueRepository.get(youngReview.id)).toMatchObject({
      resolution: null,
    });
  });
});
