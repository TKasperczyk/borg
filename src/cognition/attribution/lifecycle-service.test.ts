import { describe, expect, it, vi } from "vitest";

import type { SocialProfile } from "../../memory/social/index.js";
import type {
  PendingSocialAttribution,
  PendingTraitAttribution,
} from "../../memory/working/index.js";
import type { StreamEntry, StreamEntryInput, StreamWriter } from "../../stream/index.js";
import { DEFAULT_SESSION_ID } from "../../stream/index.js";
import { ManualClock } from "../../util/clock.js";
import type { EntityId, EpisodeId, StreamEntryId } from "../../util/ids.js";
import type { PerceptionResult } from "../types.js";
import { AttributionLifecycleService } from "./lifecycle-service.js";

const entityId = "entity_alice" as EntityId;
const otherEntityId = "entity_bob" as EntityId;
const streamEntryId = "stream_entry_1" as StreamEntryId;
const episodeId = "episode_1" as EpisodeId;

function makePerception(valence: number, degraded = false): PerceptionResult {
  return {
    mode: "relational",
    entities: ["Alice"],
    temporalCue: null,
    affectiveSignal: {
      valence,
      arousal: 0.2,
      dominant_emotion: "joy",
    },
    affectiveSignalDegraded: degraded,
  };
}

function makeProfile(): SocialProfile {
  return {
    entity_id: entityId,
    trust: 0.5,
    attachment: 0.5,
    communication_style: null,
    shared_history_summary: null,
    last_interaction_at: null,
    interaction_count: 1,
    commitment_count: 0,
    sentiment_history: [],
    notes: null,
    created_at: 1_000,
    updated_at: 1_000,
  };
}

function makeStreamWriter() {
  const appended: StreamEntryInput[] = [];
  const streamWriter: Pick<StreamWriter, "append"> = {
    append: async (input) => {
      appended.push(input);

      return {
        ...input,
        id: streamEntryId,
        timestamp: 1_000,
        session_id: DEFAULT_SESSION_ID,
        compressed: input.compressed ?? false,
      } satisfies StreamEntry;
    },
  };

  return { appended, streamWriter };
}

describe("AttributionLifecycleService", () => {
  it("attaches matching social sentiment and clears pending attribution", async () => {
    const clock = new ManualClock(2_000);
    const refreshedProfile = makeProfile();
    const attachSentiment = vi.fn();
    const getProfile = vi.fn(() => refreshedProfile);
    const service = new AttributionLifecycleService({
      socialRepository: {
        attachSentiment,
        getProfile,
      },
      traitsRepository: {
        reinforce: vi.fn(),
      },
      episodicRepository: {
        findBySourceStreamIdsContaining: vi.fn(),
      },
      clock,
    });
    const pending: PendingSocialAttribution = {
      entity_id: entityId,
      interaction_id: 42,
      agent_response_summary: "summary",
      turn_completed_ts: 1_500,
    };
    const { appended, streamWriter } = makeStreamWriter();

    const result = await service.settle({
      isUserTurn: true,
      audienceEntityId: entityId,
      perception: makePerception(0.7),
      pendingSocialAttribution: pending,
      pendingTraitAttribution: null,
      audienceProfile: null,
      streamWriter,
      onHookFailure: vi.fn(),
    });

    expect(attachSentiment).toHaveBeenCalledWith(42, {
      valence: 0.7,
      now: 2_000,
    });
    expect(result.pendingSocialAttribution).toBeNull();
    expect(result.audienceProfile).toBe(refreshedProfile);
    expect(appended).toEqual([]);
  });

  it("keeps social attribution pending when affective signal is degraded", async () => {
    const attachSentiment = vi.fn();
    const service = new AttributionLifecycleService({
      socialRepository: {
        attachSentiment,
        getProfile: vi.fn(),
      },
      traitsRepository: {
        reinforce: vi.fn(),
      },
      episodicRepository: {
        findBySourceStreamIdsContaining: vi.fn(),
      },
      clock: new ManualClock(2_000),
    });
    const pending: PendingSocialAttribution = {
      entity_id: entityId,
      interaction_id: 42,
      agent_response_summary: "summary",
      turn_completed_ts: 1_500,
    };
    const { streamWriter } = makeStreamWriter();

    const result = await service.settle({
      isUserTurn: true,
      audienceEntityId: entityId,
      perception: makePerception(0, true),
      pendingSocialAttribution: pending,
      pendingTraitAttribution: null,
      audienceProfile: null,
      streamWriter,
      onHookFailure: vi.fn(),
    });

    expect(attachSentiment).not.toHaveBeenCalled();
    expect(result.pendingSocialAttribution).toBe(pending);
  });

  it("drops mismatched trait attribution with the original internal event payload", async () => {
    const service = new AttributionLifecycleService({
      socialRepository: {
        attachSentiment: vi.fn(),
        getProfile: vi.fn(),
      },
      traitsRepository: {
        reinforce: vi.fn(),
      },
      episodicRepository: {
        findBySourceStreamIdsContaining: vi.fn(),
      },
      clock: new ManualClock(2_000),
    });
    const pending: PendingTraitAttribution = {
      trait_label: "patient",
      strength_delta: 0.05,
      source_stream_entry_ids: [streamEntryId],
      source_episode_ids: [episodeId],
      turn_completed_ts: 1_500,
      audience_entity_id: otherEntityId,
    };
    const { appended, streamWriter } = makeStreamWriter();

    const result = await service.settle({
      isUserTurn: true,
      audienceEntityId: entityId,
      perception: makePerception(0.7),
      pendingSocialAttribution: null,
      pendingTraitAttribution: pending,
      audienceProfile: null,
      streamWriter,
      onHookFailure: vi.fn(),
    });

    expect(result.pendingTraitAttribution).toBeNull();
    expect(appended).toEqual([
      {
        kind: "internal_event",
        content: {
          kind: "trait_attribution_drop",
          reason: "audience_mismatch",
          pending_trait_label: "patient",
          pending_audience_entity_id: otherEntityId,
          current_audience_entity_id: entityId,
          turn_completed_ts: 1_500,
          source_episode_ids: [episodeId],
          source_stream_entry_ids: [streamEntryId],
        },
      },
    ]);
  });

  it("keeps positive trait attribution pending when stream evidence has no episode yet", async () => {
    const reinforce = vi.fn();
    const findBySourceStreamIdsContaining = vi.fn(async () => null);
    const service = new AttributionLifecycleService({
      socialRepository: {
        attachSentiment: vi.fn(),
        getProfile: vi.fn(),
      },
      traitsRepository: {
        reinforce,
      },
      episodicRepository: {
        findBySourceStreamIdsContaining,
      },
      clock: new ManualClock(2_000),
    });
    const pending: PendingTraitAttribution = {
      trait_label: "patient",
      strength_delta: 0.05,
      source_stream_entry_ids: [streamEntryId],
      source_episode_ids: [],
      turn_completed_ts: 1_500,
      audience_entity_id: entityId,
    };
    const { streamWriter } = makeStreamWriter();

    const result = await service.settle({
      isUserTurn: true,
      audienceEntityId: entityId,
      perception: makePerception(0.7),
      pendingSocialAttribution: null,
      pendingTraitAttribution: pending,
      audienceProfile: null,
      streamWriter,
      onHookFailure: vi.fn(),
    });

    expect(findBySourceStreamIdsContaining).toHaveBeenCalledWith([streamEntryId]);
    expect(reinforce).not.toHaveBeenCalled();
    expect(result.pendingTraitAttribution).toBe(pending);
  });

  it("keeps trait attribution pending when affective signal is degraded", async () => {
    const reinforce = vi.fn();
    const findBySourceStreamIdsContaining = vi.fn();
    const service = new AttributionLifecycleService({
      socialRepository: {
        attachSentiment: vi.fn(),
        getProfile: vi.fn(),
      },
      traitsRepository: {
        reinforce,
      },
      episodicRepository: {
        findBySourceStreamIdsContaining,
      },
      clock: new ManualClock(2_000),
    });
    const pending: PendingTraitAttribution = {
      trait_label: "patient",
      strength_delta: 0.05,
      source_stream_entry_ids: [streamEntryId],
      source_episode_ids: [],
      turn_completed_ts: 1_500,
      audience_entity_id: entityId,
    };
    const { streamWriter } = makeStreamWriter();

    const result = await service.settle({
      isUserTurn: true,
      audienceEntityId: entityId,
      perception: makePerception(0, true),
      pendingSocialAttribution: null,
      pendingTraitAttribution: pending,
      audienceProfile: null,
      streamWriter,
      onHookFailure: vi.fn(),
    });

    expect(findBySourceStreamIdsContaining).not.toHaveBeenCalled();
    expect(reinforce).not.toHaveBeenCalled();
    expect(result.pendingTraitAttribution).toBe(pending);
  });
});
