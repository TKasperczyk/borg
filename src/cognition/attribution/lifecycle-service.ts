import type { EpisodicRepository } from "../../memory/episodic/index.js";
import type { SocialProfile, SocialRepository } from "../../memory/social/index.js";
import type { TraitsRepository } from "../../memory/self/index.js";
import type {
  PendingSocialAttribution,
  PendingTraitAttribution,
} from "../../memory/working/index.js";
import type { StreamWriter } from "../../stream/index.js";
import type { Clock } from "../../util/clock.js";
import type { EntityId, EpisodeId } from "../../util/ids.js";
import type { PerceptionResult } from "../types.js";

const PENDING_SOCIAL_ATTRIBUTION_TTL_MS = 60 * 60 * 1_000;
const PENDING_TRAIT_ATTRIBUTION_TTL_MS = PENDING_SOCIAL_ATTRIBUTION_TTL_MS;
const TRAIT_ATTRIBUTION_POSITIVE_VALENCE_THRESHOLD = 0.2;

export type AttributionLifecycleServiceOptions = {
  socialRepository: Pick<SocialRepository, "attachSentiment" | "getProfile">;
  traitsRepository: Pick<TraitsRepository, "reinforce">;
  episodicRepository: Pick<EpisodicRepository, "findBySourceStreamIdsContaining">;
  clock: Clock;
};

export type AttributionLifecycleInput = {
  isUserTurn: boolean;
  audienceEntityId: EntityId | null;
  perception: PerceptionResult;
  pendingSocialAttribution: PendingSocialAttribution | null;
  pendingTraitAttribution: PendingTraitAttribution | null;
  audienceProfile: SocialProfile | null;
  streamWriter: Pick<StreamWriter, "append">;
  onHookFailure: (hook: string, error: unknown) => Promise<void>;
};

export type AttributionLifecycleResult = {
  pendingSocialAttribution: PendingSocialAttribution | null;
  pendingTraitAttribution: PendingTraitAttribution | null;
  audienceProfile: SocialProfile | null;
};

export class AttributionLifecycleService {
  constructor(private readonly options: AttributionLifecycleServiceOptions) {}

  private async resolveTraitEvidenceEpisodes(
    attribution: PendingTraitAttribution,
  ): Promise<EpisodeId[]> {
    const resolved = await this.options.episodicRepository.findBySourceStreamIdsContaining(
      attribution.source_stream_entry_ids,
    );

    if (resolved !== null) {
      return [resolved.id];
    }

    return [];
  }

  async settle(input: AttributionLifecycleInput): Promise<AttributionLifecycleResult> {
    let pendingSocialAttribution = input.pendingSocialAttribution;
    let pendingTraitAttribution = input.pendingTraitAttribution;
    let audienceProfile = input.audienceProfile;

    if (input.isUserTurn && pendingSocialAttribution !== null) {
      const nowMs = this.options.clock.now();
      const expired =
        nowMs - pendingSocialAttribution.turn_completed_ts > PENDING_SOCIAL_ATTRIBUTION_TTL_MS;
      const audienceEntityId = input.audienceEntityId;

      if (
        expired ||
        audienceEntityId === null ||
        pendingSocialAttribution.entity_id !== audienceEntityId
      ) {
        await input.streamWriter.append({
          kind: "internal_event",
          content: {
            kind: "social_attribution_drop",
            reason: expired ? "expired" : "audience_mismatch",
            pending_entity_id: pendingSocialAttribution.entity_id,
            current_audience_entity_id: input.audienceEntityId,
            turn_completed_ts: pendingSocialAttribution.turn_completed_ts,
            agent_response_summary: pendingSocialAttribution.agent_response_summary,
          },
        });
        pendingSocialAttribution = null;
      } else if (input.perception.affectiveSignalDegraded === true) {
        // Keep the pending attribution alive. Neutral affect in degraded
        // mode is a contract placeholder, not observed evidence.
      } else {
        try {
          this.options.socialRepository.attachSentiment(pendingSocialAttribution.interaction_id, {
            valence: input.perception.affectiveSignal.valence,
            now: nowMs,
          });
          audienceProfile = this.options.socialRepository.getProfile(audienceEntityId);
          pendingSocialAttribution = null;
        } catch (error) {
          await input.onHookFailure("social_update", error);
        }
      }
    }

    if (input.isUserTurn && pendingTraitAttribution !== null) {
      const nowMs = this.options.clock.now();
      const expired =
        nowMs - pendingTraitAttribution.turn_completed_ts > PENDING_TRAIT_ATTRIBUTION_TTL_MS;
      const audienceMatches = pendingTraitAttribution.audience_entity_id === input.audienceEntityId;

      if (expired || !audienceMatches) {
        await input.streamWriter.append({
          kind: "internal_event",
          content: {
            kind: "trait_attribution_drop",
            reason: expired ? "expired" : "audience_mismatch",
            pending_trait_label: pendingTraitAttribution.trait_label,
            pending_audience_entity_id: pendingTraitAttribution.audience_entity_id,
            current_audience_entity_id: input.audienceEntityId,
            turn_completed_ts: pendingTraitAttribution.turn_completed_ts,
            source_stream_entry_ids: pendingTraitAttribution.source_stream_entry_ids,
          },
        });
        pendingTraitAttribution = null;
      } else if (input.perception.affectiveSignalDegraded === true) {
        // Do not clear or reinforce trait attribution from degraded affect.
      } else if (
        input.perception.affectiveSignal.valence > TRAIT_ATTRIBUTION_POSITIVE_VALENCE_THRESHOLD
      ) {
        // Sprint 56: resolve the demonstrating turn's stream entries to the
        // extracted episode. If extraction has not completed yet, keep the
        // attribution pending so a later turn can reinforce once an episode exists.
        try {
          const evidenceEpisodeIds =
            await this.resolveTraitEvidenceEpisodes(pendingTraitAttribution);

          if (evidenceEpisodeIds.length > 0) {
            this.options.traitsRepository.reinforce({
              label: pendingTraitAttribution.trait_label,
              delta: pendingTraitAttribution.strength_delta,
              provenance: {
                kind: "episodes",
                episode_ids: evidenceEpisodeIds,
              },
              timestamp: nowMs,
            });
            pendingTraitAttribution = null;
          }
        } catch (error) {
          await input.onHookFailure("trait_update", error);
        }
      } else {
        pendingTraitAttribution = null;
      }
    }

    return {
      pendingSocialAttribution,
      pendingTraitAttribution,
      audienceProfile,
    };
  }
}
