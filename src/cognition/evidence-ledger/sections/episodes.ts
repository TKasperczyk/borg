import type { BuilderSectionContext } from "../builder-context.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "../entry-metadata.js";
import { EPISODE_TRUST_RANK, addEntry, cappedTrustRank } from "../section-buckets.js";
import { persistenceClassFromProvenance, scopeFromStreamIds } from "../scope-resolver.js";
import { memoryDisclosureLabelFromEpisodeAccess } from "../../../retrieval/index.js";
import { formatRelativeAge } from "../../../util/relative-time.js";

export function addEpisodesSection(context: BuilderSectionContext): void {
  for (const result of context.input.retrievedEpisodes) {
    const disclosureLabel =
      result.disclosureLabel ?? memoryDisclosureLabelFromEpisodeAccess(result.episode);
    const scope =
      context.resolver.episodeScopesById.get(result.episode.id) ??
      scopeFromStreamIds(result.episode.source_stream_ids, context.resolver);
    addEntry(
      context.buckets,
      "episodes",
      cappedTrustRank({
        id: `episode:${result.episode.id}`,
        source_type: "episode",
        session_scope: scope,
        actor: "memory",
        trust_rank: EPISODE_TRUST_RANK,
        text: result.episode.narrative,
        value: result.episode.title,
        state: appendMemoryDisclosureState({
          state: `confidence=${result.episode.confidence.toFixed(2)} score=${result.score.toFixed(2)}`,
          disclosureLabel,
        }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: {
            episode_id: result.episode.id,
            occurred_at: new Date(result.episode.end_time).toISOString(),
            ...(context.nowMs === undefined
              ? {}
              : { relative_age: formatRelativeAge(result.episode.end_time, context.nowMs) }),
            source_stream_ids: [...result.episode.source_stream_ids],
          },
          disclosureLabel,
          currentAudienceEntityId: context.input.audienceEntityId,
        }),
        taint: "none",
        ...persistenceClassFromProvenance(
          { streamEntryIds: result.episode.source_stream_ids },
          context.resolver,
        ),
      }),
    );
  }
}
