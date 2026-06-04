import type { BuilderSectionContext } from "../builder-context.js";
import { EPISODE_TRUST_RANK, addEntry, cappedTrustRank } from "../section-buckets.js";
import { persistenceClassFromProvenance, scopeFromStreamIds } from "../scope-resolver.js";
import {
  MEMORY_DISCLOSURE_INTERNAL_USE_NOTE,
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelMetadata,
  renderMemoryDisclosureLabelForModel,
} from "../../../retrieval/index.js";

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
        state: `confidence=${result.episode.confidence.toFixed(2)} score=${result.score.toFixed(2)} ${renderMemoryDisclosureLabelForModel(disclosureLabel)}`,
        state_metadata: {
          episode_id: result.episode.id,
          source_stream_ids: [...result.episode.source_stream_ids],
          disclosure_label: memoryDisclosureLabelMetadata(disclosureLabel),
          ...(disclosureLabel.disclosureClass === "public"
            ? {}
            : {
                disclosure_note: MEMORY_DISCLOSURE_INTERNAL_USE_NOTE,
                current_audience_entity_id: context.input.audienceEntityId,
              }),
        },
        taint: "none",
        ...persistenceClassFromProvenance(
          { streamEntryIds: result.episode.source_stream_ids },
          context.resolver,
        ),
      }),
    );
  }
}
