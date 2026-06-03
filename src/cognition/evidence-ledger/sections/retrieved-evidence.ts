import type { BuilderSectionContext } from "../builder-context.js";
import { rawStreamActor } from "../entry-metadata.js";
import {
  COMMITMENT_TRUST_RANK,
  DISCOURSE_TRUST_RANK,
  IMAGE_PERCEPTION_TRUST_RANK,
  OPEN_QUESTION_TRUST_RANK,
  RAW_STREAM_TRUST_RANK,
  WARM_RECALL_TRUST_RANK,
  addEntry,
  cappedTrustRank,
} from "../section-buckets.js";
import {
  persistenceClassFromProvenance,
  persistenceClassFromStreamIds,
  scopeFromStreamIds,
  streamIndexFromSingleCurrentSessionStreamId,
} from "../scope-resolver.js";
import {
  evidenceItemProvenanceMetadata,
  evidenceItemScope,
  evidenceItemSourceType,
  evidenceItemState,
  rawStreamSourceType,
} from "../retrieved-evidence-mapping.js";

export function addRetrievedRawStreamEvidenceSection(context: BuilderSectionContext): void {
  for (const item of context.input.retrievedEvidence) {
    if (item.source !== "raw_stream" && item.source !== "recent_raw_stream") {
      continue;
    }

    const itemStreamIds = item.provenance?.streamIds ?? [];

    // If every stream ID this retrieval item points to is already in the
    // current_session_transcript section, skip emitting the duplicate
    // retrieved_raw_stream_evidence row. The transcript renders the same
    // underlying content with higher trust rank.
    if (
      itemStreamIds.length > 0 &&
      itemStreamIds.every((id) => context.transcript.rawStreamIds.has(id))
    ) {
      continue;
    }

    const scope = scopeFromStreamIds(itemStreamIds, context.resolver);
    const streamIndex = streamIndexFromSingleCurrentSessionStreamId(
      itemStreamIds,
      context.resolver,
    );
    addEntry(
      context.buckets,
      "retrieved_raw_stream_evidence",
      cappedTrustRank({
        id: `retrieved_stream:${item.id}`,
        source_type: rawStreamSourceType(scope),
        session_scope: scope,
        actor: rawStreamActor(itemStreamIds, context.resolver),
        trust_rank: RAW_STREAM_TRUST_RANK,
        text: item.text,
        value: item.source,
        ...(streamIndex === undefined ? {} : { stream_index: streamIndex }),
        state: `score=${item.score.toFixed(2)}`,
        state_metadata: itemStreamIds.length === 0 ? undefined : { stream_ids: [...itemStreamIds] },
        taint: "none",
        via_retrieval: true,
        ...persistenceClassFromStreamIds(itemStreamIds, context.resolver),
      }),
    );
  }
}

export function addRetrievedStructuredEvidenceSection(context: BuilderSectionContext): void {
  for (const item of context.input.retrievedEvidence) {
    if (item.source === "raw_stream" || item.source === "recent_raw_stream") {
      continue;
    }

    const scope = evidenceItemScope(item, context.resolver);
    const entry = cappedTrustRank({
      id: `retrieved_evidence:${item.id}`,
      source_type: evidenceItemSourceType(item, scope),
      session_scope: scope,
      actor: "memory" as const,
      trust_rank:
        item.source === "warm_recall"
          ? WARM_RECALL_TRUST_RANK
          : item.source === "image_perception"
            ? IMAGE_PERCEPTION_TRUST_RANK
            : RAW_STREAM_TRUST_RANK,
      text: item.text,
      value: item.source,
      state: evidenceItemState(item),
      state_metadata: evidenceItemProvenanceMetadata(item),
      citation_type: item.citationType,
      taint: "none" as const,
      via_retrieval: true,
      ...persistenceClassFromProvenance(
        {
          streamEntryIds: item.provenance?.streamIds ?? [],
          episodeIds: [
            ...(item.provenance?.episodeId === undefined ? [] : [item.provenance.episodeId]),
            ...(item.source_episode_ids ?? []),
          ],
        },
        context.resolver,
      ),
    });

    if (item.source === "commitment") {
      addEntry(context.buckets, "retrieved_memory_evidence", {
        ...entry,
        trust_rank: COMMITMENT_TRUST_RANK,
      });
      continue;
    }

    if (item.source === "working_state") {
      addEntry(context.buckets, "closure_discourse_state", {
        ...entry,
        actor: "system",
        trust_rank: DISCOURSE_TRUST_RANK,
      });
      continue;
    }

    if (item.source === "open_question") {
      addEntry(context.buckets, "open_questions", {
        ...entry,
        trust_rank: OPEN_QUESTION_TRUST_RANK,
      });
      continue;
    }

    addEntry(context.buckets, "retrieved_memory_evidence", entry);
  }
}
