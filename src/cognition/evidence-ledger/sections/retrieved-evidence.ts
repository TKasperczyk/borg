import type { BuilderSectionContext } from "../builder-context.js";
import {
  actorForStreamEntry,
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
  rawStreamActor,
} from "../entry-metadata.js";
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
import { unknownMemoryDisclosureLabel } from "../../../retrieval/index.js";
import {
  persistenceClassFromProvenance,
  persistenceClassFromStreamIds,
  scopeFromCarriedSessionId,
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

    // Both of the next two resolve out of the builder's entry map, and a recalled entry
    // is frequently not in it. Neither miss is visible on the row: the miss answers
    // "global" and "memory", which are also answers a hit produces. Fall back to the
    // facts the pipeline copied off the entry itself -- `kind` decides the actor on its
    // own, and the carried session gets the same comparison the resolver would apply.
    // `source_type` follows scope, so it is corrected by the same step. What this does
    // not recover is `persistence_class` below, whose absence still reads as "not an
    // assistant self-report" on an entry the map never held.
    const resolvedFromEntries = itemStreamIds.every((id) =>
      context.resolver.streamEntriesById.has(id),
    );
    const resolvedScope = scopeFromStreamIds(itemStreamIds, context.resolver);
    const scope = resolvedFromEntries
      ? resolvedScope
      : (scopeFromCarriedSessionId(item.streamEntrySessionId, context.resolver.currentSessionId) ??
        resolvedScope);
    const actor =
      resolvedFromEntries || item.streamEntryKind === undefined
        ? rawStreamActor(itemStreamIds, context.resolver)
        : actorForStreamEntry({ kind: item.streamEntryKind });
    const streamIndex = streamIndexFromSingleCurrentSessionStreamId(
      itemStreamIds,
      context.resolver,
    );
    const disclosureLabel = item.disclosureLabel ?? unknownMemoryDisclosureLabel();
    const stateMetadata = appendMemoryDisclosureStateMetadata({
      stateMetadata: itemStreamIds.length === 0 ? undefined : { stream_ids: [...itemStreamIds] },
      disclosureLabel,
    });
    addEntry(
      context.buckets,
      "retrieved_raw_stream_evidence",
      cappedTrustRank({
        id: `retrieved_stream:${item.id}`,
        source_type: rawStreamSourceType(scope),
        session_scope: scope,
        actor,
        trust_rank: RAW_STREAM_TRUST_RANK,
        text: item.text,
        value: item.source,
        ...(streamIndex === undefined ? {} : { stream_index: streamIndex }),
        state: appendMemoryDisclosureState({
          state: `score=${item.score.toFixed(2)}`,
          disclosureLabel,
        }),
        state_metadata: stateMetadata,
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
    const itemStreamIds = item.provenance?.streamIds ?? [];
    // A row carrying a stream entry's kind is that entry's own text (on this lane, a warm
    // recall of one entry): its author decides the actor exactly as on the raw-stream lane.
    // Its scope stays the resolver's, since a corrected prior_session would move the row.
    const actor =
      item.streamEntryKind === undefined
        ? ("memory" as const)
        : itemStreamIds.every((id) => context.resolver.streamEntriesById.has(id))
          ? rawStreamActor(itemStreamIds, context.resolver)
          : actorForStreamEntry({ kind: item.streamEntryKind });
    const entry = cappedTrustRank({
      id: `retrieved_evidence:${item.id}`,
      source_type: evidenceItemSourceType(item, scope),
      session_scope: scope,
      actor,
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
