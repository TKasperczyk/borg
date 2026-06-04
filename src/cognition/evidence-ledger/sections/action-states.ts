import {
  DEFAULT_ACTION_THREAD_RENDER_LIMIT,
  DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD,
  DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
  STALE_PARTICIPANT_ACTION_RENDER_LIMIT,
  actionSalienceClass,
  actionActorDisplay,
  actionThreadState,
  actionThreadStateMetadata,
  buildActionThreads,
  listActionCandidatesForCognition,
  normalizePositiveInteger,
  normalizeUnitInterval,
  orderActionThreadsBySalience,
  renderActionThreadText,
  renderOlderActionThreadsSummary,
} from "../action-threads.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "../entry-metadata.js";
import { ACTION_TRUST_RANK, addEntry, cappedTrustRank } from "../section-buckets.js";
import { persistenceClassFromProvenance } from "../scope-resolver.js";
import { combineMemoryDisclosureLabels } from "../../../retrieval/index.js";
import { actionMemoryDisclosureLabel } from "../../disclosure-labels.js";

export async function addActionStatesSection(context: BuilderSectionContext): Promise<void> {
  const sourceRecordLimit = normalizePositiveInteger(
    context.options.actionThreadSourceRecordLimit,
    DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
  );
  const renderLimit = normalizePositiveInteger(
    context.options.actionThreadRenderLimit,
    DEFAULT_ACTION_THREAD_RENDER_LIMIT,
  );
  const similarityThreshold = normalizeUnitInterval(
    context.options.actionThreadSimilarityThreshold,
    DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD,
  );
  const actionCandidates = listActionCandidatesForCognition({
    actionRepository: context.repos.actions,
    audienceEntityId: context.input.audienceEntityId,
    activeParticipants: context.input.activeParticipants,
    limit: sourceRecordLimit,
  });
  const disclosureLabelByActionId = new Map(
    actionCandidates.map((candidate) => [candidate.record.id, candidate.disclosureLabel]),
  );
  const threads = await buildActionThreads({
    records: actionCandidates.map((candidate) => candidate.record),
    repository: context.repos.actions,
    resolver: context.resolver,
    similarityThreshold,
  });
  const threadsWithSalience = threads.flatMap((thread) => {
    const salienceClass = actionSalienceClass({
      thread,
      currentUserStreamEntryId: context.input.currentUserEntry?.id,
      currentUserStreamEntryIds: context.input.currentUserEntries?.map((entry) => entry.id),
      currentTurnCounter: context.input.workingMemory.turn_counter,
    });

    return salienceClass === null ? [] : [{ ...thread, salienceClass }];
  });
  const staleParticipantThreads = threadsWithSalience.filter(
    (thread) => thread.salienceClass === "participant_pending_stale",
  );
  const cappedStaleIds = new Set(
    staleParticipantThreads.slice(STALE_PARTICIPANT_ACTION_RENDER_LIMIT).map((thread) => thread.id),
  );
  const renderedThreads = orderActionThreadsBySalience(
    threadsWithSalience.filter((thread) => !cappedStaleIds.has(thread.id)),
  ).slice(0, renderLimit);

  for (const thread of renderedThreads) {
    const disclosureLabel = combineMemoryDisclosureLabels(
      thread.records.map(
        (record) => disclosureLabelByActionId.get(record.id) ?? actionMemoryDisclosureLabel(record),
      ),
    );
    addEntry(
      context.buckets,
      "action_states",
      cappedTrustRank({
        id: `action_thread:${thread.id}`,
        source_type: "action_record",
        session_scope: thread.scope,
        actor: thread.current.actor === "borg" ? "assistant" : "user",
        trust_rank: ACTION_TRUST_RANK,
        text: [
          `salience: ${thread.salienceClass}`,
          renderActionThreadText(thread, context.repos.entities),
        ].join("\n"),
        value: actionActorDisplay(thread.current.actor, context.repos.entities),
        state: appendMemoryDisclosureState({
          state: actionThreadState(thread),
          disclosureLabel,
        }),
        salience_class: thread.salienceClass,
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: {
            ...actionThreadStateMetadata(thread, context.repos.entities),
            salience_class: thread.salienceClass,
          },
          disclosureLabel,
        }),
        taint: "none",
        ...persistenceClassFromProvenance(
          {
            streamEntryIds: thread.records.flatMap((record) => record.provenance_stream_entry_ids),
            episodeIds: thread.records.flatMap((record) => record.provenance_episode_ids),
          },
          context.resolver,
        ),
      }),
    );
  }

  const omittedStaleCount = cappedStaleIds.size;

  if (omittedStaleCount > 0) {
    addEntry(context.buckets, "action_states", {
      id: "action_threads:participant_pending_stale_summary",
      source_type: "system_metadata",
      session_scope: "global",
      actor: "system",
      trust_rank: ACTION_TRUST_RANK,
      text: `Stale participant pending actions omitted from this section: count=${omittedStaleCount}.`,
      value: "participant_pending_stale_summary",
      state: "omitted",
      salience_class: "participant_pending_stale",
      taint: "none",
    });
  }

  if (threadsWithSalience.length <= renderLimit) {
    return;
  }

  const olderThreads = orderActionThreadsBySalience(threadsWithSalience).slice(renderLimit);
  const olderDisclosureLabel = combineMemoryDisclosureLabels(
    olderThreads.flatMap((thread) =>
      thread.records.map(
        (record) => disclosureLabelByActionId.get(record.id) ?? actionMemoryDisclosureLabel(record),
      ),
    ),
  );

  addEntry(context.buckets, "action_states", {
    id: "action_threads:older_summary",
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: ACTION_TRUST_RANK,
    text: renderOlderActionThreadsSummary(olderThreads),
    value: "older_action_threads",
    state: appendMemoryDisclosureState({
      state: "omitted",
      disclosureLabel: olderDisclosureLabel,
    }),
    state_metadata: appendMemoryDisclosureStateMetadata({
      stateMetadata: undefined,
      disclosureLabel: olderDisclosureLabel,
    }),
    taint: "none",
  });
}
