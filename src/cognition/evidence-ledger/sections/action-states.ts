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
  listVisibleActions,
  normalizePositiveInteger,
  normalizeUnitInterval,
  orderActionThreadsBySalience,
  renderActionThreadText,
  renderOlderActionThreadsSummary,
} from "../action-threads.js";
import type { BuilderSectionContext } from "../builder-context.js";
import { ACTION_TRUST_RANK, addEntry, cappedTrustRank } from "../section-buckets.js";
import { persistenceClassFromProvenance } from "../scope-resolver.js";

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
  const visibleActions = listVisibleActions(
    context.repos.actions,
    context.input.audienceEntityId,
    context.input.activeParticipants,
    sourceRecordLimit,
  );
  const threads = await buildActionThreads({
    records: visibleActions,
    repository: context.repos.actions,
    resolver: context.resolver,
    similarityThreshold,
  });
  const threadsWithSalience = threads.flatMap((thread) => {
    const salienceClass = actionSalienceClass({
      thread,
      currentUserStreamEntryId: context.input.currentUserEntry?.id,
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
        state: actionThreadState(thread),
        salience_class: thread.salienceClass,
        state_metadata: {
          ...actionThreadStateMetadata(thread, context.repos.entities),
          salience_class: thread.salienceClass,
        },
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

  addEntry(context.buckets, "action_states", {
    id: "action_threads:older_summary",
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: ACTION_TRUST_RANK,
    text: renderOlderActionThreadsSummary(olderThreads),
    value: "older_action_threads",
    state: "omitted",
    taint: "none",
  });
}
