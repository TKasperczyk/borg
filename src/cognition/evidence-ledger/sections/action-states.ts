import {
  DEFAULT_ACTION_THREAD_RENDER_LIMIT,
  DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD,
  DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
  actionActorDisplay,
  actionThreadState,
  actionThreadStateMetadata,
  buildActionThreads,
  listVisibleActions,
  normalizePositiveInteger,
  normalizeUnitInterval,
  renderActionThreadText,
  renderOlderActionThreadsSummary,
} from "../action-threads.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  ACTION_TRUST_RANK,
  addEntry,
  cappedTrustRank,
} from "../section-buckets.js";
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
  const renderedThreads = threads.slice(0, renderLimit);

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
        text: renderActionThreadText(thread, context.repos.entities),
        value: actionActorDisplay(thread.current.actor, context.repos.entities),
        state: actionThreadState(thread),
        state_metadata: actionThreadStateMetadata(thread, context.repos.entities),
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

  if (threads.length <= renderLimit) {
    return;
  }

  const olderThreads = threads.slice(renderLimit);

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
