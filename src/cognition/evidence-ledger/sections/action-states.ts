import {
  DEFAULT_ACTION_THREAD_AUDIENCE_RESERVED_SLOTS,
  DEFAULT_ACTION_THREAD_RENDER_LIMIT,
  DEFAULT_ACTION_THREAD_SALIENCE_CLASS_RESERVED_SLOTS,
  DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD,
  DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
  STALE_PARTICIPANT_ACTION_RENDER_LIMIT,
  actionThreadAudienceBucket,
  actionSalienceClass,
  actionActorDisplay,
  actionThreadState,
  actionThreadStateMetadata,
  allocateActionThreadRenderSlots,
  buildActionThreads,
  listActionCandidatesForCognition,
  orderActionThreadsBySalience,
  renderActionThreadText,
  renderOlderActionThreadsSummary,
  type OlderActionThreadSummaryGroup,
} from "../action-threads.js";
import {
  clampPositiveIntegerOrFallback,
  coerceUnitIntervalOrFallback,
} from "../../../util/math.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "../entry-metadata.js";
import { ACTION_TRUST_RANK, addEntry, cappedTrustRank } from "../section-buckets.js";
import { persistenceClassFromProvenance } from "../scope-resolver.js";
import { combineMemoryDisclosureLabels } from "../../../retrieval/index.js";
import { actionMemoryDisclosureLabel } from "../../../memory/common/disclosure-serializers.js";

function clampNonnegativeIntegerOrFallback(value: number | undefined, fallback: number): number {
  if (value === undefined || !Number.isFinite(value)) {
    return fallback;
  }

  return Math.max(0, Math.floor(value));
}

export async function addActionStatesSection(context: BuilderSectionContext): Promise<void> {
  const sourceRecordLimit = clampPositiveIntegerOrFallback(
    context.options.actionThreadSourceRecordLimit,
    DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
  );
  const renderLimit = clampPositiveIntegerOrFallback(
    context.options.actionThreadRenderLimit,
    DEFAULT_ACTION_THREAD_RENDER_LIMIT,
  );
  const similarityThreshold = coerceUnitIntervalOrFallback(
    context.options.actionThreadSimilarityThreshold,
    DEFAULT_ACTION_THREAD_SIMILARITY_THRESHOLD,
  );
  const salienceClassReservedSlots = clampNonnegativeIntegerOrFallback(
    context.options.actionThreadSalienceClassReservedSlots,
    DEFAULT_ACTION_THREAD_SALIENCE_CLASS_RESERVED_SLOTS,
  );
  const audienceReservedSlots = clampNonnegativeIntegerOrFallback(
    context.options.actionThreadAudienceReservedSlots,
    DEFAULT_ACTION_THREAD_AUDIENCE_RESERVED_SLOTS,
  );
  // Future work: reservations start after the bounded source pool is classified.
  // The pre-classification 256-record pool remains repository-ranked and unreserved.
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
      currentTurnGlobal: context.input.globalTurnCounter,
    });

    return salienceClass === null ? [] : [{ ...thread, salienceClass }];
  });
  const staleParticipantThreads = threadsWithSalience.filter(
    (thread) => thread.salienceClass === "participant_pending_stale",
  );
  const cappedStaleIds = new Set(
    staleParticipantThreads.slice(STALE_PARTICIPANT_ACTION_RENDER_LIMIT).map((thread) => thread.id),
  );
  const renderedThreads = allocateActionThreadRenderSlots({
    threads: threadsWithSalience.filter((thread) => !cappedStaleIds.has(thread.id)),
    limit: renderLimit,
    salienceClassReservedSlots,
    audienceReservedSlots,
    audienceOrder: [
      ...new Set(
        actionCandidates.map((candidate) => candidate.record.audience_entity_id ?? "global"),
      ),
    ],
  });

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

  const renderedIds = new Set(renderedThreads.map((thread) => thread.id));
  const olderThreads = orderActionThreadsBySalience(
    threadsWithSalience.filter((thread) => !renderedIds.has(thread.id)),
  );

  if (olderThreads.length === 0) {
    return;
  }

  const groupedThreads: Array<
    Omit<OlderActionThreadSummaryGroup, "threads" | "disclosureLabel"> & {
      threads: typeof olderThreads;
    }
  > = [];

  for (const thread of olderThreads) {
    const audienceScope = actionThreadAudienceBucket(thread);
    const group = groupedThreads.find(
      (candidate) =>
        candidate.audienceScope === audienceScope &&
        candidate.salienceClass === thread.salienceClass,
    );

    if (group === undefined) {
      groupedThreads.push({
        audienceScope,
        salienceClass: thread.salienceClass,
        threads: [thread],
      });
    } else {
      group.threads.push(thread);
    }
  }

  const summaryGroups: OlderActionThreadSummaryGroup[] = groupedThreads.map((group) => ({
    ...group,
    disclosureLabel: combineMemoryDisclosureLabels(
      group.threads.flatMap((thread) =>
        thread.records.map(
          (record) =>
            disclosureLabelByActionId.get(record.id) ?? actionMemoryDisclosureLabel(record),
        ),
      ),
    ),
  }));

  addEntry(context.buckets, "action_states", {
    id: "action_threads:older_summary",
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: ACTION_TRUST_RANK,
    text: renderOlderActionThreadsSummary(summaryGroups),
    value: "older_action_threads",
    state: "omitted",
    taint: "none",
  });
}
