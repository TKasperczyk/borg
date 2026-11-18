import {
  DEFAULT_ACTION_THREAD_AUDIENCE_RESERVED_SLOTS,
  DEFAULT_ACTION_THREAD_SALIENCE_CLASS_RESERVED_SLOTS,
  DEFAULT_ACTION_THREAD_RENDER_LIMIT,
  DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
  ACTION_UPDATE_HISTORY_LIMIT,
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
import { clampPositiveIntegerOrFallback } from "../../../util/math.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
} from "../entry-metadata.js";
import { ACTION_TRUST_RANK, addEntry, cappedTrustRank } from "../section-buckets.js";
import { persistenceClassFromProvenance } from "../scope-resolver.js";
import { combineMemoryDisclosureLabels } from "../../../retrieval/index.js";
import { actionMemoryDisclosureLabel } from "../../../memory/common/disclosure-serializers.js";
import { isStaleOpenAction } from "../../../memory/actions/index.js";

function clampNonnegativeIntegerOrFallback(value: number | undefined, fallback: number): number {
  return value === undefined || !Number.isFinite(value) ? fallback : Math.max(0, Math.floor(value));
}

export async function addActionStatesSection(context: BuilderSectionContext): Promise<void> {
  const nowMs = context.nowMs ?? context.input.currentUserEntry?.timestamp ?? Date.now();
  const sourceRecordLimit = clampPositiveIntegerOrFallback(
    context.options.actionThreadSourceRecordLimit,
    DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
  );
  const renderLimit = clampPositiveIntegerOrFallback(
    context.options.actionThreadRenderLimit,
    DEFAULT_ACTION_THREAD_RENDER_LIMIT,
  );
  // Use one turn-time snapshot for the draw, render budget, and global omission count.
  const actionCandidates = listActionCandidatesForCognition({
    actionRepository: context.repos.actions,
    audienceEntityId: context.input.audienceEntityId,
    activeParticipants: context.input.activeParticipants,
    rankParticipantEntityIds:
      context.input.currentUserEntry?.sender_entity_id == null
        ? []
        : [context.input.currentUserEntry.sender_entity_id],
    limit: sourceRecordLimit,
    nowMs,
  });
  const disclosureLabelByActionId = new Map(
    actionCandidates.map((candidate) => [candidate.record.id, candidate.disclosureLabel]),
  );
  const threads = await buildActionThreads({
    records: actionCandidates.map((candidate) => candidate.record),
    repository: context.repos.actions,
    resolver: context.resolver,
  });
  const threadsWithSalience = threads.map((thread) => {
    const salienceClass = actionSalienceClass({
      thread,
      currentUserStreamEntryId: context.input.currentUserEntry?.id,
      currentUserStreamEntryIds: context.input.currentUserEntries?.map((entry) => entry.id),
      nowMs,
    });

    return { ...thread, salienceClass };
  });
  const renderedThreads = allocateActionThreadRenderSlots({
    threads: threadsWithSalience,
    nowMs,
    limit: renderLimit,
    audienceEntityId: context.input.audienceEntityId,
    audienceReservedSlots: clampNonnegativeIntegerOrFallback(
      context.options.actionThreadAudienceReservedSlots,
      DEFAULT_ACTION_THREAD_AUDIENCE_RESERVED_SLOTS,
    ),
    salienceClassReservedSlots: clampNonnegativeIntegerOrFallback(
      context.options.actionThreadSalienceClassReservedSlots,
      DEFAULT_ACTION_THREAD_SALIENCE_CLASS_RESERVED_SLOTS,
    ),
    reservedActors: [
      "borg",
      ...(context.input.currentUserEntry?.sender_entity_id == null
        ? []
        : [context.input.currentUserEntry.sender_entity_id]),
      ...(context.input.activeParticipants ?? [])
        .filter((p) => p.role === "speaker")
        .map((p) => p.entityId),
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
          renderActionThreadText(
            thread,
            context.repos.entities,
            context.repos.actions.readUpdateHistory?.(
              thread.current.id,
              ACTION_UPDATE_HISTORY_LIMIT,
            ),
          ),
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
    nowMs,
  );
  const salienceDroppedThreadCount = threads.length - threadsWithSalience.length;
  const drawSaturated = actionCandidates.length >= sourceRecordLimit;
  // The draw stops at the limit, so this section only ever sees a prefix of the store. The size of
  // the rest is one count away, which is what turns the below-floor field from a refusal into a
  // measurement; a repository that cannot count leaves the field saying so rather than implying 0.
  const sourceRecordTotal = context.repos.actions.count?.() ?? null;
  const storeStaleOpenCount = context.repos.actions.countStaleOpen?.(nowMs);
  const staleOpenOmitted = Math.max(
    0,
    (storeStaleOpenCount ??
      threads.filter((thread) => isStaleOpenAction(thread.current, nowMs)).length) -
      renderedThreads.filter((thread) => isStaleOpenAction(thread.current, nowMs)).length,
  );

  // Absent must not be the only way to say "nothing left over": a saturated draw or a
  // salience-dropped thread is still material this section did not show, so the entry is
  // emitted whenever any of the three populations is non-empty, including in the negative.
  if (olderThreads.length === 0 && salienceDroppedThreadCount === 0 && !drawSaturated) {
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
    text: renderOlderActionThreadsSummary({
      groups: summaryGroups,
      renderedThreadCount: renderedThreads.length,
      renderLimit,
      threadsBuiltCount: threads.length,
      consideredRecordCount: actionCandidates.length,
      sourceRecordLimit,
      sourceRecordTotal,
      salienceDroppedThreadCount,
      staleOpenOmitted,
      staleOpenCountScope: storeStaleOpenCount === undefined ? "draw" : "store",
    }),
    value: "older_action_threads",
    state: "omitted",
    taint: "none",
  });
}
