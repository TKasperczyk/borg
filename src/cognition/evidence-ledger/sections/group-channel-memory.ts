import {
  DEFAULT_ACTION_THREAD_RENDER_LIMIT,
  DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
  actionActorDisplay,
} from "../action-threads.js";
import {
  actionBelongsToGroupChannel,
  commitmentBelongsToGroupChannel,
  dedupeActions,
  goalBelongsToGroupChannel,
  scopedCommitmentsForEntity,
  scopedGoalsForEntity,
} from "../audience-visibility.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
  slotTaint,
} from "../entry-metadata.js";
import {
  ACTION_TRUST_RANK,
  COMMITMENT_TRUST_RANK,
  OPEN_QUESTION_TRUST_RANK,
  RELATIONAL_SLOT_LEDGER_LIMIT,
  SLOT_TRUST_RANK,
  addEntry,
  cappedTrustRank,
} from "../section-buckets.js";
import {
  actionScope,
  commitmentScope,
  persistenceClassFromProvenance,
  scopeFromStreamIds,
  slotScope,
} from "../scope-resolver.js";
import { commitmentDisclosureEntityIds } from "../../disclosure-labels.js";
import { relationshipPrivateMemoryDisclosureLabel } from "../../../retrieval/index.js";
import type { EntityId } from "../../../util/ids.js";

export function addGroupChannelMemorySection(context: BuilderSectionContext): void {
  const audienceEntityId = context.input.audienceEntityId;

  if (audienceEntityId === null) {
    return;
  }

  const audienceEntity = context.repos.entities?.get(audienceEntityId);

  if (audienceEntity?.kind !== "group") {
    return;
  }

  const displayName = audienceEntity.canonical_name;
  const groupDisclosureLabel = relationshipPrivateMemoryDisclosureLabel([audienceEntityId]);

  addEntry(context.buckets, "group_channel_memory", {
    id: `group_channel:${audienceEntityId}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: SLOT_TRUST_RANK,
    text: `Group/channel memory for ${displayName}. These entries belong to the channel, not to any active participant.`,
    value: displayName,
    state: appendMemoryDisclosureState({
      state: "group_channel",
      disclosureLabel: groupDisclosureLabel,
    }),
    state_metadata: appendMemoryDisclosureStateMetadata({
      stateMetadata: undefined,
      disclosureLabel: groupDisclosureLabel,
    }),
    taint: "none",
  });

  for (const slot of context.repos.relationalSlots
    .list({
      subjectEntityId: audienceEntityId,
      states: ["established", "contested", "quarantined"],
      limit: RELATIONAL_SLOT_LEDGER_LIMIT,
    })
    .slice(0, RELATIONAL_SLOT_LEDGER_LIMIT)) {
    addEntry(
      context.buckets,
      "group_channel_memory",
      cappedTrustRank({
        id: `group_relational_slot:${slot.id}`,
        source_type: "relational_slot",
        session_scope: slotScope(slot, context.resolver),
        actor: "memory",
        trust_rank: SLOT_TRUST_RANK,
        text:
          slot.alternate_values.length === 0
            ? undefined
            : `alternate_values=${slot.alternate_values.map((alternate) => alternate.value).join(", ")}`,
        value: `${slot.slot_key}=${slot.value}`,
        state: appendMemoryDisclosureState({
          state: slot.state,
          disclosureLabel: groupDisclosureLabel,
        }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: {
            subject_display_name: displayName,
            subject_role: "audience",
          },
          disclosureLabel: groupDisclosureLabel,
        }),
        taint: slotTaint(slot),
        ...persistenceClassFromProvenance(
          {
            streamEntryIds: [
              ...slot.evidence_stream_entry_ids,
              ...slot.contradicted_by_stream_entry_ids,
              ...slot.alternate_values.flatMap((alternate) => alternate.evidence_stream_entry_ids),
            ],
          },
          context.resolver,
        ),
      }),
    );
  }

  const scopedCommitments = scopedCommitmentsForEntity(
    context.repos.commitments?.list({
      activeOnly: true,
      audience: audienceEntityId,
    }) ?? context.input.applicableCommitments,
    audienceEntityId,
  ).filter((commitment) =>
    commitmentBelongsToGroupChannel(commitment, audienceEntityId, context.repos.entities),
  );

  for (const commitment of scopedCommitments) {
    const disclosureLabel = relationshipPrivateMemoryDisclosureLabel(
      commitmentDisclosureEntityIds(commitment),
    );
    addEntry(
      context.buckets,
      "group_channel_memory",
      cappedTrustRank({
        id: `group_commitment:${commitment.id}`,
        source_type: "commitment",
        session_scope: commitmentScope(commitment, context.resolver),
        actor: "memory",
        trust_rank: COMMITMENT_TRUST_RANK,
        text: commitment.directive,
        value: commitment.directive_family,
        state: appendMemoryDisclosureState({ state: "active", disclosureLabel }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: {
            commitment_kind: commitment.kind,
            commitment_type: commitment.type,
          },
          disclosureLabel,
        }),
        taint: "none",
        ...persistenceClassFromProvenance(
          { streamEntryIds: commitment.source_stream_entry_ids ?? [] },
          context.resolver,
        ),
      }),
    );
  }

  const scopedGoals = scopedGoalsForEntity(
    context.repos.goals?.list({
      status: "active",
      visibleToAudienceEntityId: audienceEntityId,
    }) ?? [],
    audienceEntityId,
  ).filter((goal) => goalBelongsToGroupChannel(goal, audienceEntityId, context.repos.entities));

  for (const goal of scopedGoals) {
    const disclosureLabel = relationshipPrivateMemoryDisclosureLabel(
      [goal.audience_entity_id, goal.owner_entity_id ?? null].filter(
        (entityId): entityId is EntityId => entityId !== null,
      ),
    );
    addEntry(context.buckets, "group_channel_memory", {
      id: `group_goal:${goal.id}`,
      source_type: "system_metadata",
      session_scope: scopeFromStreamIds(goal.source_stream_entry_ids ?? [], context.resolver),
      actor: "memory",
      trust_rank: OPEN_QUESTION_TRUST_RANK,
      text: goal.description,
      value: "goal",
      state: appendMemoryDisclosureState({ state: goal.status, disclosureLabel }),
      state_metadata: appendMemoryDisclosureStateMetadata({
        stateMetadata: undefined,
        disclosureLabel,
      }),
      taint: "none",
    });
  }

  for (const action of dedupeActions([
    ...context.repos.actions
      .list({
        audienceEntityId,
        limit: DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
      })
      .filter((record) =>
        actionBelongsToGroupChannel(record, audienceEntityId, context.repos.entities),
      ),
    ...context.repos.actions
      .list({
        actor: audienceEntityId,
        limit: DEFAULT_ACTION_THREAD_SOURCE_RECORD_LIMIT,
      })
      .filter((record) =>
        actionBelongsToGroupChannel(record, audienceEntityId, context.repos.entities),
      ),
  ]).slice(0, DEFAULT_ACTION_THREAD_RENDER_LIMIT)) {
    const disclosureLabel = relationshipPrivateMemoryDisclosureLabel(
      action.audience_entity_id === null ? [] : [action.audience_entity_id],
    );
    addEntry(
      context.buckets,
      "group_channel_memory",
      cappedTrustRank({
        id: `group_action:${action.id}`,
        source_type: "action_record",
        session_scope: actionScope(action, context.resolver),
        actor: action.actor === "borg" ? "assistant" : "user",
        trust_rank: ACTION_TRUST_RANK,
        text: action.description,
        value: actionActorDisplay(action.actor, context.repos.entities),
        state: appendMemoryDisclosureState({ state: action.state, disclosureLabel }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: undefined,
          disclosureLabel,
        }),
        taint: "none",
        ...persistenceClassFromProvenance(
          {
            streamEntryIds: action.provenance_stream_entry_ids,
            episodeIds: action.provenance_episode_ids,
          },
          context.resolver,
        ),
      }),
    );
  }
}
