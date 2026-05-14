import type { RelationalSlot } from "../../../memory/relational-slots/index.js";
import type { ActiveParticipant } from "../../participants.js";
import {
  dedupeCommitments,
  dedupeGoals,
  isCommitmentVisibleToSession,
  isGoalVisibleToSession,
  scopedCommitmentsForEntity,
  scopedGoalsForEntity,
  visibleAudienceEntityIds,
} from "../audience-visibility.js";
import type { BuilderSectionContext } from "../builder-context.js";
import {
  optionalStateMetadata,
  slotTaint,
} from "../entry-metadata.js";
import {
  COMMITMENT_TRUST_RANK,
  OPEN_QUESTION_TRUST_RANK,
  RELATIONAL_SLOT_LEDGER_LIMIT,
  SLOT_TRUST_RANK,
  addEntry,
  cappedTrustRank,
} from "../section-buckets.js";
import {
  commitmentScope,
  persistenceClassFromProvenance,
  scopeFromStreamIds,
  slotScope,
} from "../scope-resolver.js";

function participantForSlot(
  slot: RelationalSlot,
  participants: readonly ActiveParticipant[] | undefined,
): ActiveParticipant | undefined {
  return participants?.find((participant) => participant.entityId === slot.subject_entity_id);
}

function slotSubjectStateMetadata(
  slot: RelationalSlot,
  participant: ActiveParticipant | undefined,
  participantCount: number,
): Record<string, unknown> | undefined {
  if (participant === undefined || participantCount <= 1) {
    return undefined;
  }

  return {
    subject_entity_id: slot.subject_entity_id,
    subject_display_name: participant.displayName ?? slot.subject_entity_id,
    subject_role: participant.role,
  };
}

export function addRelationalSlotsSection(context: BuilderSectionContext): void {
  const audienceEntityId = context.input.audienceEntityId;
  const activeParticipants = context.input.activeParticipants;
  const activeParticipantIds = visibleAudienceEntityIds(audienceEntityId, activeParticipants);
  const slots =
    activeParticipants === undefined || activeParticipants.length === 0
      ? context.repos.relationalSlots.list({
          states: ["established", "contested", "quarantined"],
          limit: RELATIONAL_SLOT_LEDGER_LIMIT,
        })
      : activeParticipants.flatMap((participant) =>
          context.repos.relationalSlots.list({
            subjectEntityId: participant.entityId,
            states: ["established", "contested", "quarantined"],
            limit: RELATIONAL_SLOT_LEDGER_LIMIT,
          }),
        );
  const cappedSlots = slots.slice(0, RELATIONAL_SLOT_LEDGER_LIMIT);

  for (const slot of cappedSlots) {
    const participant = participantForSlot(slot, activeParticipants);
    addEntry(
      context.buckets,
      "relational_slots",
      cappedTrustRank({
        id: `relational_slot:${slot.id}`,
        source_type: "relational_slot",
        session_scope: slotScope(slot, context.resolver),
        actor: "memory",
        trust_rank: SLOT_TRUST_RANK,
        text:
          slot.alternate_values.length === 0
            ? undefined
            : `alternate_values=${slot.alternate_values.map((alternate) => alternate.value).join(", ")}`,
        value: `${slot.slot_key}=${slot.value}`,
        state: slot.state,
        ...optionalStateMetadata(
          slotSubjectStateMetadata(slot, participant, activeParticipants?.length ?? 0),
        ),
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

  for (const participant of activeParticipants ?? []) {
    const participantCommitments =
      context.repos.commitments === undefined
        ? []
        : scopedCommitmentsForEntity(
            dedupeCommitments([
              ...context.repos.commitments.list({
                activeOnly: true,
                committedByEntity: participant.entityId,
              }),
              ...context.repos.commitments.list({
                activeOnly: true,
                audience: participant.entityId,
              }),
            ]),
            participant.entityId,
          ).filter((commitment) =>
            isCommitmentVisibleToSession(commitment, audienceEntityId, activeParticipantIds),
          );

    for (const commitment of participantCommitments) {
      addEntry(
        context.buckets,
        "relational_slots",
        cappedTrustRank({
          id: `participant_commitment:${participant.entityId}:${commitment.id}`,
          source_type: "commitment",
          session_scope: commitmentScope(commitment, context.resolver),
          actor: "memory",
          trust_rank: COMMITMENT_TRUST_RANK,
          text: commitment.directive,
          value: `${participant.displayName ?? "participant"}:${commitment.directive_family}`,
          state: "active",
          state_metadata: {
            subject_display_name: participant.displayName ?? "participant",
            subject_role: participant.role,
          },
          taint: "none",
          ...persistenceClassFromProvenance(
            { streamEntryIds: commitment.source_stream_entry_ids ?? [] },
            context.resolver,
          ),
        }),
      );
    }

    const participantGoals =
      context.repos.goals === undefined
        ? []
        : scopedGoalsForEntity(
            dedupeGoals([
              ...context.repos.goals.list({
                status: "active",
                ownerEntityId: participant.entityId,
              }),
              ...context.repos.goals.list({
                status: "active",
                visibleToAudienceEntityId: participant.entityId,
              }),
            ]),
            participant.entityId,
          ).filter((goal) =>
            isGoalVisibleToSession(goal, audienceEntityId, activeParticipantIds),
          );

    for (const goal of participantGoals) {
      addEntry(context.buckets, "relational_slots", {
        id: `participant_goal:${participant.entityId}:${goal.id}`,
        source_type: "system_metadata",
        session_scope: scopeFromStreamIds(goal.source_stream_entry_ids ?? [], context.resolver),
        actor: "memory",
        trust_rank: OPEN_QUESTION_TRUST_RANK,
        text: goal.description,
        value: `${participant.displayName ?? "participant"}:goal`,
        state: goal.status,
        state_metadata: {
          subject_display_name: participant.displayName ?? "participant",
          subject_role: participant.role,
        },
        taint: "none",
      });
    }
  }
}
