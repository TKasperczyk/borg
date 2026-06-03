import type { RelationalSlot } from "../../memory/relational-slots/index.js";
import type { ActiveParticipant } from "../participants.js";
import {
  dedupeCommitments,
  dedupeGoals,
  isCommitmentVisibleToSession,
  isGoalVisibleToSession,
  scopedCommitmentsForEntity,
  scopedGoalsForEntity,
  visibleAudienceEntityIds,
} from "./audience-visibility.js";
import type { BuilderSectionContext } from "./builder-context.js";
import { optionalStateMetadata, slotTaint } from "./entry-metadata.js";
import {
  COMMITMENT_TRUST_RANK,
  CROSS_SESSION_ACTIVITY_TRUST_RANK,
  OPEN_QUESTION_TRUST_RANK,
  RELATIONAL_SLOT_LEDGER_LIMIT,
  SLOT_TRUST_RANK,
  cappedTrustRank,
} from "./section-buckets.js";
import {
  commitmentScope,
  persistenceClassFromProvenance,
  scopeFromStreamIds,
  slotScope,
} from "./scope-resolver.js";
import type { EvidenceLedgerAudienceStanding, EvidenceLedgerEntry } from "./types.js";
import {
  effectiveCommitmentCriticalDomain,
  effectiveCommitmentEnforcementClass,
} from "../../memory/commitments/index.js";

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

function buildCrossSessionActivityEntries(context: BuilderSectionContext): EvidenceLedgerEntry[] {
  const rows = context.input.crossSessionSelfActivity ?? [];

  return rows.map((row, index) => ({
    id: `cross_session_self_activity:${index + 1}`,
    source_type: "system_metadata",
    session_scope: "prior_session",
    actor: "system",
    trust_rank: CROSS_SESSION_ACTIVITY_TRUST_RANK,
    text: row.text,
    value: row.kind,
    state: "active",
    state_metadata: {
      event_kind: row.kind,
      occurred_at: row.occurredAt,
      relative_age: row.relativeAge,
    },
    taint: "none",
  }));
}

function buildCommitmentEntries(context: BuilderSectionContext): EvidenceLedgerEntry[] {
  return context.input.applicableCommitments.map((commitment) =>
    cappedTrustRank({
      id: `commitment:${commitment.id}`,
      source_type: "commitment",
      session_scope: commitmentScope(commitment, context.resolver),
      actor: "memory",
      trust_rank: COMMITMENT_TRUST_RANK,
      text: commitment.directive,
      value: commitment.directive_family,
      state:
        commitment.revoked_at !== null
          ? "revoked"
          : commitment.expired_at !== null
            ? "expired"
            : "active",
      state_metadata: {
        commitment_kind: commitment.kind,
        commitment_type: commitment.type,
        commitment_enforcement_class: effectiveCommitmentEnforcementClass(commitment),
        commitment_critical_domain: effectiveCommitmentCriticalDomain(commitment),
      },
      taint: "none",
      ...persistenceClassFromProvenance(
        { streamEntryIds: commitment.source_stream_entry_ids ?? [] },
        context.resolver,
      ),
    }),
  );
}

function buildRelationalEntries(context: BuilderSectionContext): EvidenceLedgerEntry[] {
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
  const entries: EvidenceLedgerEntry[] = [];

  for (const slot of slots.slice(0, RELATIONAL_SLOT_LEDGER_LIMIT)) {
    const participant = participantForSlot(slot, activeParticipants);
    entries.push(
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
      entries.push(
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
            commitment_kind: commitment.kind,
            commitment_type: commitment.type,
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
          ).filter((goal) => isGoalVisibleToSession(goal, audienceEntityId, activeParticipantIds));

    for (const goal of participantGoals) {
      entries.push({
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

  return entries;
}

export function buildAudienceStandingLedgerContext(
  context: BuilderSectionContext,
): EvidenceLedgerAudienceStanding {
  return {
    crossSessionActivityEntries: buildCrossSessionActivityEntries(context),
    commitmentEntries: buildCommitmentEntries(context),
    relationalEntries: buildRelationalEntries(context),
  };
}
