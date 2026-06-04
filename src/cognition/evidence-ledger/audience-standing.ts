import type { RelationalSlot } from "../../memory/relational-slots/index.js";
import {
  relationshipPrivateMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../../retrieval/index.js";
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
import {
  appendMemoryDisclosureState,
  appendMemoryDisclosureStateMetadata,
  slotTaint,
} from "./entry-metadata.js";
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
import { resolveSpeakerDisplayName } from "../speaker-tags.js";
import type { CommitmentRecord, EntityKind } from "../../memory/commitments/index.js";
import type { GoalRecord } from "../../memory/self/index.js";
import type { EntityId } from "../../util/ids.js";
import {
  effectiveCommitmentCriticalDomain,
  effectiveCommitmentEnforcementClass,
} from "../../memory/commitments/index.js";
import { commitmentDisclosureEntityIds } from "../disclosure-labels.js";

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
  const disclosureLabel = selfPrivateMemoryDisclosureLabel();

  return rows.map((row, index) => ({
    id: `cross_session_self_activity:${index + 1}`,
    source_type: "system_metadata",
    session_scope: "prior_session",
    actor: "system",
    trust_rank: CROSS_SESSION_ACTIVITY_TRUST_RANK,
    text: row.text,
    value: row.kind,
    state: appendMemoryDisclosureState({ state: "active", disclosureLabel }),
    state_metadata: appendMemoryDisclosureStateMetadata({
      stateMetadata: {
        event_kind: row.kind,
        occurred_at: row.occurredAt,
        relative_age: row.relativeAge,
      },
      disclosureLabel,
    }),
    taint: "none",
  }));
}

function buildSelfDecisionIntrospectionEntries(
  context: BuilderSectionContext,
): EvidenceLedgerEntry[] {
  // Visibility is resolved upstream by selectSelfDecisionIntrospection.
  const rows = context.input.selfDecisionIntrospection ?? [];
  const disclosureLabel = selfPrivateMemoryDisclosureLabel();

  return rows.map((row, index) => ({
    id: `self_decision_introspection:${index + 1}`,
    source_type: "system_metadata",
    session_scope: "global",
    actor: "system",
    trust_rank: CROSS_SESSION_ACTIVITY_TRUST_RANK,
    text: row.text,
    value: row.triggerName,
    state: appendMemoryDisclosureState({ state: "active", disclosureLabel }),
    state_metadata: appendMemoryDisclosureStateMetadata({
      stateMetadata: {
        trigger_name: row.triggerName,
        trigger_type: row.triggerType,
        occurred_at: row.occurredAt,
        relative_age: row.relativeAge,
        disclosure_class: "self_private",
      },
      disclosureLabel,
    }),
    taint: "none",
  }));
}

function originAudienceKind(
  context: BuilderSectionContext,
  audienceEntityId: EntityId | null,
): EntityKind | null {
  if (audienceEntityId === null) {
    return null;
  }

  return context.repos.entities?.get(audienceEntityId)?.kind ?? null;
}

function originDescriptor(kind: EntityKind | null): string | null {
  if (kind === null) {
    return null;
  }

  return kind === "group" ? "group" : "one-to-one";
}

function observedEventText(input: {
  rowText: string;
  speakerDisplayName: string | null;
  originDescriptor: string | null;
}): string {
  const speakerSegment = input.speakerDisplayName;
  const originSegment = input.originDescriptor === null ? null : `in a ${input.originDescriptor}`;
  const provenance = [speakerSegment, originSegment]
    .filter((segment): segment is string => segment !== null)
    .join(" ");

  return provenance.length === 0 ? input.rowText : `${provenance}: ${input.rowText}`;
}

function buildObservedEventIntrospectionEntries(
  context: BuilderSectionContext,
): EvidenceLedgerEntry[] {
  const rows = context.input.observedEventIntrospection ?? [];

  return rows.map((row, index) => {
    const speakerDisplayName = resolveSpeakerDisplayName(
      context.repos.entities,
      row.speakerEntityId,
    );
    const audienceKind = originAudienceKind(context, row.audienceEntityId);
    const descriptor = originDescriptor(audienceKind);
    const disclosureLabel = observedEventDisclosureLabel(row);

    return {
      id: `observed_event_introspection:${index + 1}`,
      source_type: "system_metadata",
      session_scope: "prior_session",
      actor: "system",
      trust_rank: CROSS_SESSION_ACTIVITY_TRUST_RANK,
      text: observedEventText({
        rowText: row.text,
        speakerDisplayName,
        originDescriptor: descriptor,
      }),
      value: row.stance,
      state: appendMemoryDisclosureState({ state: "active", disclosureLabel }),
      state_metadata: appendMemoryDisclosureStateMetadata({
        stateMetadata: {
          observed_event_disclosure_class: row.disclosureClass,
          disclosure_class: row.disclosureClass,
          stance: row.stance,
          taint: row.taint,
          belief_effect: row.beliefEffect,
          recurrence_count: row.recurrenceCount,
          occurred_at: row.occurredAt,
          relative_age: row.relativeAge,
          speaker_entity_id: row.speakerEntityId,
          speaker_display_name: speakerDisplayName,
          audience_entity_id: row.audienceEntityId,
          origin_audience_kind: audienceKind,
        },
        disclosureLabel,
      }),
      taint: "none",
    };
  });
}

function observedEventDisclosureLabel(row: {
  disclosureClass: "social_observed" | "self_private";
  speakerEntityId: EntityId | null;
  audienceEntityId: EntityId | null;
}): MemoryDisclosureLabel {
  const originEntityIds = [row.audienceEntityId, row.speakerEntityId].filter(
    (entityId): entityId is EntityId => entityId !== null,
  );

  return row.disclosureClass === "self_private"
    ? selfPrivateMemoryDisclosureLabel(originEntityIds)
    : relationshipPrivateMemoryDisclosureLabel(originEntityIds);
}

function commitmentDisclosureLabel(commitment: CommitmentRecord): MemoryDisclosureLabel {
  return relationshipPrivateMemoryDisclosureLabel(commitmentDisclosureEntityIds(commitment));
}

function relationalSlotDisclosureLabel(slot: RelationalSlot): MemoryDisclosureLabel {
  return relationshipPrivateMemoryDisclosureLabel([slot.subject_entity_id]);
}

function goalDisclosureLabel(goal: GoalRecord): MemoryDisclosureLabel {
  return relationshipPrivateMemoryDisclosureLabel(
    [goal.audience_entity_id, goal.owner_entity_id ?? null].filter(
      (entityId): entityId is EntityId => entityId !== null,
    ),
  );
}

function buildCommitmentEntries(context: BuilderSectionContext): EvidenceLedgerEntry[] {
  return context.input.applicableCommitments.map((commitment) => {
    const disclosureLabel = commitmentDisclosureLabel(commitment);

    return cappedTrustRank({
      id: `commitment:${commitment.id}`,
      source_type: "commitment",
      session_scope: commitmentScope(commitment, context.resolver),
      actor: "memory",
      trust_rank: COMMITMENT_TRUST_RANK,
      text: commitment.directive,
      value: commitment.directive_family,
      state: appendMemoryDisclosureState({
        state:
          commitment.revoked_at !== null
            ? "revoked"
            : commitment.expired_at !== null
              ? "expired"
              : "active",
        disclosureLabel,
      }),
      state_metadata: appendMemoryDisclosureStateMetadata({
        stateMetadata: {
          commitment_kind: commitment.kind,
          commitment_type: commitment.type,
          commitment_enforcement_class: effectiveCommitmentEnforcementClass(commitment),
          commitment_critical_domain: effectiveCommitmentCriticalDomain(commitment),
        },
        disclosureLabel,
      }),
      taint: "none",
      ...persistenceClassFromProvenance(
        { streamEntryIds: commitment.source_stream_entry_ids ?? [] },
        context.resolver,
      ),
    });
  });
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
    const disclosureLabel = relationalSlotDisclosureLabel(slot);
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
        state: appendMemoryDisclosureState({ state: slot.state, disclosureLabel }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: slotSubjectStateMetadata(
            slot,
            participant,
            activeParticipants?.length ?? 0,
          ),
          disclosureLabel,
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
      const disclosureLabel = commitmentDisclosureLabel(commitment);
      entries.push(
        cappedTrustRank({
          id: `participant_commitment:${participant.entityId}:${commitment.id}`,
          source_type: "commitment",
          session_scope: commitmentScope(commitment, context.resolver),
          actor: "memory",
          trust_rank: COMMITMENT_TRUST_RANK,
          text: commitment.directive,
          value: `${participant.displayName ?? "participant"}:${commitment.directive_family}`,
          state: appendMemoryDisclosureState({ state: "active", disclosureLabel }),
          state_metadata: appendMemoryDisclosureStateMetadata({
            stateMetadata: {
              subject_display_name: participant.displayName ?? "participant",
              subject_role: participant.role,
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
      const disclosureLabel = goalDisclosureLabel(goal);
      entries.push({
        id: `participant_goal:${participant.entityId}:${goal.id}`,
        source_type: "system_metadata",
        session_scope: scopeFromStreamIds(goal.source_stream_entry_ids ?? [], context.resolver),
        actor: "memory",
        trust_rank: OPEN_QUESTION_TRUST_RANK,
        text: goal.description,
        value: `${participant.displayName ?? "participant"}:goal`,
        state: appendMemoryDisclosureState({ state: goal.status, disclosureLabel }),
        state_metadata: appendMemoryDisclosureStateMetadata({
          stateMetadata: {
            subject_display_name: participant.displayName ?? "participant",
            subject_role: participant.role,
          },
          disclosureLabel,
        }),
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
    selfDecisionIntrospectionEntries: buildSelfDecisionIntrospectionEntries(context),
    observedEventIntrospectionEntries: buildObservedEventIntrospectionEntries(context),
    commitmentEntries: buildCommitmentEntries(context),
    relationalEntries: buildRelationalEntries(context),
  };
}
