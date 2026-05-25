import type { EntityRecord, EntityRepository } from "../../memory/commitments/index.js";
import type {
  RelationalSlot,
  RelationalSlotRepository,
} from "../../memory/relational-slots/index.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import type { ActiveParticipant, ActiveParticipantRole } from "../participants.js";

const DEFAULT_ROSTER_RELATIONAL_SLOT_LIMIT = 64;
const ROSTER_SLOT_STATES = ["established", "contested", "quarantined"] as const;

export type ParticipantRosterAudienceRole = "speaker" | "active_participant" | "audience";

export type ParticipantRosterMember = {
  entity_id: EntityId;
  display_name: string;
  known_relationships: string[];
  audience_role: ParticipantRosterAudienceRole;
  relationship_source: string | null;
  relationship_sources?: string[];
};

export type ParticipantRosterSubject = {
  entity_id: EntityId;
  display_name: string;
  known_relationships: string[];
  relationship_source: string | null;
  relationship_sources?: string[];
};

export type ParticipantRosterUncertain = {
  entity_id: EntityId | null;
  display_name: string | null;
  known_relationships: string[];
  reason: string;
  relationship_source: string | null;
  relationship_sources?: string[];
};

export type ParticipantRoster = {
  participants: ParticipantRosterMember[];
  non_chat_subjects: ParticipantRosterSubject[];
  unknown_or_uncertain: ParticipantRosterUncertain[];
};

export type ParticipantRosterStreamEvidence = {
  entity_id: EntityId;
  display_name?: string | null;
  known_relationship: string;
  source_stream_entry_id: StreamEntryId;
};

export type BuildParticipantRosterInput = {
  activeParticipants: readonly ActiveParticipant[];
  audienceEntityId?: EntityId | null;
  entityRepository: Pick<EntityRepository, "get" | "findByName">;
  relationalSlots: readonly RelationalSlot[];
  streamEvidence?: readonly ParticipantRosterStreamEvidence[];
};

export type BuildParticipantRosterFromRepositoriesInput = {
  activeParticipants: readonly ActiveParticipant[];
  audienceEntityId?: EntityId | null;
  entityRepository: Pick<EntityRepository, "get" | "findByName">;
  relationalSlotRepository?: Pick<RelationalSlotRepository, "list">;
  slotLimit?: number;
  streamEvidence?: readonly ParticipantRosterStreamEvidence[];
};

function rosterScopeEntityIds(input: {
  activeParticipants: readonly ActiveParticipant[];
  audienceEntityId?: EntityId | null;
}): Set<EntityId> {
  return new Set([
    ...input.activeParticipants.map((participant) => participant.entityId),
    ...(input.audienceEntityId === undefined || input.audienceEntityId === null
      ? []
      : [input.audienceEntityId]),
  ]);
}

function displayNameForEntity(entity: Pick<EntityRecord, "canonical_name"> | null): string | null {
  return entity?.canonical_name ?? null;
}

function displayNameForEntityId(
  entityRepository: Pick<EntityRepository, "get">,
  entityId: EntityId,
): string {
  return displayNameForEntity(entityRepository.get(entityId)) ?? entityId;
}

function rosterRole(role: ActiveParticipantRole): ParticipantRosterAudienceRole {
  if (role === "speaker") {
    return "speaker";
  }

  if (role === "audience") {
    return "audience";
  }

  return "active_participant";
}

function relationshipSourceForSlot(slot: Pick<RelationalSlot, "id">): string {
  return `relational_slot:${slot.id}`;
}

function relationshipTextForSlot(slot: Pick<RelationalSlot, "slot_key" | "value">): string {
  return `${slot.slot_key}:${slot.value}`;
}

function relationalSlotScopesToRoster(input: {
  slot: RelationalSlot;
  scopedEntityIds: ReadonlySet<EntityId>;
  entityRepository: Pick<EntityRepository, "findByName">;
}): boolean {
  if (input.scopedEntityIds.has(input.slot.subject_entity_id)) {
    return true;
  }

  const valueEntityId = input.entityRepository.findByName(input.slot.value);

  return valueEntityId !== null && input.scopedEntityIds.has(valueEntityId);
}

function pushUnique(target: string[], value: string): void {
  if (!target.some((existing) => existing === value)) {
    target.push(value);
  }
}

function primarySource(sources: readonly string[]): string | null {
  return sources[0] ?? null;
}

function createSubjectRecord(input: {
  entityId: EntityId;
  displayName: string;
  relationship: string;
  source: string;
}): ParticipantRosterSubject {
  return {
    entity_id: input.entityId,
    display_name: input.displayName,
    known_relationships: [input.relationship],
    relationship_source: input.source,
  };
}

export function buildParticipantRoster(input: BuildParticipantRosterInput): ParticipantRoster {
  const participantRecords = new Map<
    EntityId,
    ParticipantRosterMember & { relationship_sources: string[] }
  >();
  const nonChatRecords = new Map<
    EntityId,
    ParticipantRosterSubject & { relationship_sources: string[] }
  >();
  const uncertain: ParticipantRosterUncertain[] = [];

  for (const participant of input.activeParticipants) {
    const existing = participantRecords.get(participant.entityId);
    const displayName =
      participant.displayName ??
      displayNameForEntityId(input.entityRepository, participant.entityId);

    if (existing !== undefined) {
      if (existing.audience_role !== "speaker" && participant.role === "speaker") {
        existing.audience_role = "speaker";
      }
      continue;
    }

    participantRecords.set(participant.entityId, {
      entity_id: participant.entityId,
      display_name: displayName,
      known_relationships: [],
      audience_role: rosterRole(participant.role),
      relationship_source: null,
      relationship_sources: [],
    });
  }

  if (input.audienceEntityId !== undefined && input.audienceEntityId !== null) {
    const audience = input.entityRepository.get(input.audienceEntityId);

    if (audience !== null && audience.kind !== "group" && !participantRecords.has(audience.id)) {
      participantRecords.set(audience.id, {
        entity_id: audience.id,
        display_name: audience.canonical_name,
        known_relationships: [],
        audience_role: "audience",
        relationship_source: null,
        relationship_sources: [],
      });
    }
  }

  for (const slot of input.relationalSlots) {
    const relationship = relationshipTextForSlot(slot);
    const source = relationshipSourceForSlot(slot);
    const subjectParticipant = participantRecords.get(slot.subject_entity_id);

    if (slot.state !== "established") {
      uncertain.push({
        entity_id: slot.subject_entity_id,
        display_name: displayNameForEntityId(input.entityRepository, slot.subject_entity_id),
        known_relationships: [relationship],
        reason: `relational_slot_state:${slot.state}`,
        relationship_source: source,
        relationship_sources: [source],
      });
      continue;
    }

    if (subjectParticipant !== undefined) {
      pushUnique(subjectParticipant.known_relationships, relationship);
      pushUnique(subjectParticipant.relationship_sources, source);
      subjectParticipant.relationship_source = primarySource(
        subjectParticipant.relationship_sources,
      );
    } else {
      const subjectEntity = input.entityRepository.get(slot.subject_entity_id);

      if (subjectEntity !== null) {
        const existing = nonChatRecords.get(slot.subject_entity_id);

        if (existing === undefined) {
          nonChatRecords.set(slot.subject_entity_id, {
            ...createSubjectRecord({
              entityId: slot.subject_entity_id,
              displayName: subjectEntity.canonical_name,
              relationship,
              source,
            }),
            relationship_sources: [source],
          });
        } else {
          pushUnique(existing.known_relationships, relationship);
          pushUnique(existing.relationship_sources, source);
          existing.relationship_source = primarySource(existing.relationship_sources);
        }
      }
    }

    const valueEntityId = input.entityRepository.findByName(slot.value);

    if (
      valueEntityId !== null &&
      !participantRecords.has(valueEntityId) &&
      !nonChatRecords.has(valueEntityId)
    ) {
      nonChatRecords.set(valueEntityId, {
        entity_id: valueEntityId,
        display_name: displayNameForEntityId(input.entityRepository, valueEntityId),
        known_relationships: [],
        relationship_source: source,
        relationship_sources: [source],
      });
    }
  }

  for (const evidence of input.streamEvidence ?? []) {
    const source = `stream_entry:${evidence.source_stream_entry_id}`;
    const participant = participantRecords.get(evidence.entity_id);

    if (participant !== undefined) {
      pushUnique(participant.known_relationships, evidence.known_relationship);
      pushUnique(participant.relationship_sources, source);
      participant.relationship_source = primarySource(participant.relationship_sources);
      continue;
    }

    const existing = nonChatRecords.get(evidence.entity_id);
    const displayName =
      evidence.display_name ?? displayNameForEntityId(input.entityRepository, evidence.entity_id);

    if (existing === undefined) {
      nonChatRecords.set(evidence.entity_id, {
        entity_id: evidence.entity_id,
        display_name: displayName,
        known_relationships: [evidence.known_relationship],
        relationship_source: source,
        relationship_sources: [source],
      });
    } else {
      pushUnique(existing.known_relationships, evidence.known_relationship);
      pushUnique(existing.relationship_sources, source);
      existing.relationship_source = primarySource(existing.relationship_sources);
    }
  }

  return {
    participants: [...participantRecords.values()].map(
      ({ relationship_sources: _, ...record }) => ({
        ...record,
        relationship_sources: [..._],
        known_relationships: [...record.known_relationships].sort((left, right) =>
          left.localeCompare(right),
        ),
      }),
    ),
    non_chat_subjects: [...nonChatRecords.values()].map(
      ({ relationship_sources: _, ...record }) => ({
        ...record,
        relationship_sources: [..._],
        known_relationships: [...record.known_relationships].sort((left, right) =>
          left.localeCompare(right),
        ),
      }),
    ),
    unknown_or_uncertain: uncertain,
  };
}

export function buildParticipantRosterFromRepositories(
  input: BuildParticipantRosterFromRepositoriesInput,
): ParticipantRoster {
  const scopedEntityIds = rosterScopeEntityIds(input);

  if (scopedEntityIds.size === 0) {
    return buildParticipantRoster({
      activeParticipants: input.activeParticipants,
      audienceEntityId: input.audienceEntityId,
      entityRepository: input.entityRepository,
      relationalSlots: [],
      streamEvidence: input.streamEvidence,
    });
  }

  const slotLimit = input.slotLimit ?? DEFAULT_ROSTER_RELATIONAL_SLOT_LIMIT;
  const slotsById = new Map<RelationalSlot["id"], RelationalSlot>();

  for (const subjectEntityId of scopedEntityIds) {
    for (const slot of input.relationalSlotRepository?.list({
      subjectEntityId,
      states: ROSTER_SLOT_STATES,
      limit: slotLimit,
    }) ?? []) {
      slotsById.set(slot.id, slot);
    }
  }

  for (const slot of input.relationalSlotRepository?.list({
    states: ROSTER_SLOT_STATES,
    limit: slotLimit,
  }) ?? []) {
    if (
      relationalSlotScopesToRoster({
        slot,
        scopedEntityIds,
        entityRepository: input.entityRepository,
      })
    ) {
      slotsById.set(slot.id, slot);
    }
  }

  return buildParticipantRoster({
    activeParticipants: input.activeParticipants,
    audienceEntityId: input.audienceEntityId,
    entityRepository: input.entityRepository,
    relationalSlots: [...slotsById.values()],
    streamEvidence: input.streamEvidence,
  });
}

function renderRelationshipSummary(relationships: readonly string[]): string | null {
  if (relationships.length === 0) {
    return null;
  }

  return `relationships: ${relationships.join(", ")}`;
}

function renderSource(source: string | null): string | null {
  return source === null ? null : `source: ${source}`;
}

function renderParts(parts: readonly (string | null)[]): string {
  return parts.filter((part): part is string => part !== null).join("; ");
}

export function renderParticipantRoster(
  roster: ParticipantRoster | null | undefined,
): string | null {
  if (
    roster === null ||
    roster === undefined ||
    (roster.participants.length === 0 &&
      roster.non_chat_subjects.length === 0 &&
      roster.unknown_or_uncertain.length === 0)
  ) {
    return null;
  }

  const lines = ["Thread roster:"];

  for (const participant of roster.participants) {
    lines.push(
      `- ${participant.display_name} (${renderParts([
        `id: ${participant.entity_id}`,
        participant.audience_role,
        renderRelationshipSummary(participant.known_relationships),
        renderSource(participant.relationship_source),
      ])})`,
    );
  }

  if (roster.non_chat_subjects.length > 0) {
    lines.push("Non-chat subjects:");

    for (const subject of roster.non_chat_subjects) {
      lines.push(
        `- ${subject.display_name} (${renderParts([
          `id: ${subject.entity_id}`,
          renderRelationshipSummary(subject.known_relationships),
          renderSource(subject.relationship_source),
        ])})`,
      );
    }
  }

  if (roster.unknown_or_uncertain.length > 0) {
    lines.push("Unknown or uncertain:");

    for (const item of roster.unknown_or_uncertain) {
      lines.push(
        `- ${item.display_name ?? "unknown entity"} (${renderParts([
          item.entity_id === null ? null : `id: ${item.entity_id}`,
          item.reason,
          renderRelationshipSummary(item.known_relationships),
          renderSource(item.relationship_source),
        ])})`,
      );
    }
  }

  return lines.join("\n");
}

export function participantRosterRelationalSlotIds(
  roster: ParticipantRoster | null | undefined,
): Set<string> {
  const ids = new Set<string>();

  for (const source of [
    ...(roster?.participants ?? []).map((participant) => participant.relationship_source),
    ...(roster?.participants ?? []).flatMap(
      (participant) => participant.relationship_sources ?? [],
    ),
    ...(roster?.non_chat_subjects ?? []).map((subject) => subject.relationship_source),
    ...(roster?.non_chat_subjects ?? []).flatMap((subject) => subject.relationship_sources ?? []),
  ]) {
    if (source === null || !source.startsWith("relational_slot:")) {
      continue;
    }

    ids.add(source.slice("relational_slot:".length));
  }

  return ids;
}
