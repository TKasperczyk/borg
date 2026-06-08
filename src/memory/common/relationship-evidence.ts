const RELATIONAL_SLOT_SOURCE_PREFIX = "relational_slot:";

type RelationshipSourceRecord = {
  relationship_source: string | null;
  relationship_sources?: readonly string[];
  [key: string]: unknown;
};

export type ParticipantRosterRelationshipEvidence = {
  participants?: readonly RelationshipSourceRecord[];
  non_chat_subjects?: readonly RelationshipSourceRecord[];
  unknown_or_uncertain?: readonly RelationshipSourceRecord[];
};

export function participantRosterRelationalSlotIds(
  roster: ParticipantRosterRelationshipEvidence | null | undefined,
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
    if (source === null || !source.startsWith(RELATIONAL_SLOT_SOURCE_PREFIX)) {
      continue;
    }

    ids.add(source.slice(RELATIONAL_SLOT_SOURCE_PREFIX.length));
  }

  return ids;
}
