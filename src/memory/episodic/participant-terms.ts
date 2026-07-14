import { entityIdHelpers, type EntityId } from "../../util/ids.js";

const PARTICIPANT_ENTITY_ID_TERM_PREFIX = "entity_id:";

export function episodeParticipantEntityIdTerm(entityId: EntityId): string {
  return `${PARTICIPANT_ENTITY_ID_TERM_PREFIX}${entityId}`;
}

export function parseEpisodeParticipantEntityIdTerm(value: string): EntityId | null {
  const candidate = value.slice(PARTICIPANT_ENTITY_ID_TERM_PREFIX.length);

  if (`${PARTICIPANT_ENTITY_ID_TERM_PREFIX}${candidate}` !== value) {
    return null;
  }

  return entityIdHelpers.is(candidate) ? candidate : null;
}

export function episodeParticipantEntityIds(participants: readonly string[]): EntityId[] {
  const ids = participants.flatMap((participant) => {
    const entityId = parseEpisodeParticipantEntityIdTerm(participant);

    return entityId === null ? [] : [entityId];
  });

  return [...new Set(ids)];
}

export function episodeParticipantDisplayNames(participants: readonly string[]): string[] {
  return participants.filter(
    (participant) => parseEpisodeParticipantEntityIdTerm(participant) === null,
  );
}
