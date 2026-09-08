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

// The harness mints participant ids in the prefixed form, but the extraction
// model also sees raw entity ids in its own prompt and sometimes copies one
// into the participants array bare. Both forms name an entity, so both resolve
// here: an id is never a display name, whichever way it was written.
export function episodeParticipantEntityId(value: string): EntityId | null {
  const prefixed = parseEpisodeParticipantEntityIdTerm(value);

  if (prefixed !== null) {
    return prefixed;
  }

  return entityIdHelpers.is(value) ? value : null;
}

export function episodeParticipantEntityIds(participants: readonly string[]): EntityId[] {
  const ids = participants.flatMap((participant) => {
    const entityId = episodeParticipantEntityId(participant);

    return entityId === null ? [] : [entityId];
  });

  return [...new Set(ids)];
}

export function episodeParticipantDisplayNames(participants: readonly string[]): string[] {
  return participants.filter((participant) => episodeParticipantEntityId(participant) === null);
}
