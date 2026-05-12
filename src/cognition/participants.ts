import type { EntityRepository } from "../memory/commitments/index.js";
import type { StreamEntry } from "../stream/index.js";
import type { EntityId } from "../util/ids.js";
import { resolveSpeakerDisplayName } from "./speaker-tags.js";

export const DEFAULT_ACTIVE_PARTICIPANT_LIMIT = 8;

export type ActiveParticipantRole = "speaker" | "participant" | "audience";

export type ActiveParticipant = {
  entityId: EntityId;
  displayName: string | null;
  role: ActiveParticipantRole;
};

export type ResolveActiveParticipantsInput = {
  audienceEntityId: EntityId | null;
  senderEntityId?: EntityId | null;
  streamEntries: readonly StreamEntry[];
  entityRepository: Pick<EntityRepository, "get">;
  limit?: number;
};

type MutableParticipant = {
  entityId: EntityId;
  role: ActiveParticipantRole;
};

function appendParticipant(
  participants: MutableParticipant[],
  seen: Set<EntityId>,
  entityId: EntityId | null | undefined,
  role: ActiveParticipantRole,
): void {
  if (entityId === null || entityId === undefined || seen.has(entityId)) {
    return;
  }

  seen.add(entityId);
  participants.push({
    entityId,
    role,
  });
}

function recentSenderEntityIds(streamEntries: readonly StreamEntry[]): EntityId[] {
  const seen = new Set<EntityId>();
  const senders: EntityId[] = [];

  for (let index = streamEntries.length - 1; index >= 0; index -= 1) {
    const entry = streamEntries[index];

    if (entry === undefined || entry.kind !== "user_msg") {
      continue;
    }

    const senderEntityId = entry.sender_entity_id;

    if (senderEntityId === null || senderEntityId === undefined || seen.has(senderEntityId)) {
      continue;
    }

    seen.add(senderEntityId);
    senders.push(senderEntityId);
  }

  return senders;
}

export function resolveActiveParticipants(
  input: ResolveActiveParticipantsInput,
): ActiveParticipant[] {
  const limit = input.limit ?? DEFAULT_ACTIVE_PARTICIPANT_LIMIT;
  const participants: MutableParticipant[] = [];
  const seen = new Set<EntityId>();
  const audienceEntity =
    input.audienceEntityId === null ? null : input.entityRepository.get(input.audienceEntityId);
  const audienceKind = audienceEntity?.kind ?? null;

  appendParticipant(participants, seen, input.senderEntityId, "speaker");

  if (audienceKind === "group") {
    for (const senderEntityId of recentSenderEntityIds(input.streamEntries)) {
      appendParticipant(participants, seen, senderEntityId, "participant");
    }
  } else if (input.audienceEntityId !== null) {
    appendParticipant(participants, seen, input.audienceEntityId, "audience");
  }

  return participants.slice(0, limit).map((participant) => ({
    entityId: participant.entityId,
    displayName: resolveSpeakerDisplayName(input.entityRepository, participant.entityId),
    role: participant.role,
  }));
}
