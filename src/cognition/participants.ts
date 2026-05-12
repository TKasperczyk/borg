import { DEFAULT_ACTIVE_PARTICIPANT_LIMIT } from "../config/index.js";
import type { EntityRepository } from "../memory/commitments/index.js";
import type { SocialProfile, SocialRepository } from "../memory/social/index.js";
import type { StreamEntry, StreamReader } from "../stream/index.js";
import type { EntityId } from "../util/ids.js";
import { resolveSpeakerDisplayName } from "./speaker-tags.js";

const ACTIVE_PARTICIPANT_SCAN_MULTIPLIER = 4;

export type ActiveParticipantRole = "speaker" | "participant" | "audience";

export type ActiveParticipant = {
  entityId: EntityId;
  displayName: string | null;
  role: ActiveParticipantRole;
};

export type ParticipantProfileContext = ActiveParticipant & {
  profile: SocialProfile | null;
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

export function activeParticipantStreamEntryScanLimit(participantLimit: number): number {
  return Math.max(1, Math.floor(participantLimit)) * ACTIVE_PARTICIPANT_SCAN_MULTIPLIER;
}

export function loadRecentParticipantStreamEntries(
  reader: Pick<StreamReader, "tail">,
  participantLimit: number = DEFAULT_ACTIVE_PARTICIPANT_LIMIT,
): StreamEntry[] {
  return reader.tail(activeParticipantStreamEntryScanLimit(participantLimit));
}

function recentSenderEntityIds(streamEntries: readonly StreamEntry[], limit: number): EntityId[] {
  if (limit <= 0) {
    return [];
  }

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

    if (senders.length >= limit) {
      break;
    }
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
    for (const senderEntityId of recentSenderEntityIds(input.streamEntries, limit)) {
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

export function resolveParticipantProfiles(
  participants: readonly ActiveParticipant[],
  socialRepository: Pick<SocialRepository, "getProfile">,
): ParticipantProfileContext[] {
  return participants.map((participant) => ({
    ...participant,
    profile: socialRepository.getProfile(participant.entityId),
  }));
}
