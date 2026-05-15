import { DEFAULT_ACTIVE_PARTICIPANT_LIMIT } from "../config/index.js";
import type { EntityRepository } from "../memory/commitments/index.js";
import type { SocialProfile, SocialRepository } from "../memory/social/index.js";
import {
  filterActiveStreamEntries,
  isAbortedTurnMarker,
  isQuarantinedUserEntryMarker,
  type StreamEntry,
  type StreamReader,
  type StreamReverseScanResult,
} from "../stream/index.js";
import type { EntityId } from "../util/ids.js";
import { resolveSpeakerDisplayName } from "./speaker-tags.js";

const ACTIVE_PARTICIPANT_MAX_SCAN_ENTRIES = 500;
const ACTIVE_PARTICIPANT_MAX_SCAN_BYTES = 512 * 1024;

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

export type RecentParticipantStreamEntryScanResult = StreamReverseScanResult & {
  foundUniqueParticipants: number;
};

type ParticipantStreamReader = Pick<StreamReader, "tail"> &
  Partial<Pick<StreamReader, "scanReverse">>;

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

function normalizedParticipantLimit(participantLimit: number): number {
  return Math.max(1, Math.floor(participantLimit));
}

export function activeParticipantStreamEntryScanLimit(_participantLimit: number): number {
  return ACTIVE_PARTICIPANT_MAX_SCAN_ENTRIES;
}

function participantScanEntryFilter(entry: StreamEntry): boolean {
  return (
    entry.kind === "user_msg" || isAbortedTurnMarker(entry) || isQuarantinedUserEntryMarker(entry)
  );
}

function activeParticipantUserEntries(entries: readonly StreamEntry[]): StreamEntry[] {
  return filterActiveStreamEntries(entries).filter(
    (entry) =>
      entry.kind === "user_msg" &&
      entry.sender_entity_id !== null &&
      entry.sender_entity_id !== undefined,
  );
}

function estimatedStreamEntryBytes(entry: StreamEntry): number {
  return Buffer.byteLength(JSON.stringify(entry), "utf8");
}

function fallbackScanRecentParticipantStreamEntries(
  reader: Pick<StreamReader, "tail">,
  participantLimit: number,
): RecentParticipantStreamEntryScanResult {
  const limit = normalizedParticipantLimit(participantLimit);
  const tailEntries = reader.tail(ACTIVE_PARTICIPANT_MAX_SCAN_ENTRIES);
  const scanned: StreamEntry[] = [];
  let scannedEntries = 0;
  let scannedBytes = 0;
  let capReached: RecentParticipantStreamEntryScanResult["capReached"] = null;

  for (let index = tailEntries.length - 1; index >= 0; index -= 1) {
    const entry = tailEntries[index];

    if (entry === undefined) {
      continue;
    }

    const entryBytes = estimatedStreamEntryBytes(entry);

    if (scannedBytes + entryBytes > ACTIVE_PARTICIPANT_MAX_SCAN_BYTES) {
      scannedBytes = ACTIVE_PARTICIPANT_MAX_SCAN_BYTES;
      capReached = "bytes";
      break;
    }

    scannedBytes += entryBytes;
    scannedEntries += 1;

    if (participantScanEntryFilter(entry)) {
      scanned.push(entry);
    }

    if (recentSenderEntityIds(activeParticipantUserEntries(scanned), limit).length >= limit) {
      break;
    }

    if (scannedEntries >= ACTIVE_PARTICIPANT_MAX_SCAN_ENTRIES) {
      capReached = "entries";
      break;
    }
  }

  const entries = activeParticipantUserEntries(scanned.reverse());

  return {
    entries,
    scannedEntries,
    scannedBytes,
    capReached,
    foundUniqueParticipants: recentSenderEntityIds(entries, limit).length,
  };
}

export function scanRecentParticipantStreamEntries(
  reader: ParticipantStreamReader,
  participantLimit: number = DEFAULT_ACTIVE_PARTICIPANT_LIMIT,
): RecentParticipantStreamEntryScanResult {
  const limit = normalizedParticipantLimit(participantLimit);

  if (reader.scanReverse === undefined) {
    return fallbackScanRecentParticipantStreamEntries(reader, limit);
  }

  const scan = reader.scanReverse({
    maxEntries: ACTIVE_PARTICIPANT_MAX_SCAN_ENTRIES,
    maxBytes: ACTIVE_PARTICIPANT_MAX_SCAN_BYTES,
    filter: participantScanEntryFilter,
    stop: (entries) =>
      recentSenderEntityIds(activeParticipantUserEntries(entries), limit).length >= limit,
  });
  const entries = activeParticipantUserEntries(scan.entries);

  return {
    ...scan,
    entries,
    foundUniqueParticipants: recentSenderEntityIds(entries, limit).length,
  };
}

export function loadRecentParticipantStreamEntries(
  reader: ParticipantStreamReader,
  participantLimit: number = DEFAULT_ACTIVE_PARTICIPANT_LIMIT,
): StreamEntry[] {
  return scanRecentParticipantStreamEntries(reader, participantLimit).entries;
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

    if (participants.length <= 1) {
      appendParticipant(participants, seen, input.audienceEntityId, "audience");
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
