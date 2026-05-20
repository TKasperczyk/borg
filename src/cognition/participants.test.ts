import { describe, expect, it } from "vitest";

import type { EntityRecord } from "../memory/commitments/index.js";
import type { SocialProfile } from "../memory/social/index.js";
import {
  ABORTED_TURN_EVENT,
  QUARANTINED_USER_ENTRY_EVENT,
  type StreamEntry,
  type StreamReverseScanOptions,
  type StreamReverseScanResult,
} from "../stream/index.js";
import {
  DEFAULT_SESSION_ID,
  createEntityId,
  createStreamEntryId,
  type EntityId,
} from "../util/ids.js";
import {
  loadRecentParticipantStreamEntries,
  resolveActiveParticipants,
  resolveParticipantProfiles,
  scanRecentParticipantStreamEntries,
} from "./participants.js";

const BASE_TS = 1_700_000_000_000;

function entity(id: EntityId, name: string, kind: EntityRecord["kind"]): EntityRecord {
  return {
    id,
    canonical_name: name,
    aliases: [],
    kind,
    name_provenance: "unknown",
    created_at: BASE_TS,
  };
}

function repository(records: readonly EntityRecord[]) {
  const byId = new Map(records.map((record) => [record.id, record]));

  return {
    get: (id: EntityId) => byId.get(id) ?? null,
  };
}

function userEntry(senderEntityId: EntityId | null, offset: number): StreamEntry {
  return {
    id: createStreamEntryId(),
    timestamp: BASE_TS + offset,
    kind: "user_msg",
    content: `message-${offset}`,
    turn_status: "active",
    sender_entity_id: senderEntityId,
    reply_target_entity_id: null,
    session_id: DEFAULT_SESSION_ID,
    compressed: false,
  };
}

function internalEntry(offset: number, content: unknown = `internal-${offset}`): StreamEntry {
  return {
    id: createStreamEntryId(),
    timestamp: BASE_TS + offset,
    kind: "internal_event",
    content,
    turn_status: "active",
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: DEFAULT_SESSION_ID,
    compressed: false,
  };
}

function agentEntry(offset: number): StreamEntry {
  return {
    id: createStreamEntryId(),
    timestamp: BASE_TS + offset,
    kind: "agent_msg",
    content: `agent-${offset}`,
    turn_status: "active",
    sender_entity_id: null,
    reply_target_entity_id: null,
    session_id: DEFAULT_SESSION_ID,
    compressed: false,
  };
}

function profile(entityId: EntityId): SocialProfile {
  return {
    entity_id: entityId,
    trust: 0.7,
    attachment: 0.2,
    communication_style: null,
    shared_history_summary: null,
    last_interaction_at: BASE_TS,
    interaction_count: 3,
    commitment_count: 0,
    sentiment_history: [],
    notes: null,
    created_at: BASE_TS,
    updated_at: BASE_TS,
  };
}

function scanReverseEntries(
  entries: readonly StreamEntry[],
  options: StreamReverseScanOptions = {},
): StreamReverseScanResult {
  const maxEntries = options.maxEntries ?? 500;
  const maxBytes = options.maxBytes ?? 512 * 1024;
  const scanned: StreamEntry[] = [];
  let scannedEntries = 0;
  let scannedBytes = 0;
  let capReached: StreamReverseScanResult["capReached"] = null;

  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index];

    if (entry === undefined) {
      continue;
    }

    const entryBytes = Buffer.byteLength(JSON.stringify(entry), "utf8");

    if (scannedBytes + entryBytes > maxBytes) {
      scannedBytes = maxBytes;
      capReached = "bytes";
      break;
    }

    scannedBytes += entryBytes;
    scannedEntries += 1;

    if (options.filter === undefined || options.filter(entry)) {
      scanned.push(entry);
    }

    if (options.stop?.(scanned) === true) {
      break;
    }

    if (scannedEntries >= maxEntries) {
      capReached = "entries";
      break;
    }
  }

  return {
    entries: scanned.reverse(),
    scannedEntries,
    scannedBytes,
    capReached,
  };
}

function readerFor(entries: readonly StreamEntry[]) {
  return {
    tail: (limit: number) => entries.slice(-limit),
    scanReverse: (options: StreamReverseScanOptions = {}) => scanReverseEntries(entries, options),
  };
}

describe("resolveActiveParticipants", () => {
  it("uses current speaker first, then recent distinct senders for group audiences", () => {
    const groupId = createEntityId();
    const aliceId = createEntityId();
    const bobId = createEntityId();
    const entities = repository([
      entity(groupId, "Planning Room", "group"),
      entity(aliceId, "Alice", "person"),
      entity(bobId, "Bob", "person"),
    ]);

    const participants = resolveActiveParticipants({
      audienceEntityId: groupId,
      senderEntityId: aliceId,
      streamEntries: [userEntry(aliceId, 1), userEntry(bobId, 2), userEntry(aliceId, 3)],
      entityRepository: entities,
    });

    expect(participants).toEqual([
      {
        entityId: aliceId,
        displayName: "Alice",
        role: "speaker",
      },
      {
        entityId: bobId,
        displayName: "Bob",
        role: "participant",
      },
    ]);
  });

  it("scans noisy stream tails until all recent group speakers are found", () => {
    const groupId = createEntityId();
    const aliceId = createEntityId();
    const bobId = createEntityId();
    const carolId = createEntityId();
    const entities = repository([
      entity(groupId, "Planning Room", "group"),
      entity(aliceId, "Alice", "person"),
      entity(bobId, "Bob", "person"),
      entity(carolId, "Carol", "person"),
    ]);
    const userEntries = new Map<number, EntityId>([
      [10, carolId],
      [30, bobId],
      [60, carolId],
      [90, aliceId],
      [95, bobId],
    ]);
    const agentIndexes = new Set([15, 35, 55, 75, 97]);
    let maintenanceEvents = 0;
    const entries = Array.from({ length: 100 }, (_, index) => {
      const senderEntityId = userEntries.get(index);

      if (senderEntityId !== undefined) {
        return userEntry(senderEntityId, index);
      }

      if (agentIndexes.has(index)) {
        return agentEntry(index);
      }

      if (maintenanceEvents < 10) {
        maintenanceEvents += 1;
        return internalEntry(index, { event: "maintenance_audit" });
      }

      return internalEntry(index);
    });

    const recentEntries = loadRecentParticipantStreamEntries(readerFor(entries), 8);
    const participants = resolveActiveParticipants({
      audienceEntityId: groupId,
      senderEntityId: null,
      streamEntries: recentEntries,
      entityRepository: entities,
      limit: 8,
    });

    expect(recentEntries).toHaveLength(5);
    expect(participants).toEqual([
      {
        entityId: bobId,
        displayName: "Bob",
        role: "participant",
      },
      {
        entityId: aliceId,
        displayName: "Alice",
        role: "participant",
      },
      {
        entityId: carolId,
        displayName: "Carol",
        role: "participant",
      },
    ]);
  });

  it("skips aborted and quarantined user entries during participant scanning", () => {
    const groupId = createEntityId();
    const aliceId = createEntityId();
    const bobId = createEntityId();
    const abortedByStatusId = createEntityId();
    const abortedByMarkerId = createEntityId();
    const quarantinedId = createEntityId();
    const entities = repository([
      entity(groupId, "Planning Room", "group"),
      entity(aliceId, "Alice", "person"),
      entity(bobId, "Bob", "person"),
      entity(abortedByStatusId, "Aborted Status", "person"),
      entity(abortedByMarkerId, "Aborted Marker", "person"),
      entity(quarantinedId, "Quarantined", "person"),
    ]);
    const abortedTurnId = "turn.rejected_marker";
    const abortedByStatus = {
      ...userEntry(abortedByStatusId, 1),
      turn_status: "aborted" as const,
    };
    const abortedByMarker = {
      ...userEntry(abortedByMarkerId, 2),
      turn_id: abortedTurnId,
    };
    const quarantined = userEntry(quarantinedId, 4);
    const entries = [
      abortedByStatus,
      abortedByMarker,
      internalEntry(3, {
        event: ABORTED_TURN_EVENT,
        turn_id: abortedTurnId,
      }),
      quarantined,
      internalEntry(5, {
        event: QUARANTINED_USER_ENTRY_EVENT,
        source_stream_entry_id: quarantined.id,
      }),
      userEntry(aliceId, 6),
      userEntry(bobId, 7),
    ];

    const participants = resolveActiveParticipants({
      audienceEntityId: groupId,
      senderEntityId: null,
      streamEntries: loadRecentParticipantStreamEntries(readerFor(entries), 8),
      entityRepository: entities,
      limit: 8,
    });

    expect(participants).toEqual([
      {
        entityId: bobId,
        displayName: "Bob",
        role: "participant",
      },
      {
        entityId: aliceId,
        displayName: "Alice",
        role: "participant",
      },
    ]);
  });

  it("reports entry-cap scans with partial participant coverage", () => {
    const aliceId = createEntityId();
    const entries = [
      ...Array.from({ length: 101 }, (_, index) => internalEntry(index)),
      userEntry(aliceId, 101),
      ...Array.from({ length: 499 }, (_, index) => internalEntry(102 + index)),
    ];

    const scan = scanRecentParticipantStreamEntries(readerFor(entries), 8);

    expect(scan.capReached).toBe("entries");
    expect(scan.scannedEntries).toBe(500);
    expect(scan.foundUniqueParticipants).toBe(1);
    expect(scan.entries.map((entry) => entry.sender_entity_id)).toEqual([aliceId]);
  });

  it("preserves most-recent-first speaker order after deduplication", () => {
    const groupId = createEntityId();
    const aliceId = createEntityId();
    const bobId = createEntityId();
    const carolId = createEntityId();
    const entities = repository([
      entity(groupId, "Planning Room", "group"),
      entity(aliceId, "Alice", "person"),
      entity(bobId, "Bob", "person"),
      entity(carolId, "Carol", "person"),
    ]);
    const entries = [
      userEntry(aliceId, 1),
      userEntry(bobId, 2),
      userEntry(aliceId, 3),
      userEntry(carolId, 4),
    ];

    const participants = resolveActiveParticipants({
      audienceEntityId: groupId,
      senderEntityId: null,
      streamEntries: loadRecentParticipantStreamEntries(readerFor(entries), 8),
      entityRepository: entities,
      limit: 8,
    });

    expect(participants.map((participant) => participant.entityId)).toEqual([
      carolId,
      aliceId,
      bobId,
    ]);
  });

  it("uses the person audience for legacy single-user turns", () => {
    const aliceId = createEntityId();
    const bobId = createEntityId();
    const entities = repository([
      entity(aliceId, "Alice", "person"),
      entity(bobId, "Bob", "person"),
    ]);

    const participants = resolveActiveParticipants({
      audienceEntityId: aliceId,
      senderEntityId: null,
      streamEntries: [userEntry(bobId, 1)],
      entityRepository: entities,
    });

    expect(participants).toEqual([
      {
        entityId: aliceId,
        displayName: "Alice",
        role: "audience",
      },
    ]);
  });

  it("returns the group audience marker when no person can be resolved", () => {
    const groupId = createEntityId();
    const entities = repository([entity(groupId, "Planning Room", "group")]);

    const participants = resolveActiveParticipants({
      audienceEntityId: groupId,
      senderEntityId: null,
      streamEntries: [userEntry(null, 1)],
      entityRepository: entities,
    });

    expect(participants).toEqual([
      {
        entityId: groupId,
        displayName: "Planning Room",
        role: "audience",
      },
    ]);
  });

  it("preserves a group-audience signal when recency has no other speakers", () => {
    const groupId = createEntityId();
    const aliceId = createEntityId();
    const entities = repository([
      entity(groupId, "Planning Room", "group"),
      entity(aliceId, "Alice", "person"),
    ]);

    const participants = resolveActiveParticipants({
      audienceEntityId: groupId,
      senderEntityId: aliceId,
      streamEntries: [],
      entityRepository: entities,
    });

    expect(participants).toEqual([
      {
        entityId: aliceId,
        displayName: "Alice",
        role: "speaker",
      },
      {
        entityId: groupId,
        displayName: "Planning Room",
        role: "audience",
      },
    ]);
  });

  it("loads social profiles for each active participant", () => {
    const aliceId = createEntityId();
    const bobId = createEntityId();
    const aliceProfile = profile(aliceId);
    const profiles = new Map<EntityId, SocialProfile>([[aliceId, aliceProfile]]);

    const participantProfiles = resolveParticipantProfiles(
      [
        {
          entityId: aliceId,
          displayName: "Alice",
          role: "speaker",
        },
        {
          entityId: bobId,
          displayName: "Bob",
          role: "participant",
        },
      ],
      {
        getProfile: (entityId) => profiles.get(entityId) ?? null,
      },
    );

    expect(participantProfiles).toEqual([
      {
        entityId: aliceId,
        displayName: "Alice",
        role: "speaker",
        profile: aliceProfile,
      },
      {
        entityId: bobId,
        displayName: "Bob",
        role: "participant",
        profile: null,
      },
    ]);
  });
});
