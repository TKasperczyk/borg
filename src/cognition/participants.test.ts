import { describe, expect, it } from "vitest";

import type { EntityRecord } from "../memory/commitments/index.js";
import type { SocialProfile } from "../memory/social/index.js";
import type { StreamEntry } from "../stream/index.js";
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

  it("loads only a bounded recent stream slice for group participant resolution", () => {
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
    const senders = [aliceId, bobId, carolId];
    const entries = Array.from({ length: 500 }, (_, index) =>
      userEntry(senders[index % senders.length] ?? null, index),
    );
    let requestedTailLimit = 0;

    const recentEntries = loadRecentParticipantStreamEntries(
      {
        tail: (limit) => {
          requestedTailLimit = limit;
          return entries.slice(-limit);
        },
      },
      8,
    );
    const participants = resolveActiveParticipants({
      audienceEntityId: groupId,
      senderEntityId: null,
      streamEntries: recentEntries,
      entityRepository: entities,
      limit: 8,
    });

    expect(requestedTailLimit).toBe(32);
    expect(recentEntries).toHaveLength(32);
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
