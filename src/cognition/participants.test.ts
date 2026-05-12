import { describe, expect, it } from "vitest";

import type { EntityRecord } from "../memory/commitments/index.js";
import type { StreamEntry } from "../stream/index.js";
import {
  DEFAULT_SESSION_ID,
  createEntityId,
  createStreamEntryId,
  type EntityId,
} from "../util/ids.js";
import { resolveActiveParticipants } from "./participants.js";

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

  it("returns no active people for group audiences without speaker history", () => {
    const groupId = createEntityId();
    const entities = repository([entity(groupId, "Planning Room", "group")]);

    const participants = resolveActiveParticipants({
      audienceEntityId: groupId,
      senderEntityId: null,
      streamEntries: [userEntry(null, 1)],
      entityRepository: entities,
    });

    expect(participants).toEqual([]);
  });
});
