import { describe, expect, it, vi } from "vitest";

import type { Borg } from "../borg.js";
import type { EntityRecord } from "../memory/commitments/index.js";
import type { SessionRecord } from "../sessions/index.js";
import { createEntityId, createSessionId } from "../util/ids.js";
import { resolveTeamAgentIdentity } from "./team-agent-identity.js";

function harness(existing: SessionRecord | null) {
  const senderId = createEntityId();
  const audienceId = createEntityId();
  const entities = new Map<string, EntityRecord>([
    [
      senderId,
      {
        id: senderId,
        canonical_name: "Sender",
        aliases: [],
        kind: "person",
        borg_role: null,
        name_provenance: "transport_sender",
        created_at: 1,
      },
    ],
    [
      audienceId,
      {
        id: audienceId,
        canonical_name: "Room",
        aliases: [],
        kind: "group",
        borg_role: null,
        name_provenance: "transport_audience_label",
        created_at: 1,
      },
    ],
  ]);
  const resolveExternal = vi.fn((input: { kind: string }) =>
    input.kind === "group" ? audienceId : senderId,
  );
  const borg = {
    entities: {
      resolveExternal,
      get: (id: string) => entities.get(id) ?? null,
    },
    sessions: { get: () => existing },
  } as unknown as Borg;
  return { borg };
}

describe("resolveTeamAgentIdentity", () => {
  it("lets enqueue claim teams_inbox ownership using the conversation external id", () => {
    const session = createSessionId();
    const { borg } = harness(null);
    const identity = resolveTeamAgentIdentity({
      borg,
      session,
      rawSession: "raw-thread",
      sender: { externalId: "user", displayName: "Sender", operator: false },
      conversation: { external_id: "conversation", type: "groupChat", name: "Room" },
      claimInbox: true,
    });

    expect(identity.sessionEnsureInput).toMatchObject({
      source_type: "teams_inbox",
      source_external_id: "conversation",
    });
  });

  it("preserves sticky teams_inbox ownership during context and append-turn refreshes", () => {
    const session = createSessionId();
    const existing = {
      session_id: session,
      source_type: "teams_inbox",
      source_external_id: "original-conversation",
      source_url: null,
      label: "Old",
      audience_label: "Old",
      audience_entity_id: null,
      conversation_kind: "thread",
      created_at: 1,
      last_activity_at: 1,
      last_turn_id: null,
      message_count: 1,
      status: "active",
      privacy_level: "payload_off",
      participation_policy: "active",
      audience_role: "participant",
    } satisfies SessionRecord;
    const { borg } = harness(existing);
    const identity = resolveTeamAgentIdentity({
      borg,
      session,
      rawSession: "different-raw-thread",
      sender: { externalId: "user", displayName: "Sender", operator: false },
      conversation: { external_id: "different-conversation", type: "groupChat", name: "Room" },
    });

    expect(identity.sessionEnsureInput).toMatchObject({
      source_type: "teams_inbox",
      source_external_id: "original-conversation",
    });
  });
});
