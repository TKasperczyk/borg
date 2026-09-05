import { createHash } from "node:crypto";

import { z } from "zod";

import type { Borg } from "../borg.js";
import type { EntityRecord } from "../memory/commitments/index.js";
import type { SessionEnsureInput } from "../sessions/index.js";
import { streamConversationSchema, type StreamConversation } from "../stream/index.js";
import { parseSessionId, type EntityId, type SessionId } from "../util/ids.js";

export const TEAM_AGENT_SENDER_EXTERNAL_ID_SOURCE = "team-agent.sender";
export const TEAM_AGENT_CONVERSATION_EXTERNAL_ID_SOURCE = "team-agent.conversation";

export const sidecarConversationSchema = streamConversationSchema.extend({
  external_id: z.string().trim().min(1).optional(),
});

export type SidecarConversation = z.infer<typeof sidecarConversationSchema>;
export type TeamAgentIdentitySender = {
  externalId: string;
  displayName: string;
  operator: boolean;
};

export type TeamAgentIdentity = {
  session: SessionId;
  audienceEntity: EntityRecord;
  audienceRole: "participant" | "operator";
  conversation: StreamConversation;
  sessionEnsureInput: SessionEnsureInput;
  senderEntityId: EntityId | null;
};

export function sessionFromCaller(value: string): SessionId {
  try {
    return parseSessionId(value);
  } catch {
    const hash = createHash("sha256").update(value).digest("hex").slice(0, 16);
    return parseSessionId(`sess_${hash}`);
  }
}

function conversationKindForSidecar(
  type: SidecarConversation["type"],
): "dm" | "thread" | "channel" {
  switch (type) {
    case "personal":
      return "dm";
    case "groupChat":
      return "thread";
    case "channel":
      return "channel";
  }
}

export function resolveTeamAgentIdentity(input: {
  borg: Borg;
  session: SessionId;
  rawSession: string;
  sender: TeamAgentIdentitySender | null;
  conversation: SidecarConversation;
  claimInbox?: boolean;
}): TeamAgentIdentity {
  const senderEntityId =
    input.sender === null
      ? null
      : input.borg.entities.resolveExternal({
          source: TEAM_AGENT_SENDER_EXTERNAL_ID_SOURCE,
          externalId: input.sender.externalId,
          canonicalName: input.sender.displayName,
          kind: "person",
          provenance: "transport_sender",
        });
  let audienceEntityId = senderEntityId;

  if (input.conversation.type !== "personal") {
    const externalId = input.conversation.external_id;
    if (externalId === undefined) {
      throw new Error("group conversation identity requires an external id");
    }
    audienceEntityId = input.borg.entities.resolveExternal({
      source: TEAM_AGENT_CONVERSATION_EXTERNAL_ID_SOURCE,
      externalId,
      canonicalName:
        input.conversation.name.length > 0
          ? input.conversation.name
          : `${input.conversation.type}:${externalId}`,
      kind: "group",
      provenance: "transport_audience_label",
    });
  }

  if (audienceEntityId === null) {
    throw new Error("personal conversation identity requires a sender entity");
  }
  const audienceEntity = input.borg.entities.get(audienceEntityId);
  if (audienceEntity === null) {
    throw new Error(`resolved audience entity ${audienceEntityId} is missing`);
  }

  const existing = input.borg.sessions.get(input.session);
  const stickyInbox = existing?.source_type === "teams_inbox";
  const inboxExternalId = input.conversation.external_id;
  if (input.claimInbox === true && inboxExternalId === undefined) {
    throw new Error("inbox conversation identity requires an external id");
  }
  const sourceType = input.claimInbox === true || stickyInbox ? "teams_inbox" : "team_agent";
  const sourceExternalId =
    input.claimInbox === true
      ? inboxExternalId!
      : stickyInbox
        ? existing.source_external_id
        : input.rawSession;
  const audienceRole = input.sender?.operator === true ? "operator" : "participant";
  const conversation = {
    type: input.conversation.type,
    name: input.conversation.name,
  } satisfies StreamConversation;

  return {
    session: input.session,
    audienceEntity,
    audienceRole,
    conversation,
    senderEntityId,
    sessionEnsureInput: {
      session_id: input.session,
      source_type: sourceType,
      source_external_id: sourceExternalId,
      label: input.conversation.name || audienceEntity.canonical_name,
      audience_label: audienceEntity.canonical_name,
      audience_entity_id: audienceEntity.id,
      conversation_kind: conversationKindForSidecar(input.conversation.type),
      audience_role: audienceRole,
      status: "active",
    },
  };
}
