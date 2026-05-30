import type { BorgUserContentBlock, TurnInputAttachment } from "../attachments/index.js";
import type {
  StreamCursor,
  StreamEntry,
  StreamResponseTo,
  StreamSourceMessageKey,
} from "../stream/index.js";
import type { EntityId, SessionId, StreamEntryId } from "../util/ids.js";
import type { AutonomyTriggerContext } from "./autonomy-trigger.js";
import type { TurnStakes } from "./deliberation/deliberator.js";
import type { TurnOrigin } from "./types.js";

export type HydratedInboundMessage = {
  id: StreamEntryId;
  session_id: SessionId;
  entry_index: number;
  timestamp: number;
  kind: "user_msg";
  content: string;
  sender_entity_id?: EntityId | null;
  audience?: string;
  source_message_key?: StreamSourceMessageKey;
};

export type TurnLockMode = "block" | "try" | { timeoutMs: number };

export type SingleMessageTurnInput = TurnInputBase & {
  userMessage: string;
  attachments?: readonly TurnInputAttachment[];
  inboundBatch?: never;
};

export type TurnInput = SingleMessageTurnInput;

export type InternalSingleMessageTurnInput = SingleMessageTurnInput & {
  lockMode?: TurnLockMode;
};

export type InboundBatchTurnInput = TurnInputBase & {
  lockMode?: TurnLockMode;
  inboundBatch: {
    kind: "stream_backlog";
    entryIds: readonly StreamEntryId[];
    throughCursorInclusive?: StreamCursor;
  };
  userMessage?: never;
  attachments?: never;
};

export type TurnOrchestratorInput = InternalSingleMessageTurnInput | InboundBatchTurnInput;

export type CurrentTurnUserInputSenderAttribution = {
  entryId: StreamEntryId;
  senderEntityId: EntityId | null;
  senderDisplayName?: string;
};

export type CurrentTurnUserInput = {
  renderedText: string;
  currentUserContent: readonly BorgUserContentBlock[];
  sourceUserEntries: readonly StreamEntry[];
  sourceUserEntryIds: readonly StreamEntryId[];
  senderAttribution: readonly CurrentTurnUserInputSenderAttribution[];
  // Scalar/display sender for rendering and non-authority attribution only.
  // Privileged authority checks must use the coordinator's authority sender fields.
  effectiveSenderEntityId: EntityId | null;
  responseTo?: StreamResponseTo;
  recencyBeforeEntryIdExclusive?: StreamEntryId;
  persistUserMessage: boolean;
};

type TurnInputBase = {
  audience?: string;
  senderEntityId?: EntityId;
  stakes?: TurnStakes;
  sessionId?: SessionId;
  globalTurnCounter?: number;
  origin?: TurnOrigin;
  autonomyTrigger?: AutonomyTriggerContext | null;
};

export function isInboundBatchTurnInput<T extends TurnOrchestratorInput>(
  input: T,
): input is Extract<T, InboundBatchTurnInput> {
  return input.inboundBatch !== undefined;
}

export function orderedInboundBatchEntries(
  entries: readonly HydratedInboundMessage[],
): HydratedInboundMessage[] {
  return [...entries].sort(
    (left, right) =>
      left.entry_index - right.entry_index ||
      left.timestamp - right.timestamp ||
      left.id.localeCompare(right.id),
  );
}

function escapeXmlText(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
}

function escapeXmlAttribute(value: string): string {
  return escapeXmlText(value).replaceAll('"', "&quot;");
}

function xmlAttribute(name: string, value: string | number | null | undefined): string {
  if (value === null || value === undefined) {
    return "";
  }

  return ` ${name}="${escapeXmlAttribute(String(value))}"`;
}

export function renderInboundBatch(input: {
  entries: readonly HydratedInboundMessage[];
  senderDisplayNameById?: (entityId: EntityId) => string | null | undefined;
}): string {
  const entries = orderedInboundBatchEntries(input.entries);
  const renderedEntries = entries.map((entry, index) => {
    const senderEntityId = entry.sender_entity_id ?? null;
    const senderDisplayName =
      senderEntityId === null ? null : (input.senderDisplayNameById?.(senderEntityId) ?? null);

    return [
      `<inbound_message index="${index + 1}"`,
      xmlAttribute("stream_entry_id", entry.id),
      xmlAttribute("timestamp_ms", entry.timestamp),
      xmlAttribute("sender_entity_id", senderEntityId),
      xmlAttribute("sender_display_name", senderDisplayName),
      ">",
      escapeXmlText(entry.content),
      "</inbound_message>",
    ].join("");
  });

  return [
    `<inbound_batch kind="stream_backlog" count="${entries.length}">`,
    ...renderedEntries,
    "</inbound_batch>",
  ].join("\n");
}
