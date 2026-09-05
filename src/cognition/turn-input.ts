import type {
  BorgUserContentBlock,
  ImageKind,
  ImageMediaType,
  TurnInputAttachment,
} from "../attachments/index.js";
import type {
  StreamCursor,
  StreamConversation,
  StreamEntry,
  StreamEntryMetadata,
  StreamResponseTo,
  StreamSourceMessageKey,
} from "../stream/index.js";
import { escapeXmlText } from "../util/prompt-tags.js";
import type {
  AttachmentId,
  EntityId,
  ImagePerceptionId,
  SessionId,
  StreamEntryId,
} from "../util/ids.js";
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
  observed_at?: number;
  conversation?: StreamConversation;
  metadata?: StreamEntryMetadata;
  attachments?: readonly HydratedInboundAttachment[];
};

export type HydratedInboundAttachment = {
  attachment_id: AttachmentId;
  media_type: ImageMediaType;
  width: number;
  height: number;
  perception: HydratedInboundImagePerception | null;
};

export type HydratedInboundImagePerception = {
  perception_id: ImagePerceptionId;
  caption: string;
  image_kind: ImageKind;
  visible_text: readonly string[];
  search_terms: readonly string[];
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

function escapeXmlAttribute(value: string): string {
  return escapeXmlText(value).replaceAll('"', "&quot;");
}

function xmlAttribute(name: string, value: string | number | null | undefined): string {
  if (value === null || value === undefined) {
    return "";
  }

  return ` ${name}="${escapeXmlAttribute(String(value))}"`;
}

function renderTextList(input: {
  containerTag: string;
  itemTag: string;
  values: readonly string[];
}): string {
  return [
    `<${input.containerTag} count="${input.values.length}">`,
    ...input.values.map(
      (value, index) =>
        `<${input.itemTag} index="${index + 1}">${escapeXmlText(value)}</${input.itemTag}>`,
    ),
    `</${input.containerTag}>`,
  ].join("\n");
}

function renderInboundAttachment(attachment: HydratedInboundAttachment, index: number): string {
  const perception = attachment.perception;
  const openTag = [
    `<attachment index="${index + 1}" kind="image"`,
    xmlAttribute("attachment_id", attachment.attachment_id),
    xmlAttribute("media_type", attachment.media_type),
    xmlAttribute("width", attachment.width),
    xmlAttribute("height", attachment.height),
    ">",
  ].join("");

  if (perception === null) {
    return [openTag, '<perception status="unavailable" />', "</attachment>"].join("\n");
  }

  return [
    openTag,
    [
      `<perception status="available"`,
      xmlAttribute("perception_id", perception.perception_id),
      ">",
    ].join(""),
    `<caption>${escapeXmlText(perception.caption)}</caption>`,
    `<image_kind>${escapeXmlText(perception.image_kind)}</image_kind>`,
    renderTextList({
      containerTag: "visible_text",
      itemTag: "text",
      values: perception.visible_text,
    }),
    renderTextList({
      containerTag: "search_terms",
      itemTag: "term",
      values: perception.search_terms,
    }),
    "</perception>",
    "</attachment>",
  ].join("\n");
}

function renderInboundAttachments(attachments: readonly HydratedInboundAttachment[]): string {
  return [
    `<attachments count="${attachments.length}">`,
    ...attachments.map((attachment, index) => renderInboundAttachment(attachment, index)),
    "</attachments>",
  ].join("\n");
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
    const attachments = entry.attachments ?? [];

    if (attachments.length === 0) {
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
    }

    const openTag = [
      `<inbound_message index="${index + 1}"`,
      xmlAttribute("stream_entry_id", entry.id),
      xmlAttribute("timestamp_ms", entry.timestamp),
      xmlAttribute("sender_entity_id", senderEntityId),
      xmlAttribute("sender_display_name", senderDisplayName),
      ">",
    ].join("");

    return [
      openTag,
      escapeXmlText(entry.content),
      renderInboundAttachments(attachments),
      "</inbound_message>",
    ].join("\n");
  });

  return [
    `<inbound_batch kind="stream_backlog" count="${entries.length}">`,
    ...renderedEntries,
    "</inbound_batch>",
  ].join("\n");
}
