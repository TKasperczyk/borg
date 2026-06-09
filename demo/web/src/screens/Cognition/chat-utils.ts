import type { StreamEntry } from "../../api/types";
import {
  agentObservedContent,
  agentSuppressedContent,
  streamOutcomeSummary,
  type StreamOutcomeSummary,
} from "../../lib/stream-outcomes";
import { sortStreamEntries, streamContentText } from "../../lib/stream-utils";

export type ChatDeliveryStatus = "queued" | "sent";

export type ChatStreamEntry = StreamEntry & {
  external_message_id?: string;
  optimistic_status?: ChatDeliveryStatus;
};

export type AttachmentRef = {
  attachmentId: string;
  mediaType?: string;
  entryId: string;
};

export type ChatTurn = {
  itemType: "message";
  entry: ChatStreamEntry;
  role: "user" | "borg";
  text: string;
  attachments: AttachmentRef[];
  thought?: string;
  refs?: Array<{ id: string; kind: string; label: string; trust?: string }>;
  sourceEntryIds: string[];
};

export type ChatMarker = {
  itemType: "marker";
  entry: ChatStreamEntry;
  summary: StreamOutcomeSummary;
  reason: string | null;
  turnId?: string;
  userEntryIds: string[];
};

export type ChatLaneItem = ChatTurn | ChatMarker;

type ImageRefContent = {
  type?: unknown;
  attachment_id?: unknown;
  media_type?: unknown;
  parent_entry_id?: unknown;
};

type RichContent = {
  text?: unknown;
  thought?: unknown;
  refs?: unknown;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function contentThought(content: unknown): string | undefined {
  if (!isRecord(content)) {
    return undefined;
  }
  const rich = content as RichContent;
  return typeof rich.thought === "string" ? rich.thought : undefined;
}

function contentRefs(content: unknown): ChatTurn["refs"] {
  if (!isRecord(content) || !Array.isArray((content as RichContent).refs)) {
    return undefined;
  }

  return ((content as RichContent).refs as unknown[]).flatMap((item) => {
    if (!isRecord(item) || typeof item.id !== "string" || typeof item.kind !== "string") {
      return [];
    }
    return [
      {
        id: item.id,
        kind: item.kind,
        label: typeof item.label === "string" ? item.label : item.id,
        trust: typeof item.trust === "string" ? item.trust : undefined,
      },
    ];
  });
}

function attachmentFromEntry(entry: StreamEntry): AttachmentRef | null {
  if (!isRecord(entry.content)) {
    return null;
  }

  const content = entry.content as ImageRefContent;
  if (content.type !== "image_ref" || typeof content.attachment_id !== "string") {
    return null;
  }

  return {
    attachmentId: content.attachment_id,
    mediaType: typeof content.media_type === "string" ? content.media_type : undefined,
    entryId: entry.id,
  };
}

function attachmentParentEntryId(content: unknown): string | null {
  if (!isRecord(content)) {
    return null;
  }

  const parentEntryId = (content as ImageRefContent).parent_entry_id;
  return typeof parentEntryId === "string" && parentEntryId.length > 0 ? parentEntryId : null;
}

function responseToSourceEntryIds(entry: StreamEntry): string[] {
  const sourceEntryIds = entry.response_to?.source_entry_ids;
  if (!Array.isArray(sourceEntryIds)) {
    return [];
  }

  return sourceEntryIds.filter((entryId): entryId is string => typeof entryId === "string");
}

function uniqueStrings(values: readonly (string | undefined)[]): string[] {
  return [...new Set(values.filter((value): value is string => typeof value === "string"))];
}

function markerFromEntry(entry: ChatStreamEntry): ChatMarker | null {
  if (entry.kind !== "agent_suppressed" && entry.kind !== "agent_observed") {
    return null;
  }

  const summary = streamOutcomeSummary(entry);
  if (summary === null) {
    return null;
  }

  const content =
    entry.kind === "agent_suppressed"
      ? agentSuppressedContent(entry.content)
      : agentObservedContent(entry.content);
  const userEntryIds = uniqueStrings([
    content?.user_entry_id,
    ...(content?.user_entry_ids ?? []),
  ]);
  const turnId = content?.turn_id ?? entry.turn_id;

  return {
    itemType: "marker",
    entry,
    summary,
    reason: summary.reason,
    ...(turnId === undefined ? {} : { turnId }),
    userEntryIds,
  };
}

export function streamEntriesToChatTurns(entries: readonly ChatStreamEntry[]): ChatLaneItem[] {
  const ordered = sortStreamEntries(entries) as ChatStreamEntry[];
  const turns: ChatLaneItem[] = [];
  const byTurnId = new Map<string, ChatTurn>();
  const byEntryId = new Map<string, ChatTurn>();

  for (const entry of ordered) {
    if (entry.kind === "user_msg" || entry.kind === "agent_msg") {
      const turn: ChatTurn = {
        itemType: "message",
        entry,
        role: entry.kind === "user_msg" ? "user" : "borg",
        text: streamContentText(entry.content),
        attachments: [],
        thought: contentThought(entry.content),
        refs: contentRefs(entry.content),
        sourceEntryIds: responseToSourceEntryIds(entry),
      };
      turns.push(turn);
      byEntryId.set(entry.id, turn);
      if (entry.turn_id !== undefined) {
        byTurnId.set(`${entry.turn_id}:${entry.kind === "user_msg" ? "user" : "borg"}`, turn);
      }
      continue;
    }

    if (entry.kind === "user_image_attachment") {
      const attachment = attachmentFromEntry(entry);
      if (attachment === null) {
        continue;
      }
      const key = entry.turn_id === undefined ? null : `${entry.turn_id}:user`;
      const parentEntryId = attachmentParentEntryId(entry.content);
      const parent =
        parentEntryId === null
          ? key === null
            ? undefined
            : byTurnId.get(key)
          : byEntryId.get(parentEntryId) ?? (key === null ? undefined : byTurnId.get(key));
      if (parent !== undefined) {
        parent.attachments.push(attachment);
      } else {
        turns.push({
          itemType: "message",
          entry,
          role: "user",
          text: "image attachment",
          attachments: [attachment],
          sourceEntryIds: responseToSourceEntryIds(entry),
        });
      }
      continue;
    }

    const marker = markerFromEntry(entry);
    if (marker !== null) {
      turns.push(marker);
    }
  }

  return turns;
}
