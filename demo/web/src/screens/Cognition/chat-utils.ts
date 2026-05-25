import type { StreamEntry } from "../../api/types";

export type AttachmentRef = {
  attachmentId: string;
  mediaType?: string;
  entryId: string;
};

export type ChatTurn = {
  entry: StreamEntry;
  role: "user" | "borg";
  text: string;
  attachments: AttachmentRef[];
  thought?: string;
  refs?: Array<{ id: string; kind: string; label: string; trust?: string }>;
};

type ImageRefContent = {
  type?: unknown;
  attachment_id?: unknown;
  media_type?: unknown;
};

type RichContent = {
  text?: unknown;
  thought?: unknown;
  refs?: unknown;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function timestampLabel(timestamp: number): string {
  return new Date(timestamp).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit"
  });
}

export function contentText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    const blocks = content.flatMap((item) => {
      if (isRecord(item) && item.type === "text" && typeof item.text === "string") {
        return [item.text];
      }
      return [];
    });
    if (blocks.length > 0) {
      return blocks.join("\n");
    }
  }

  if (isRecord(content) && typeof content.text === "string") {
    return content.text;
  }

  try {
    return JSON.stringify(content ?? null);
  } catch {
    return String(content);
  }
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
        trust: typeof item.trust === "string" ? item.trust : undefined
      }
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
    entryId: entry.id
  };
}

export function streamEntriesToChatTurns(entries: readonly StreamEntry[]): ChatTurn[] {
  const ordered = [...entries].sort((left, right) => {
    if (left.timestamp !== right.timestamp) {
      return left.timestamp - right.timestamp;
    }
    return left.id.localeCompare(right.id);
  });
  const turns: ChatTurn[] = [];
  const byTurnId = new Map<string, ChatTurn>();

  for (const entry of ordered) {
    if (entry.kind === "user_msg" || entry.kind === "agent_msg") {
      const turn: ChatTurn = {
        entry,
        role: entry.kind === "user_msg" ? "user" : "borg",
        text: contentText(entry.content),
        attachments: [],
        thought: contentThought(entry.content),
        refs: contentRefs(entry.content)
      };
      turns.push(turn);
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
      const parent = key === null ? undefined : byTurnId.get(key);
      if (parent !== undefined) {
        parent.attachments.push(attachment);
      } else {
        turns.push({
          entry,
          role: "user",
          text: "image attachment",
          attachments: [attachment]
        });
      }
    }
  }

  return turns;
}
