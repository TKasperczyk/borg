import { AttachmentChip } from "../../components/AttachmentChip";
import { formatTime } from "../../lib/stream-utils";
import type { ChatTurn } from "./chat-utils";

export type ChatMessageProps = {
  turn: ChatTurn;
  audience: string;
};

export function ChatMessage({ turn, audience }: ChatMessageProps) {
  return (
    <div className={`chat-msg ${turn.role === "borg" ? "borg" : "user"}`}>
      <div className="meta">
        {turn.role === "user" ? (
          <span className="role user">{audience} ⟶ borg</span>
        ) : (
          <span className="role borg">borg ⟶ {audience}</span>
        )}
        <span className="dim">{formatTime(turn.entry.timestamp)}</span>
        {turn.attachments.length === 0 ? null : (
          <span className="dim">
            · {turn.attachments.length} attachment{turn.attachments.length > 1 ? "s" : ""}
          </span>
        )}
      </div>
      <div className="body">{turn.text}</div>
      {turn.attachments.length === 0 ? null : (
        <div className="msg-attachments">
          {turn.attachments.map((attachment) => (
            <AttachmentChip
              key={attachment.entryId}
              attachmentId={attachment.attachmentId}
              mediaType={attachment.mediaType}
              audience={turn.entry.audience}
            />
          ))}
        </div>
      )}
    </div>
  );
}
