import { AttachmentChip } from "../../components/AttachmentChip";
import { Tag } from "../../components/Tag";
import type { ChatTurn } from "./chat-utils";
import { timestampLabel } from "./chat-utils";

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
        <span className="dim">{timestampLabel(turn.entry.timestamp)}</span>
        {turn.refs === undefined ? null : <span className="dim">· {turn.refs.length} refs</span>}
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
      {turn.thought === undefined ? null : (
        <div className="thought">
          <span className="purple">thought ▸</span> {turn.thought}
        </div>
      )}
      {turn.refs === undefined || turn.refs.length === 0 ? null : (
        <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: 6 }}>
          {turn.refs.map((ref) => (
            <Tag key={`${ref.kind}:${ref.id}`} kind={ref.kind === "cm" ? "bad" : ""}>
              [{ref.kind}:{ref.id}] {ref.label}
            </Tag>
          ))}
        </div>
      )}
    </div>
  );
}
