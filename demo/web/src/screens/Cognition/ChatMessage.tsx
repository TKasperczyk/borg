import { AttachmentChip } from "../../components/AttachmentChip";
import { IdChip } from "../../components/Inspector/IdChip";
import { IdRef } from "../../components/Inspector/IdRef";
import { formatTimestamp } from "../../lib/stream-utils";
import { shortId } from "../screen-utils";
import type { ChatTurn } from "./chat-utils";

export type ChatMessageProps = {
  turn: ChatTurn;
  audience: string;
  audienceLabel?: string | null;
};

function userInitial(label: string): string {
  if (label.length === 0) {
    return "?";
  }
  const first = label.trim()[0];
  if (first === undefined) {
    return "?";
  }
  return first.toUpperCase();
}

export function ChatMessage({ turn, audienceLabel = null }: ChatMessageProps) {
  const senderLabel =
    turn.entry.sender_label?.trim() ||
    (turn.entry.sender_entity_id === null ? audienceLabel : null);
  const name = turn.role === "borg" ? "borg" : (senderLabel ?? "unknown speaker");
  const initial = turn.role === "borg" ? "ψ" : userInitial(name);
  const turnId = turn.entry.turn_id;
  const deliveryStatus = turn.entry.optimistic_status;
  const messageAudience = turn.entry.audience;
  const messageAudienceLabel = turn.entry.audience_label ?? audienceLabel;

  return (
    <div
      className={`chat-msg ${turn.role}${deliveryStatus === undefined ? "" : " optimistic"}`}
      data-chat-entry-id={turn.entry.id}
      data-delivery-status={deliveryStatus ?? ""}
    >
      <div className="avatar" aria-hidden="true">
        {initial}
      </div>
      <div className="content">
        <div className="meta">
          <span className={`role ${turn.role}`}>{name}</span>
          {turn.role === "borg" || turn.entry.sender_entity_id === null ? null : (
            <>
              <span className="sep">·</span>
              <IdChip id={turn.entry.sender_entity_id} type="entity" />
            </>
          )}
          {messageAudience === undefined ? null : (
            <>
              <span className="sep">·</span>
              <span className="chat-chip">
                aud {messageAudienceLabel ?? "unknown"}
                {messageAudienceLabel === null ? (
                  <>
                    {" "}
                    <IdChip id={messageAudience} type={null} />
                  </>
                ) : null}
              </span>
            </>
          )}
          {turnId === undefined ? null : (
            <>
              <span className="sep">·</span>
              <span className="turn">
                <IdRef id={turnId} type="turn" label={`turn ${shortId(turnId)}`} />
              </span>
            </>
          )}
          <span className="sep">·</span>
          <span className="when">{formatTimestamp(turn.entry.timestamp)}</span>
          {deliveryStatus === undefined ? null : (
            <>
              <span className="sep">·</span>
              <span className={`delivery ${deliveryStatus}`}>{deliveryStatus}</span>
            </>
          )}
          {turn.attachments.length === 0 ? null : (
            <>
              <span className="sep">·</span>
              <span className="when">
                {turn.attachments.length} attachment
                {turn.attachments.length > 1 ? "s" : ""}
              </span>
            </>
          )}
          {turn.sourceEntryIds.length === 0 ? null : (
            <>
              <span className="sep">·</span>
              <span className="response-refs">
                src{" "}
                {turn.sourceEntryIds.map((entryId, index) => (
                  <span key={entryId}>
                    {index === 0 ? null : ", "}
                    <IdRef id={entryId} type="stream_entry" label={shortId(entryId)} />
                  </span>
                ))}
              </span>
            </>
          )}
        </div>
        {turn.thought === undefined ? null : <div className="thought">{turn.thought}</div>}
        <div className="body">{turn.text}</div>
        {turn.refs === undefined || turn.refs.length === 0 ? null : (
          <div className="refs">
            {turn.refs.map((ref) => (
              <span
                key={ref.id}
                className={`ref ${ref.trust === "low" || ref.trust === "untrusted" ? "bad" : ""}`}
              >
                {ref.label}
              </span>
            ))}
          </div>
        )}
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
    </div>
  );
}
