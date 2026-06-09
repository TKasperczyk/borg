import { AttachmentChip } from "../../components/AttachmentChip";
import { IdRef } from "../../components/Inspector/IdRef";
import { formatTime } from "../../lib/stream-utils";
import { shortId } from "../screen-utils";
import type { ChatTurn } from "./chat-utils";

export type ChatMessageProps = {
  turn: ChatTurn;
  audience: string;
};

function userInitial(audience: string): string {
  if (audience.length === 0) {
    return "?";
  }
  const first = audience.trim()[0];
  if (first === undefined) {
    return "?";
  }
  return first.toUpperCase();
}

function roleName(turn: ChatTurn, audience: string): string {
  return turn.role === "borg" ? "borg" : audience;
}

function shortTurnId(turnId: string | undefined): string | null {
  if (typeof turnId !== "string" || turnId.length === 0) {
    return null;
  }
  const trimmed = turnId.replace(/^turn[-_]?/i, "");
  return trimmed.slice(0, 6);
}

export function ChatMessage({ turn, audience }: ChatMessageProps) {
  const initial = turn.role === "borg" ? "ψ" : userInitial(audience);
  const name = roleName(turn, audience);
  const turnId = turn.entry.turn_id;
  const turnShort = shortTurnId(turnId);
  const deliveryStatus = turn.entry.optimistic_status;
  const messageAudience = turn.entry.audience;

  return (
    <div
      className={`chat-msg ${turn.role}${deliveryStatus === undefined ? "" : " optimistic"}`}
      data-delivery-status={deliveryStatus ?? ""}
    >
      <div className="avatar" aria-hidden="true">
        {initial}
      </div>
      <div className="content">
        <div className="meta">
          <span className={`role ${turn.role}`}>{name}</span>
          {messageAudience === undefined ? null : (
            <>
              <span className="sep">·</span>
              <span className="chat-chip">aud {messageAudience}</span>
            </>
          )}
          {turnShort === null || turnId === undefined ? null : (
            <>
              <span className="sep">·</span>
              <span className="turn">
                <IdRef id={turnId} type="turn" label={`turn ${turnShort}`} />
              </span>
            </>
          )}
          <span className="sep">·</span>
          <span className="when">{formatTime(turn.entry.timestamp)}</span>
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
