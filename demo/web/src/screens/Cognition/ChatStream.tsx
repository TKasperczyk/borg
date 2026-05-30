import { useEffect, useMemo, useRef } from "react";

import { Empty } from "../../components/Empty";
import { streamEntriesToChatTurns, type ChatStreamEntry } from "./chat-utils";
import { ChatMessage } from "./ChatMessage";

export type ChatStreamProps = {
  entries: readonly ChatStreamEntry[];
  sessionId: string;
  audience: string;
  running: boolean;
};

function shortSession(sessionId: string): string {
  if (sessionId.length <= 10) {
    return sessionId;
  }
  return sessionId.slice(0, 10);
}

export function ChatStream({ entries, sessionId, audience, running }: ChatStreamProps) {
  const chatRef = useRef<HTMLDivElement | null>(null);
  const turns = useMemo(() => streamEntriesToChatTurns(entries), [entries]);

  useEffect(() => {
    if (chatRef.current !== null) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [turns.length, running]);

  return (
    <>
      <div className="chat-head">
        <span className="title">transcript · {shortSession(sessionId)}</span>
        <span className="chat-audience">
          <span className="label">audience</span>
          <span className="val">{audience}</span>
        </span>
      </div>
      <div className="chat-stream" ref={chatRef}>
        {turns.length === 0 ? <Empty>no chat history for this audience</Empty> : null}
        {turns.map((turn) => (
          <ChatMessage key={turn.entry.id} turn={turn} audience={audience} />
        ))}
        {running ? (
          <div className="chat-msg borg">
            <div className="avatar" aria-hidden="true">
              ψ
            </div>
            <div className="content">
              <div className="meta">
                <span className="role borg">borg</span>
                <span className="sep">·</span>
                <span className="acc">thinking</span>
              </div>
              <div className="body" style={{ minHeight: 22 }}>
                borg is thinking...
                <span className="acc">▍</span>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </>
  );
}
