import { useEffect, useMemo, useRef } from "react";

import type { StreamEntry } from "../../api/types";
import { Empty } from "../../components/Empty";
import { streamEntriesToChatTurns } from "./chat-utils";
import { ChatMessage } from "./ChatMessage";

export type ChatStreamProps = {
  entries: readonly StreamEntry[];
  sessionId: string;
  audience: string;
  running: boolean;
};

export function ChatStream({ entries, sessionId, audience, running }: ChatStreamProps) {
  const chatRef = useRef<HTMLDivElement | null>(null);
  const turns = useMemo(() => streamEntriesToChatTurns(entries), [entries]);

  useEffect(() => {
    if (chatRef.current !== null) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight;
    }
  }, [turns.length, running]);

  return (
    <div className="chat-stream" ref={chatRef}>
      <div
        style={{
          marginBottom: 22,
          padding: "8px 12px",
          border: "1px solid var(--line-soft)",
          background: "var(--bg-1)",
          fontSize: "10.5px",
          color: "var(--text-mute)",
          display: "flex",
          gap: 14,
          alignItems: "center",
          whiteSpace: "nowrap",
          overflow: "hidden"
        }}
      >
        <span>
          <span className="dim">session</span> <span className="acc">{sessionId}</span>
        </span>
        <span className="dim">·</span>
        <span>
          <span className="dim">audience</span> <span className="acc">{audience}</span>
        </span>
        <span className="dim">·</span>
        <span>
          <span className="dim">kind</span> 1:1
        </span>
        <span style={{ flex: 1 }}></span>
        <span className="acc">loop attached ▸</span>
      </div>

      {turns.length === 0 ? <Empty>no chat history for this audience</Empty> : null}
      {turns.map((turn) => (
        <ChatMessage key={turn.entry.id} turn={turn} audience={audience} />
      ))}
      {running ? (
        <div className="chat-msg borg">
          <div className="meta">
            <span className="role borg">borg ⟶ {audience}</span>
            <span className="acc">· thinking</span>
          </div>
          <div className="body" style={{ minHeight: 22 }}>
            borg is thinking...
            <span className="acc">▍</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}
