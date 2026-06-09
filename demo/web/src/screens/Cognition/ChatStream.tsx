import { useEffect, useMemo, useRef } from "react";

import { Empty } from "../../components/Empty";
import { IdRef } from "../../components/Inspector/IdRef";
import { Tag } from "../../components/Tag";
import { formatTime } from "../../lib/stream-utils";
import { shortId } from "../screen-utils";
import { streamEntriesToChatTurns, type ChatMarker, type ChatStreamEntry } from "./chat-utils";
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

function IdRefList({ ids, type }: { ids: readonly string[]; type: "stream_entry" | "turn" }) {
  if (ids.length === 0) {
    return <span className="dim">none</span>;
  }

  return (
    <>
      {ids.map((id, index) => (
        <span key={id}>
          {index === 0 ? null : ", "}
          <IdRef id={id} type={type} label={shortId(id)} />
        </span>
      ))}
    </>
  );
}

function ChatMarkerRow({ marker }: { marker: ChatMarker }) {
  const invalidTool = marker.summary.finalizerInvalidTool;

  return (
    <details className={`chat-marker ${marker.entry.kind}`}>
      <summary>
        <Tag kind={marker.summary.outcome.tagKind}>{marker.summary.outcome.label}</Tag>
        <span className="chat-marker-kind">{marker.entry.kind}</span>
        {marker.reason === null ? null : (
          <span className="chat-marker-reason">{marker.reason}</span>
        )}
        <span className="chat-marker-time">{formatTime(marker.entry.timestamp)}</span>
      </summary>
      <div className="chat-marker-details">
        <div>
          <span className="k">reason</span>
          <span className="v">{marker.reason ?? "none"}</span>
        </div>
        {marker.summary.primaryNoOutputReason === undefined ? null : (
          <div>
            <span className="k">primary</span>
            <span className="v">{marker.summary.primaryNoOutputReason}</span>
          </div>
        )}
        {marker.summary.noOutputCategories.length === 0 ? null : (
          <div>
            <span className="k">categories</span>
            <span className="v">{marker.summary.noOutputCategories.join(", ")}</span>
          </div>
        )}
        {marker.summary.structuralNoOutputFlags.length === 0 ? null : (
          <div>
            <span className="k">flags</span>
            <span className="v">{marker.summary.structuralNoOutputFlags.join(", ")}</span>
          </div>
        )}
        {invalidTool === undefined ? null : (
          <div>
            <span className="k">invalid tool</span>
            <span className="v">
              {invalidTool.tool_name} · {invalidTool.reason} · {invalidTool.attempt}
            </span>
          </div>
        )}
        <div>
          <span className="k">turn</span>
          <span className="v">
            {marker.turnId === undefined ? (
              <span className="dim">none</span>
            ) : (
              <IdRef id={marker.turnId} type="turn" label={shortId(marker.turnId)} />
            )}
          </span>
        </div>
        <div>
          <span className="k">user entries</span>
          <span className="v">
            <IdRefList ids={marker.userEntryIds} type="stream_entry" />
          </span>
        </div>
      </div>
    </details>
  );
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
        {turns.map((turn) =>
          turn.itemType === "marker" ? (
            <ChatMarkerRow key={turn.entry.id} marker={turn} />
          ) : (
            <ChatMessage key={turn.entry.id} turn={turn} audience={audience} />
          ),
        )}
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
