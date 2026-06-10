import { useLayoutEffect, useMemo, useRef, useState } from "react";

import { Empty } from "../../components/Empty";
import { IdChip } from "../../components/Inspector/IdChip";
import { IdRef } from "../../components/Inspector/IdRef";
import { Tag } from "../../components/Tag";
import type { SessionRecord } from "../../api/types";
import type { AudienceDisplayIdentity } from "../../lib/audience-identity";
import { formatTimestamp } from "../../lib/stream-utils";
import { shortId } from "../screen-utils";
import { streamEntriesToChatTurns, type ChatMarker, type ChatStreamEntry } from "./chat-utils";
import { ChatMessage } from "./ChatMessage";

export type ChatStreamProps = {
  entries: readonly ChatStreamEntry[];
  sessionId: string;
  session?: SessionRecord | null;
  audienceValue: string | null;
  audienceDisplay: AudienceDisplayIdentity;
  running: boolean;
  hasOlder: boolean;
  loadingOlder: boolean;
  olderError: Error | null;
  onLoadOlder: () => Promise<boolean>;
};

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
    <details className={`chat-marker ${marker.entry.kind}`} data-chat-entry-id={marker.entry.id}>
      <summary>
        <Tag kind={marker.summary.outcome.tagKind}>{marker.summary.outcome.label}</Tag>
        <span className="chat-marker-kind">{marker.entry.kind}</span>
        {marker.reason === null ? null : (
          <span className="chat-marker-reason">{marker.reason}</span>
        )}
        <span className="chat-marker-time">{formatTimestamp(marker.entry.timestamp)}</span>
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

export function ChatStream({
  entries,
  sessionId,
  session,
  audienceValue,
  audienceDisplay,
  running,
  hasOlder,
  loadingOlder,
  olderError,
  onLoadOlder,
}: ChatStreamProps) {
  const chatRef = useRef<HTMLDivElement | null>(null);
  const followRef = useRef(true);
  const olderLoadSeqRef = useRef(0);
  const olderScrollAnchorRef = useRef<{
    seq: number;
    id: string;
    top: number;
  } | null>(null);
  const [olderRestoreSeq, setOlderRestoreSeq] = useState(0);
  const turns = useMemo(() => streamEntriesToChatTurns(entries), [entries]);
  const audienceEntityId = audienceDisplay.entityId;
  const audienceLabel = audienceDisplay.label ?? "unknown";

  useLayoutEffect(() => {
    const chat = chatRef.current;
    if (chat === null) {
      return;
    }

    const olderAnchor = olderScrollAnchorRef.current;
    if (olderAnchor !== null) {
      if (olderAnchor.seq !== olderRestoreSeq) {
        return;
      }

      const anchoredElement =
        [...chat.querySelectorAll<HTMLElement>("[data-chat-entry-id]")].find(
          (element) => element.dataset.chatEntryId === olderAnchor.id,
        ) ?? null;
      olderScrollAnchorRef.current = null;
      if (anchoredElement === null) {
        return;
      }

      chat.scrollTop += anchoredElement.getBoundingClientRect().top - olderAnchor.top;
      return;
    }

    if (followRef.current) {
      chat.scrollTop = chat.scrollHeight;
      followRef.current = true;
    }
  }, [entries.length, olderRestoreSeq, running]);

  async function loadOlder(): Promise<void> {
    const seq = olderLoadSeqRef.current + 1;
    olderLoadSeqRef.current = seq;
    const chat = chatRef.current;
    if (chat !== null) {
      const chatTop = chat.getBoundingClientRect().top;
      const anchors = [...chat.querySelectorAll<HTMLElement>("[data-chat-entry-id]")];
      const anchor =
        anchors.find((element) => element.getBoundingClientRect().bottom >= chatTop) ?? null;
      const anchorId = anchor?.dataset.chatEntryId;
      olderScrollAnchorRef.current =
        anchor === null || anchorId === undefined
          ? null
          : {
              seq,
              id: anchorId,
              top: anchor.getBoundingClientRect().top,
            };
    }

    const loaded = await onLoadOlder();
    if (olderScrollAnchorRef.current?.seq !== seq) {
      return;
    }
    if (loaded) {
      setOlderRestoreSeq(seq);
    } else {
      olderScrollAnchorRef.current = null;
    }
  }

  function updateFollowMode(): void {
    const chat = chatRef.current;
    if (chat === null) {
      return;
    }
    followRef.current = chat.scrollHeight - chat.scrollTop - chat.clientHeight <= 24;
  }

  return (
    <>
      <div className="chat-head">
        <span className="title transcript-title">
          transcript · {session?.label ?? "unknown session"}{" "}
          <IdChip id={sessionId} type="session" />
        </span>
        <span className="chat-audience">
          <span className="label">audience</span>
          <span className="val identity-inline">
            <span>{audienceLabel}</span>
            {audienceEntityId === null ? null : <IdChip id={audienceEntityId} type="entity" />}
          </span>
        </span>
      </div>
      <div className="chat-stream" ref={chatRef} onScroll={updateFollowMode}>
        {hasOlder ? (
          <div className="chat-history-control">
            <button
              type="button"
              className="btn sm ghost"
              onClick={() => {
                void loadOlder();
              }}
              disabled={loadingOlder}
            >
              {loadingOlder ? "loading older" : "load older"}
            </button>
          </div>
        ) : null}
        {olderError === null ? null : (
          <div className="chat-history-note">
            Older transcript unavailable: {olderError.message}
          </div>
        )}
        {turns.length === 0 ? <Empty>no chat history for this audience</Empty> : null}
        {turns.map((turn) =>
          turn.itemType === "marker" ? (
            <ChatMarkerRow key={turn.entry.id} marker={turn} />
          ) : (
            <ChatMessage
              key={turn.entry.id}
              turn={turn}
              audience={audienceValue ?? ""}
              audienceLabel={audienceDisplay.label}
            />
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
