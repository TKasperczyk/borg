import { useEffect, useRef, useState } from "react";

import { deleteAdvice, getAdviceHistory, getAdvicePending, getStream, postAdvice } from "../../api/client";
import type {
  OperatorAdviceRecord,
  StreamChatKind,
  StreamEntry,
  TurnStakes,
} from "../../api/types";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import type { TurnStreamState } from "../../hooks/use-turn-stream";
import { mergeEntries } from "../../lib/stream-utils";
import { ChatInput } from "./ChatInput";
import { ChatStream } from "./ChatStream";
import { Xray } from "./Xray";

const CHAT_KINDS: readonly StreamChatKind[] = ["user_msg", "agent_msg", "user_image_attachment"];
const CHAT_PANEL_LIMIT = 16;
const ADVICE_REFRESH_DELAY_MS = 350;

export type CognitionScreenProps = {
  sessionId: string;
  audience: string;
  turnStream: TurnStreamState;
};

function isChatEntry(entry: StreamEntry, sessionId: string, audience: string): boolean {
  return (
    entry.session_id === sessionId &&
    CHAT_KINDS.includes(entry.kind as StreamChatKind) &&
    entry.audience === audience
  );
}

function adviceStatus(record: OperatorAdviceRecord): "pending" | "consumed" | "canceled" | "expired" {
  if (record.consumed_at !== null) {
    return "consumed";
  }
  if (record.canceled_at !== null) {
    return "canceled";
  }
  if (record.expires_at !== null && record.expires_at <= Date.now()) {
    return "expired";
  }
  return "pending";
}

function formatAdviceTimestamp(record: OperatorAdviceRecord): string {
  const timestamp =
    record.consumed_at ?? record.canceled_at ?? record.expires_at ?? record.created_at;

  return new Date(timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function OperatorAdvicePanel({
  sessionId,
  pendingItems,
  historyItems,
  loading,
  onRefresh,
}: {
  sessionId: string;
  pendingItems: readonly OperatorAdviceRecord[];
  historyItems: readonly OperatorAdviceRecord[];
  loading: boolean;
  onRefresh: () => Promise<void>;
}) {
  const [text, setText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const trimmed = text.trim();

  const submit = () => {
    if (trimmed.length === 0 || submitting) {
      return;
    }

    void (async () => {
      setSubmitting(true);
      setError(null);
      try {
        await postAdvice({ text: trimmed, session_id: sessionId });
        setText("");
        await onRefresh();
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      } finally {
        setSubmitting(false);
      }
    })();
  };

  const cancel = (id: string) => {
    void (async () => {
      setError(null);
      try {
        await deleteAdvice(id);
        await onRefresh();
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    })();
  };

  return (
    <section className="operator-advice" aria-label="Advice for next turn">
      <div className="operator-advice-head">
        <span className="operator-advice-title">Advice for next turn</span>
        <span className="operator-advice-count">{pendingItems.length} pending</span>
      </div>
      <div className="operator-advice-compose">
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="creator guidance"
          rows={2}
        />
        <button
          className="btn sm primary"
          type="button"
          onClick={submit}
          disabled={trimmed.length === 0 || submitting}
        >
          queue
        </button>
      </div>
      {error === null ? null : <div className="operator-advice-error">{error}</div>}
      <div className="operator-advice-list">
        {loading && pendingItems.length === 0 ? (
          <div className="operator-advice-empty">Loading advice...</div>
        ) : pendingItems.length === 0 ? (
          <div className="operator-advice-empty">No pending advice.</div>
        ) : (
          pendingItems.map((item) => (
            <div key={item.id} className="operator-advice-item">
              <p>{item.text}</p>
              <button
                className="btn sm ghost"
                type="button"
                onClick={() => cancel(item.id)}
                aria-label={`cancel advice ${item.id}`}
              >
                cancel
              </button>
            </div>
          ))
        )}
      </div>
      <details className="operator-advice-history">
        <summary>Recent history</summary>
        {historyItems.length === 0 ? (
          <div className="operator-advice-empty">No advice history.</div>
        ) : (
          historyItems.map((item) => (
            <div key={item.id} className="operator-advice-history-row">
              <span>{adviceStatus(item)}</span>
              <span>{formatAdviceTimestamp(item)}</span>
              <p>{item.text}</p>
            </div>
          ))
        )}
      </details>
    </section>
  );
}

export function CognitionScreen({ sessionId, audience, turnStream }: CognitionScreenProps) {
  const live = useLiveEventsContext();
  const [chatEntries, setChatEntries] = useState<StreamEntry[]>([]);
  const previousConnectionCountRef = useRef(live.connectionCount);
  const adviceRefreshTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const streamApi = useApi(
    () => getStream({ session: sessionId, audience, kinds: CHAT_KINDS, limit: 50 }),
    [audience, sessionId],
  );
  const pendingAdviceApi = useApi(() => getAdvicePending(sessionId), [sessionId]);
  const adviceHistoryApi = useApi(() => getAdviceHistory(sessionId, 12), [sessionId]);
  const resetForReconnect = turnStream.resetForReconnect;
  const replaceTailFromEntries = turnStream.replaceTailFromEntries;

  const refreshAdvice = async () => {
    await pendingAdviceApi.refetch();
    await adviceHistoryApi.refetch();
  };

  useEffect(() => {
    setChatEntries([]);
  }, [audience, sessionId]);

  useEffect(
    () => () => {
      if (adviceRefreshTimerRef.current !== null) {
        clearTimeout(adviceRefreshTimerRef.current);
      }
    },
    [],
  );

  useEffect(() => {
    const streamData = streamApi.data;

    if (streamData !== null) {
      setChatEntries((current) =>
        mergeEntries(
          current.filter((entry) => entry.session_id === sessionId && entry.audience === audience),
          streamData.entries,
        ),
      );
    }
  }, [audience, streamApi.data]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type !== "stream:append") {
        return;
      }

      const matching = frame.entries.filter((entry) => isChatEntry(entry, sessionId, audience));
      if (matching.length > 0) {
        setChatEntries((current) => mergeEntries(current, matching));
      }
    });
  }, [audience, live, sessionId]);

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const stream = await getStream({
          session: sessionId,
          audience,
          kinds: CHAT_KINDS,
          limit: 50,
        });

        if (cancelled) {
          return;
        }

        setChatEntries((current) =>
          mergeEntries(
            current.filter(
              (entry) => entry.session_id === sessionId && entry.audience === audience,
            ),
            stream.entries,
          ),
        );
        replaceTailFromEntries(stream.entries);
      } catch {
        // The standing useApi calls retain the previous visible error/data state.
      } finally {
        if (!cancelled) {
          resetForReconnect();
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [audience, live.connectionCount, replaceTailFromEntries, resetForReconnect, sessionId]);

  const send = async (input: {
    message: string;
    stakes: TurnStakes;
    attachments?: readonly File[];
  }) => {
    const accepted = await turnStream.runTurn({ ...input, audience, session: sessionId });

    if (accepted) {
      if (adviceRefreshTimerRef.current !== null) {
        clearTimeout(adviceRefreshTimerRef.current);
      }
      adviceRefreshTimerRef.current = setTimeout(() => {
        void refreshAdvice();
      }, ADVICE_REFRESH_DELAY_MS);
    }

    return accepted;
  };

  return (
    <div className="cog">
      <div className="chat">
        <ChatStream
          entries={chatEntries.slice(-CHAT_PANEL_LIMIT)}
          sessionId={sessionId}
          audience={audience}
          running={turnStream.running}
        />
        <OperatorAdvicePanel
          sessionId={sessionId}
          pendingItems={pendingAdviceApi.data?.items ?? []}
          historyItems={adviceHistoryApi.data?.items ?? []}
          loading={pendingAdviceApi.loading}
          onRefresh={refreshAdvice}
        />
        <ChatInput audience={audience} running={turnStream.running} onSend={send} />
      </div>
      <div className="cog-divider"></div>
      <Xray
        phases={turnStream.phases}
        activeTurnId={turnStream.activeTurnId}
        tokenTextByPhase={turnStream.tokenTextByPhase}
        terminalOutcome={turnStream.terminalOutcome}
        delibPath={turnStream.delibPath}
        finalAttempt={turnStream.finalAttempt}
      />
    </div>
  );
}
