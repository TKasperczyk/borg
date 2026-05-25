import { useEffect, useMemo, useRef, useState } from "react";

import { getSharedState, getStream } from "../../api/client";
import type { SharedStateEntry, StreamChatKind, StreamEntry, TurnStakes } from "../../api/types";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import type { TurnStreamState } from "../../hooks/use-turn-stream";
import { ChatInput } from "./ChatInput";
import { ChatStream } from "./ChatStream";
import { Xray } from "./Xray";

const CHAT_KINDS: readonly StreamChatKind[] = ["user_msg", "agent_msg", "user_image_attachment"];

export type CognitionScreenProps = {
  sessionId: string;
  audience: string;
  turnStream: TurnStreamState;
};

function mergeEntries(current: readonly StreamEntry[], incoming: readonly StreamEntry[]): StreamEntry[] {
  const byId = new Map(current.map((entry) => [entry.id, entry]));
  for (const entry of incoming) {
    byId.set(entry.id, entry);
  }
  return [...byId.values()].sort((left, right) => {
    if (left.timestamp !== right.timestamp) {
      return left.timestamp - right.timestamp;
    }
    return left.id.localeCompare(right.id);
  });
}

function isChatEntry(entry: StreamEntry, audience: string): boolean {
  return CHAT_KINDS.includes(entry.kind as StreamChatKind) && entry.audience === audience;
}

export function CognitionScreen({ sessionId, audience, turnStream }: CognitionScreenProps) {
  const live = useLiveEventsContext();
  const [chatEntries, setChatEntries] = useState<StreamEntry[]>([]);
  const [sharedEntries, setSharedEntries] = useState<SharedStateEntry[]>([]);
  const previousConnectionCountRef = useRef(live.connectionCount);

  const streamApi = useApi(
    () => getStream({ audience, kinds: CHAT_KINDS, limit: 50 }),
    [audience]
  );
  const sharedApi = useApi(() => getSharedState(audience), [audience]);
  const refetchShared = sharedApi.refetch;
  const resetForReconnect = turnStream.resetForReconnect;
  const replaceTailFromEntries = turnStream.replaceTailFromEntries;

  useEffect(() => {
    if (streamApi.data !== null) {
      setChatEntries(mergeEntries([], streamApi.data.entries));
    }
  }, [streamApi.data]);

  useEffect(() => {
    if (sharedApi.data !== null) {
      setSharedEntries(sharedApi.data.entries);
    }
  }, [sharedApi.data]);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type !== "stream:append") {
        return;
      }

      const matching = frame.entries.filter((entry) => isChatEntry(entry, audience));
      if (matching.length > 0) {
        setChatEntries((current) => mergeEntries(current, matching));
      }

      void refetchShared();
    });
  }, [audience, live, refetchShared]);

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const [stream, shared] = await Promise.all([
          getStream({ audience, kinds: CHAT_KINDS, limit: 50 }),
          getSharedState(audience)
        ]);

        if (cancelled) {
          return;
        }

        setChatEntries(mergeEntries([], stream.entries));
        setSharedEntries(shared.entries);
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
  }, [audience, live.connectionCount, replaceTailFromEntries, resetForReconnect]);

  const activeLedger = useMemo(() => {
    if (turnStream.activeTurnId === null) {
      return undefined;
    }
    return turnStream.ledgerByTurn.get(turnStream.activeTurnId);
  }, [turnStream.activeTurnId, turnStream.ledgerByTurn]);

  const send = ({ message, stakes }: { message: string; stakes: TurnStakes }) => {
    void turnStream.runTurn({ message, audience, stakes });
  };

  return (
    <div className="cog">
      <div className="chat">
        <ChatStream
          entries={chatEntries}
          sessionId={sessionId}
          audience={audience}
          running={turnStream.running}
        />
        <ChatInput audience={audience} running={turnStream.running} onSend={send} />
      </div>
      <div className="cog-divider"></div>
      <Xray
        phases={turnStream.phases}
        activeTurnId={turnStream.activeTurnId}
        ledger={activeLedger}
        sharedEntries={sharedEntries}
        audience={audience}
        tailEvents={turnStream.eventTail}
      />
    </div>
  );
}
