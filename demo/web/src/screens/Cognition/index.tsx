import { useEffect, useRef, useState } from "react";

import { getStream } from "../../api/client";
import type { StreamChatKind, StreamEntry, TurnStakes } from "../../api/types";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import type { TurnStreamState } from "../../hooks/use-turn-stream";
import { mergeEntries } from "../../lib/stream-utils";
import { ChatInput } from "./ChatInput";
import { ChatStream } from "./ChatStream";
import { Xray } from "./Xray";

const CHAT_KINDS: readonly StreamChatKind[] = ["user_msg", "agent_msg", "user_image_attachment"];
const CHAT_PANEL_LIMIT = 16;

export type CognitionScreenProps = {
  sessionId: string;
  audience: string;
  turnStream: TurnStreamState;
};

function isChatEntry(entry: StreamEntry, audience: string): boolean {
  return CHAT_KINDS.includes(entry.kind as StreamChatKind) && entry.audience === audience;
}

export function CognitionScreen({ sessionId, audience, turnStream }: CognitionScreenProps) {
  const live = useLiveEventsContext();
  const [chatEntries, setChatEntries] = useState<StreamEntry[]>([]);
  const previousConnectionCountRef = useRef(live.connectionCount);

  const streamApi = useApi(() => getStream({ audience, kinds: CHAT_KINDS, limit: 50 }), [audience]);
  const resetForReconnect = turnStream.resetForReconnect;
  const replaceTailFromEntries = turnStream.replaceTailFromEntries;

  useEffect(() => {
    setChatEntries([]);
  }, [audience]);

  useEffect(() => {
    const streamData = streamApi.data;

    if (streamData !== null) {
      setChatEntries((current) =>
        mergeEntries(
          current.filter((entry) => entry.audience === audience),
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

      const matching = frame.entries.filter((entry) => isChatEntry(entry, audience));
      if (matching.length > 0) {
        setChatEntries((current) => mergeEntries(current, matching));
      }
    });
  }, [audience, live]);

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const stream = await getStream({ audience, kinds: CHAT_KINDS, limit: 50 });

        if (cancelled) {
          return;
        }

        setChatEntries((current) =>
          mergeEntries(
            current.filter((entry) => entry.audience === audience),
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
  }, [audience, live.connectionCount, replaceTailFromEntries, resetForReconnect]);

  const send = ({ message, stakes }: { message: string; stakes: TurnStakes }) => {
    void turnStream.runTurn({ message, audience, stakes });
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
