import { useEffect, useState } from "react";

import { getSessions, getState } from "./api/client";
import { LiveEventsProvider } from "./hooks/live-context";
import { useApi } from "./hooks/use-api";
import { useLiveEvents } from "./hooks/use-live-events";
import { useSession } from "./hooks/use-session";
import { useTurnStream } from "./hooks/use-turn-stream";
import { Rail, type RouteId } from "./components/Rail";
import { SessionsSidebar } from "./components/SessionsSidebar";
import { StatusBar } from "./components/StatusBar";
import { Topbar } from "./components/Topbar";
import { CognitionScreen } from "./screens/Cognition";
import { CommitScreen } from "./screens/Commit";
import { DreamScreen } from "./screens/Dream";
import { GraphScreen } from "./screens/Graph";
import { IdentityScreen } from "./screens/Identity";
import { MemoryScreen } from "./screens/Memory";
import { PromptsScreen } from "./screens/Prompts";
import { SharedScreen } from "./screens/Shared";
import { StreamScreen } from "./screens/Stream";

const AUDIENCE = "alice";

function formatNow(): string {
  const date = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

export function App() {
  const [route, setRoute] = useState<RouteId>("cognition");
  const [now, setNow] = useState(formatNow);
  const { sessionId, setSessionId } = useSession();
  const stateApi = useApi(() => getState({ session: sessionId }), [sessionId]);
  const sessionsApi = useApi(getSessions, []);
  const refetchState = stateApi.refetch;
  const refetchSessions = sessionsApi.refetch;
  const live = useLiveEvents({ onReconnected: refetchState, sessionId });
  const turnStream = useTurnStream(live, { sessionId });

  useEffect(() => {
    const timer = window.setInterval(() => setNow(formatNow()), 1_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type === "stream:append") {
        void refetchState();
        void refetchSessions();
      }
      if (frame.type === "borg:reset") {
        window.location.reload();
      }
    });
  }, [live, refetchSessions, refetchState]);

  return (
    <LiveEventsProvider value={live}>
      <div className="app">
        <Rail route={route} setRoute={setRoute} />
        <Topbar
          session_id={sessionId}
          audience={AUDIENCE}
          turns={stateApi.data?.counts.turns ?? 0}
          ws_state={live.wsState}
          now={now}
        />
        <div className="main">
          <SessionsSidebar
            sessions={sessionsApi.data?.sessions ?? []}
            activeSessionId={sessionId}
            onSelect={setSessionId}
          />
          <div className="screen-shell">
            {route === "cognition" ? (
              <CognitionScreen
                sessionId={sessionId}
                audience={AUDIENCE}
                turnStream={turnStream}
              />
            ) : null}
            {route === "stream" ? <StreamScreen sessionId={sessionId} /> : null}
            {route === "memory" ? <MemoryScreen sessionId={sessionId} /> : null}
            {route === "identity" ? <IdentityScreen /> : null}
            {route === "commit" ? <CommitScreen /> : null}
            {route === "shared" ? <SharedScreen /> : null}
            {route === "dream" ? <DreamScreen /> : null}
            {route === "graph" ? <GraphScreen /> : null}
            {route === "prompts" ? <PromptsScreen /> : null}
          </div>
        </div>
        <StatusBar state={stateApi.data} lastPhase={turnStream.lastPhase} />
      </div>
    </LiveEventsProvider>
  );
}
