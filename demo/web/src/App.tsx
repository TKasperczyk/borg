import { useEffect, useState } from "react";

import { getState } from "./api/client";
import { LiveEventsProvider } from "./hooks/live-context";
import { useApi } from "./hooks/use-api";
import { useLiveEvents } from "./hooks/use-live-events";
import { useTurnStream } from "./hooks/use-turn-stream";
import { Rail, type RouteId } from "./components/Rail";
import { StatusBar } from "./components/StatusBar";
import { Topbar } from "./components/Topbar";
import { CognitionScreen } from "./screens/Cognition";
import { CommitScreen } from "./screens/Commit";
import { DreamScreen } from "./screens/Dream";
import { GraphScreen } from "./screens/Graph";
import { IdentityScreen } from "./screens/Identity";
import { MemoryScreen } from "./screens/Memory";
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
  const stateApi = useApi(getState, []);
  const refetchState = stateApi.refetch;
  const live = useLiveEvents({ onReconnected: refetchState });
  const turnStream = useTurnStream(live);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(formatNow()), 1_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    return live.subscribe((frame) => {
      if (frame.type === "stream:append") {
        void refetchState();
      }
    });
  }, [live, refetchState]);

  const sessionId = stateApi.data?.active_session ?? "default";

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
          {route === "cognition" ? <CognitionScreen sessionId={sessionId} audience={AUDIENCE} turnStream={turnStream} /> : null}
          {route === "stream" ? <StreamScreen /> : null}
          {route === "memory" ? <MemoryScreen /> : null}
          {route === "identity" ? <IdentityScreen /> : null}
          {route === "commit" ? <CommitScreen /> : null}
          {route === "shared" ? <SharedScreen /> : null}
          {route === "dream" ? <DreamScreen /> : null}
          {route === "graph" ? <GraphScreen /> : null}
        </div>
        <StatusBar state={stateApi.data} lastPhase={turnStream.lastPhase} />
      </div>
    </LiveEventsProvider>
  );
}
