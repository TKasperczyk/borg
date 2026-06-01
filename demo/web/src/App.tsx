import { useEffect, useState } from "react";

import {
  ApiError,
  getCreatorEntity,
  getSessions,
  getState,
  openOperatorSession,
  setCreatorByName,
} from "./api/client";
import type { MaintenanceTickFrame } from "./api/types";
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
import { DirectivesScreen } from "./screens/Directives";
import { DreamScreen } from "./screens/Dream";
import { GraphScreen } from "./screens/Graph";
import { IdentityScreen } from "./screens/Identity";
import { MemoryScreen } from "./screens/Memory";
import { PromptsScreen } from "./screens/Prompts";
import { SharedScreen } from "./screens/Shared";
import { StreamScreen } from "./screens/Stream";

const DEFAULT_AUDIENCE = "alice";

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
  const creatorApi = useApi(getCreatorEntity, []);
  const refetchState = stateApi.refetch;
  const refetchSessions = sessionsApi.refetch;
  const refetchCreator = creatorApi.refetch;
  const [operatorChatError, setOperatorChatError] = useState<string | null>(null);
  const [lastMaintenanceTick, setLastMaintenanceTick] = useState<MaintenanceTickFrame | null>(null);
  const live = useLiveEvents({ onReconnected: refetchState, sessionId });
  const turnStream = useTurnStream(live, { sessionId });
  const activeSession =
    sessionsApi.data?.sessions.find((session) => session.session_id === sessionId) ?? null;
  const activeAudience = activeSession?.audience_label ?? DEFAULT_AUDIENCE;

  const refetchSessionState = async () => {
    await Promise.all([refetchSessions(), refetchState()]);
  };

  const openOperatorChat = async () => {
    try {
      const session = await openOperatorSession();
      setSessionId(session.session_id);
      setOperatorChatError(null);
      await Promise.all([refetchSessions(), refetchState(), refetchCreator()]);
    } catch (error) {
      if (error instanceof ApiError && error.status === 409) {
        setOperatorChatError("mark a creator first");
        return;
      }
      throw error;
    }
  };

  const markCreatorByName = async (name: string) => {
    await setCreatorByName(name);
    setOperatorChatError(null);
    await Promise.all([refetchCreator(), refetchSessions(), refetchState()]);
  };

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
      if (frame.type === "maintenance:tick") {
        setLastMaintenanceTick(frame);
        void refetchState();
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
          audience={activeAudience}
          turns={stateApi.data?.counts.turns ?? 0}
          ws_state={live.wsState}
          now={now}
          route={route}
        />
        <div className="main">
          <SessionsSidebar
            sessions={sessionsApi.data?.sessions ?? []}
            activeSessionId={sessionId}
            onSelect={setSessionId}
            creator={creatorApi.data ?? null}
            operatorChatError={operatorChatError}
            onOpenOperatorChat={openOperatorChat}
            onSetCreatorByName={markCreatorByName}
          />
          <div className="screen-shell">
            {route === "cognition" ? (
              <CognitionScreen
                sessionId={sessionId}
                audience={activeAudience}
                audienceEntityId={activeSession?.audience_entity_id ?? null}
                turnStream={turnStream}
                session={activeSession}
                onSessionPolicyChanged={refetchSessionState}
              />
            ) : null}
            {route === "stream" ? <StreamScreen sessionId={sessionId} /> : null}
            {route === "memory" ? <MemoryScreen sessionId={sessionId} /> : null}
            {route === "identity" ? <IdentityScreen /> : null}
            {route === "commit" ? <CommitScreen /> : null}
            {route === "directives" ? <DirectivesScreen /> : null}
            {route === "shared" ? <SharedScreen /> : null}
            {route === "dream" ? <DreamScreen /> : null}
            {route === "graph" ? <GraphScreen /> : null}
            {route === "prompts" ? <PromptsScreen /> : null}
          </div>
        </div>
        <StatusBar
          state={stateApi.data}
          lastPhase={turnStream.lastPhase}
          lastMaintenanceTick={lastMaintenanceTick}
        />
      </div>
    </LiveEventsProvider>
  );
}
