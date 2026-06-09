import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  getCreatorEntity,
  getSessions,
  getState,
  openOperatorSession,
  setCreatorByName,
} from "./api/client";
import type { MaintenanceTickFrame, StateSnapshot } from "./api/types";
import { AppErrorBoundary } from "./components/AppErrorBoundary";
import { InstrumentStrip } from "./components/InstrumentStrip";
import { Rail, type RailBadge, type RouteId } from "./components/Rail";
import { SessionFleet } from "./components/SessionFleet";
import { StatusBar } from "./components/StatusBar";
import { LiveEventsProvider } from "./hooks/live-context";
import { useApi } from "./hooks/use-api";
import { useLiveEvents } from "./hooks/use-live-events";
import { useSession } from "./hooks/use-session";
import { useTurnStream } from "./hooks/use-turn-stream";
import { useView } from "./hooks/use-view";
import { CognitionScreen } from "./screens/Cognition";
import { CommitScreen } from "./screens/Commit";
import { DirectivesScreen } from "./screens/Directives";
import { DreamScreen } from "./screens/Dream";
import { IdentityScreen } from "./screens/Identity";
import { MemoryScreen } from "./screens/Memory";
import { PromptsScreen } from "./screens/Prompts";
import { ReviewScreen } from "./screens/Review";
import { StreamScreen } from "./screens/Stream";

const DEFAULT_AUDIENCE = "alice";

function formatNow(): string {
  const date = new Date();
  const pad = (value: number) => String(value).padStart(2, "0");
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function countBadge(
  count: number | undefined,
  severity: RailBadge["severity"],
  label: string,
): RailBadge | undefined {
  if (count === undefined) {
    return undefined;
  }

  return { count, severity, label };
}

function railBadges(state: StateSnapshot | null): Partial<Record<RouteId, RailBadge>> {
  return {
    identity: countBadge(state?.counts.open_qs, 1, "open questions"),
    commit: countBadge(state?.counts.commitments, 1, "commitments"),
    review: countBadge(state?.counts.open_reviews, 2, "open reviews"),
    dream: countBadge(state?.counts.dream_audit_rows, 1, "dream audit rows"),
  };
}

export function AppShell() {
  const { view, setView } = useView();
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
  const badges = useMemo(() => railBadges(stateApi.data), [stateApi.data]);

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
        <Rail route={view} setRoute={setView} badges={badges} />
        <InstrumentStrip
          sessionId={sessionId}
          activeSession={activeSession}
          audience={activeAudience}
          creator={creatorApi.data ?? null}
          state={stateApi.data}
          wsState={live.wsState}
          now={now}
          route={view}
        />
        <div className="main">
          <SessionFleet
            sessions={sessionsApi.data?.sessions ?? []}
            activeSessionId={sessionId}
            onSelect={setSessionId}
            creator={creatorApi.data ?? null}
            operatorChatError={operatorChatError}
            onOpenOperatorChat={openOperatorChat}
            onSetCreatorByName={markCreatorByName}
          />
          <div className="screen-shell">
            <AppErrorBoundary resetKey={view}>
              {view === "cognition" ? (
                <CognitionScreen
                  sessionId={sessionId}
                  audience={activeAudience}
                  audienceEntityId={activeSession?.audience_entity_id ?? null}
                  turnStream={turnStream}
                  session={activeSession}
                  onSessionPolicyChanged={refetchSessionState}
                />
              ) : null}
              {view === "stream" ? <StreamScreen sessionId={sessionId} /> : null}
              {view === "memory" ? (
                <MemoryScreen sessionId={sessionId} onOpenReview={() => setView("review")} />
              ) : null}
              {view === "identity" ? <IdentityScreen /> : null}
              {view === "commit" ? <CommitScreen /> : null}
              {view === "directives" ? <DirectivesScreen sessionId={sessionId} /> : null}
              {view === "review" ? <ReviewScreen /> : null}
              {view === "dream" ? <DreamScreen onOpenReview={() => setView("review")} /> : null}
              {view === "prompts" ? <PromptsScreen /> : null}
            </AppErrorBoundary>
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
