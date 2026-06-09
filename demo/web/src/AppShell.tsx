import { useEffect, useMemo, useState } from "react";

import { ApiError, getCreatorEntity, openOperatorSession, setCreatorByName } from "./api/client";
import type { EntityRecord, StateSnapshot } from "./api/types";
import { AppErrorBoundary } from "./components/AppErrorBoundary";
import { CommandPalette } from "./components/CommandPalette/CommandPalette";
import { Inspector } from "./components/Inspector/Inspector";
import { InspectorProvider } from "./components/Inspector/inspector-context";
import { InstrumentStrip } from "./components/InstrumentStrip";
import { Rail, type RailBadge, type RouteId } from "./components/Rail";
import { ResetButton } from "./components/ResetButton";
import { SessionFleet } from "./components/SessionFleet";
import { StatusBar } from "./components/StatusBar";
import { LiveEventsProvider } from "./hooks/live-context";
import { useApi } from "./hooks/use-api";
import { useLiveEvents } from "./hooks/use-live-events";
import { LiveCacheProvider, useLiveCache } from "./hooks/use-live-cache";
import { usePaletteHotkey } from "./hooks/use-palette-hotkey";
import { useSession } from "./hooks/use-session";
import { useTurnStream } from "./hooks/use-turn-stream";
import { useView } from "./hooks/use-view";
import { recordClientError } from "./lib/client-error-log";
import { AdminScreen } from "./screens/Admin";
import { CognitionScreen } from "./screens/Cognition";
import { DreamScreen } from "./screens/Dream";
import { GovernanceScreen } from "./screens/Governance";
import { IdentityScreen } from "./screens/Identity";
import { MemoryScreen } from "./screens/Memory";
import { MissionControlScreen } from "./screens/MissionControl";
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

function railBadges(counts: StateSnapshot["counts"] | null): Partial<Record<RouteId, RailBadge>> {
  return {
    identity: countBadge(counts?.open_qs, 1, "open questions"),
    governance: countBadge(counts?.commitments, 1, "commitments"),
    review: countBadge(counts?.open_reviews, 2, "open reviews"),
    dream: countBadge(counts?.dream_audit_rows, 1, "dream audit rows"),
  };
}

export function AppShell() {
  const { view, governanceTab, setView, setGovernanceTab } = useView();
  const [now, setNow] = useState(formatNow);
  const { sessionId, setSessionId } = useSession();
  const creatorApi = useApi(getCreatorEntity, []);
  const [operatorChatError, setOperatorChatError] = useState<string | null>(null);
  const live = useLiveEvents({ sessionId });
  const turnStream = useTurnStream(live, { sessionId });
  const refetchCreator = creatorApi.refetch;

  useEffect(() => {
    const timer = window.setInterval(() => setNow(formatNow()), 1_000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <LiveEventsProvider value={live}>
      <LiveCacheProvider sessionId={sessionId}>
        <AppShellContent
          view={view}
          governanceTab={governanceTab}
          setView={setView}
          setGovernanceTab={setGovernanceTab}
          now={now}
          sessionId={sessionId}
          setSessionId={setSessionId}
          creator={creatorApi.data ?? null}
          refetchCreator={refetchCreator}
          operatorChatError={operatorChatError}
          setOperatorChatError={setOperatorChatError}
          turnStream={turnStream}
        />
      </LiveCacheProvider>
    </LiveEventsProvider>
  );
}

type AppShellContentProps = {
  view: RouteId;
  governanceTab: ReturnType<typeof useView>["governanceTab"];
  setView: ReturnType<typeof useView>["setView"];
  setGovernanceTab: ReturnType<typeof useView>["setGovernanceTab"];
  now: string;
  sessionId: string;
  setSessionId: (sessionId: string) => void;
  creator: EntityRecord | null;
  refetchCreator: () => Promise<void>;
  operatorChatError: string | null;
  setOperatorChatError: (error: string | null) => void;
  turnStream: ReturnType<typeof useTurnStream>;
};

function AppShellContent({
  view,
  governanceTab,
  setView,
  setGovernanceTab,
  now,
  sessionId,
  setSessionId,
  creator,
  refetchCreator,
  operatorChatError,
  setOperatorChatError,
  turnStream,
}: AppShellContentProps) {
  const {
    stateApi,
    counts,
    sessionsApi,
    sessionActivity,
    lastMaintenanceTick,
    dreamActivity,
    wsState,
  } = useLiveCache();
  const refetchState = stateApi.refetch;
  const refetchSessions = sessionsApi.refetch;
  const activeSession =
    sessionsApi.data?.sessions.find((session) => session.session_id === sessionId) ?? null;
  const activeAudience = activeSession?.audience_label ?? DEFAULT_AUDIENCE;
  const badges = useMemo(() => railBadges(counts), [counts]);
  const [resetOpen, setResetOpen] = useState(false);
  const palette = usePaletteHotkey({ disabled: resetOpen });

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

  const refetchAll = async () => {
    await Promise.all([refetchCreator(), refetchSessions(), refetchState()]);
    return { turnCachesReset: turnStream.resetCaches() };
  };

  return (
    <InspectorProvider
      setView={setView}
      setSessionId={setSessionId}
      sessionId={sessionId}
      audience={activeAudience}
    >
      <div className="app">
        <Rail route={view} setRoute={setView} badges={badges} />
        <InstrumentStrip
          sessionId={sessionId}
          activeSession={activeSession}
          audience={activeAudience}
          creator={creator}
          state={stateApi.data}
          wsState={wsState}
          dreamActivity={dreamActivity}
          now={now}
          route={view}
        />
        <div className="main">
          <SessionFleet
            sessions={sessionsApi.data?.sessions ?? []}
            activeSessionId={sessionId}
            onSelect={setSessionId}
            creator={creator}
            operatorChatError={operatorChatError}
            onOpenOperatorChat={openOperatorChat}
            onSetCreatorByName={markCreatorByName}
            sessionActivity={sessionActivity}
          />
          <div className="screen-shell">
            <AppErrorBoundary
              resetKey={view}
              onError={(error) => {
                recordClientError({
                  source: "boundary",
                  boundarySource: `screen:${view}`,
                  message: error.message,
                });
              }}
            >
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
              {view === "mission" ? (
                <MissionControlScreen
                  sessionId={sessionId}
                  turnStream={turnStream}
                  onNavigate={setView}
                />
              ) : null}
              {view === "stream" ? <StreamScreen sessionId={sessionId} /> : null}
              {view === "memory" ? (
                <MemoryScreen
                  sessionId={sessionId}
                  onOpenReview={() => setView("review")}
                  onOpenIdentity={() => setView("identity")}
                  onOpenCommitments={() => setView("governance", { governanceTab: "commitments" })}
                />
              ) : null}
              {view === "identity" ? <IdentityScreen /> : null}
              {view === "governance" ? (
                <GovernanceScreen
                  sessionId={sessionId}
                  activeSessionId={sessionId}
                  activeTab={governanceTab}
                  sessions={sessionsApi.data?.sessions ?? []}
                  sessionsError={sessionsApi.error}
                  creator={creator}
                  operatorChatError={operatorChatError}
                  onSelectSession={setSessionId}
                  onOpenOperatorChat={openOperatorChat}
                  onSetCreatorByName={markCreatorByName}
                  onSessionPolicyChanged={refetchSessionState}
                  onTabChange={setGovernanceTab}
                />
              ) : null}
              {view === "review" ? <ReviewScreen /> : null}
              {view === "dream" ? <DreamScreen onOpenReview={() => setView("review")} /> : null}
              {view === "prompts" ? <PromptsScreen /> : null}
              {view === "admin" ? (
                <AdminScreen
                  route={view}
                  sessionId={sessionId}
                  onRefetchAll={refetchAll}
                  onOpenResetConfirm={() => setResetOpen(true)}
                />
              ) : null}
            </AppErrorBoundary>
          </div>
        </div>
        <StatusBar
          state={stateApi.data}
          lastPhase={turnStream.lastPhase}
          lastMaintenanceTick={lastMaintenanceTick}
        />
        <CommandPalette
          open={palette.open}
          onOpenChange={palette.setOpen}
          setView={setView}
          setSessionId={setSessionId}
          onOpenReset={() => setResetOpen(true)}
        />
        <ResetButton open={resetOpen} onOpenChange={setResetOpen} showTrigger={false} />
        <Inspector />
      </div>
    </InspectorProvider>
  );
}
