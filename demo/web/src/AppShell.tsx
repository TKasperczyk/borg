import { useCallback, useMemo, useState } from "react";

import { ApiError, getCreatorEntity, openOperatorSession, setCreatorByName } from "./api/client";
import type { EntityRecord, SessionRecord, StateSnapshot } from "./api/types";
import { AppErrorBoundary } from "./components/AppErrorBoundary";
import { CommandPalette } from "./components/CommandPalette/CommandPalette";
import { Inspector } from "./components/Inspector/Inspector";
import { InspectorProvider } from "./components/Inspector/inspector-context";
import { InstrumentStrip } from "./components/InstrumentStrip";
import { Modal } from "./components/Modal";
import { Rail, type RailBadge, type RouteId } from "./components/Rail";
import { ResetButton } from "./components/ResetButton";
import { SessionFleet } from "./components/SessionFleet";
import { ShortcutLegend } from "./components/ShortcutLegend";
import { StatusBar } from "./components/StatusBar";
import { LiveEventsProvider } from "./hooks/live-context";
import { useApi } from "./hooks/use-api";
import { useLiveEvents } from "./hooks/use-live-events";
import { LiveCacheProvider, useLiveCache } from "./hooks/use-live-cache";
import { usePaletteHotkey } from "./hooks/use-palette-hotkey";
import { useSession } from "./hooks/use-session";
import { useTurnStream } from "./hooks/use-turn-stream";
import { useView, type ViewState } from "./hooks/use-view";
import type { AudienceDisplayIdentity } from "./lib/audience-identity";
import { recordClientError } from "./lib/client-error-log";
import type { RouteNavigationOptions } from "./routes";
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

function countBadge(
  count: number | undefined,
  severity: RailBadge["severity"],
  label: string,
): RailBadge | undefined {
  if (count === undefined || count <= 0) {
    return undefined;
  }

  return { count, severity, label };
}

function railBadges(counts: StateSnapshot["counts"] | null): Partial<Record<RouteId, RailBadge>> {
  return {
    identity: countBadge(counts?.open_qs, 1, "open questions"),
    review: countBadge(counts?.open_reviews, 2, "open reviews"),
  };
}

type PendingRoute = {
  route: RouteId;
  options?: RouteNavigationOptions;
};

function routeOptionsForViewState(state: ViewState): RouteNavigationOptions | undefined {
  const options: RouteNavigationOptions = {};

  if (state.view === "governance") {
    options.governanceTab = state.governanceTab;
  }
  if (state.view === "memory" && state.memoryBand !== null) {
    options.memoryBand = state.memoryBand;
  }
  if (state.view === "dream" && state.dreamProcess !== null) {
    options.dreamProcess = state.dreamProcess;
  }

  return Object.keys(options).length === 0 ? undefined : options;
}

type AudienceTransportIdentity = {
  audienceValue: string | null;
  display: AudienceDisplayIdentity;
};

function audienceIdentity(
  sessionId: string,
  activeSession: SessionRecord | null,
): AudienceTransportIdentity {
  if (activeSession === null) {
    return {
      audienceValue: null,
      display: { label: null, entityId: null, fallbackId: sessionId },
    };
  }

  const entityId = activeSession.audience_entity_id;
  const hasTrustedDisplayLabel = entityId !== null || activeSession.source_type === "demo";

  return {
    audienceValue: activeSession.audience_label,
    display: {
      label: hasTrustedDisplayLabel ? activeSession.audience_label : null,
      entityId,
      fallbackId: activeSession.source_external_id ?? sessionId,
    },
  };
}

export function AppShell() {
  const [promptsDirty, setPromptsDirty] = useState(false);
  const [pendingRoute, setPendingRoute] = useState<PendingRoute | null>(null);
  const shouldBlockPromptPopState = useCallback(
    (current: ViewState, next: ViewState) =>
      current.view === "prompts" && promptsDirty && next.view !== "prompts",
    [promptsDirty],
  );
  const onBlockedPromptPopState = useCallback((next: ViewState) => {
    setPendingRoute({ route: next.view, options: routeOptionsForViewState(next) });
  }, []);
  const { view, governanceTab, memoryBand, dreamProcess, setView, setGovernanceTab } = useView({
    shouldBlockPopState: shouldBlockPromptPopState,
    onBlockedPopState: onBlockedPromptPopState,
  });
  const { sessionId, setSessionId } = useSession();
  const creatorApi = useApi(getCreatorEntity, []);
  const [operatorChatError, setOperatorChatError] = useState<string | null>(null);
  const live = useLiveEvents({ sessionId });
  const turnStream = useTurnStream(live, { sessionId });
  const refetchCreator = creatorApi.refetch;
  const requestView = useCallback(
    (route: RouteId, options?: RouteNavigationOptions) => {
      if (view === "prompts" && promptsDirty && route !== "prompts") {
        setPendingRoute({ route, options });
        return;
      }
      setView(route, options);
    },
    [promptsDirty, setView, view],
  );
  const cancelRouteChange = useCallback(() => setPendingRoute(null), []);
  const confirmRouteChange = useCallback(() => {
    if (pendingRoute === null) {
      return;
    }
    const next = pendingRoute;
    setPendingRoute(null);
    setPromptsDirty(false);
    setView(next.route, next.options);
  }, [pendingRoute, setView]);

  return (
    <LiveEventsProvider value={live}>
      <LiveCacheProvider sessionId={sessionId}>
        <AppShellContent
          view={view}
          governanceTab={governanceTab}
          memoryBand={memoryBand}
          dreamProcess={dreamProcess}
          setView={requestView}
          setGovernanceTab={setGovernanceTab}
          sessionId={sessionId}
          setSessionId={setSessionId}
          creator={creatorApi.data ?? null}
          refetchCreator={refetchCreator}
          operatorChatError={operatorChatError}
          setOperatorChatError={setOperatorChatError}
          turnStream={turnStream}
          pendingRoute={pendingRoute}
          onCancelRouteChange={cancelRouteChange}
          onConfirmRouteChange={confirmRouteChange}
          onPromptsDirtyChange={setPromptsDirty}
        />
      </LiveCacheProvider>
    </LiveEventsProvider>
  );
}

type AppShellContentProps = {
  view: RouteId;
  governanceTab: ReturnType<typeof useView>["governanceTab"];
  memoryBand: ReturnType<typeof useView>["memoryBand"];
  dreamProcess: ReturnType<typeof useView>["dreamProcess"];
  setView: (view: RouteId, options?: RouteNavigationOptions) => void;
  setGovernanceTab: ReturnType<typeof useView>["setGovernanceTab"];
  sessionId: string;
  setSessionId: (sessionId: string) => void;
  creator: EntityRecord | null;
  refetchCreator: () => Promise<void>;
  operatorChatError: string | null;
  setOperatorChatError: (error: string | null) => void;
  turnStream: ReturnType<typeof useTurnStream>;
  pendingRoute: PendingRoute | null;
  onCancelRouteChange: () => void;
  onConfirmRouteChange: () => void;
  onPromptsDirtyChange: (dirty: boolean) => void;
};

function AppShellContent({
  view,
  governanceTab,
  memoryBand,
  dreamProcess,
  setView,
  setGovernanceTab,
  sessionId,
  setSessionId,
  creator,
  refetchCreator,
  operatorChatError,
  setOperatorChatError,
  turnStream,
  pendingRoute,
  onCancelRouteChange,
  onConfirmRouteChange,
  onPromptsDirtyChange,
}: AppShellContentProps) {
  const { stateApi, counts, sessionsApi, sessionActivity, dreamActivity, wsState } = useLiveCache();
  const refetchState = stateApi.refetch;
  const refetchSessions = sessionsApi.refetch;
  const activeSession =
    sessionsApi.data?.sessions.find((session) => session.session_id === sessionId) ?? null;
  const activeAudience = audienceIdentity(sessionId, activeSession);
  const badges = useMemo(() => railBadges(counts), [counts]);
  const [resetOpen, setResetOpen] = useState(false);
  const [shortcutLegendOpen, setShortcutLegendOpen] = useState(false);
  const palette = usePaletteHotkey({
    disabled: resetOpen,
    onRouteChord: setView,
    onHelpChord: () => setShortcutLegendOpen(true),
  });

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
      audience={activeAudience.audienceValue}
    >
      <div className="app">
        <Rail route={view} setRoute={setView} badges={badges} />
        <InstrumentStrip
          sessionId={sessionId}
          activeSession={activeSession}
          audienceDisplay={activeAudience.display}
          creator={creator}
          state={stateApi.data}
          dreamActivity={dreamActivity}
          route={view}
          onOpenPalette={() => palette.setOpen(true)}
          onOpenHelp={() => setShortcutLegendOpen(true)}
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
                  audienceValue={activeAudience.audienceValue}
                  audienceDisplay={activeAudience.display}
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
              {view === "stream" ? (
                <StreamScreen sessionId={sessionId} activeSession={activeSession} />
              ) : null}
              {view === "memory" ? (
                <MemoryScreen
                  sessionId={sessionId}
                  initialBand={memoryBand}
                  onOpenWorkbench={() => setView("cognition")}
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
              {view === "dream" ? (
                <DreamScreen initialProcess={dreamProcess} onOpenReview={() => setView("review")} />
              ) : null}
              {view === "prompts" ? <PromptsScreen onDirtyChange={onPromptsDirtyChange} /> : null}
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
        <StatusBar state={stateApi.data} wsState={wsState} />
        <CommandPalette
          open={palette.open}
          onOpenChange={palette.setOpen}
          setView={setView}
          setSessionId={setSessionId}
          onOpenReset={() => setResetOpen(true)}
        />
        <ShortcutLegend open={shortcutLegendOpen} onClose={() => setShortcutLegendOpen(false)} />
        <ResetButton open={resetOpen} onOpenChange={setResetOpen} showTrigger={false} />
        <Modal
          open={pendingRoute !== null}
          title="discard prompt drafts?"
          onClose={onCancelRouteChange}
          footer={
            <>
              <button type="button" className="btn sm ghost" onClick={onCancelRouteChange}>
                stay
              </button>
              <button type="button" className="btn sm danger" onClick={onConfirmRouteChange}>
                discard drafts
              </button>
            </>
          }
        >
          <p>Leaving Prompt Lab will discard unsaved prompt drafts.</p>
        </Modal>
        <Inspector />
      </div>
    </InspectorProvider>
  );
}
