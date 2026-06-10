import { useCallback, useEffect, useRef, useState } from "react";

import { getCommitments, getCreatorDirectives, getReviews } from "../../api/client";
import type { EntityRecord, SessionRecord } from "../../api/types";
import { useApi, type ApiHookState } from "../../hooks/use-api";
import type { GovernanceTabId } from "../../routes";
import { CommitmentsPanel } from "./CommitmentsTab";
import { DirectivesPanel, loadDirectiveSupportData } from "./DirectivesTab";
import { ScopeMatrixTab } from "./ScopeMatrixTab";
import { SessionsEntitiesTab } from "./SessionsEntitiesTab";

type GovernanceScreenProps = {
  sessionId: string;
  activeSessionId: string;
  activeTab: GovernanceTabId;
  sessions: readonly SessionRecord[];
  sessionsError?: Error | null;
  creator: EntityRecord | null;
  operatorChatError: string | null;
  onSelectSession: (sessionId: string) => void;
  onOpenOperatorChat: () => Promise<void> | void;
  onSetCreatorByName: (name: string) => Promise<void> | void;
  onSessionPolicyChanged: () => Promise<void>;
  onTabChange: (tab: GovernanceTabId) => void;
};

const GOVERNANCE_TABS: ReadonlyArray<{ id: GovernanceTabId; label: string }> = [
  { id: "commitments", label: "Commitments" },
  { id: "shared_state", label: "Directives & shared state" },
  { id: "scope", label: "Scope matrix" },
  { id: "sessions", label: "Sessions & entities" },
];

function useLazyApi<T>(
  loader: () => Promise<T>,
  deps: readonly unknown[],
  enabled: boolean,
): ApiHookState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const startedRef = useRef(false);
  const requestSeqRef = useRef(0);
  const mountedRef = useRef(true);

  useEffect(
    () => () => {
      mountedRef.current = false;
    },
    [],
  );

  useEffect(() => {
    startedRef.current = false;
    requestSeqRef.current += 1;
    setData(null);
    setLoading(false);
    setError(null);
  }, deps);

  const refetch = useCallback(async () => {
    const requestSeq = requestSeqRef.current + 1;
    requestSeqRef.current = requestSeq;
    startedRef.current = true;
    setLoading(true);
    setError(null);
    try {
      const result = await loader();
      if (mountedRef.current && requestSeqRef.current === requestSeq) {
        setData(result);
      }
    } catch (caught) {
      if (mountedRef.current && requestSeqRef.current === requestSeq) {
        setError(caught instanceof Error ? caught : new Error(String(caught)));
      }
    } finally {
      if (mountedRef.current && requestSeqRef.current === requestSeq) {
        setLoading(false);
      }
    }
  }, deps);

  useEffect(() => {
    if (!enabled || startedRef.current) {
      return;
    }

    void refetch();
  }, [enabled, refetch]);

  return {
    data,
    loading: enabled && !startedRef.current && data === null && error === null ? true : loading,
    error,
    refetch,
    retry: refetch,
    isStale: data !== null && (loading || error !== null),
    degraded: error !== null,
    retrying: false,
  };
}

export function GovernanceScreen({
  sessionId,
  activeSessionId,
  activeTab,
  sessions,
  sessionsError = null,
  creator,
  operatorChatError,
  onSelectSession,
  onOpenOperatorChat,
  onSetCreatorByName,
  onSessionPolicyChanged,
  onTabChange,
}: GovernanceScreenProps) {
  const commitmentsApi = useApi(() => getCommitments({ state: "all" }), []);
  const needsDirectives =
    activeTab === "shared_state" || activeTab === "scope" || activeTab === "sessions";
  const needsDirectiveSupport = activeTab === "shared_state" || activeTab === "scope";
  const needsReviews = activeTab === "scope";
  const commitments = commitmentsApi.data?.commitments ?? [];
  const directivesApi = useLazyApi(
    () => getCreatorDirectives({ status: "all" }),
    [],
    needsDirectives,
  );
  const directiveSupportApi = useLazyApi(
    () => loadDirectiveSupportData(sessionId, commitments),
    [sessionId, commitmentsApi.data],
    needsDirectiveSupport && commitmentsApi.data !== null,
  );
  const reviewsApi = useLazyApi(() => getReviews({ openOnly: true }), [], needsReviews);
  const directives = directivesApi.data?.directives ?? [];
  const directiveSupportViewApi: typeof directiveSupportApi =
    needsDirectiveSupport && commitmentsApi.data === null && commitmentsApi.error === null
      ? { ...directiveSupportApi, loading: true }
      : directiveSupportApi;

  return (
    <div className="full-page governance-studio">
      <div className="page-head governance-head">
        <h1>governance</h1>
        <span className="desc">commitments · directives · shared state · scope policy labels</span>
        <span className="spacer"></span>
        <div className="governance-tabs" role="tablist" aria-label="governance sections">
          {GOVERNANCE_TABS.map((tab) => (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              className={`governance-tab ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => onTabChange(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      {sessionsError === null ? null : (
        <div className="notice bad" style={{ padding: 12 }}>
          sessions unavailable: {sessionsError.message}
        </div>
      )}
      <div className="governance-tab-shell" role="tabpanel">
        {activeTab === "commitments" ? <CommitmentsPanel api={commitmentsApi} embedded /> : null}
        {activeTab === "shared_state" ? (
          <DirectivesPanel api={directivesApi} supportApi={directiveSupportViewApi} embedded />
        ) : null}
        {activeTab === "scope" ? (
          <ScopeMatrixTab
            sessions={sessions}
            commitments={commitments}
            directives={directives}
            supportData={directiveSupportViewApi.data}
            reviews={reviewsApi.error === null ? (reviewsApi.data?.rows ?? null) : undefined}
            reviewsLoading={reviewsApi.loading}
            reviewsError={reviewsApi.error}
          />
        ) : null}
        {activeTab === "sessions" ? (
          <SessionsEntitiesTab
            sessions={sessions}
            directives={directives}
            activeSessionId={activeSessionId}
            creator={creator}
            operatorChatError={operatorChatError}
            onSelectSession={onSelectSession}
            onOpenOperatorChat={onOpenOperatorChat}
            onSetCreatorByName={onSetCreatorByName}
            onSessionPolicyChanged={onSessionPolicyChanged}
          />
        ) : null}
      </div>
    </div>
  );
}
