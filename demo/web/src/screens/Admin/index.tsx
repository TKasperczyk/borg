import { useEffect, useState } from "react";

import { apiBase, liveUrl, wsBase } from "../../api/client";
import type { StateSnapshot, WsState } from "../../api/types";
import { Empty } from "../../components/Empty";
import { Panel } from "../../components/Panel";
import { maintenanceTickLabel, maintenanceTickTone } from "../../components/StatusBar";
import { Tag, type TagKind } from "../../components/Tag";
import { useLiveCache, type DreamActivity } from "../../hooks/use-live-cache";
import { DOWN_AFTER_FAILED_ATTEMPTS } from "../../hooks/use-live-events";
import { LEGACY_VIEW_ALIASES } from "../../hooks/use-view";
import {
  clearClientErrors,
  getClientErrors,
  subscribeClientErrorLog,
  type ClientErrorLogEntry,
} from "../../lib/client-error-log";
import { RAIL_ITEMS, type RouteId } from "../../routes";

export type AdminRefetchResult = {
  turnCachesReset: boolean;
};

export type AdminScreenProps = {
  route: RouteId;
  sessionId: string;
  onRefetchAll: () => Promise<AdminRefetchResult>;
  onOpenResetConfirm: () => void;
};

function baseLabel(value: string): string {
  return value.length === 0 ? "same-origin" : value;
}

function timestampLabel(ts: number): string {
  return new Date(ts).toISOString();
}

function wsKind(wsState: WsState): TagKind {
  if (wsState === "live") {
    return "acc";
  }
  if (wsState === "down") {
    return "bad";
  }
  return "warn";
}

function maintenanceKind(tone: "ok" | "warn" | "bad"): TagKind {
  return tone === "ok" ? "acc" : tone;
}

function runtimeVersion(state: StateSnapshot | null, loading: boolean): string {
  if (state !== null) {
    return state.version;
  }
  return loading ? "loading..." : "unknown";
}

function dreamActivityLabel(activity: DreamActivity | null): string {
  if (activity === null) {
    return "none";
  }

  return `${activity.process} · ${activity.phase} · ${activity.run_id ?? "no run id"}`;
}

function RuntimePanel() {
  const { stateApi, wsState, connectionCount, lastMaintenanceTick, dreamActivity } = useLiveCache();
  const maintenanceLabel = maintenanceTickLabel(lastMaintenanceTick);

  return (
    <Panel title="runtime" badge={stateApi.data?.version ?? "snapshot"} className="admin-panel">
      <div className="panel-body pad">
        {stateApi.error === null ? null : (
          <div className="warn admin-inline-alert" role="alert">
            state snapshot unavailable: {stateApi.error.message}{" "}
            <button type="button" className="btn sm ghost" onClick={() => void stateApi.refetch()}>
              retry
            </button>
          </div>
        )}
        <div className="props admin-props">
          <div className="row">
            <span className="k">version</span>
            <span className="v tab-num">{runtimeVersion(stateApi.data, stateApi.loading)}</span>
          </div>
          <div className="row">
            <span className="k">api base</span>
            <span className="v">{baseLabel(apiBase())}</span>
          </div>
          <div className="row">
            <span className="k">ws base</span>
            <span className="v">{baseLabel(wsBase())}</span>
          </div>
          <div className="row">
            <span className="k">live url</span>
            <span className="v">{liveUrl()}</span>
          </div>
          <div className="row">
            <span className="k">live state</span>
            <span className="v admin-tagline">
              <span data-testid="ws-state">
                <Tag kind={wsKind(wsState)} dot>
                  {wsState}
                </Tag>
              </span>
              <span className="tab-num">{connectionCount} connections</span>
            </span>
          </div>
          <div className="row">
            <span className="k">down threshold</span>
            <span className="v tab-num">
              down after {DOWN_AFTER_FAILED_ATTEMPTS} failed reconnect attempts
            </span>
          </div>
          <div className="row">
            <span className="k">maintenance</span>
            <span className="v admin-tagline">
              {lastMaintenanceTick === null || maintenanceLabel === null ? (
                <span>none this session</span>
              ) : (
                <>
                  <Tag kind={maintenanceKind(maintenanceTickTone(lastMaintenanceTick))}>
                    {lastMaintenanceTick.status}
                  </Tag>
                  <span>{maintenanceLabel}</span>
                  <span className="tab-num">{timestampLabel(lastMaintenanceTick.ts)}</span>
                </>
              )}
              <Tag kind="info">session-only</Tag>
            </span>
          </div>
          <div className="row">
            <span className="k">dream process</span>
            <span className="v">{dreamActivityLabel(dreamActivity)}</span>
          </div>
        </div>
      </div>
    </Panel>
  );
}

function errorTarget(entry: ClientErrorLogEntry): string {
  if (entry.source === "api") {
    return entry.endpoint;
  }

  return entry.boundarySource;
}

function errorStatus(entry: ClientErrorLogEntry): string {
  if (entry.source === "api") {
    return entry.status === undefined ? "network" : String(entry.status);
  }

  return "render";
}

function useClientErrorEntries(): readonly ClientErrorLogEntry[] {
  const [entries, setEntries] = useState(() => getClientErrors());

  useEffect(
    () =>
      subscribeClientErrorLog(() => {
        setEntries(getClientErrors());
      }),
    [],
  );

  return entries;
}

function ClientErrorLogPanel() {
  const entries = useClientErrorEntries();

  return (
    <Panel
      title="client error log"
      badge={`${entries.length}`}
      action="clear log"
      onAction={clearClientErrors}
      className="admin-panel"
    >
      <div className="panel-body admin-log-body">
        <div className="panel-note admin-panel-note">
          in-memory only; captures transport and render failures for this browser session
        </div>
        {entries.length === 0 ? (
          <Empty>no client-captured errors this session</Empty>
        ) : (
          <div className="list admin-error-list">
            {entries.map((entry) => (
              <div
                className="list-row admin-error-row"
                data-testid="client-error-row"
                key={entry.id}
              >
                <div className="ttl admin-error-head">
                  <Tag kind={entry.source === "api" && entry.status === undefined ? "warn" : "bad"}>
                    {errorStatus(entry)}
                  </Tag>
                  <span>{errorTarget(entry)}</span>
                </div>
                <div className="meta">
                  <span className="tab-num">{timestampLabel(entry.ts)}</span>
                  <span>{entry.message}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Panel>
  );
}

function RouteDiagnosticsPanel({ route, sessionId }: { route: RouteId; sessionId: string }) {
  return (
    <Panel title="route diagnostics" badge={`${RAIL_ITEMS.length} routes`} className="admin-panel">
      <div className="panel-body">
        <table className="tbl admin-route-table">
          <thead>
            <tr>
              <th>route</th>
              <th>label</th>
              <th>num</th>
              <th>chord</th>
              <th>current</th>
            </tr>
          </thead>
          <tbody>
            {RAIL_ITEMS.map((item) => (
              <tr className={route === item.id ? "selected" : ""} key={item.id}>
                <td>
                  <span aria-hidden="true">{item.glyph}</span> {item.id}
                </td>
                <td>{item.label}</td>
                <td className="tab-num">{item.num}</td>
                <td className="tab-num">⌘{item.num}</td>
                <td>{route === item.id ? <Tag kind="acc">current</Tag> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="admin-route-meta">
          <span className="k">session</span>
          <span className="v">{sessionId}</span>
        </div>
        <table className="tbl admin-route-table admin-alias-table">
          <thead>
            <tr>
              <th>legacy alias</th>
              <th>route</th>
              <th>tab</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(LEGACY_VIEW_ALIASES).map(([alias, target]) => (
              <tr key={alias}>
                <td>{alias}</td>
                <td>{target.view}</td>
                <td>{target.governanceTab}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function DangerZonePanel({
  onRefetchAll,
  onOpenResetConfirm,
}: {
  onRefetchAll: () => Promise<AdminRefetchResult>;
  onOpenResetConfirm: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  async function refetchAll(): Promise<void> {
    if (busy) {
      return;
    }

    setBusy(true);
    setMessage(null);
    try {
      const result = await onRefetchAll();
      setMessage(
        result.turnCachesReset
          ? "client caches refetched"
          : "refetched; turn cache kept during active turn",
      );
    } catch (cause) {
      setMessage(cause instanceof Error ? cause.message : "refetch failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel title="danger zone" badge="operator" className="admin-panel admin-danger-panel">
      <div className="panel-body pad admin-danger-body">
        <div>
          <div className="admin-danger-title">reset demo</div>
          <div className="dim admin-danger-copy">
            Deletes the borg substrate through the sanctioned admin reset endpoint and requires the
            RESET confirmation token.
          </div>
          <button type="button" className="btn sm danger" onClick={onOpenResetConfirm}>
            reset
          </button>
        </div>
        <div className="admin-danger-separator" />
        <div>
          <div className="admin-danger-title">client cache</div>
          <div className="dim admin-danger-copy">
            Refetches state and session data, then clears retained turn-stream caches when no turn
            is active. This does not mutate memory.
          </div>
          <button
            type="button"
            className="btn sm ghost"
            disabled={busy}
            onClick={() => void refetchAll()}
          >
            {busy ? "refetching..." : "refetch all"}
          </button>
          {message === null ? null : <div className="admin-action-note">{message}</div>}
        </div>
      </div>
    </Panel>
  );
}

export function AdminScreen({
  route,
  sessionId,
  onRefetchAll,
  onOpenResetConfirm,
}: AdminScreenProps) {
  return (
    <div className="full-page admin-page">
      <div className="page-head">
        <h1>admin</h1>
        <span className="desc">runtime console · destructive actions quarantined</span>
        <span className="spacer" />
        <Tag kind="warn">operator surface</Tag>
      </div>
      <div className="page-body admin-body">
        <div className="admin-grid">
          <RuntimePanel />
          <DangerZonePanel onRefetchAll={onRefetchAll} onOpenResetConfirm={onOpenResetConfirm} />
          <ClientErrorLogPanel />
          <RouteDiagnosticsPanel route={route} sessionId={sessionId} />
        </div>
      </div>
    </div>
  );
}
