import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  applyDream,
  fetchDreamAudit,
  fetchDreamState,
  planDream,
  revertDreamAudit,
} from "../api/client";
import type {
  DreamPlanResponse,
  DreamProcessRow,
  DreamReport,
  MaintenanceAuditRow,
  OfflineProcessName,
} from "../api/types";
import { useQuery } from "../api/useQuery";
import { dayLabel, hm, hms, humanMs } from "../format/time";
import { useLive } from "../live/useLive";
import {
  appendDreamRequest,
  EMPTY_DREAM_RUN_FEED,
  reduceDreamRunFeed,
  type DreamRunFeedEntry,
  type DreamRunFeedState,
} from "./dream/runFeed";

function formatError(error: unknown): string {
  if (error instanceof ApiError) {
    return `${error.status} ${error.message}`;
  }
  return error instanceof Error ? error.message : String(error);
}

function processLabel(name: string): string {
  return name.replace(/-/g, " ").toUpperCase();
}

export function sameProcessSet(
  left: readonly OfflineProcessName[],
  right: readonly OfflineProcessName[],
): boolean {
  if (left.length !== right.length) {
    return false;
  }

  const rightSet = new Set(right);
  return left.every((name) => rightSet.has(name));
}

function selectedRows(
  rows: readonly DreamProcessRow[],
  selected: ReadonlySet<OfflineProcessName>,
): OfflineProcessName[] {
  return rows
    .map((row) => row.name)
    .filter((name): name is OfflineProcessName => selected.has(name));
}

function statusLabel(status: DreamProcessRow["last_status"]): string {
  if (status === null) {
    return "—";
  }
  return status.toUpperCase();
}

function auditRevertable(row: MaintenanceAuditRow): boolean {
  return row.reverted_at === null && Object.keys(row.reversal).length > 0;
}

function summarizePlan(plan: DreamPlanResponse): string {
  const changed = plan.processes.filter((process) => process.would_change).length;
  return `${plan.processes.length} processes · ${changed} would change · ${plan.total_budget_used} budget`;
}

function RunFeedRow({ entry }: { entry: DreamRunFeedEntry }) {
  if (entry.kind === "request") {
    const target =
      entry.plan_id === undefined
        ? entry.processes.map(processLabel).join(", ")
        : `plan ${entry.plan_id}`;
    return (
      <div className="dream-feed-row dream-feed-request">
        <span>○ {entry.action.toUpperCase()}</span>
        <span>{target}</span>
        <time>{hms(new Date(entry.ts))}</time>
      </div>
    );
  }

  if (entry.kind === "started") {
    return (
      <div className="dream-feed-row">
        <span>▸ {entry.process}</span>
        <span>{entry.phase}</span>
        <time>{hms(new Date(entry.ts))}</time>
      </div>
    );
  }

  if (entry.kind === "completed") {
    const errored = entry.errors > 0;
    return (
      <div className={errored ? "dream-feed-row dream-feed-error" : "dream-feed-row"}>
        <span>{errored ? "✕" : "✓"} {entry.process}</span>
        <span>accepted {entry.candidates_accepted}</span>
        <time>{humanMs(entry.duration_ms)}</time>
      </div>
    );
  }

  return (
    <div className={entry.errors > 0 ? "dream-feed-row dream-feed-error" : "dream-feed-row"}>
      <span>{entry.cadence} tick · {entry.status}</span>
      <span>{entry.changes} changes · {entry.errors} errors</span>
      <time>{humanMs(entry.duration_ms)}</time>
    </div>
  );
}

function DreamReportRow({ report }: { report: DreamReport }) {
  const errorCount = report.errors.length;
  return (
    <div className="dream-report-row">
      <div className="dream-report-top">
        <time>{report.planned_at === null ? "—" : hm(new Date(report.planned_at))}</time>
        <span>{report.processes.map(processLabel).join(", ")}</span>
        <b className={errorCount > 0 ? "tone-error" : "tone-ok"}>
          {errorCount > 0 ? `${errorCount} errors` : "clean"}
        </b>
      </div>
      <div className="dream-report-summary">
        {report.dry_run ? "dry run" : "applied"} · {report.changes} changes · {report.tokens_used} budget
      </div>
      {report.notes.length > 0 ? (
        <div className="dream-report-note">{report.notes.join(" · ")}</div>
      ) : null}
      {report.errors.map((error, index) => (
        <div className="dream-report-error" key={`${report.run_id}:error:${index}`}>
          {error.process ?? "process"} · {error.message ?? "error"}
        </div>
      ))}
    </div>
  );
}

export function DreamPage() {
  const dream = useQuery("dream:state", fetchDreamState);
  const audit = useQuery("dream:audit", () => fetchDreamAudit(50));
  const live = useLive();
  const [selected, setSelected] = useState<ReadonlySet<OfflineProcessName>>(() => new Set());
  const [plan, setPlan] = useState<DreamPlanResponse | null>(null);
  const [planSelection, setPlanSelection] = useState<OfflineProcessName[]>([]);
  const [runFeed, setRunFeed] = useState<DreamRunFeedState>(EMPTY_DREAM_RUN_FEED);
  const [planning, setPlanning] = useState(false);
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmRevert, setConfirmRevert] = useState<number | null>(null);
  const [pendingRevert, setPendingRevert] = useState<number | null>(null);

  const processes = dream.data?.processes ?? [];
  const selectedProcesses = useMemo(() => selectedRows(processes, selected), [processes, selected]);
  const allSelected = processes.length > 0 && selectedProcesses.length === processes.length;
  const planMatchesSelection = plan !== null && sameProcessSet(planSelection, selectedProcesses);
  const runLocked = planning || applying;
  const disabledForRun = runLocked || selectedProcesses.length === 0;

  useEffect(() => {
    const offStarted = live.onFrame("dream:process:started", (frame) => {
      setRunFeed((state) => reduceDreamRunFeed(state, frame));
    });
    const offCompleted = live.onFrame("dream:process:completed", (frame) => {
      setRunFeed((state) => reduceDreamRunFeed(state, frame));
    });
    const offTick = live.onFrame("maintenance:tick", (frame) => {
      setRunFeed((state) => reduceDreamRunFeed(state, frame));
    });

    return () => {
      offStarted();
      offCompleted();
      offTick();
    };
  }, [live]);

  const scheduler = dream.data?.scheduler;
  const runState = planning
    ? "planning…"
    : applying || runFeed.inFlight
      ? "IN FLIGHT — streaming dream:process:* frames"
      : runFeed.summary === null
        ? "idle"
        : `complete · ${runFeed.summary.run_id ?? "manual"}`;

  const toggleProcess = (name: OfflineProcessName) => {
    if (runLocked) {
      return;
    }

    setSelected((current) => {
      const next = new Set(current);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  const setAll = () => {
    if (runLocked) {
      return;
    }

    setSelected(allSelected ? new Set() : new Set(processes.map((process) => process.name)));
  };

  const clearSelection = () => {
    if (runLocked) {
      return;
    }

    setSelected(new Set());
  };

  const handlePlan = async () => {
    if (disabledForRun) {
      return;
    }
    setPlanning(true);
    setError(null);
    setRunFeed((state) =>
      appendDreamRequest(state, {
        action: "plan",
        processes: selectedProcesses,
      }),
    );
    try {
      const nextPlan = await planDream({ processes: selectedProcesses });
      setPlan(nextPlan);
      setPlanSelection(selectedProcesses);
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setPlanning(false);
    }
  };

  const handleApply = async () => {
    if (disabledForRun) {
      return;
    }
    setApplying(true);
    setError(null);
    const reusePlan = planMatchesSelection && plan !== null;
    const payload = reusePlan ? { plan_id: plan.plan_id } : { processes: selectedProcesses };
    setRunFeed((state) =>
      appendDreamRequest(state, {
        action: "apply",
        processes: selectedProcesses,
        ...(reusePlan ? { plan_id: plan.plan_id } : {}),
      }),
    );
    try {
      await applyDream(payload);
      dream.refetch();
      audit.refetch();
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setApplying(false);
    }
  };

  const handleRevert = async (row: MaintenanceAuditRow) => {
    if (pendingRevert !== null || !auditRevertable(row)) {
      return;
    }
    setPendingRevert(row.id);
    setError(null);
    try {
      await revertDreamAudit(row.id);
      setConfirmRevert(null);
      audit.refetch();
      dream.refetch();
    } catch (caught) {
      setError(formatError(caught));
    } finally {
      setPendingRevert(null);
    }
  };

  return (
    <main className="page dream-page">
      <div className="page-header">
        <div className="page-title">DREAM</div>
        <div className="page-subtitle">offline mind maintenance -- plan before apply</div>
        <div className="dream-header-status">
          {scheduler === undefined ? null : (
            <span>
              scheduler {scheduler.enabled ? "ENABLED" : "DISABLED"} · light{" "}
              {humanMs(scheduler.light_interval_ms)} · heavy {humanMs(scheduler.heavy_interval_ms)}
            </span>
          )}
          {(dream.data?.pending_extraction_episodes ?? 0) > 0 ? (
            <b>{dream.data?.pending_extraction_episodes} episodes pending extraction</b>
          ) : null}
        </div>
      </div>

      <div className="dream-main">
        <section className="dream-processes">
          <div className="dream-toolbar">
            <button className="ghost-button" type="button" disabled={runLocked} onClick={setAll}>
              {allSelected ? "[✓] ALL" : "[ ] ALL"}
            </button>
            <button className="ghost-button" type="button" disabled={runLocked} onClick={clearSelection}>
              [ ] NONE
            </button>
            <span>{selectedProcesses.length} selected</span>
            <button
              className="outline-button dream-toolbar-push"
              type="button"
              disabled={disabledForRun}
              onClick={handlePlan}
            >
              PLAN
            </button>
            <button
              className="solid-button"
              type="button"
              disabled={disabledForRun}
              onClick={handleApply}
            >
              ▶ APPLY
            </button>
          </div>
          <div className="dream-table-head">
            <span>PROCESS</span>
            <span>DOES</span>
            <span>LAST RUN</span>
            <span>STATUS</span>
            <span>BUDGET</span>
          </div>
          <div className="dream-process-list">
            {processes.map((process) => {
              const isSelected = selected.has(process.name);
              return (
                <button
                  className={isSelected ? "dream-process-row dream-process-row-selected" : "dream-process-row"}
                  key={process.name}
                  type="button"
                  disabled={runLocked}
                  onClick={() => toggleProcess(process.name)}
                >
                  <span className="dream-process-name">
                    <i>{isSelected ? "✓" : ""}</i>
                    <b>{processLabel(process.name)}</b>
                  </span>
                  <span>{process.description}</span>
                  <span>{process.last_run_at === null ? "—" : hm(new Date(process.last_run_at))}</span>
                  <span className={process.last_status === "error" ? "tone-error" : "tone-ok"}>
                    {statusLabel(process.last_status)}
                  </span>
                  <span>{process.budget ?? "—"}</span>
                </button>
              );
            })}
          </div>
        </section>

        <aside className="dream-run-panel">
          <div className="panel-head">
            <b>RUN</b>
            <span className={applying || runFeed.inFlight ? "pulse" : ""}>{runState}</span>
          </div>
          {plan === null ? null : (
            <div className="dream-plan-card">
              <div>
                <b>PLAN {plan.plan_id}</b>
                <span>{planMatchesSelection ? "dry-run preview" : "selection changed"}</span>
              </div>
              <p>{summarizePlan(plan)}</p>
              <ul>
                {plan.processes.map((process) => (
                  <li key={process.name}>
                    <span>{processLabel(process.name)}</span>
                    <span>{process.summary}</span>
                  </li>
                ))}
              </ul>
              <small>
                {planMatchesSelection
                  ? "apply will reuse this plan via plan_id"
                  : "apply will use the selected processes"}
              </small>
            </div>
          )}
          <div className="dream-feed">
            {runFeed.entries.length === 0 ? (
              <div className="quiet-line">no run frames yet</div>
            ) : (
              runFeed.entries.map((entry) => <RunFeedRow entry={entry} key={entry.id} />)
            )}
          </div>
          {runFeed.summary === null ? null : (
            <div className="dream-summary-strip">
              <span>{runFeed.summary.changes} changes</span>
              <span>{runFeed.summary.errors} errors</span>
              <span>{humanMs(runFeed.summary.duration_ms)}</span>
              <span>{runFeed.summary.run_id ?? "manual"}</span>
            </div>
          )}
          {error === null ? null : <div className="inline-error">{error}</div>}
        </aside>
      </div>

      <div className="dream-bottom">
        <section className="dream-audit">
          <div className="panel-head">
            <b>AUDIT</b>
            <span>{audit.data?.rows.length ?? 0} rows</span>
          </div>
          <div className="dream-audit-list">
            {(audit.data?.rows ?? []).map((row) => (
              <div className={row.reverted_at === null ? "dream-audit-row" : "dream-audit-row dream-row-dim"} key={row.id}>
                <span>#{row.id}</span>
                <b>{row.process}</b>
                <span>{row.action}</span>
                <time>{dayLabel(new Date(row.applied_at))} {hm(new Date(row.applied_at))}</time>
                {row.reverted_at !== null ? (
                  <span className="tone-ok">REVERTED ✓</span>
                ) : auditRevertable(row) ? (
                  confirmRevert === row.id ? (
                    <span className="inline-actions">
                      <button
                        className="outline-button danger"
                        type="button"
                        disabled={pendingRevert === row.id}
                        onClick={() => handleRevert(row)}
                      >
                        CONFIRM
                      </button>
                      <button className="ghost-button" type="button" onClick={() => setConfirmRevert(null)}>
                        CANCEL
                      </button>
                    </span>
                  ) : (
                    <button className="outline-button" type="button" onClick={() => setConfirmRevert(row.id)}>
                      REVERT
                    </button>
                  )
                ) : (
                  <span />
                )}
              </div>
            ))}
          </div>
        </section>

        <section className="dream-reports">
          <div className="panel-head">
            <b>DREAM REPORTS</b>
            <span>{dream.data?.dream_reports.length ?? 0}</span>
          </div>
          <div className="dream-report-list">
            {(dream.data?.dream_reports ?? []).map((report) => (
              <DreamReportRow key={report.run_id} report={report} />
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
