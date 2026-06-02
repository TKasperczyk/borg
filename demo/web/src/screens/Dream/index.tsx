import { Fragment, useEffect, useMemo, useRef, useState } from "react";

import {
  getDreamAudit,
  getDreamState,
  postDreamApply,
  postDreamPlan,
  revertDreamAudit,
} from "../../api/client";
import type {
  DreamApplyResponse,
  DreamReport,
  MaintenanceTickFrame,
  MaintenanceAuditRow,
  DreamPlanResponse,
  DreamProcessName,
  DreamProcessSummary,
} from "../../api/types";
import { Modal } from "../../components/Modal";
import { Tag } from "../../components/Tag";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import { formatTime } from "../../lib/stream-utils";
import {
  displayTargetSummary,
  displayValue,
  fieldLabel,
  isRecord,
  jsonText,
  shortId,
} from "../screen-utils";

const PROCESS_NAMES: DreamProcessName[] = [
  "consolidator",
  "reflector",
  "semantic-extractor",
  "curator",
  "overseer",
  "review-resolver",
  "ruminator",
  "self-narrator",
  "procedural-synthesizer",
  "belief-reviser",
  "creator-directive-reconciler",
  "commitment-reconciler",
];

function statusTag(status: DreamProcessSummary["last_status"]) {
  if (status === "ok") {
    return "acc";
  }
  if (status === "error") {
    return "bad";
  }
  return "";
}

function maintenanceTickTone(frame: MaintenanceTickFrame): "acc" | "warn" | "bad" {
  if (frame.status === "error" || frame.errors > 0) {
    return "bad";
  }

  return frame.changed ? "acc" : "warn";
}

function maintenanceTickSummary(frame: MaintenanceTickFrame): string {
  const processLabel =
    frame.processes.length === 1 ? "1 process" : `${frame.processes.length} processes`;
  const changeLabel = frame.changes === 1 ? "1 change" : `${frame.changes} changes`;
  const pendingLabel =
    frame.pending_extraction_episodes === undefined
      ? null
      : `${frame.pending_extraction_episodes} pending`;
  return [`last ${frame.cadence}`, processLabel, changeLabel, pendingLabel]
    .filter((part): part is string => part !== null)
    .join(" / ");
}

function auditHasReversal(row: MaintenanceAuditRow): boolean {
  return Object.keys(row.reversal).length > 0;
}

function auditCanRevert(row: MaintenanceAuditRow): boolean {
  return auditHasReversal(row) && row.reverted_at === null;
}

function auditRevertLabel(row: MaintenanceAuditRow): string {
  if (!auditHasReversal(row)) {
    return "not undoable";
  }

  return row.reverted_at === null ? "revert" : "reverted";
}

function auditRevertTitle(row: MaintenanceAuditRow): string {
  if (!auditHasReversal(row)) {
    return "No reversal payload was recorded for this audit row";
  }

  return row.reverted_at === null
    ? "Revert this audited maintenance change"
    : "Already reverted";
}

function auditStatusLabel(row: MaintenanceAuditRow): string {
  if (row.reverted_at !== null) {
    return "reverted";
  }

  if (
    auditHasReversal(row) &&
    (row.process === "creator-directive-reconciler" ||
      row.process === "commitment-reconciler")
  ) {
    return "auto-resolved";
  }

  return "ok";
}

function auditStatusTone(row: MaintenanceAuditRow): "" | "acc" | "warn" | "bad" {
  if (row.reverted_at !== null) {
    return "warn";
  }

  return auditHasReversal(row) ? "acc" : "";
}

type AuditRunGroup = {
  runId: string;
  rows: MaintenanceAuditRow[];
  latestAppliedAt: number;
  processes: string[];
  revertedCount: number;
};

type OldNewPair = {
  field: string;
  oldKey: string;
  newKey: string;
  before: unknown;
  after: unknown;
};

function auditRunGroups(rows: readonly MaintenanceAuditRow[]): AuditRunGroup[] {
  return [
    ...rows.reduce((groups, row) => {
      const group = groups.get(row.run_id) ?? [];
      group.push(row);
      groups.set(row.run_id, group);
      return groups;
    }, new Map<string, MaintenanceAuditRow[]>()),
  ]
    .map(([runId, runRows]) => ({
      runId,
      rows: runRows,
      latestAppliedAt: Math.max(...runRows.map((row) => row.applied_at)),
      processes: [...new Set(runRows.map((row) => row.process))],
      revertedCount: runRows.filter((row) => row.reverted_at !== null).length,
    }))
    .sort((left, right) => right.latestAppliedAt - left.latestAppliedAt);
}

function oldNewPairs(record: Record<string, unknown>): OldNewPair[] {
  return Object.entries(record).flatMap(([key, before]) => {
    if (!key.startsWith("old_")) {
      return [];
    }

    const field = key.slice("old_".length);
    const newKey = `new_${field}`;
    if (!Object.prototype.hasOwnProperty.call(record, newKey)) {
      return [];
    }

    return [
      {
        field,
        oldKey: key,
        newKey,
        before,
        after: record[newKey],
      },
    ];
  });
}

function pairedKeys(pairs: readonly OldNewPair[]): Set<string> {
  return new Set(pairs.flatMap((pair) => [pair.oldKey, pair.newKey]));
}

function hasPayload(value: Record<string, unknown>): boolean {
  return Object.keys(value).length > 0;
}

export function DreamScreen({ onOpenReview }: { onOpenReview?: () => void }) {
  const live = useLiveEventsContext();
  const api = useApi(getDreamState, []);
  const refetch = api.refetch;
  const previousConnectionCountRef = useRef(live.connectionCount);
  const [selected, setSelected] = useState<DreamProcessName>("belief-reviser");
  const [plan, setPlan] = useState<DreamPlanResponse | null>(null);
  const [planOpen, setPlanOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [applyResult, setApplyResult] = useState<DreamApplyResponse | null>(null);
  const [busy, setBusy] = useState<"plan" | "apply" | "revert" | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
  const [lastMaintenanceTick, setLastMaintenanceTick] = useState<MaintenanceTickFrame | null>(null);
  const [expandedAuditRows, setExpandedAuditRows] = useState<Record<number, boolean>>({});
  const [revertCandidate, setRevertCandidate] = useState<MaintenanceAuditRow | null>(null);
  // Per-process live status tracked across a dream run. A run does TWO sweeps
  // through every process: plan (dry-run, possibly LLM work) then apply
  // (commit). We surface both so the user sees activity through the full
  // ~30-120s cycle instead of only the apply phase.
  // queued (no key) → planning → planned → running → done | fail
  type DreamRunStatus = "planning" | "planned" | "running" | "done" | "fail";
  const [runStatus, setRunStatus] = useState<Map<string, DreamRunStatus>>(() => new Map());

  useEffect(() => {
    return live.subscribe((frame) => {
      if (
        frame.type === "stream:append" &&
        frame.entries.some((entry) => entry.kind === "dream_report")
      ) {
        void refetch();
        return;
      }

      if (frame.type === "dream:process:started") {
        setRunStatus((current) => {
          const next = new Map(current);
          next.set(frame.process, frame.phase === "plan" ? "planning" : "running");
          return next;
        });
        return;
      }

      if (frame.type === "dream:process:completed") {
        setRunStatus((current) => {
          const next = new Map(current);
          if (frame.phase === "plan") {
            // Plan completed — apply hasn't started yet for this process. Mark
            // "planned" so the tile shows it's ready, then "running" will
            // overwrite when apply sweep reaches it.
            next.set(frame.process, frame.errors > 0 ? "fail" : "planned");
          } else {
            next.set(frame.process, frame.errors > 0 ? "fail" : "done");
          }
          return next;
        });
        return;
      }

      if (frame.type === "maintenance:tick") {
        setLastMaintenanceTick(frame);
        setRunStatus((current) => {
          const next = new Map(current);
          const status = frame.status === "error" || frame.errors > 0 ? "fail" : "done";

          for (const process of frame.processes) {
            next.set(process, status);
          }

          return next;
        });
        void refetch();
        return;
      }
    });
  }, [live, refetch]);

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    void refetch();
  }, [live.connectionCount, refetch]);

  const state = api.data;
  const processes = PROCESS_NAMES.map(
    (name) =>
      state?.processes.find((process) => process.name === name) ?? {
        name,
        description: name,
        last_run_at: null,
        last_status: null,
        last_audit_id: null,
        budget: null,
        enabled: false,
      },
  );
  const selectedProcess = processes.find((process) => process.name === selected) ?? processes[0];
  const reportsByRunId = useMemo(
    () => new Map((state?.dream_reports ?? []).map((report) => [report.run_id, report])),
    [state?.dream_reports],
  );
  const groupedAuditRows = useMemo(
    () => auditRunGroups(state?.audit_rows ?? []),
    [state?.audit_rows],
  );

  async function loadPlan(openConfirm: boolean): Promise<void> {
    setBusy("plan");
    setOperatorError(null);
    setApplyResult(null);
    try {
      const nextPlan = await postDreamPlan({});
      setPlan(nextPlan);
      if (openConfirm) {
        setConfirmOpen(true);
      } else {
        setPlanOpen(true);
      }
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  function openApplyConfirm(): void {
    // Apply opens the confirm modal immediately — no upstream plan call.
    // A dream cycle runs all 12 offline processes (~30-120s); making the
    // user wait that long for the confirm dialog to appear is bad UX.
    // Users who want a preview can hit the `plan` button first; the
    // cached plan_id (if any) is passed through, otherwise the server
    // runs a fresh dry-run inside the apply path.
    setOperatorError(null);
    setApplyResult(null);
    setConfirmOpen(true);
  }

  async function applyDreamPlan(): Promise<void> {
    setBusy("apply");
    setOperatorError(null);
    // Close the modal immediately so the user can see per-process progress
    // light up on the main Dream screen via the live WS dream:process:*
    // frames. The apply call continues in the background.
    setConfirmOpen(false);
    setPlanOpen(false);
    // Reset previous run's tile state so progress on this run starts clean.
    setRunStatus(new Map());
    try {
      const result = await postDreamApply(plan === null ? {} : { plan_id: plan.plan_id });
      setApplyResult(result);
      await Promise.all([refetch(), getDreamAudit()]);
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  function toggleAuditRow(rowId: number): void {
    setExpandedAuditRows((current) => ({ ...current, [rowId]: current[rowId] !== true }));
  }

  function openRevertConfirm(row: MaintenanceAuditRow): void {
    if (!auditCanRevert(row)) {
      return;
    }

    setOperatorError(null);
    setRevertCandidate(row);
  }

  async function revertAuditRow(row: MaintenanceAuditRow): Promise<void> {
    if (!auditCanRevert(row)) {
      return;
    }

    setBusy("revert");
    setOperatorError(null);
    try {
      await revertDreamAudit(row.id);
      await Promise.all([refetch(), getDreamAudit()]);
      setRevertCandidate(null);
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(null);
    }
  }

  if (api.loading && state === null) {
    return <div className="notice">loading dream cycle</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <h1>dream cycle</h1>
        <span className="desc">synthesized from audit log · dream reports · review queue</span>
        <span className="spacer"></span>
        <span
          style={{
            fontSize: 10.5,
            color: "var(--text-mute)",
            display: "flex",
            alignItems: "center",
            gap: 6,
            whiteSpace: "nowrap",
          }}
        >
          <span className={state?.scheduler.enabled === true ? "live-dot" : "dot mute"}></span>
          <span className="acc upper">
            {state?.scheduler.enabled === true ? "scheduler enabled" : "scheduler disabled"}
          </span>
        </span>
        {lastMaintenanceTick === null ? null : (
          <span
            className={`dream-live-note ${maintenanceTickTone(lastMaintenanceTick)}`}
            aria-live="polite"
          >
            <span className="dot" aria-hidden="true"></span>
            {maintenanceTickSummary(lastMaintenanceTick)}
          </span>
        )}
        <button
          className="btn sm"
          disabled={busy !== null}
          aria-label="plan dream"
          onClick={() => void loadPlan(false)}
        >
          {busy === "plan" ? "planning" : "plan"}
        </button>
        <button
          className="btn sm primary"
          disabled={busy !== null}
          aria-label="apply dream"
          onClick={() => openApplyConfirm()}
        >
          {busy === "apply"
            ? (() => {
                const states = Array.from(runStatus.values());
                const planned = states.filter(
                  (s) => s === "planned" || s === "running" || s === "done" || s === "fail",
                ).length;
                const applied = states.filter((s) => s === "done" || s === "fail").length;
                const total = processes.length;
                // Two sweeps: plan then apply. Show whichever sweep is in flight.
                if (planned < total) {
                  return `planning ${planned}/${total}`;
                }
                return `applying ${applied}/${total}`;
              })()
            : "apply"}
        </button>
      </div>

      <div className="page-body">
        {operatorError === null ? null : <div className="notice bad">{operatorError}</div>}
        <div style={{ padding: "14px 20px 16px 20px", borderBottom: "1px solid var(--line)" }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: 10,
            }}
          >
            <div className="upper dim">schedule · recent synthesized rows</div>
            <div className="dim" style={{ fontSize: 10.5 }}>
              {state?.schedule.length ?? 0} rows · {state?.audit_rows.length ?? 0} audit ·{" "}
              {state?.belief_revision_rows.length ?? 0} belief-revision reviews
            </div>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "150px 1fr",
              gap: 14,
              alignItems: "center",
              rowGap: 5,
            }}
          >
            {processes.slice(0, 6).map((process) => {
              const runs = (state?.schedule ?? [])
                .filter((item) => item.process === process.name)
                .slice(0, 6);
              return (
                <Fragment key={process.name}>
                  <div style={{ fontSize: 11, color: "var(--text-dim)" }}>{process.name}</div>
                  <div
                    style={{
                      position: "relative",
                      height: 12,
                      background: "var(--bg-1)",
                      border: "1px solid var(--line-soft)",
                    }}
                  >
                    {[1, 2, 3, 4, 5].map((index) => (
                      <div
                        key={index}
                        style={{
                          position: "absolute",
                          top: 0,
                          bottom: 0,
                          left: `${(index / 6) * 100}%`,
                          width: 1,
                          background: "var(--line-soft)",
                        }}
                      ></div>
                    ))}
                    {runs.map((run, index) => (
                      <div
                        key={`${run.process}-${run.scheduled_at}-${index}`}
                        title={`${run.source} · ${formatTime(run.scheduled_at)}`}
                        style={{
                          position: "absolute",
                          left: `${Math.max(0, 95 - index * 15)}%`,
                          width: "4%",
                          top: 1,
                          bottom: 1,
                          background: run.source === "audit" ? "var(--acc)" : "var(--purple)",
                          opacity: 0.75,
                        }}
                      ></div>
                    ))}
                  </div>
                </Fragment>
              );
            })}
          </div>
          {(state?.schedule.length ?? 0) === 0 ? (
            <div className="dim" style={{ fontSize: 10.5, marginTop: 8 }}>
              no scheduled runs synthesized yet
            </div>
          ) : null}
        </div>

        <div className="dream-grid">
          {processes.map((process) => (
            <DreamCard
              key={process.name}
              process={process}
              selected={process.name === selectedProcess?.name}
              onSelect={() => setSelected(process.name)}
              runStatus={runStatus.get(process.name)}
            />
          ))}
        </div>

        <div style={{ padding: "0 20px 24px 20px" }}>
          <div className="divider" style={{ marginTop: 12 }}>
            selected process
          </div>
          {selectedProcess === undefined ? null : (
            <div className="panel" style={{ marginBottom: 14 }}>
              <div className="panel-header">
                <span className="title">{selectedProcess.name}</span>
                <span className="badge">{selectedProcess.last_status ?? "never"}</span>
              </div>
              <div className="panel-body pad">
                <div className="props">
                  <div className="row">
                    <span className="k">description</span>
                    <span className="v">{selectedProcess.description}</span>
                  </div>
                  <div className="row">
                    <span className="k">last run</span>
                    <span className="v">{formatTime(selectedProcess.last_run_at)}</span>
                  </div>
                  <div className="row">
                    <span className="k">budget cap</span>
                    <span className="v">
                      {selectedProcess.budget === null
                        ? "uncapped / process-local"
                        : selectedProcess.budget}
                    </span>
                  </div>
                  <div className="row">
                    <span className="k">last audit</span>
                    <span className="v">{selectedProcess.last_audit_id ?? "—"}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="divider">review rows</div>
          <div className="panel" style={{ marginBottom: 18 }}>
            <div className="panel-body pad">
              <div className="props">
                <div className="row">
                  <span className="k">belief revisions</span>
                  <span className="v">{state?.belief_revision_rows.length ?? 0} open</span>
                </div>
              </div>
              <div className="operator-actions" style={{ marginTop: 10 }}>
                <button className="btn sm primary" type="button" onClick={onOpenReview}>
                  open review
                </button>
              </div>
            </div>
          </div>

          <div className="divider">audit log · last 50</div>
          <table className="tbl">
            <thead>
              <tr>
                <th>ts</th>
                <th>process</th>
                <th>op</th>
                <th>target</th>
                <th>reverter</th>
                <th>status</th>
                <th>detail</th>
                <th>undo</th>
              </tr>
            </thead>
            <tbody>
              {groupedAuditRows.map((group) => (
                <Fragment key={group.runId}>
                  <AuditRunHeader group={group} report={reportsByRunId.get(group.runId)} />
                  {group.rows.map((row) => {
                    const expanded = expandedAuditRows[row.id] === true;
                    return (
                      <Fragment key={row.id}>
                        <tr>
                          <td className="dim">{formatTime(row.applied_at)}</td>
                          <td>
                            <span className="purple">{row.process}</span>
                          </td>
                          <td className="dim">{row.action}</td>
                          <td className="wrap" style={{ fontFamily: "var(--sans)" }}>
                            {displayTargetSummary(row.targets)}
                          </td>
                          <td>
                            {auditHasReversal(row) ? (
                              <Tag kind="acc" dot>
                                reverter
                              </Tag>
                            ) : (
                              <Tag kind="warn">no_reverser</Tag>
                            )}
                          </td>
                          <td>
                            <Tag kind={auditStatusTone(row)} dot>
                              {auditStatusLabel(row)}
                            </Tag>
                          </td>
                          <td>
                            <button
                              type="button"
                              className="btn sm ghost"
                              aria-label={`${expanded ? "hide" : "show"} audit ${row.id} payload`}
                              onClick={() => toggleAuditRow(row.id)}
                            >
                              {expanded ? "hide" : "payload"}
                            </button>
                          </td>
                          <td>
                            <button
                              type="button"
                              className={auditCanRevert(row) ? "btn sm primary" : "btn sm ghost"}
                              disabled={busy !== null || !auditCanRevert(row)}
                              title={auditRevertTitle(row)}
                              aria-label={`revert audit ${row.id}`}
                              onClick={() => openRevertConfirm(row)}
                            >
                              {auditRevertLabel(row)}
                            </button>
                          </td>
                        </tr>
                        {expanded ? (
                          <tr>
                            <td colSpan={8}>
                              <AuditPayloadDetail row={row} />
                            </td>
                          </tr>
                        ) : null}
                      </Fragment>
                    );
                  })}
                </Fragment>
              ))}
              {(state?.audit_rows.length ?? 0) === 0 ? (
                <tr>
                  <td colSpan={8} className="dim">
                    no audit rows yet
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>
      <Modal
        open={planOpen}
        title="dream plan"
        onClose={() => setPlanOpen(false)}
        footer={
          <>
            <button className="btn sm ghost" onClick={() => setPlanOpen(false)}>
              close
            </button>
            <button
              className="btn sm primary"
              disabled={busy !== null || plan === null}
              onClick={() => setConfirmOpen(true)}
            >
              apply
            </button>
          </>
        }
      >
        {plan === null ? (
          <div className="dim">no plan loaded</div>
        ) : (
          <div className="modal-form">
            <div className="dim">
              {plan.changes} proposed changes · {plan.total_budget_used} tokens ·{" "}
              {plan.processes.length} processes
            </div>
            {plan.processes.map((process) => (
              <div key={process.name} className="item">
                <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                  <span className="acc">{process.name}</span>
                  <Tag kind={process.would_change ? "acc" : ""}>{process.summary}</Tag>
                  <span className="dim tab-num">{process.budget_used} tok</span>
                </div>
                {process.changes.length === 0 ? (
                  <div className="dim">no proposed changes</div>
                ) : (
                  <div className="props">
                    {process.changes.map((change, index) => (
                      <div key={`${process.name}-${change.action}-${index}`} className="row">
                        <span className="k">{change.action}</span>
                        <span className="v">{displayTargetSummary(change.targets)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {applyResult === null ? null : (
              <div className="dim">
                applied {applyResult.applied.length} processes · {applyResult.duration_ms} ms
              </div>
            )}
          </div>
        )}
      </Modal>
      <Modal
        open={confirmOpen}
        title={busy === "apply" ? "running dream cycle..." : "apply dream cycle"}
        onClose={busy === "apply" ? () => undefined : () => setConfirmOpen(false)}
        footer={
          busy === "apply" ? null : (
            <>
              <button className="btn sm ghost" onClick={() => setConfirmOpen(false)}>
                cancel
              </button>
              <button className="btn sm primary" onClick={() => void applyDreamPlan()}>
                apply
              </button>
            </>
          )
        }
      >
        {busy === "apply" ? (
          <div className="modal-form" aria-live="polite">
            <div className="dream-running">
              <span className="dream-running-spinner" aria-hidden="true" />
              <div>
                <div style={{ color: "var(--text)", fontFamily: "var(--sans)", lineHeight: 1.5 }}>
                  Running all 12 maintenance processes. Typical runtime is 30-120 seconds depending
                  on substrate size.
                </div>
                <div className="dim" style={{ marginTop: 4 }}>
                  The dialog will close and the audit table will refresh when apply completes.
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="modal-form">
            <div style={{ color: "var(--text)", fontFamily: "var(--sans)", lineHeight: 1.5 }}>
              {plan === null ? (
                <>
                  Run the dream cycle? This executes all 12 offline maintenance processes
                  (consolidator, reflector, semantic extractor, curator, overseer, review resolver,
                  ruminator, self-narrator, procedural synthesizer, belief reviser,
                  creator-directive reconciler, commitment reconciler) and takes roughly 30-120
                  seconds.
                </>
              ) : (
                <>
                  Apply {plan.changes} changes from {plan.processes.length} processes?
                </>
              )}
            </div>
            <div className="dim">
              This writes audit rows and a dream report for the default maintenance substrate. Audit
              rows can be reverted via `borg audit revert &lt;id&gt;`.
            </div>
          </div>
        )}
      </Modal>
      <Modal
        open={revertCandidate !== null}
        title="revert maintenance change?"
        onClose={busy === "revert" ? () => undefined : () => setRevertCandidate(null)}
        footer={
          <>
            <button
              className="btn sm ghost"
              disabled={busy === "revert"}
              onClick={() => setRevertCandidate(null)}
            >
              cancel
            </button>
            <button
              className="btn sm primary"
              disabled={busy !== null || revertCandidate === null}
              onClick={() => {
                if (revertCandidate !== null) {
                  void revertAuditRow(revertCandidate);
                }
              }}
            >
              {busy === "revert" ? "reverting" : "confirm revert"}
            </button>
          </>
        }
      >
        {revertCandidate === null ? (
          <div className="dim">no audit row selected</div>
        ) : (
          <div className="modal-form">
            <div className="props">
              <div className="row">
                <span className="k">process</span>
                <span className="v">{revertCandidate.process}</span>
              </div>
              <div className="row">
                <span className="k">action</span>
                <span className="v">{revertCandidate.action}</span>
              </div>
              <div className="row">
                <span className="k">target</span>
                <span className="v">{displayTargetSummary(revertCandidate.targets)}</span>
              </div>
            </div>
            <AuditPayloadDetail row={revertCandidate} compact />
          </div>
        )}
      </Modal>
    </div>
  );
}

function AuditRunHeader({ group, report }: { group: AuditRunGroup; report?: DreamReport }) {
  return (
    <tr aria-label={`audit run ${group.runId}`}>
      <td colSpan={8} style={{ background: "var(--bg-0)" }}>
        <div style={{ display: "grid", gap: 7 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <Tag kind="info">run {shortId(group.runId)}</Tag>
            <span className="dim" style={{ fontSize: 10.5 }}>
              {group.rows.length} rows
            </span>
            <span className="dim" style={{ fontSize: 10.5 }}>
              {group.processes.join(", ")}
            </span>
            <Tag kind={group.revertedCount > 0 ? "warn" : ""}>
              {group.revertedCount} reverted
            </Tag>
          </div>
          <DreamReportInlineSummary report={report} />
        </div>
      </td>
    </tr>
  );
}

function DreamReportInlineSummary({ report }: { report?: DreamReport }) {
  if (report === undefined) {
    return (
      <div className="dim" style={{ fontSize: 10.5 }}>
        no matching dream_report in state window
      </div>
    );
  }

  const errorCount = report.errors.length;
  const budgetCount = report.budget_exhausted_processes.length;

  return (
    <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
      <Tag kind="purple" dot>
        dream_report
      </Tag>
      <Tag kind={report.changes > 0 ? "acc" : ""}>{report.changes} changes</Tag>
      <Tag>{report.tokens_used} tok</Tag>
      <Tag kind={errorCount > 0 ? "bad" : "acc"}>{errorCount} errors</Tag>
      {report.dry_run ? <Tag kind="warn">dry run</Tag> : null}
      {budgetCount > 0 ? (
        <Tag kind="warn">budget {report.budget_exhausted_processes.join(", ")}</Tag>
      ) : null}
      {report.notes.map((note, index) => (
        <span key={`${report.run_id}-note-${index}`} className="dim" style={{ fontSize: 10.5 }}>
          {note}
        </span>
      ))}
    </div>
  );
}

function AuditPayloadDetail({
  row,
  compact = false,
}: {
  row: MaintenanceAuditRow;
  compact?: boolean;
}) {
  return (
    <div
      style={{
        display: "grid",
        gap: 10,
        padding: compact ? 0 : 12,
        background: compact ? undefined : "var(--bg-0)",
      }}
    >
      <div className="dim" style={{ fontSize: 10.5, lineHeight: 1.45 }}>
        Reversal payloads are process-specific restore data. Explicit old/new machine fields are
        highlighted; raw JSON remains available below.
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: 10,
        }}
      >
        <PayloadPanel title="applied target" value={row.targets} rawLabel="raw target JSON" />
        <PayloadPanel
          title="undo/change payload"
          value={row.reversal}
          rawLabel="raw reversal JSON"
          emptyText="no reversal payload recorded"
        />
      </div>
    </div>
  );
}

function PayloadPanel({
  title,
  value,
  rawLabel,
  emptyText,
}: {
  title: string;
  value: Record<string, unknown>;
  rawLabel: string;
  emptyText?: string;
}) {
  return (
    <div className="item" style={{ padding: 12, border: "1px solid var(--line)" }}>
      <div className="upper dim" style={{ marginBottom: 8 }}>
        {title}
      </div>
      {hasPayload(value) ? (
        <StructuredPayload value={value} />
      ) : (
        <div className="dim">{emptyText ?? "empty payload"}</div>
      )}
      <RawJsonDisclosure label={rawLabel} value={value} />
    </div>
  );
}

function RawJsonDisclosure({ label, value }: { label: string; value: unknown }) {
  return (
    <details style={{ marginTop: 10 }}>
      <summary className="dim" style={{ cursor: "pointer", fontSize: 10.5 }}>
        {label}
      </summary>
      <pre style={{ marginTop: 8, maxHeight: 260, overflow: "auto" }}>{jsonText(value)}</pre>
    </details>
  );
}

function StructuredPayload({ value }: { value: Record<string, unknown> }) {
  return <PayloadRecord record={value} />;
}

function PayloadRecord({ record }: { record: Record<string, unknown> }) {
  const pairs = oldNewPairs(record);
  const consumed = pairedKeys(pairs);

  return (
    <div className="props">
      {pairs.length === 0 ? null : <OldNewPairs pairs={pairs} />}
      {Object.entries(record)
        .filter(([key]) => !consumed.has(key))
        .map(([key, value]) => (
          <div className="row" key={key}>
            <span className="k">{fieldLabel(key)}</span>
            <div className="v" style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>
              {key === "previous_fields" && isRecord(value) ? (
                <PreviousFields record={value} />
              ) : (
                <StructuredValue value={value} />
              )}
            </div>
          </div>
        ))}
    </div>
  );
}

function OldNewPairs({ pairs }: { pairs: readonly OldNewPair[] }) {
  return (
    <>
      {pairs.map((pair) => (
        <div className="row" key={pair.field}>
          <span className="k warn">{fieldLabel(pair.field)}</span>
          <div
            className="v warn"
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1fr) auto minmax(0, 1fr)",
              gap: 8,
              alignItems: "center",
              whiteSpace: "pre-wrap",
            }}
          >
            <span>{displayValue(pair.before)}</span>
            <span className="dim">-&gt;</span>
            <span>{displayValue(pair.after)}</span>
          </div>
        </div>
      ))}
    </>
  );
}

function PreviousFields({ record }: { record: Record<string, unknown> }) {
  const entries = Object.entries(record);

  if (entries.length === 0) {
    return <span className="dim">empty restore field set</span>;
  }

  return (
    <div className="props">
      {entries.map(([key, value]) => (
        <div className="row" key={key}>
          <span className="k warn">restore {fieldLabel(key)}</span>
          <span className="v warn" style={{ whiteSpace: "pre-wrap" }}>
            {displayValue(value)}
          </span>
        </div>
      ))}
    </div>
  );
}

function StructuredValue({ value }: { value: unknown }) {
  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="dim">empty array</span>;
    }

    if (!value.some((item) => isRecord(item))) {
      return <span>{displayValue(value)}</span>;
    }

    return (
      <div style={{ display: "grid", gap: 6 }}>
        {value.map((item, index) =>
          isRecord(item) ? (
            <div
              key={index}
              className="item"
              style={{ padding: 8, border: "1px solid var(--line-soft)" }}
            >
              <div className="upper dim" style={{ marginBottom: 6 }}>
                item {index + 1}
              </div>
              <PayloadRecord record={item} />
            </div>
          ) : (
            <div className="row" key={index}>
              <span className="k">{index + 1}</span>
              <span className="v">{displayValue(item)}</span>
            </div>
          ),
        )}
      </div>
    );
  }

  if (isRecord(value)) {
    return <PayloadRecord record={value} />;
  }

  return <span>{displayValue(value)}</span>;
}

function DreamCard({
  process,
  selected,
  onSelect,
  runStatus,
}: {
  process: DreamProcessSummary;
  selected: boolean;
  onSelect: () => void;
  runStatus?: "planning" | "planned" | "running" | "done" | "fail";
}) {
  // Live run status takes precedence over the historical last_status when
  // present, so the user sees the in-flight dream cycle progress on the tile.
  const liveTag =
    runStatus === undefined
      ? null
      : runStatus === "planning"
        ? { kind: "warn" as const, label: "planning" }
        : runStatus === "planned"
          ? { kind: "info" as const, label: "planned" }
          : runStatus === "running"
            ? { kind: "warn" as const, label: "applying" }
            : runStatus === "done"
              ? { kind: "acc" as const, label: "just ran" }
              : { kind: "bad" as const, label: "failed" };

  const activeRun = runStatus === "planning" || runStatus === "running";

  return (
    <div
      className={`dream-card${activeRun ? " dream-card-running" : ""}`}
      onClick={onSelect}
      style={{ borderColor: selected ? "var(--acc-dim)" : undefined, cursor: "pointer" }}
    >
      <div className="h">
        <div>
          <div className="name">{process.name}</div>
          <div className="sub">{process.description}</div>
        </div>
        <div style={{ flex: 1 }}></div>
        {liveTag === null ? (
          <Tag kind={statusTag(process.last_status)} dot>
            {process.last_status ?? "never"}
          </Tag>
        ) : (
          <Tag kind={liveTag.kind} dot>
            {liveTag.label}
          </Tag>
        )}
      </div>
      <div className="body">
        <div
          style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10, marginBottom: 10 }}
        >
          <div>
            <div className="upper dim">budget</div>
            <div style={{ color: "var(--text)", fontVariantNumeric: "tabular-nums", fontSize: 14 }}>
              {process.budget === null ? (
                <span className="dim" style={{ fontSize: 12 }}>
                  uncapped
                </span>
              ) : (
                process.budget.toLocaleString()
              )}
            </div>
          </div>
          <div>
            <div className="upper dim">audit</div>
            <div style={{ color: "var(--text)", fontVariantNumeric: "tabular-nums", fontSize: 14 }}>
              {process.last_audit_id ?? "—"}
            </div>
          </div>
          <div>
            <div className="upper dim">last</div>
            <div style={{ color: "var(--text-dim)", fontSize: 12 }}>
              {formatTime(process.last_run_at)}
            </div>
          </div>
        </div>
        <div className="dim" style={{ fontSize: 10.5, lineHeight: 1.4 }}>
          synthesized state; live budget metering ships in v2
        </div>
        <div style={{ display: "flex", gap: 6, marginTop: 10, alignItems: "center" }}>
          <span style={{ flex: 1 }}></span>
          <Tag kind={process.enabled ? "acc" : ""}>{process.enabled ? "enabled" : "off"}</Tag>
        </div>
      </div>
    </div>
  );
}
