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
  DreamPlanRequest,
  DreamProcessName,
  DreamProcessSummary,
  DreamScheduleItem,
} from "../../api/types";
import { ErrorState } from "../../components/ErrorState";
import { IdRef } from "../../components/Inspector/IdRef";
import { Loading } from "../../components/Loading";
import { resolveObjectType, type ObjectType } from "../../components/Inspector/inspector-id";
import { Modal } from "../../components/Modal";
import { Tag } from "../../components/Tag";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import { activateOnEnterOrSpace } from "../../lib/keyboard";
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

function formatIntervalMs(milliseconds: number): string {
  if (milliseconds < 1000) {
    return `${milliseconds} ms`;
  }

  if (milliseconds < 60_000) {
    const seconds = milliseconds / 1000;
    return `${Number.isInteger(seconds) ? seconds : seconds.toFixed(1)}s`;
  }

  const minutes = Math.floor(milliseconds / 60_000);
  const seconds = Math.round((milliseconds % 60_000) / 1000);
  return seconds === 0 ? `${minutes}m` : `${minutes}m ${seconds}s`;
}

function formatDurationMs(milliseconds: number): string {
  if (milliseconds < 1000) {
    return `${milliseconds} ms`;
  }

  const seconds = milliseconds / 1000;
  return `${Number.isInteger(seconds) ? seconds : seconds.toFixed(1)}s`;
}

function processListTitle(label: string, processes: readonly DreamProcessName[]): string {
  return `${label}: ${processes.length === 0 ? "none" : processes.join(", ")}`;
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

  return row.reverted_at === null ? "Revert this audited maintenance change" : "Already reverted";
}

function auditStatusLabel(row: MaintenanceAuditRow): string {
  if (row.reverted_at !== null) {
    return "reverted";
  }

  if (
    auditHasReversal(row) &&
    (row.process === "creator-directive-reconciler" || row.process === "commitment-reconciler")
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

type AuditObjectRef = {
  id: string;
  type: ObjectType;
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

function auditObjectRefsFromRecord(
  record: Record<string, unknown>,
  deep: boolean,
): AuditObjectRef[] {
  const refs: AuditObjectRef[] = [];
  const seen = new Set<string>();

  function addId(value: string): void {
    const type = resolveObjectType(value);
    if (type === null || seen.has(value)) {
      return;
    }

    seen.add(value);
    refs.push({ id: value, type });
  }

  function walk(value: unknown): void {
    if (typeof value === "string") {
      addId(value);
      return;
    }

    if (Array.isArray(value)) {
      for (const item of value) {
        if (typeof item === "string") {
          addId(item);
        } else if (deep && (Array.isArray(item) || isRecord(item))) {
          walk(item);
        }
      }
      return;
    }

    if (deep && isRecord(value)) {
      for (const child of Object.values(value)) {
        walk(child);
      }
    }
  }

  for (const value of Object.values(record)) {
    walk(value);
  }

  return refs;
}

export function DreamScreen({ onOpenReview }: { onOpenReview?: () => void }) {
  const live = useLiveEventsContext();
  const api = useApi(getDreamState, []);
  const refetch = api.refetch;
  const previousConnectionCountRef = useRef(live.connectionCount);
  const [selected, setSelected] = useState<DreamProcessName>("belief-reviser");
  const [selectedPlanProcesses, setSelectedPlanProcesses] = useState<ReadonlySet<DreamProcessName>>(
    () => new Set(PROCESS_NAMES),
  );
  const [budgetInput, setBudgetInput] = useState("");
  const [plan, setPlan] = useState<DreamPlanResponse | null>(null);
  const [planStale, setPlanStale] = useState(false);
  const [planOpen, setPlanOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [confirmApplyPlanId, setConfirmApplyPlanId] = useState<string | null>(null);
  const [confirmApplyProcessCount, setConfirmApplyProcessCount] = useState<number | null>(null);
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
        if (plan !== null) {
          setPlanStale(true);
        }
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
        if (plan !== null) {
          setPlanStale(true);
        }
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
  }, [live, plan, refetch]);

  useEffect(() => {
    const previousConnectionCount = previousConnectionCountRef.current;
    previousConnectionCountRef.current = live.connectionCount;

    if (live.connectionCount <= 1 || live.connectionCount === previousConnectionCount) {
      return;
    }

    if (plan !== null) {
      setPlanStale(true);
    }
    void refetch();
  }, [live.connectionCount, plan, refetch]);

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
  const selectedPlanProcessNames = useMemo(
    () => PROCESS_NAMES.filter((name) => selectedPlanProcesses.has(name)),
    [selectedPlanProcesses],
  );
  const schedulerTitle =
    state === null
      ? ""
      : [
          processListTitle("light", state.scheduler.light_processes),
          processListTitle("heavy", state.scheduler.heavy_processes),
        ].join("\n");
  const recentErrorCount = processes.filter((process) => process.last_status === "error").length;
  const applyingPlan = confirmApplyPlanId !== null && plan !== null;

  function setAllPlanProcesses(selected: boolean): void {
    setSelectedPlanProcesses(selected ? new Set(PROCESS_NAMES) : new Set());
  }

  function togglePlanProcess(name: DreamProcessName): void {
    setSelectedPlanProcesses((current) => {
      const next = new Set(current);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  }

  function buildPlanRequest(): DreamPlanRequest | null {
    if (selectedPlanProcessNames.length === 0) {
      setOperatorError("Select at least one process to plan.");
      return null;
    }

    const trimmedBudget = budgetInput.trim();
    const budget = trimmedBudget.length === 0 ? undefined : Number.parseInt(trimmedBudget, 10);

    if (budget !== undefined) {
      if (!Number.isFinite(budget) || String(budget) !== trimmedBudget || budget <= 0) {
        setOperatorError("Budget must be a positive integer.");
        return null;
      }
    }

    return {
      processes: selectedPlanProcessNames,
      ...(budget === undefined ? {} : { budget }),
    };
  }

  async function loadPlan(): Promise<void> {
    const request = buildPlanRequest();
    if (request === null) {
      return;
    }

    setBusy("plan");
    setOperatorError(null);
    setApplyResult(null);
    try {
      const nextPlan = await postDreamPlan(request);
      setPlan(nextPlan);
      setPlanStale(false);
      setPlanOpen(true);
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
    // Users who want a preview can hit the `plan` button first and apply
    // from the plan modal; this direct apply path remains unplanned.
    setOperatorError(null);
    setApplyResult(null);
    setConfirmApplyPlanId(null);
    setConfirmApplyProcessCount(null);
    setConfirmOpen(true);
  }

  function openPlanApplyConfirm(): void {
    if (plan === null || planStale) {
      return;
    }

    setOperatorError(null);
    setApplyResult(null);
    setConfirmApplyPlanId(plan.plan_id);
    setConfirmApplyProcessCount(plan.processes.length);
    setConfirmOpen(true);
  }

  async function applyDreamPlan(): Promise<void> {
    if (confirmApplyPlanId !== null && planStale) {
      setOperatorError("state changed since this plan -- re-plan to apply");
      setConfirmOpen(false);
      setConfirmApplyPlanId(null);
      setConfirmApplyProcessCount(null);
      return;
    }

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
      const result = await postDreamApply(
        confirmApplyPlanId === null ? {} : { plan_id: confirmApplyPlanId },
      );
      setApplyResult(result);
      await Promise.all([refetch(), getDreamAudit()]);
    } catch (caught) {
      setOperatorError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setConfirmApplyPlanId(null);
      setConfirmApplyProcessCount(null);
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
    return <Loading>loading dream cycle</Loading>;
  }

  if (api.error !== null) {
    return <ErrorState onRetry={api.refetch}>{api.error.message}</ErrorState>;
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <h1>dream ops</h1>
        <span className="desc">maintenance planning · schedule · audit</span>
        <span className="spacer"></span>
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
                const total = confirmApplyProcessCount ?? processes.length;
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
        {operatorError === null ? null : <ErrorState>{operatorError}</ErrorState>}
        <div className="dream-ops-strip">
          <div className="dream-health-grid">
            <div className="dream-health-card" title={schedulerTitle}>
              <div className="upper dim">scheduler</div>
              <div className={state?.scheduler.enabled === true ? "value acc" : "value dim"}>
                {state?.scheduler.enabled === true ? "enabled" : "disabled"}
              </div>
              <div className="sub">
                light {formatIntervalMs(state?.scheduler.light_interval_ms ?? 0)} / heavy{" "}
                {formatIntervalMs(state?.scheduler.heavy_interval_ms ?? 0)}
              </div>
            </div>
            <div className="dream-health-card">
              <div className="upper dim">pending extraction</div>
              <div className="value">{state?.pending_extraction_episodes ?? "—"}</div>
              <div className="sub">episodes</div>
            </div>
            <div className="dream-health-card">
              <div className="upper dim">belief revision</div>
              <div className="value">{state?.belief_revision_rows.length ?? 0}</div>
              <div className="sub">open reviews</div>
            </div>
            <div
              className="dream-health-card"
              title={
                lastMaintenanceTick === null
                  ? "No maintenance tick observed in this browser session"
                  : `Observed at ${formatTime(lastMaintenanceTick.ts)}`
              }
            >
              <div className="upper dim">last tick this session</div>
              {lastMaintenanceTick === null ? (
                <>
                  <div className="value dim">none</div>
                  <div className="sub">live frame only</div>
                </>
              ) : (
                <>
                  <div className={`value ${maintenanceTickTone(lastMaintenanceTick)}`}>
                    {maintenanceTickSummary(lastMaintenanceTick)}
                  </div>
                  <div className="sub">live frame only</div>
                </>
              )}
            </div>
            <div className="dream-health-card">
              <div className="upper dim">recent errors</div>
              <div className={recentErrorCount > 0 ? "value bad" : "value"}>{recentErrorCount}</div>
              <div className="sub">process statuses</div>
            </div>
          </div>

          <div className="dream-workbench panel">
            <div className="panel-header">
              <span className="title">plan/apply workbench</span>
              <span className="badge">{selectedPlanProcessNames.length}/12 selected</span>
            </div>
            <div className="panel-body pad">
              <div className="dream-workbench-controls">
                <div className="dream-process-picker" aria-label="dream process subset">
                  {PROCESS_NAMES.map((name) => (
                    <label
                      key={name}
                      className={`dream-process-toggle${
                        selectedPlanProcesses.has(name) ? " on" : ""
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={selectedPlanProcesses.has(name)}
                        aria-label={`include ${name}`}
                        onChange={() => togglePlanProcess(name)}
                      />
                      <span>{name}</span>
                    </label>
                  ))}
                </div>
                <div className="dream-budget-control">
                  <label className="modal-field">
                    <span>budget</span>
                    <input
                      type="number"
                      inputMode="numeric"
                      min={1}
                      step={1}
                      placeholder="optional"
                      value={budgetInput}
                      onChange={(event) => setBudgetInput(event.target.value)}
                    />
                  </label>
                  <div className="operator-actions">
                    <button
                      type="button"
                      className="btn sm ghost"
                      disabled={busy !== null}
                      onClick={() => setAllPlanProcesses(true)}
                    >
                      all
                    </button>
                    <button
                      type="button"
                      className="btn sm ghost"
                      disabled={busy !== null}
                      onClick={() => setAllPlanProcesses(false)}
                    >
                      clear
                    </button>
                    <button
                      className="btn sm"
                      disabled={busy !== null}
                      aria-label="plan dream"
                      onClick={() => void loadPlan()}
                    >
                      {busy === "plan" ? "planning" : "plan"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <DreamScheduleLane schedule={state?.schedule ?? []} />
          {applyResult === null ? null : <DreamApplyResultPanel result={applyResult} />}
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
            <div className="panel dream-selected-process" style={{ marginBottom: 14 }}>
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
                    <span className="v">
                      {selectedProcess.last_audit_id === null ? (
                        "—"
                      ) : (
                        <IdRef
                          id={String(selectedProcess.last_audit_id)}
                          type="dream_audit"
                          label={String(selectedProcess.last_audit_id)}
                        />
                      )}
                    </span>
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
                          <td className="dim wrap" style={{ minWidth: "6.5rem" }}>
                            {row.action}
                          </td>
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
              disabled={busy !== null || plan === null || planStale}
              onClick={() => openPlanApplyConfirm()}
            >
              apply plan
            </button>
          </>
        }
      >
        {plan === null ? (
          <div className="dim">no plan loaded</div>
        ) : (
          <div className="modal-form">
            {planStale ? (
              <div className="notice warn">state changed since this plan -- re-plan to apply</div>
            ) : null}
            <div className="dream-plan-summary">
              <Tag kind={plan.changes > 0 ? "acc" : ""}>{plan.changes} changes</Tag>
              <Tag>{plan.total_budget_used} budget used</Tag>
              <Tag>{plan.processes.length} processes</Tag>
            </div>
            <div className="dim">total budget used: {plan.total_budget_used}</div>
            {plan.processes.map((process) => (
              <DreamPlanProcessResult key={process.name} process={process} />
            ))}
          </div>
        )}
      </Modal>
      <Modal
        open={confirmOpen}
        title={busy === "apply" ? "running dream cycle..." : "apply dream cycle"}
        onClose={
          busy === "apply"
            ? () => undefined
            : () => {
                setConfirmOpen(false);
                setConfirmApplyPlanId(null);
                setConfirmApplyProcessCount(null);
              }
        }
        footer={
          busy === "apply" ? null : (
            <>
              <button
                className="btn sm ghost"
                onClick={() => {
                  setConfirmOpen(false);
                  setConfirmApplyPlanId(null);
                  setConfirmApplyProcessCount(null);
                }}
              >
                cancel
              </button>
              <button
                className="btn sm primary"
                disabled={confirmApplyPlanId !== null && planStale}
                onClick={() => void applyDreamPlan()}
              >
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
            {confirmApplyPlanId !== null && planStale ? (
              <div className="notice warn">state changed since this plan -- re-plan to apply</div>
            ) : null}
            <div style={{ color: "var(--text)", fontFamily: "var(--sans)", lineHeight: 1.5 }}>
              {!applyingPlan ? (
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
                <span className="v">
                  {displayTargetSummary(revertCandidate.targets)}
                  <AuditTargetRefs targets={revertCandidate.targets} />
                </span>
              </div>
            </div>
            <RevertPayloadComparison row={revertCandidate} />
          </div>
        )}
      </Modal>
    </div>
  );
}

function DreamScheduleLane({ schedule }: { schedule: readonly DreamScheduleItem[] }) {
  return (
    <div className="panel dream-schedule-panel">
      <div className="panel-header">
        <span className="title">schedule lane</span>
        <span className="badge">{schedule.length} rows</span>
      </div>
      <div className="panel-body">
        {schedule.length === 0 ? (
          <div className="dim" style={{ padding: 12 }}>
            no scheduled runs synthesized yet
          </div>
        ) : (
          <table className="tbl dream-schedule-table">
            <thead>
              <tr>
                <th>process</th>
                <th>scheduled</th>
                <th>source</th>
                <th>related ids</th>
              </tr>
            </thead>
            <tbody>
              {schedule.map((item, index) => (
                <tr key={`${item.process}-${item.scheduled_at}-${item.source}-${index}`}>
                  <td>
                    <span className="purple">{item.process}</span>
                  </td>
                  <td className="dim">{formatTime(item.scheduled_at)}</td>
                  <td>
                    <Tag kind={item.source === "audit" ? "acc" : "info"}>{item.source}</Tag>
                  </td>
                  <td>
                    <ScheduleRelatedRefs item={item} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function ScheduleRelatedRefs({ item }: { item: DreamScheduleItem }) {
  const streamEntryType =
    item.stream_entry_id === undefined ? null : resolveObjectType(item.stream_entry_id);

  if (item.audit_id === undefined && item.stream_entry_id === undefined) {
    return <span className="dim">—</span>;
  }

  return (
    <div className="dream-ref-list">
      {item.audit_id === undefined ? null : (
        <IdRef
          id={String(item.audit_id)}
          type="dream_audit"
          label={`audit ${item.audit_id}`}
          ariaLabel={`jump to audit ${item.audit_id}`}
        />
      )}
      {item.stream_entry_id === undefined ? null : streamEntryType === null ? (
        <span className="dim">{item.stream_entry_id}</span>
      ) : (
        <IdRef id={item.stream_entry_id} type={streamEntryType} />
      )}
    </div>
  );
}

function DreamApplyResultPanel({ result }: { result: DreamApplyResponse }) {
  return (
    <div className="panel dream-apply-result">
      <div className="panel-header">
        <span className="title">apply result</span>
        <span className="badge">{formatDurationMs(result.duration_ms)}</span>
      </div>
      <div className="panel-body pad">
        <div className="dream-plan-summary">
          <Tag kind="info">
            <IdRef
              id={result.run_id}
              type="maintenance_run"
              label={`run ${shortId(result.run_id)}`}
              hint={result}
            />
          </Tag>
          <Tag kind={result.applied.length > 0 ? "acc" : ""}>{result.applied.length} applied</Tag>
          <Tag kind={result.failed.length > 0 ? "bad" : "acc"}>{result.failed.length} failed</Tag>
          <Tag>{result.total_budget_used} budget used</Tag>
          <Tag>{result.duration_ms} ms</Tag>
        </div>
        <div className="dream-result-grid">
          <div>
            <div className="upper dim" style={{ marginBottom: 6 }}>
              applied
            </div>
            {result.applied.length === 0 ? (
              <div className="dim">no processes applied</div>
            ) : (
              <div className="props">
                {result.applied.map((entry) => (
                  <div key={entry.name} className="row">
                    <span className="k">{entry.name}</span>
                    <span className="v">
                      {entry.changes} changes
                      <span className="dream-ref-list">
                        {entry.audit_id === null ? null : (
                          <IdRef
                            id={String(entry.audit_id)}
                            type="dream_audit"
                            label={`audit ${entry.audit_id}`}
                            ariaLabel={`jump to audit ${entry.audit_id}`}
                          />
                        )}
                        {entry.audit_ids
                          .filter((auditId) => auditId !== entry.audit_id)
                          .map((auditId) => (
                            <IdRef
                              key={`${entry.name}-${auditId}`}
                              id={String(auditId)}
                              type="dream_audit"
                              label={`audit ${auditId}`}
                              ariaLabel={`jump to audit ${auditId}`}
                            />
                          ))}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div>
            <div className="upper dim" style={{ marginBottom: 6 }}>
              failed
            </div>
            {result.failed.length === 0 ? (
              <div className="dim">no process failures</div>
            ) : (
              <div className="props">
                {result.failed.map((entry, index) => (
                  <div key={`${entry.name}-${index}`} className="row">
                    <span className="k">{entry.name}</span>
                    <span className="v">
                      {entry.message}
                      {entry.code === undefined ? null : (
                        <span className="dim"> · code {entry.code}</span>
                      )}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function DreamPlanProcessResult({ process }: { process: DreamPlanResponse["processes"][number] }) {
  return (
    <div className="item dream-plan-process">
      <div className="dream-plan-process-head">
        <span className="acc">{process.name}</span>
        <Tag kind={process.would_change ? "acc" : ""}>
          {process.would_change ? "would change" : "no change"}
        </Tag>
        <Tag kind={process.budget_exhausted ? "warn" : ""}>
          {process.budget_exhausted ? "budget exhausted" : "budget ok"}
        </Tag>
        <span className="dim tab-num">{process.budget_used} budget used</span>
      </div>
      <div className="dim" style={{ marginBottom: 8 }}>
        {process.summary}
      </div>
      <div className="props">
        <div className="row">
          <span className="k">would change</span>
          <span className="v">{process.would_change ? "true" : "false"}</span>
        </div>
        <div className="row">
          <span className="k">budget used</span>
          <span className="v">{process.budget_used}</span>
        </div>
        <div className="row">
          <span className="k">budget exhausted</span>
          <span className="v">{process.budget_exhausted ? "true" : "false"}</span>
        </div>
      </div>
      <div className="upper dim" style={{ marginTop: 10, marginBottom: 6 }}>
        changes
      </div>
      {process.changes.length === 0 ? (
        <div className="dim">no proposed changes</div>
      ) : (
        <div className="props">
          {process.changes.map((change, index) => (
            <div key={`${process.name}-${change.action}-${index}`} className="row">
              <span className="k">{change.action}</span>
              <div className="v">
                {displayTargetSummary(change.targets)}
                <AuditTargetRefs targets={change.targets} />
                {change.preview === undefined ? null : (
                  <RawJsonDisclosure label="preview JSON" value={change.preview} />
                )}
              </div>
            </div>
          ))}
        </div>
      )}
      <div className="upper dim" style={{ marginTop: 10, marginBottom: 6 }}>
        errors
      </div>
      {process.errors.length === 0 ? (
        <div className="dim">no errors</div>
      ) : (
        <div className="props">
          {process.errors.map((error, index) => (
            <div key={`${process.name}-error-${index}`} className="row">
              <span className="k">{error.code ?? "error"}</span>
              <span className="v">
                {error.message}
                {error.target_type === undefined && error.target_id === undefined ? null : (
                  <span className="dim">
                    {" "}
                    · target {error.target_type ?? "unknown"} {error.target_id ?? "—"}
                  </span>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RevertPayloadComparison({ row }: { row: MaintenanceAuditRow }) {
  return (
    <div className="dream-revert-compare">
      <div className="dim" style={{ fontSize: "var(--fs-xs)", lineHeight: 1.45 }}>
        Current shows the audited target row. After revert shows the recorded reversal payload the
        server will apply; opaque process-specific shapes are shown as payload panels.
      </div>
      <div className="dream-diff-grid">
        <PayloadPanel
          title="current / audited target"
          value={row.targets}
          rawLabel="raw current target JSON"
          emptyText="empty target payload"
        />
        <PayloadPanel
          title="after revert / reversal payload"
          value={row.reversal}
          rawLabel="raw after-revert JSON"
          emptyText="empty reversal payload"
        />
      </div>
    </div>
  );
}

function AuditRunHeader({ group, report }: { group: AuditRunGroup; report?: DreamReport }) {
  return (
    <tr aria-label={`audit run ${group.runId}`}>
      <td colSpan={8} style={{ background: "var(--bg-0)" }}>
        <div style={{ display: "grid", gap: 7 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <Tag kind="info">
              <IdRef
                id={group.runId}
                type="maintenance_run"
                label={`run ${shortId(group.runId)}`}
                hint={group}
              />
            </Tag>
            <span className="dim" style={{ fontSize: "var(--fs-xs)" }}>
              {group.rows.length} rows
            </span>
            <span className="dim" style={{ fontSize: "var(--fs-xs)" }}>
              {group.processes.join(", ")}
            </span>
            <Tag kind={group.revertedCount > 0 ? "warn" : ""}>{group.revertedCount} reverted</Tag>
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
      <div className="dim" style={{ fontSize: "var(--fs-xs)" }}>
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
        <span
          key={`${report.run_id}-note-${index}`}
          className="dim"
          style={{ fontSize: "var(--fs-xs)" }}
        >
          {note}
        </span>
      ))}
    </div>
  );
}

function AuditObjectRefList({ refs }: { refs: readonly AuditObjectRef[] }) {
  if (refs.length === 0) {
    return null;
  }

  return (
    <span
      style={{
        display: "inline-flex",
        gap: 6,
        alignItems: "center",
        flexWrap: "wrap",
        marginLeft: 6,
      }}
    >
      {refs.map((ref) => (
        <IdRef key={ref.id} id={ref.id} type={ref.type} />
      ))}
    </span>
  );
}

function AuditTargetRefs({ targets }: { targets: Record<string, unknown> }) {
  return <AuditObjectRefList refs={auditObjectRefsFromRecord(targets, true)} />;
}

function PayloadIdRefs({ value }: { value: Record<string, unknown> }) {
  const refs = auditObjectRefsFromRecord(value, false);

  if (refs.length === 0) {
    return null;
  }

  return (
    <div style={{ marginBottom: 8 }}>
      <AuditObjectRefList refs={refs} />
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
      <div className="dim" style={{ fontSize: "var(--fs-xs)", lineHeight: 1.45 }}>
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
        <AuditTargetPanel row={row} />
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

function AuditTargetPanel({ row }: { row: MaintenanceAuditRow }) {
  return (
    <div className="item" style={{ padding: 12, border: "1px solid var(--line)" }}>
      <div className="upper dim" style={{ marginBottom: 8 }}>
        applied target
      </div>
      <div
        style={{
          marginBottom: 8,
          color: "var(--text)",
          fontFamily: "var(--sans)",
          fontSize: 12.5,
          lineHeight: 1.45,
          overflowWrap: "anywhere",
        }}
      >
        {displayTargetSummary(row.targets)}
        <AuditTargetRefs targets={row.targets} />
      </div>
      {hasPayload(row.targets) ? (
        <StructuredPayload value={row.targets} />
      ) : (
        <div className="dim">empty payload</div>
      )}
      <RawJsonDisclosure label="raw target JSON" value={row.targets} />
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
      <PayloadIdRefs value={value} />
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
      <summary className="dim" style={{ cursor: "pointer", fontSize: "var(--fs-xs)" }}>
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
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      onClick={onSelect}
      onKeyDown={(event) => activateOnEnterOrSpace(event, onSelect)}
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
        <div className="dim" style={{ fontSize: "var(--fs-xs)", lineHeight: 1.4 }}>
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
