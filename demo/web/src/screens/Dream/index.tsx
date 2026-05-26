import { Fragment, useEffect, useRef, useState } from "react";

import {
  getDreamAudit,
  getDreamState,
  patchReviewItem,
  postDreamApply,
  postDreamPlan,
} from "../../api/client";
import type {
  DreamApplyResponse,
  DreamPlanResponse,
  DreamProcessName,
  DreamProcessSummary,
  ReviewRow,
} from "../../api/types";
import { Modal } from "../../components/Modal";
import { Tag } from "../../components/Tag";
import { useLiveEventsContext } from "../../hooks/live-context";
import { useApi } from "../../hooks/use-api";
import { formatTime } from "../../lib/stream-utils";
import { jsonText } from "../screen-utils";

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

type ReviewAction = {
  row: ReviewRow;
  action: "dismiss";
  note: string;
};

export function DreamScreen() {
  const live = useLiveEventsContext();
  const api = useApi(getDreamState, []);
  const refetch = api.refetch;
  const previousConnectionCountRef = useRef(live.connectionCount);
  const [selected, setSelected] = useState<DreamProcessName>("belief-reviser");
  const [plan, setPlan] = useState<DreamPlanResponse | null>(null);
  const [planOpen, setPlanOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [applyResult, setApplyResult] = useState<DreamApplyResponse | null>(null);
  const [reviewAction, setReviewAction] = useState<ReviewAction | null>(null);
  const [busy, setBusy] = useState<"plan" | "apply" | "review" | null>(null);
  const [operatorError, setOperatorError] = useState<string | null>(null);
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
    // A dream cycle runs all 10 offline processes (~30-120s); making the
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

  async function submitReviewAction(): Promise<void> {
    if (reviewAction === null) {
      return;
    }

    setBusy("review");
    setOperatorError(null);
    try {
      await patchReviewItem(reviewAction.row.id, {
        action: reviewAction.action,
        ...(reviewAction.note.trim().length === 0 ? {} : { note: reviewAction.note.trim() }),
      });
      setReviewAction(null);
      await refetch();
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

          <div className="divider">belief-revision review rows</div>
          <table className="tbl" style={{ marginBottom: 18 }}>
            <thead>
              <tr>
                <th>id</th>
                <th>target</th>
                <th>invalidated edge</th>
                <th>reason</th>
                <th>created</th>
                <th>actions</th>
              </tr>
            </thead>
            <tbody>
              {(state?.belief_revision_rows ?? []).map((row) => (
                <tr key={row.id}>
                  <td className="acc">{row.id}</td>
                  <td>
                    {String(row.refs.target_type ?? "target")}:{String(row.refs.target_id ?? "—")}
                  </td>
                  <td className="dim">{String(row.refs.invalidated_edge_id ?? "—")}</td>
                  <td className="wrap" style={{ fontFamily: "var(--sans)" }}>
                    {row.reason}
                  </td>
                  <td className="dim">{formatTime(row.created_at)}</td>
                  <td>
                    <div className="operator-actions">
                      {/*
                        belief_revision review rows only allow the "dismiss"
                        resolution (BELIEF_REVISION_REVIEW_RESOLUTIONS) — applying
                        a revision goes through the belief-reviser apply step,
                        not the review queue. Single dismiss button reflects that.
                      */}
                      <button
                        className="btn sm ghost"
                        disabled={busy !== null}
                        onClick={() => setReviewAction({ row, action: "dismiss", note: "" })}
                      >
                        dismiss
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
              {(state?.belief_revision_rows.length ?? 0) === 0 ? (
                <tr>
                  <td colSpan={6} className="dim">
                    no open belief-revision rows
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>

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
              </tr>
            </thead>
            <tbody>
              {(state?.audit_rows ?? []).map((row) => (
                <tr key={row.id}>
                  <td className="dim">{formatTime(row.applied_at)}</td>
                  <td>
                    <span className="purple">{row.process}</span>
                  </td>
                  <td className="dim">{row.action}</td>
                  <td className="wrap" style={{ fontFamily: "var(--sans)" }}>
                    {jsonText(row.targets)}
                  </td>
                  <td>
                    {Object.keys(row.reversal).length > 0 ? (
                      <Tag kind="acc" dot>
                        reverter
                      </Tag>
                    ) : (
                      <Tag kind="warn">no_reverser</Tag>
                    )}
                  </td>
                  <td>
                    <Tag kind={row.reverted_at === null ? "acc" : "warn"} dot>
                      {row.reverted_at === null ? "ok" : "reverted"}
                    </Tag>
                  </td>
                </tr>
              ))}
              {(state?.audit_rows.length ?? 0) === 0 ? (
                <tr>
                  <td colSpan={6} className="dim">
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
                        <span className="v">{jsonText(change.targets)}</span>
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
                  Running all 10 maintenance processes. Typical runtime is 30-120 seconds depending
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
                  Run the dream cycle? This executes all 10 offline maintenance processes
                  (consolidator, reflector, semantic extractor, curator, overseer, review resolver,
                  ruminator, self-narrator, procedural synthesizer, belief reviser) and takes
                  roughly 30-120 seconds.
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
        open={reviewAction !== null}
        title={
          reviewAction === null
            ? "review row"
            : `${reviewAction.action} review ${reviewAction.row.id}`
        }
        onClose={() => setReviewAction(null)}
        footer={
          <>
            <button
              className="btn sm ghost"
              disabled={busy !== null}
              onClick={() => setReviewAction(null)}
            >
              cancel
            </button>
            <button
              className="btn sm primary"
              disabled={busy !== null}
              onClick={() => void submitReviewAction()}
            >
              {busy === "review" ? "saving" : (reviewAction?.action ?? "save")}
            </button>
          </>
        }
      >
        {reviewAction === null ? null : (
          <div className="modal-form">
            <label className="modal-field">
              <span>note</span>
              <textarea
                value={reviewAction.note}
                onChange={(event) => setReviewAction({ ...reviewAction, note: event.target.value })}
                placeholder="operator note"
              />
            </label>
          </div>
        )}
      </Modal>
    </div>
  );
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
              {process.budget === null ? "—" : process.budget}
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
          <span title="v1 read-only" style={{ display: "inline-flex" }}>
            <button className="btn sm" disabled>
              plan
            </button>
          </span>
          <span title="v1 read-only" style={{ display: "inline-flex" }}>
            <button className="btn sm" disabled>
              apply
            </button>
          </span>
          <span title="v1 read-only" style={{ display: "inline-flex" }}>
            <button className="btn sm ghost" disabled>
              audit
            </button>
          </span>
          <span style={{ flex: 1 }}></span>
          <Tag kind={process.enabled ? "acc" : ""}>{process.enabled ? "enabled" : "off"}</Tag>
        </div>
      </div>
    </div>
  );
}
