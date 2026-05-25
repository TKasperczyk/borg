import { Fragment, useEffect, useState } from "react";

import { getDreamState } from "../../api/client";
import type { DreamProcessName, DreamProcessSummary } from "../../api/types";
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

export function DreamScreen() {
  const live = useLiveEventsContext();
  const api = useApi(getDreamState, []);
  const refetch = api.refetch;
  const [selected, setSelected] = useState<DreamProcessName>("belief-reviser");

  useEffect(() => {
    return live.subscribe((frame) => {
      if (
        frame.type === "stream:append" &&
        frame.entries.some((entry) => entry.kind === "dream_report")
      ) {
        void refetch();
      }
    });
  }, [live, refetch]);

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
        <button className="btn sm" disabled title="v1 read-only">
          dry-run all
        </button>
        <button className="btn sm primary" disabled title="v1 read-only">
          trigger dream
        </button>
      </div>

      <div className="page-body">
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
                </tr>
              ))}
              {(state?.belief_revision_rows.length ?? 0) === 0 ? (
                <tr>
                  <td colSpan={5} className="dim">
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
    </div>
  );
}

function DreamCard({
  process,
  selected,
  onSelect,
}: {
  process: DreamProcessSummary;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      className="dream-card"
      onClick={onSelect}
      style={{ borderColor: selected ? "var(--acc-dim)" : undefined, cursor: "pointer" }}
    >
      <div className="h">
        <div>
          <div className="name">{process.name}</div>
          <div className="sub">{process.description}</div>
        </div>
        <div style={{ flex: 1 }}></div>
        <Tag kind={statusTag(process.last_status)} dot>
          {process.last_status ?? "never"}
        </Tag>
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
