import { useMemo, useState } from "react";

import { getCommitments } from "../../api/client";
import type { CommitmentEnforcement, CommitmentItem, CommitmentState } from "../../api/types";
import { Tag } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { dateLabel, shortId } from "../screen-utils";

type StateFilter = CommitmentState | "all";
type EnforcementFilter = CommitmentEnforcement | "all";

function stateTag(state: CommitmentState) {
  if (state === "active") {
    return "acc";
  }
  if (state === "revoked") {
    return "bad";
  }
  return "warn";
}

export function CommitScreen() {
  const api = useApi(() => getCommitments({ state: "all" }), []);
  const [state, setState] = useState<StateFilter>("active");
  const [enforcement, setEnforcement] = useState<EnforcementFilter>("all");
  const [audience, setAudience] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const commitments = api.data?.commitments ?? [];
  const audiences = useMemo(
    () => ["all", ...[...new Set(commitments.map((item) => item.audience ?? "global"))].sort()],
    [commitments]
  );
  const filtered = useMemo(
    () =>
      commitments.filter((item) => {
        if (state !== "all" && item.state !== state) {
          return false;
        }
        if (enforcement !== "all" && item.enforcement_class !== enforcement) {
          return false;
        }
        if (audience !== "all" && (item.audience ?? "global") !== audience) {
          return false;
        }
        return true;
      }),
    [audience, commitments, enforcement, state]
  );
  const selected = filtered.find((item) => item.id === selectedId) ?? filtered[0] ?? null;

  if (api.loading && api.data === null) {
    return <div className="notice">loading commitments</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <h1>commitments</h1>
        <span className="desc">scoped promises · rules · preferences · boundaries</span>
        <span className="spacer"></span>
        <div className="filter-pills">
          {(["all", "active", "revoked", "expired"] as const).map((value) => (
            <span key={value} className={`pill ${state === value ? "on" : ""}`} onClick={() => setState(value)}>
              {value}
            </span>
          ))}
        </div>
        <span className="sep">|</span>
        <div className="filter-pills">
          {(["all", "critical", "advisory"] as const).map((value) => (
            <span
              key={value}
              className={`pill ${enforcement === value ? "on" : ""}`}
              onClick={() => setEnforcement(value)}
            >
              {value}
            </span>
          ))}
        </div>
        <span className="sep">|</span>
        <div className="filter-pills">
          {audiences.map((value) => (
            <span key={value} className={`pill ${audience === value ? "on" : ""}`} onClick={() => setAudience(value)}>
              {value}
            </span>
          ))}
        </div>
      </div>

      <div className="page-body" style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 320px" }}>
        <div style={{ overflow: "auto", borderRight: "1px solid var(--line)" }}>
          <table className="tbl">
            <thead>
              <tr>
                <th style={{ width: 92 }}>id</th>
                <th style={{ minWidth: 240 }}>text</th>
                <th style={{ width: 96 }}>audience</th>
                <th style={{ width: 96 }}>enforce</th>
                <th style={{ width: 84 }}>state</th>
                <th style={{ width: 50, textAlign: "right" }}>p</th>
                <th style={{ width: 100 }}>since</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((item) => (
                <tr
                  key={item.id}
                  onClick={() => setSelectedId(item.id)}
                  className={item.id === selected?.id ? "selected" : ""}
                  style={{ cursor: "pointer" }}
                >
                  <td>
                    <span className="acc">{shortId(item.id)}</span>
                  </td>
                  <td className="wrap" style={{ fontFamily: "var(--sans)", fontSize: "12px", lineHeight: 1.45 }}>
                    {item.text}
                    <div className="dim" style={{ fontSize: 10, marginTop: 2 }}>
                      {item.type} · {item.kind}
                      {item.about === null ? "" : ` · about:${item.about}`}
                    </div>
                  </td>
                  <td>
                    <span className={item.audience === null ? "mute" : "acc"}>{item.audience ?? "global"}</span>
                  </td>
                  <td>
                    <Tag kind={item.enforcement_class === "critical" ? "bad" : ""} dot>
                      {item.enforcement_class}
                    </Tag>
                  </td>
                  <td>
                    <Tag kind={stateTag(item.state)} dot>
                      {item.state}
                    </Tag>
                  </td>
                  <td
                    className="tab-num"
                    style={{
                      textAlign: "right",
                      color: item.priority >= 8 ? "var(--bad)" : "var(--text-dim)"
                    }}
                  >
                    {item.priority}
                  </td>
                  <td className="dim" style={{ fontSize: 11 }}>
                    {dateLabel(item.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ overflowY: "auto", background: "var(--bg-0)" }}>
          {selected === null ? (
            <div className="notice">no commitments in filter</div>
          ) : (
            <CommitmentDetail commitment={selected} />
          )}
        </div>
      </div>
    </div>
  );
}

function CommitmentDetail({ commitment }: { commitment: CommitmentItem }) {
  return (
    <>
      <div style={{ padding: "16px 16px 10px 16px", borderBottom: "1px solid var(--line)" }}>
        <div style={{ fontSize: 10.5, color: "var(--text-mute)" }}>commitment</div>
        <div style={{ fontSize: 14, color: "var(--text)", fontFamily: "var(--sans)", margin: "6px 0 10px 0" }}>
          {commitment.text}
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Tag kind={commitment.enforcement_class === "critical" ? "bad" : ""} dot>
            {commitment.enforcement_class}
          </Tag>
          <Tag kind={stateTag(commitment.state)} dot>
            {commitment.state}
          </Tag>
          <Tag>{commitment.type}</Tag>
          <Tag>{commitment.kind}</Tag>
        </div>
      </div>
      <div style={{ padding: 16 }}>
        <div className="props">
          <div className="row">
            <span className="k">id</span>
            <span className="v acc">{commitment.id}</span>
          </div>
          <div className="row">
            <span className="k">audience</span>
            <span className="v">{commitment.audience ?? "global"}</span>
          </div>
          <div className="row">
            <span className="k">made to</span>
            <span className="v">{commitment.made_to ?? "—"}</span>
          </div>
          <div className="row">
            <span className="k">about</span>
            <span className="v">{commitment.about ?? "—"}</span>
          </div>
          <div className="row">
            <span className="k">priority</span>
            <span className="v tab-num">{commitment.priority}</span>
          </div>
          <div className="row">
            <span className="k">source</span>
            <span className="v">{commitment.source}</span>
          </div>
          <div className="row">
            <span className="k">created</span>
            <span className="v">{dateLabel(commitment.created_at)}</span>
          </div>
          {commitment.revoked_at === null ? null : (
            <div className="row">
              <span className="k">revoked at</span>
              <span className="v">{dateLabel(commitment.revoked_at)}</span>
            </div>
          )}
          {commitment.expired_at === null ? null : (
            <div className="row">
              <span className="k">expired at</span>
              <span className="v">{dateLabel(commitment.expired_at)}</span>
            </div>
          )}
          {commitment.superseded_by_id === null ? null : (
            <div className="row">
              <span className="k">superseded by</span>
              <span className="v">{commitment.superseded_by_id}</span>
            </div>
          )}
        </div>

        <div className="divider">enforcement</div>
        <div style={{ fontSize: 11.5, color: "var(--text-dim)", lineHeight: 1.55 }}>
          {commitment.enforcement_class === "critical" ? (
            <>
              checked as a hard constraint. critical domain:{" "}
              <span className="bad">{commitment.critical_domain ?? "unspecified"}</span>.
            </>
          ) : (
            <>tracked as advisory context and surfaced before generation.</>
          )}
        </div>

        <div className="divider">provenance</div>
        <div style={{ fontSize: 11, color: "var(--text-dim)", display: "flex", flexDirection: "column", gap: 4 }}>
          {commitment.source_stream_entry_ids.length === 0 ? (
            <div className="dim">no stream source ids recorded</div>
          ) : (
            commitment.source_stream_entry_ids.map((id) => (
              <div key={id}>
                <span className="acc">[{id}]</span> source stream entry
              </div>
            ))
          )}
        </div>

        <div className="divider">operations</div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
          <button className="btn sm ghost" disabled title="v1 read-only">
            revoke
          </button>
          <button className="btn sm ghost" disabled title="v1 read-only">
            supersede
          </button>
          <button className="btn sm ghost" disabled title="v1 read-only">
            view evaluation history
          </button>
        </div>
      </div>
    </>
  );
}
