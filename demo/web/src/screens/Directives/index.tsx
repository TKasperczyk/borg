import { useState } from "react";

import { getCreatorDirectives } from "../../api/client";
import type { CreatorDirectiveItem, CreatorDirectiveStatus } from "../../api/types";
import { Tag } from "../../components/Tag";
import { useApi } from "../../hooks/use-api";
import { dateLabel, shortId } from "../screen-utils";

function statusTag(status: CreatorDirectiveStatus) {
  if (status === "active") {
    return "acc";
  }
  if (status === "revoked") {
    return "bad";
  }
  return "warn";
}

function emptyLabel(value: string | null): string {
  return value === null || value.length === 0 ? "—" : value;
}

function joinedIds(ids: readonly string[]): string {
  return ids.length === 0 ? "—" : ids.map(shortId).join(", ");
}

export function DirectivesScreen() {
  const api = useApi(getCreatorDirectives, []);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const directives = api.data?.directives ?? [];
  const selected = directives.find((item) => item.id === selectedId) ?? directives[0] ?? null;

  if (api.loading && api.data === null) {
    return <div className="notice">loading creator directives</div>;
  }

  if (api.error !== null) {
    return <div className="notice bad">{api.error.message}</div>;
  }

  return (
    <div className="full-page">
      <div className="page-head">
        <h1>creator directives</h1>
        <span className="desc">identity · subject facts · disclosure · response policies</span>
      </div>

      <div
        className="page-body"
        style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) 320px" }}
      >
        <div style={{ overflow: "auto", borderRight: "1px solid var(--line)" }}>
          <table className="tbl">
            <thead>
              <tr>
                <th style={{ width: 92 }}>id</th>
                <th style={{ width: 150 }}>kind</th>
                <th style={{ minWidth: 260 }}>text</th>
                <th style={{ width: 132 }}>scope</th>
                <th style={{ width: 126 }}>content</th>
                <th style={{ width: 142 }}>mention</th>
                <th style={{ width: 84 }}>status</th>
                <th style={{ width: 100 }}>since</th>
              </tr>
            </thead>
            <tbody>
              {directives.map((item) => (
                <tr
                  key={item.id}
                  onClick={() => setSelectedId(item.id)}
                  className={item.id === selected?.id ? "selected" : ""}
                  style={{ cursor: "pointer" }}
                >
                  <td>
                    <span className="acc">{shortId(item.id)}</span>
                  </td>
                  <td>
                    <Tag>{item.kind}</Tag>
                  </td>
                  <td
                    className="wrap"
                    style={{ fontFamily: "var(--sans)", fontSize: "12px", lineHeight: 1.45 }}
                  >
                    {emptyLabel(item.text)}
                    <div className="dim" style={{ fontSize: 10, marginTop: 2 }}>
                      subject:{item.subject_entity_name ?? item.subject_kind} · p:{item.priority}
                    </div>
                  </td>
                  <td>{item.activation_scope}</td>
                  <td>{item.content_scope}</td>
                  <td>{item.mention_policy}</td>
                  <td>
                    <Tag kind={statusTag(item.status)} dot>
                      {item.status}
                    </Tag>
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
            <div className="notice">no creator directives</div>
          ) : (
            <CreatorDirectiveDetail directive={selected} />
          )}
        </div>
      </div>
    </div>
  );
}

function CreatorDirectiveDetail({ directive }: { directive: CreatorDirectiveItem }) {
  return (
    <>
      <div style={{ padding: "16px 16px 10px 16px", borderBottom: "1px solid var(--line)" }}>
        <div style={{ fontSize: 10.5, color: "var(--text-mute)" }}>creator directive</div>
        <div
          style={{
            fontSize: 14,
            color: "var(--text)",
            fontFamily: "var(--sans)",
            margin: "6px 0 10px 0",
          }}
        >
          {emptyLabel(directive.text)}
        </div>
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          <Tag kind={statusTag(directive.status)} dot>
            {directive.status}
          </Tag>
          <Tag>{directive.kind}</Tag>
          <Tag>{directive.activation_scope}</Tag>
        </div>
      </div>
      <div style={{ padding: 16 }}>
        <div className="props">
          <div className="row">
            <span className="k">id</span>
            <span className="v acc">{directive.id}</span>
          </div>
          <div className="row">
            <span className="k">subject</span>
            <span className="v">{directive.subject_entity_name ?? directive.subject_kind}</span>
          </div>
          <div className="row">
            <span className="k">subject id</span>
            <span className="v">{directive.subject_entity_id ?? "—"}</span>
          </div>
          <div className="row">
            <span className="k">content scope</span>
            <span className="v">{directive.content_scope}</span>
          </div>
          <div className="row">
            <span className="k">mention policy</span>
            <span className="v">{directive.mention_policy}</span>
          </div>
          <div className="row">
            <span className="k">priority</span>
            <span className="v tab-num">{directive.priority}</span>
          </div>
          <div className="row">
            <span className="k">created</span>
            <span className="v">{dateLabel(directive.created_at)}</span>
          </div>
        </div>

        <div className="divider">activation</div>
        <div className="props">
          <div className="row">
            <span className="k">scope</span>
            <span className="v">{directive.activation_scope}</span>
          </div>
          <div className="row">
            <span className="k">allowed ids</span>
            <span className="v">{joinedIds(directive.activation_allowed_entity_ids)}</span>
          </div>
          <div className="row">
            <span className="k">excluded ids</span>
            <span className="v">{joinedIds(directive.activation_excluded_entity_ids)}</span>
          </div>
        </div>

        <div className="divider">content</div>
        <div className="props">
          <div className="row">
            <span className="k">canonical fact</span>
            <span className="v">{directive.canonical_fact ?? "—"}</span>
          </div>
          <div className="row">
            <span className="k">operational directive</span>
            <span className="v">{directive.operational_directive ?? "—"}</span>
          </div>
        </div>
      </div>
    </>
  );
}
