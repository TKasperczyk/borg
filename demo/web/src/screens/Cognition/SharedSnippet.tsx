import type { SharedStateEntry } from "../../api/types";
import { Empty } from "../../components/Empty";
import { Tag, type TagKind } from "../../components/Tag";

export type SharedSnippetProps = {
  audience: string;
  entries: readonly SharedStateEntry[];
};

function tagKind(kind: SharedStateEntry["kind"]): TagKind {
  if (kind === "locked") {
    return "acc";
  }
  if (kind === "live") {
    return "info";
  }
  if (kind === "tentative") {
    return "warn";
  }
  if (kind === "pending") {
    return "purple";
  }
  if (kind === "invalidated") {
    return "bad";
  }
  return "";
}

function visualClass(kind: SharedStateEntry["kind"]): string {
  if (kind === "low_salience_live") {
    return "live";
  }
  if (kind === "dormant_live") {
    return "dormant";
  }
  return kind;
}

export function SharedSnippet({ audience, entries }: SharedSnippetProps) {
  return (
    <div>
      <div style={{ padding: "10px 14px", borderBottom: "1px solid var(--line)", color: "var(--text-mute)", fontSize: "10.5px" }}>
        audience-scoped durable shared state · {audience}
      </div>
      <div style={{ padding: 14, display: "flex", flexDirection: "column", gap: 8 }}>
        {entries.length === 0 ? <Empty>no shared state for {audience}</Empty> : null}
        {entries.map((entry) => (
          <div key={entry.id} className={`ss-entry ${entry.kind} ${visualClass(entry.kind)}`}>
            <div className="h">
              <Tag kind={tagKind(entry.kind)} dot>
                {entry.kind}
              </Tag>
              <span className="id">[{entry.id}]</span>
              <span style={{ flex: 1 }}></span>
              <span className="dim" style={{ fontSize: 10 }}>
                rank {entry.rank} · {entry.provenance_stream_entry_ids.length} src
              </span>
            </div>
            <div className="text">{entry.text}</div>
            <div className="meta">
              {entry.state_key === null ? null : <span>state key <span className="acc">{entry.state_key}</span></span>}
              {entry.superseded_by_id === null ? null : (
                <span>
                  superseded by <span className="info">{entry.superseded_by_id}</span>
                </span>
              )}
              {entry.canonicalizes.goal_ids.length === 0 ? null : (
                <span>goals {entry.canonicalizes.goal_ids.length}</span>
              )}
              {entry.canonicalizes.commitment_ids.length === 0 ? null : (
                <span>commitments {entry.canonicalizes.commitment_ids.length}</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
