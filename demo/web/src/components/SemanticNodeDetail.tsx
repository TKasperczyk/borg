import type { LabelRef, SemanticMemoryNode } from "../api/types";
import { isInternalId } from "../screens/screen-utils";
import { DisclosureLabel } from "./DisclosureLabel";
import { IdChip } from "./Inspector/IdChip";
import { Tag } from "./Tag";

type SemanticNodeDetailProps = {
  node: SemanticMemoryNode;
  label?: string;
};

function statusTagKind(status: SemanticMemoryNode["status"]): "acc" | "warn" | "bad" | "" {
  if (status === "active") {
    return "acc";
  }
  if (status === "quarantined") {
    return "bad";
  }
  return "warn";
}

function emptyFallback(values: readonly string[], fallback: string): string {
  return values.length === 0 ? fallback : values.join(", ");
}

function originAudienceRefs(node: SemanticMemoryNode): LabelRef[] {
  return (
    node.origin_audience_refs ??
    (node.origin_audience_entity_ids ?? []).map((id) => ({
      value: id,
      id,
      label: null,
    }))
  );
}

export function SemanticNodeDetail({ node, label }: SemanticNodeDetailProps) {
  const disclosureClass = node.disclosure_class ?? node.disclosure_label?.disclosure_class;
  const origins = originAudienceRefs(node);

  return (
    <div className="item" style={{ padding: 12, border: "1px solid var(--line)" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        {label === undefined ? null : <span className="acc">{label}</span>}
        <Tag>{node.kind}</Tag>
        <Tag kind={statusTagKind(node.status)} dot>
          {node.status}
        </Tag>
        <DisclosureLabel value={disclosureClass} />
        <Tag>confidence {node.confidence.toFixed(2)}</Tag>
      </div>
      <div
        style={{
          marginTop: 8,
          color: "var(--text)",
          fontFamily: "var(--sans)",
          fontSize: 13,
          lineHeight: 1.45,
          overflowWrap: "anywhere",
        }}
      >
        {node.display_label ?? (isInternalId(node.label) ? "unknown entity" : node.label)}
        {node.display_label === null && isInternalId(node.label) ? (
          <>
            {" "}
            <IdChip id={node.label} type="entity" />
          </>
        ) : null}
      </div>
      <div
        style={{
          marginTop: 6,
          color: "var(--text-dim)",
          fontFamily: "var(--sans)",
          fontSize: 12.5,
          lineHeight: 1.55,
          whiteSpace: "pre-wrap",
        }}
      >
        {node.description}
      </div>
      <div className="props" style={{ marginTop: 10 }}>
        <div className="row">
          <span className="k">id</span>
          <span className="v">
            <IdChip id={node.id} type="semantic_node" hint={node} />
          </span>
        </div>
        <div className="row">
          <span className="k">domain</span>
          <span className="v">{node.domain ?? "none"}</span>
        </div>
        <div className="row">
          <span className="k">aliases</span>
          <span className="v">{emptyFallback(node.aliases, "none")}</span>
        </div>
        {origins.length === 0 ? null : (
          <div className="row">
            <span className="k">origin audiences</span>
            <span className="v">
              {origins.map((origin, index) => (
                <span key={origin.value}>
                  {index === 0 ? null : ", "}
                  <span>{origin.label ?? origin.value}</span>
                  {origin.id === null ? null : (
                    <>
                      {" "}
                      <IdChip id={origin.id} type="entity" />
                    </>
                  )}
                </span>
              ))}
            </span>
          </div>
        )}
        <div className="row">
          <span className="k">source episodes</span>
          <span className="v">
            {node.source_episode_ids.length === 0
              ? "none"
              : node.source_episode_ids.map((id, index) => (
                  <span key={id}>
                    {index === 0 ? null : ", "}
                    <IdChip id={id} type="episode" />
                  </span>
                ))}
          </span>
        </div>
      </div>
    </div>
  );
}
