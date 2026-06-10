import type { LabelRef, SemanticMemoryEdge, SemanticMemoryNode } from "../api/types";
import { formatTimestamp } from "../lib/stream-utils";
import { shortId } from "../screens/screen-utils";
import { DisclosureLabel } from "./DisclosureLabel";
import { IdChip } from "./Inspector/IdChip";
import { Tag } from "./Tag";

function edgeEndpointLabel(id: string, nodes: readonly SemanticMemoryNode[]): string {
  return nodes.find((node) => node.id === id)?.label ?? shortId(id);
}

function originAudienceRefs(edge: SemanticMemoryEdge): LabelRef[] {
  return (
    edge.origin_audience_refs ??
    (edge.origin_audience_entity_ids ?? []).map((id) => ({
      value: id,
      id,
      label: null,
    }))
  );
}

export function SemanticEdgeDetail({
  edge,
  nodes,
}: {
  edge: SemanticMemoryEdge;
  nodes: readonly SemanticMemoryNode[];
}) {
  const disclosureClass = edge.disclosure_class ?? edge.disclosure_label?.disclosure_class;
  const origins = originAudienceRefs(edge);

  return (
    <div className="item" style={{ padding: 12, border: "1px solid var(--line)" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <Tag kind={edge.relation === "contradicts" ? "warn" : "info"}>{edge.relation}</Tag>
        <DisclosureLabel value={disclosureClass} />
        <Tag>confidence {edge.confidence.toFixed(2)}</Tag>
        <Tag>{edge.invalidated_at === null ? "active" : "invalidated"}</Tag>
      </div>
      <div
        style={{
          marginTop: 8,
          color: "var(--text)",
          fontFamily: "var(--sans)",
          fontSize: 12.5,
          lineHeight: 1.45,
          overflowWrap: "anywhere",
        }}
      >
        {edgeEndpointLabel(edge.from_node_id, nodes)}
        {" -> "}
        {edgeEndpointLabel(edge.to_node_id, nodes)}
      </div>
      <div className="props" style={{ marginTop: 10 }}>
        <div className="row">
          <span className="k">edge id</span>
          <span className="v">
            <IdChip id={edge.id} type="semantic_edge" hint={edge} />
          </span>
        </div>
        <div className="row">
          <span className="k">from</span>
          <span className="v">
            <IdChip id={edge.from_node_id} type="semantic_node" />
          </span>
        </div>
        <div className="row">
          <span className="k">to</span>
          <span className="v">
            <IdChip id={edge.to_node_id} type="semantic_node" />
          </span>
        </div>
        <div className="row">
          <span className="k">valid from</span>
          <span className="v">{formatTimestamp(edge.valid_from)}</span>
        </div>
        <div className="row">
          <span className="k">valid to</span>
          <span className="v">
            {edge.valid_to === null ? "open" : formatTimestamp(edge.valid_to)}
          </span>
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
          <span className="k">evidence episodes</span>
          <span className="v">
            {edge.evidence_episode_ids.length === 0
              ? "none"
              : edge.evidence_episode_ids.map((id, index) => (
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
