import type { SemanticMemoryEdge, SemanticMemoryNode } from "../api/types";
import { formatTime } from "../lib/stream-utils";
import { shortId } from "../screens/screen-utils";
import { IdChip } from "./Inspector/IdChip";
import { Tag } from "./Tag";

function edgeEndpointLabel(id: string, nodes: readonly SemanticMemoryNode[]): string {
  return nodes.find((node) => node.id === id)?.label ?? shortId(id);
}

export function SemanticEdgeDetail({
  edge,
  nodes,
}: {
  edge: SemanticMemoryEdge;
  nodes: readonly SemanticMemoryNode[];
}) {
  return (
    <div className="item" style={{ padding: 12, border: "1px solid var(--line)" }}>
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <Tag kind={edge.relation === "contradicts" ? "warn" : "info"}>{edge.relation}</Tag>
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
          <span className="v">{formatTime(edge.valid_from)}</span>
        </div>
        <div className="row">
          <span className="k">valid to</span>
          <span className="v">{edge.valid_to === null ? "open" : formatTime(edge.valid_to)}</span>
        </div>
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
