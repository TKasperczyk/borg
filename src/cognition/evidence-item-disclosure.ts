import {
  renderMemoryDisclosureLabelForModel,
  unknownMemoryDisclosureLabel,
  type EvidenceItem,
  type MemoryDisclosureLabelRenderContext,
} from "../retrieval/index.js";

export function isSemanticEvidenceItem(item: EvidenceItem): boolean {
  return (
    item.source === "semantic_node" ||
    item.source === "semantic_edge" ||
    item.provenance?.nodeId !== undefined ||
    item.provenance?.edgeId !== undefined
  );
}

export function evidenceItemDisclosureRenderContext(
  item: EvidenceItem,
): MemoryDisclosureLabelRenderContext {
  return isSemanticEvidenceItem(item) ? "semantic_source" : "memory";
}

export function renderEvidenceItemDisclosureLabel(item: EvidenceItem): string {
  return renderMemoryDisclosureLabelForModel(item.disclosureLabel ?? unknownMemoryDisclosureLabel(), {
    context: evidenceItemDisclosureRenderContext(item),
  });
}
