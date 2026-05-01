import type { EvidenceItem } from "./recall-types.js";

export function rankEvidenceItems(items: readonly EvidenceItem[]): EvidenceItem[] {
  return [...dedupeEvidenceItems(items)].sort(compareEvidenceItems);
}

function dedupeEvidenceItems(items: readonly EvidenceItem[]): EvidenceItem[] {
  const byKey = new Map<string, EvidenceItem>();

  for (const item of items) {
    const key = evidenceDedupeKey(item);
    const current = byKey.get(key);

    if (current === undefined || compareEvidenceItems(item, current) < 0) {
      byKey.set(key, item);
    }
  }

  return [...byKey.values()];
}

function evidenceDedupeKey(item: EvidenceItem): string {
  const provenance = item.provenance;

  if (provenance?.episodeId !== undefined) {
    return `episode:${provenance.episodeId}`;
  }

  if (provenance?.streamIds !== undefined && provenance.streamIds.length > 0) {
    return `raw_stream:${provenance.streamIds.join("|")}`;
  }

  if (provenance?.commitmentId !== undefined) {
    return `commitment:${provenance.commitmentId}`;
  }

  if (provenance?.openQuestionId !== undefined) {
    return `open_question:${provenance.openQuestionId}`;
  }

  if (provenance?.edgeId !== undefined) {
    return `semantic_edge:${provenance.edgeId}`;
  }

  if (provenance?.nodeId !== undefined) {
    return `semantic_node:${provenance.nodeId}`;
  }

  return `${item.source}:${item.id}`;
}

function compareEvidenceItems(left: EvidenceItem, right: EvidenceItem): number {
  return (
    evidenceTruthRank(right) - evidenceTruthRank(left) ||
    right.score - left.score ||
    right.id.localeCompare(left.id)
  );
}

function evidenceTruthRank(item: EvidenceItem): number {
  if (item.source === "raw_stream") {
    return 60;
  }

  if (item.source === "episode") {
    return item.provenance?.streamIds === undefined || item.provenance.streamIds.length === 0
      ? 40
      : 50;
  }

  if (item.source === "commitment" || item.source === "open_question") {
    return 30;
  }

  if (item.source === "semantic_node" || item.source === "semantic_edge") {
    return 20;
  }

  if (item.source === "recent_raw_stream") {
    return 5;
  }

  if (item.source === "warm_recall") {
    return 3;
  }

  return 10;
}
