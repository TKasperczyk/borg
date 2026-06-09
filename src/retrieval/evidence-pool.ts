import type { EvidenceItem } from "./recall-types.js";

// Tunes truth priority for raw stream evidence.
const RAW_STREAM_TRUTH_RANK = 60;

// Tunes truth priority for episodes with direct stream provenance.
const EPISODE_WITH_SOURCE_TRUTH_RANK = 50;

// Tunes truth priority for episodes without direct stream provenance.
const EPISODE_WITHOUT_SOURCE_TRUTH_RANK = 40;

// Tunes truth priority for durable action-state evidence.
const DURABLE_STATE_TRUTH_RANK = 30;

// Tunes truth priority for semantic graph evidence.
const SEMANTIC_TRUTH_RANK = 20;

// Tunes truth priority for uncategorized evidence.
const DEFAULT_EVIDENCE_TRUTH_RANK = 10;

// Tunes truth priority for recent raw stream evidence.
const RECENT_RAW_STREAM_TRUTH_RANK = 5;

// Tunes truth priority for warm recall evidence.
const WARM_RECALL_TRUTH_RANK = 3;

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
    return RAW_STREAM_TRUTH_RANK;
  }

  if (item.source === "episode") {
    return item.provenance?.streamIds === undefined || item.provenance.streamIds.length === 0
      ? EPISODE_WITHOUT_SOURCE_TRUTH_RANK
      : EPISODE_WITH_SOURCE_TRUTH_RANK;
  }

  if (item.source === "commitment" || item.source === "open_question") {
    return DURABLE_STATE_TRUTH_RANK;
  }

  if (item.source === "semantic_node" || item.source === "semantic_edge") {
    return SEMANTIC_TRUTH_RANK;
  }

  if (item.source === "recent_raw_stream") {
    return RECENT_RAW_STREAM_TRUTH_RANK;
  }

  if (item.source === "warm_recall") {
    return WARM_RECALL_TRUTH_RANK;
  }

  return DEFAULT_EVIDENCE_TRUTH_RANK;
}
