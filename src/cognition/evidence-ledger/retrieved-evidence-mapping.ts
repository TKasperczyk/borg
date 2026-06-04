import {
  MEMORY_DISCLOSURE_INTERNAL_USE_NOTE,
  SEMANTIC_SOURCE_DISCLOSURE_INTERNAL_USE_NOTE,
  memoryDisclosureLabelMetadata,
  renderMemoryDisclosureLabelForModel,
  renderSemanticSourceDisclosureLabelForModel,
  type EvidenceItem,
} from "../../retrieval/index.js";
import {
  combineScopes,
  scopeFromEpisodeIds,
  scopeFromStreamIds,
  type ScopeResolver,
} from "./scope-resolver.js";
import type { EvidenceLedgerSessionScope, EvidenceLedgerSourceType } from "./types.js";

export function rawStreamSourceType(scope: EvidenceLedgerSessionScope): EvidenceLedgerSourceType {
  if (scope === "current_session") {
    return "current_session_stream";
  }

  if (scope === "prior_session") {
    return "prior_session_stream";
  }

  return "system_metadata";
}

export function evidenceItemSourceType(
  item: EvidenceItem,
  scope: EvidenceLedgerSessionScope,
): EvidenceLedgerSourceType {
  if (item.provenance?.attachmentId !== undefined || item.source === "image_perception") {
    return "image_attachment";
  }

  if (item.provenance?.streamIds !== undefined && item.provenance.streamIds.length > 0) {
    return rawStreamSourceType(scope);
  }

  if (item.provenance?.episodeId !== undefined || item.source === "episode") {
    return "episode";
  }

  if (item.provenance?.nodeId !== undefined || item.source === "semantic_node") {
    return "semantic_node";
  }

  if (item.provenance?.edgeId !== undefined || item.source === "semantic_edge") {
    return "semantic_edge";
  }

  if (item.provenance?.commitmentId !== undefined || item.source === "commitment") {
    return "commitment";
  }

  return "system_metadata";
}

export function evidenceItemScope(
  item: EvidenceItem,
  resolver: ScopeResolver,
): EvidenceLedgerSessionScope {
  return combineScopes([
    scopeFromStreamIds(item.provenance?.streamIds, resolver),
    scopeFromEpisodeIds(
      [
        ...(item.provenance?.episodeId === undefined ? [] : [item.provenance.episodeId]),
        ...(item.source_episode_ids ?? []),
      ],
      resolver,
    ),
  ]);
}

export function evidenceItemState(item: EvidenceItem): string {
  const parts = [
    `score=${item.score.toFixed(2)}`,
    `intent=${item.recallIntentId}`,
    item.matchedTerms.length === 0 ? null : `terms=${item.matchedTerms.join(", ")}`,
    item.source_episode_ids === undefined || item.source_episode_ids.length === 0
      ? null
      : `sources=${item.source_episode_ids.slice(0, 3).join(", ")}`,
    item.partial_source_visibility === true ? "partial_sources=true" : null,
    item.source_visibility_fraction === undefined
      ? null
      : `visible_fraction=${item.source_visibility_fraction.toFixed(2)}`,
    item.imageUnavailableReason === undefined
      ? null
      : `image_unavailable=${item.imageUnavailableReason}`,
    item.disclosureLabel === undefined ? null : renderEvidenceItemDisclosureLabel(item),
  ].filter((part): part is string => part !== null);

  return parts.join(" ");
}

function renderEvidenceItemDisclosureLabel(item: EvidenceItem): string {
  if (item.disclosureLabel === undefined) {
    return "";
  }

  return isSemanticEvidenceItem(item)
    ? renderSemanticSourceDisclosureLabelForModel(item.disclosureLabel)
    : renderMemoryDisclosureLabelForModel(item.disclosureLabel);
}

function evidenceItemDisclosureNote(item: EvidenceItem): string {
  return isSemanticEvidenceItem(item)
    ? SEMANTIC_SOURCE_DISCLOSURE_INTERNAL_USE_NOTE
    : MEMORY_DISCLOSURE_INTERNAL_USE_NOTE;
}

function isSemanticEvidenceItem(item: EvidenceItem): boolean {
  return (
    item.source === "semantic_node" ||
    item.source === "semantic_edge" ||
    item.provenance?.nodeId !== undefined ||
    item.provenance?.edgeId !== undefined
  );
}

export function evidenceItemProvenanceMetadata(
  item: EvidenceItem,
): Record<string, unknown> | undefined {
  const provenance = item.provenance;

  if (provenance === undefined) {
    return undefined;
  }

  return {
    ...(provenance.episodeId === undefined ? {} : { episode_id: provenance.episodeId }),
    ...(provenance.parentEpisodeId === undefined
      ? {}
      : { parent_episode_id: provenance.parentEpisodeId }),
    ...(provenance.nodeId === undefined ? {} : { node_id: provenance.nodeId }),
    ...(provenance.edgeId === undefined ? {} : { edge_id: provenance.edgeId }),
    ...(provenance.commitmentId === undefined ? {} : { commitment_id: provenance.commitmentId }),
    ...(provenance.openQuestionId === undefined
      ? {}
      : { open_question_id: provenance.openQuestionId }),
    ...(provenance.imagePerceptionId === undefined
      ? {}
      : { image_perception_id: provenance.imagePerceptionId }),
    ...(provenance.attachmentId === undefined ? {} : { attachment_id: provenance.attachmentId }),
    ...(item.citationType === undefined ? {} : { citation_type: item.citationType }),
    ...(provenance.streamIds === undefined || provenance.streamIds.length === 0
      ? {}
      : { stream_ids: provenance.streamIds }),
    ...(item.disclosureLabel === undefined
      ? {}
      : {
          disclosure_label: memoryDisclosureLabelMetadata(item.disclosureLabel),
          ...(item.disclosureLabel.disclosureClass === "public"
            ? {}
            : { disclosure_note: evidenceItemDisclosureNote(item) }),
        }),
  };
}
