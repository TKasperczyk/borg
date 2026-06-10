import {
  memoryDisclosurePayloadFields,
  semanticEdgeMemoryDisclosureLabel,
  semanticNodeMemoryDisclosureLabel,
} from "../memory/common/disclosure-serializers.js";
import {
  memoryDisclosureLabelFromEpisodeAccess,
  resolveDisclosureLabelsByEpisodeId,
  type MemoryDisclosureLabel,
} from "../memory/common/disclosure-label.js";
import type { Episode } from "../memory/episodic/index.js";
import type { SemanticEdge, SemanticNode } from "../memory/semantic/index.js";
import { episodeEvidencePromptRow } from "./evidence-labels.js";
import type { OfflineContext } from "./types.js";

export type DisclosureLabeledTarget =
  | {
      type: "episode";
      content: Episode;
    }
  | {
      type: "semantic_node";
      content: SemanticNode;
    }
  | {
      type: "semantic_edge";
      content: SemanticEdge;
    };

type MemoryDisclosurePromptFields = ReturnType<typeof memoryDisclosurePayloadFields>;

export type DisclosureLabeledTargetPayload = MemoryDisclosurePromptFields & {
  type: DisclosureLabeledTarget["type"];
  content: Record<string, unknown>;
};

export async function disclosureLabelForLoadedReviewTarget(
  ctx: Pick<OfflineContext, "episodicRepository">,
  target: DisclosureLabeledTarget,
): Promise<MemoryDisclosureLabel> {
  if (target.type === "episode") {
    return memoryDisclosureLabelFromEpisodeAccess(target.content);
  }

  if (target.type === "semantic_node") {
    const labelsByEpisodeId = await resolveDisclosureLabelsByEpisodeId(
      target.content.source_episode_ids,
      (episodeIds) => ctx.episodicRepository.getMany(episodeIds),
    );

    return semanticNodeMemoryDisclosureLabel(labelsByEpisodeId, target.content);
  }

  const labelsByEpisodeId = await resolveDisclosureLabelsByEpisodeId(
    target.content.evidence_episode_ids,
    (episodeIds) => ctx.episodicRepository.getMany(episodeIds),
  );

  return semanticEdgeMemoryDisclosureLabel(labelsByEpisodeId, target.content);
}

export async function serializeDisclosureLabeledTargetPayload(
  ctx: Pick<OfflineContext, "episodicRepository">,
  target: DisclosureLabeledTarget,
): Promise<DisclosureLabeledTargetPayload> {
  const disclosureLabel = await disclosureLabelForLoadedReviewTarget(ctx, target);

  if (target.type === "episode") {
    return {
      type: target.type,
      content: episodeEvidencePromptRow(target.content, {
        participants: target.content.participants,
        location: target.content.location,
        start_time: target.content.start_time,
        end_time: target.content.end_time,
        source_stream_ids: target.content.source_stream_ids,
        significance: target.content.significance,
        tags: target.content.tags,
        confidence: target.content.confidence,
        emotional_arc: target.content.emotional_arc,
      }),
      ...memoryDisclosurePayloadFields(disclosureLabel),
    };
  }

  if (target.type === "semantic_node") {
    return {
      type: target.type,
      content: {
        id: target.content.id,
        kind: target.content.kind,
        label: target.content.label,
        description: target.content.description,
        aliases: target.content.aliases,
        confidence: target.content.confidence,
        source_episode_ids: target.content.source_episode_ids,
        archived: target.content.archived,
        superseded_by: target.content.superseded_by,
        ...memoryDisclosurePayloadFields(disclosureLabel),
      },
      ...memoryDisclosurePayloadFields(disclosureLabel),
    };
  }

  return {
    type: target.type,
    content: {
      id: target.content.id,
      from_node_id: target.content.from_node_id,
      to_node_id: target.content.to_node_id,
      relation: target.content.relation,
      confidence: target.content.confidence,
      evidence_episode_ids: target.content.evidence_episode_ids,
      created_at: target.content.created_at,
      last_verified_at: target.content.last_verified_at,
      valid_from: target.content.valid_from,
      valid_to: target.content.valid_to,
      invalidated_at: target.content.invalidated_at,
      invalidated_by_edge_id: target.content.invalidated_by_edge_id,
      invalidated_by_review_id: target.content.invalidated_by_review_id,
      invalidated_by_process: target.content.invalidated_by_process,
      invalidated_reason: target.content.invalidated_reason,
      ...memoryDisclosurePayloadFields(disclosureLabel),
    },
    ...memoryDisclosurePayloadFields(disclosureLabel),
  };
}
