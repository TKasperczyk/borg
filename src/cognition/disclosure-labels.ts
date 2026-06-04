import type { ImagePerceptionRecord } from "../attachments/index.js";
import type { CommitmentRecord } from "../memory/commitments/index.js";
import type { Episode, EpisodicRepository } from "../memory/episodic/index.js";
import {
  parseIdentityEventDisclosureSources,
  type IdentityEvent,
} from "../memory/identity/index.js";
import type { SharedStateEntry } from "../memory/decision-artifacts/index.js";
import type { RelationalSlot } from "../memory/relational-slots/index.js";
import type { SemanticEdge, SemanticNode } from "../memory/semantic/index.js";
import type { ActionRecord } from "../memory/actions/index.js";
import type { GoalRecord, OpenQuestion } from "../memory/self/index.js";
import {
  MEMORY_DISCLOSURE_CLASSES,
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelMetadata,
  relationshipPrivateMemoryDisclosureLabel,
  renderMemoryDisclosureLabelForModel,
  renderSemanticSourceDisclosureLabelForModel,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureLabel,
} from "../retrieval/recall-context.js";
import type { EntityId } from "../util/ids.js";

export function uniqueDisclosureEntityIds(
  entityIds: readonly (EntityId | null | undefined)[],
): EntityId[] {
  return [...new Set(entityIds.filter((entityId): entityId is EntityId => entityId != null))];
}

export function commitmentDisclosureEntityIds(
  commitment: Pick<CommitmentRecord, "restricted_audience" | "made_to_entity">,
): EntityId[] {
  return uniqueDisclosureEntityIds([commitment.restricted_audience, commitment.made_to_entity]);
}

export function commitmentMemoryDisclosureLabel(
  commitment: Pick<CommitmentRecord, "restricted_audience" | "made_to_entity">,
): MemoryDisclosureLabel {
  return relationshipPrivateMemoryDisclosureLabel(commitmentDisclosureEntityIds(commitment));
}

export function goalMemoryDisclosureLabel(
  goal: Pick<GoalRecord, "owner_entity_id"> & { audience_entity_id?: EntityId | null },
): MemoryDisclosureLabel {
  const entityIds = uniqueDisclosureEntityIds([
    goal.audience_entity_id ?? null,
    goal.owner_entity_id,
  ]);

  return entityIds.length === 0
    ? selfPrivateMemoryDisclosureLabel()
    : relationshipPrivateMemoryDisclosureLabel(entityIds);
}

export function openQuestionMemoryDisclosureLabel(
  question: Pick<OpenQuestion, "audience_entity_id">,
): MemoryDisclosureLabel {
  const entityIds = uniqueDisclosureEntityIds([question.audience_entity_id]);

  return entityIds.length === 0
    ? selfPrivateMemoryDisclosureLabel()
    : relationshipPrivateMemoryDisclosureLabel(entityIds);
}

export function actionMemoryDisclosureLabel(
  action: Pick<ActionRecord, "actor" | "audience_entity_id">,
): MemoryDisclosureLabel {
  const entityIds = uniqueDisclosureEntityIds([action.audience_entity_id]);

  if (entityIds.length > 0) {
    return relationshipPrivateMemoryDisclosureLabel(entityIds);
  }

  return action.actor === "borg"
    ? selfPrivateMemoryDisclosureLabel()
    : unknownMemoryDisclosureLabel();
}

export function observedEventMemoryDisclosureLabel(event: {
  disclosureClass: "social_observed" | "self_private";
  speakerEntityId: EntityId | null;
  audienceEntityId: EntityId | null;
}): MemoryDisclosureLabel {
  const originIds = uniqueDisclosureEntityIds([event.audienceEntityId, event.speakerEntityId]);

  return event.disclosureClass === "self_private"
    ? selfPrivateMemoryDisclosureLabel(originIds)
    : relationshipPrivateMemoryDisclosureLabel(originIds);
}

export function sharedStateMemoryDisclosureLabel(
  entry: Pick<SharedStateEntry, "audience_entity_id" | "owner_entity_id">,
): MemoryDisclosureLabel {
  return relationshipPrivateMemoryDisclosureLabel(
    uniqueDisclosureEntityIds([entry.audience_entity_id, entry.owner_entity_id]),
  );
}

export function relationalSlotMemoryDisclosureLabel(
  slot: Pick<RelationalSlot, "subject_entity_id">,
): MemoryDisclosureLabel {
  return relationshipPrivateMemoryDisclosureLabel([slot.subject_entity_id]);
}

export function imagePerceptionMemoryDisclosureLabel(
  record: Pick<ImagePerceptionRecord, "audience_entity_id">,
): MemoryDisclosureLabel {
  return relationshipPrivateMemoryDisclosureLabel([record.audience_entity_id]);
}

async function resolveEpisodeSourceDisclosureLabel(
  episodeIds: readonly Episode["id"][],
  options: { episodicRepository?: Pick<EpisodicRepository, "getMany"> },
): Promise<MemoryDisclosureLabel | null> {
  if (episodeIds.length === 0) {
    return null;
  }
  if (options.episodicRepository === undefined) {
    return unknownMemoryDisclosureLabel();
  }

  const episodes = await options.episodicRepository.getMany(episodeIds);
  const labelsByEpisodeId = new Map(
    episodes.map((episode) => [episode.id, memoryDisclosureLabelFromEpisodeAccess(episode)]),
  );
  return combineMemoryDisclosureLabels(
    episodeIds.map(
      (episodeId) => labelsByEpisodeId.get(episodeId) ?? unknownMemoryDisclosureLabel(),
    ),
  );
}

export async function identityEventMemoryDisclosureLabel(
  event: IdentityEvent,
  options: { episodicRepository?: Pick<EpisodicRepository, "getMany"> } = {},
): Promise<MemoryDisclosureLabel> {
  const sources = parseIdentityEventDisclosureSources(event);
  const labels: MemoryDisclosureLabel[] = [
    ...sources.disclosureLabels,
    ...sources.episodeAccesses.map((access) => memoryDisclosureLabelFromEpisodeAccess(access)),
    ...sources.commitmentAccesses.map((commitment) => commitmentMemoryDisclosureLabel(commitment)),
  ];

  if (sources.audienceEntityIds.length > 0) {
    labels.push(relationshipPrivateMemoryDisclosureLabel(sources.audienceEntityIds));
  }
  if (sources.malformed) {
    labels.push(unknownMemoryDisclosureLabel());
  }

  const sourceEpisodeLabel = await resolveEpisodeSourceDisclosureLabel(
    sources.sourceEpisodeIds,
    options,
  );
  if (sourceEpisodeLabel !== null) {
    labels.push(sourceEpisodeLabel);
  }

  return labels.length === 0
    ? unknownMemoryDisclosureLabel()
    : combineMemoryDisclosureLabels(labels);
}

export function semanticSourceMemoryDisclosureLabel(
  labels: readonly MemoryDisclosureLabel[],
): MemoryDisclosureLabel {
  return combineMemoryDisclosureLabels(labels);
}

export function semanticNodeMemoryDisclosureLabel(
  labelsByEpisodeId: ReadonlyMap<string, MemoryDisclosureLabel>,
  node: Pick<SemanticNode, "source_episode_ids">,
): MemoryDisclosureLabel {
  return semanticSourceMemoryDisclosureLabel(
    node.source_episode_ids.map(
      (episodeId) => labelsByEpisodeId.get(episodeId) ?? unknownMemoryDisclosureLabel(),
    ),
  );
}

export function semanticEdgeMemoryDisclosureLabel(
  labelsByEpisodeId: ReadonlyMap<string, MemoryDisclosureLabel>,
  edge: Pick<SemanticEdge, "evidence_episode_ids">,
): MemoryDisclosureLabel {
  return semanticSourceMemoryDisclosureLabel(
    edge.evidence_episode_ids.map(
      (episodeId) => labelsByEpisodeId.get(episodeId) ?? unknownMemoryDisclosureLabel(),
    ),
  );
}

export function memoryDisclosurePayloadFields(label: MemoryDisclosureLabel): {
  disclosure: string;
  disclosure_label: ReturnType<typeof memoryDisclosureLabelMetadata>;
} {
  return {
    disclosure: renderMemoryDisclosureLabelForModel(label),
    disclosure_label: memoryDisclosureLabelMetadata(label),
  };
}

export function semanticSourceDisclosurePayloadFields(label: MemoryDisclosureLabel): {
  disclosure: string;
  disclosure_label: ReturnType<typeof memoryDisclosureLabelMetadata>;
} {
  return {
    disclosure: renderSemanticSourceDisclosureLabelForModel(label),
    disclosure_label: memoryDisclosureLabelMetadata(label),
  };
}

export function correctionDisclosureEntityIds(refs: Record<string, unknown>): EntityId[] {
  const origins = refs.origin_audience_entity_ids;

  if (Array.isArray(origins) && origins.every((origin) => typeof origin === "string")) {
    return [...new Set(origins)] as EntityId[];
  }

  return typeof refs.audience_entity_id === "string" ? [refs.audience_entity_id as EntityId] : [];
}

function isDisclosureClass(value: unknown): value is MemoryDisclosureLabel["disclosureClass"] {
  return (
    typeof value === "string" && (MEMORY_DISCLOSURE_CLASSES as readonly string[]).includes(value)
  );
}

function metadataEntityIds(value: unknown): EntityId[] | null {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
    ? ([...new Set(value)] as EntityId[])
    : null;
}

export function memoryDisclosureLabelFromMetadata(value: unknown): MemoryDisclosureLabel | null {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  const record = value as Record<string, unknown>;
  const disclosureClass = record.disclosure_class;
  const originAudienceEntityIds = metadataEntityIds(record.origin_audience_entity_ids);
  const privateToEntityIds = metadataEntityIds(record.private_to_entity_ids);
  const publicToEntityIds = metadataEntityIds(record.public_to_entity_ids);

  if (
    !isDisclosureClass(disclosureClass) ||
    originAudienceEntityIds === null ||
    privateToEntityIds === null ||
    publicToEntityIds === null
  ) {
    return unknownMemoryDisclosureLabel();
  }

  return {
    disclosureClass,
    originAudienceEntityIds,
    privateToEntityIds,
    publicToEntityIds,
  };
}

export function correctionMemoryDisclosureLabel(
  refs: Record<string, unknown>,
): MemoryDisclosureLabel {
  const metadataLabel = memoryDisclosureLabelFromMetadata(refs.disclosure_label);

  if (metadataLabel !== null) {
    return metadataLabel;
  }

  const origins = refs.origin_audience_entity_ids;

  if (
    origins !== undefined &&
    (!Array.isArray(origins) || !origins.every((origin) => typeof origin === "string"))
  ) {
    return unknownMemoryDisclosureLabel();
  }

  return relationshipPrivateMemoryDisclosureLabel(correctionDisclosureEntityIds(refs));
}
