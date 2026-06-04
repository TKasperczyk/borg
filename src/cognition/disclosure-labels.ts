import type { CommitmentRecord } from "../memory/commitments/index.js";
import type { GoalRecord, OpenQuestion } from "../memory/self/index.js";
import {
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelMetadata,
  relationshipPrivateMemoryDisclosureLabel,
  renderMemoryDisclosureLabelForModel,
  selfPrivateMemoryDisclosureLabel,
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

export function memoryDisclosurePayloadFields(label: MemoryDisclosureLabel): {
  disclosure: string;
  disclosure_label: ReturnType<typeof memoryDisclosureLabelMetadata>;
} {
  return {
    disclosure: renderMemoryDisclosureLabelForModel(label),
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

export function correctionMemoryDisclosureLabel(refs: Record<string, unknown>): MemoryDisclosureLabel {
  const origins = refs.origin_audience_entity_ids;

  if (
    origins !== undefined &&
    (!Array.isArray(origins) || !origins.every((origin) => typeof origin === "string"))
  ) {
    return combineMemoryDisclosureLabels([]);
  }

  return relationshipPrivateMemoryDisclosureLabel(correctionDisclosureEntityIds(refs));
}
