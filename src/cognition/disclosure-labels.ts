import type { CommitmentRecord } from "../memory/commitments/index.js";
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

export function correctionDisclosureEntityIds(refs: Record<string, unknown>): EntityId[] {
  const origins = refs.origin_audience_entity_ids;

  if (Array.isArray(origins) && origins.every((origin) => typeof origin === "string")) {
    return [...new Set(origins)] as EntityId[];
  }

  return typeof refs.audience_entity_id === "string" ? [refs.audience_entity_id as EntityId] : [];
}
