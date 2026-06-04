import type { BorgRole } from "../memory/commitments/index.js";
import type { SessionAudienceRole } from "../sessions/index.js";
import type { EntityId, SessionId } from "../util/ids.js";

/**
 * Transitional Sprint-1 memory label classes for recalled records. This sits beside existing
 * disclosure vocabularies: observed-events `disclosure_class`, self-decisions `self_private`,
 * and creator-directives `content_scope` / allow-list policy. It is metadata for recall results,
 * not an authorization vocabulary; Sprint 5 must reconcile these into one disclosure-policy layer.
 */
export const MEMORY_DISCLOSURE_CLASSES = [
  "public",
  "relationship_private",
  "operator_private",
  "self_private",
  "sensitive",
  "unknown",
] as const;

export type MemoryDisclosureClass = (typeof MEMORY_DISCLOSURE_CLASSES)[number];

/**
 * Label attached to recalled memory so render/disclosure can decide what may be said after recall.
 * It must not be used to hide records from Sol cognition.
 */
export type MemoryDisclosureLabel = {
  readonly disclosureClass: MemoryDisclosureClass;
  readonly originAudienceEntityIds: readonly EntityId[];
  readonly privateToEntityIds: readonly EntityId[];
  readonly publicToEntityIds: readonly EntityId[];
};

export const MEMORY_DISCLOSURE_INTERNAL_USE_NOTE =
  "usable internally; do not disclose to current audience unless authorized";

export type MemoryDisclosureLabelMetadata = {
  disclosure_class: MemoryDisclosureClass;
  origin_audience_entity_ids: string[];
  private_to_entity_ids: string[];
  public_to_entity_ids: string[];
};

export function memoryDisclosureLabelMetadata(
  label: MemoryDisclosureLabel,
): MemoryDisclosureLabelMetadata {
  return {
    disclosure_class: label.disclosureClass,
    origin_audience_entity_ids: [...label.originAudienceEntityIds],
    private_to_entity_ids: [...label.privateToEntityIds],
    public_to_entity_ids: [...label.publicToEntityIds],
  };
}

export function renderMemoryDisclosureLabelForModel(label: MemoryDisclosureLabel): string {
  const fragments = [`disclosure_class=${label.disclosureClass}`];

  if (label.originAudienceEntityIds.length > 0) {
    fragments.push(`origin_audience=${label.originAudienceEntityIds.join(",")}`);
  }

  if (label.disclosureClass !== "public") {
    const privateTo =
      label.privateToEntityIds.length === 0 ? "unknown" : label.privateToEntityIds.join(",");
    fragments.push(`private-to=${privateTo}; ${MEMORY_DISCLOSURE_INTERNAL_USE_NOTE}`);
  }

  return fragments.join(" ");
}

export type CognitionRecallContext = {
  readonly reader: "sol";
  readonly currentSessionId: SessionId;
  readonly currentAudienceEntityId: EntityId | null;
  readonly currentParticipantEntityIds: readonly EntityId[];
};

export type DisclosureContext = {
  readonly currentSessionId: SessionId;
  readonly currentAudienceEntityId: EntityId | null;
  readonly audienceRole: SessionAudienceRole;
  readonly senderEntityId: EntityId | null;
  readonly senderRole: BorgRole | null;
  readonly participantEntityIds: readonly EntityId[];
  readonly isPrivateSelfCognition: boolean;
};

export function memoryDisclosureLabelFromEpisodeAccess(input: {
  readonly audience_entity_id?: EntityId | null;
  readonly shared?: boolean;
}): MemoryDisclosureLabel {
  const originAudienceEntityIds =
    input.audience_entity_id === null || input.audience_entity_id === undefined
      ? []
      : [input.audience_entity_id];

  if (input.audience_entity_id === null || input.audience_entity_id === undefined || input.shared) {
    return {
      disclosureClass: "public",
      originAudienceEntityIds,
      privateToEntityIds: [],
      publicToEntityIds: [],
    };
  }

  return {
    disclosureClass: "relationship_private",
    originAudienceEntityIds,
    privateToEntityIds: [input.audience_entity_id],
    publicToEntityIds: [],
  };
}
