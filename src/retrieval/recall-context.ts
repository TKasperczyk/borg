import type { BorgRole } from "../memory/commitments/index.js";
import {
  MEMORY_DISCLOSURE_CLASSES,
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureClass,
  type MemoryDisclosureLabel,
} from "../memory/common/disclosure-label.js";
import { normalizeEpisodeAccess, type EpisodeAccessLike } from "../memory/episodic/index.js";
import type { SessionAudienceRole } from "../sessions/index.js";
import type { EntityId, SessionId } from "../util/ids.js";

export {
  MEMORY_DISCLOSURE_CLASSES,
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureClass,
  type MemoryDisclosureLabel,
} from "../memory/common/disclosure-label.js";

/**
 * Memory disclosure label classes for recalled records. These are metadata for render/emission
 * judgment after global cognition recall; they are not predicates for hiding records from Sol.
 * `operator_private` and `sensitive` are reserved classes for future per-record disclosure
 * policies and may appear in persisted review/tool metadata.
 */
/**
 * Label attached to recalled memory so render/disclosure can decide what may be said after recall.
 * It must not be used to hide records from Sol cognition.
 */
export const MEMORY_DISCLOSURE_INTERNAL_USE_NOTE =
  "usable internally; do not disclose to current audience unless authorized";
export const SEMANTIC_SOURCE_DISCLOSURE_INTERNAL_USE_NOTE =
  "supported by private source episodes; usable internally; do not reveal source details to current audience unless authorized";
export const MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL = [
  "Memory disclosure labels are input-side guidance for reasoning with recalled memory.",
  "Some entries may be labeled relationship_private, operator_private, self_private, sensitive, or unknown, with private-to=<ids> and origin_audience=<ids> metadata.",
  "Use labeled-private memories internally to inform judgment, empathy, caution, continuity, and uncertainty.",
  "Do not reveal labeled-private content, source details, or the existence of a private memory to the current audience unless the rendered disclosure policy, creator/operator context, or current audience authorization permits it.",
  "Operator or creator context may permit fuller discussion; use the rendered authority and disclosure context to decide how much can be discussed.",
].join("\n");

export type MemoryDisclosureLabelRenderContext = "memory" | "semantic_source";

const MEMORY_DISCLOSURE_CLASS_RESTRICTION_RANK = {
  public: 0,
  relationship_private: 1,
  operator_private: 2,
  self_private: 3,
  sensitive: 4,
  unknown: 5,
} as const satisfies Record<MemoryDisclosureClass, number>;

export type MemoryDisclosureLabelMetadata = {
  disclosure_class: MemoryDisclosureClass;
  origin_audience_entity_ids: EntityId[];
  private_to_entity_ids: EntityId[];
  public_to_entity_ids: EntityId[];
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

export function memoryDisclosureInternalUseNote(
  context: MemoryDisclosureLabelRenderContext = "memory",
): string {
  return context === "semantic_source"
    ? SEMANTIC_SOURCE_DISCLOSURE_INTERNAL_USE_NOTE
    : MEMORY_DISCLOSURE_INTERNAL_USE_NOTE;
}

export function renderMemoryDisclosureLabelForModel(
  label: MemoryDisclosureLabel,
  options: {
    context?: MemoryDisclosureLabelRenderContext;
  } = {},
): string {
  const fragments = [`disclosure_class=${label.disclosureClass}`];

  if (label.originAudienceEntityIds.length > 0) {
    fragments.push(`origin_audience=${label.originAudienceEntityIds.join(",")}`);
  }

  if (label.disclosureClass !== "public") {
    const privateTo =
      label.privateToEntityIds.length === 0 ? "unknown" : label.privateToEntityIds.join(",");
    fragments.push(`private-to=${privateTo}; ${memoryDisclosureInternalUseNote(options.context)}`);
  }

  return fragments.join(" ");
}

export function renderSemanticSourceDisclosureLabelForModel(label: MemoryDisclosureLabel): string {
  return renderMemoryDisclosureLabelForModel(label, { context: "semantic_source" });
}

type MemoryDisclosureEntityIdListKey =
  | "originAudienceEntityIds"
  | "privateToEntityIds"
  | "publicToEntityIds";

function mergeEntityIds(
  labels: readonly MemoryDisclosureLabel[],
  key: MemoryDisclosureEntityIdListKey,
) {
  // Canonical sorted order: a disclosure label's entity-id lists are conceptually sets, and
  // combined labels are persisted as JSON. Sorting makes the merged label deterministic
  // regardless of the (possibly racing) order the source labels were assembled in.
  return [...new Set(labels.flatMap((label) => label[key]))].sort();
}

export function combineMemoryDisclosureLabels(
  labels: readonly MemoryDisclosureLabel[],
): MemoryDisclosureLabel {
  if (labels.length === 0) {
    return {
      disclosureClass: "unknown",
      originAudienceEntityIds: [],
      privateToEntityIds: [],
      publicToEntityIds: [],
    };
  }

  const disclosureClass = labels.reduce((mostRestrictive, label) => {
    return MEMORY_DISCLOSURE_CLASS_RESTRICTION_RANK[label.disclosureClass] >
      MEMORY_DISCLOSURE_CLASS_RESTRICTION_RANK[mostRestrictive]
      ? label.disclosureClass
      : mostRestrictive;
  }, "public" as MemoryDisclosureClass);

  return {
    disclosureClass,
    originAudienceEntityIds: mergeEntityIds(labels, "originAudienceEntityIds"),
    privateToEntityIds: mergeEntityIds(labels, "privateToEntityIds"),
    publicToEntityIds: mergeEntityIds(labels, "publicToEntityIds"),
  };
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

export function memoryDisclosureLabelFromEpisodeAccess(
  input: EpisodeAccessLike,
): MemoryDisclosureLabel {
  const normalized = normalizeEpisodeAccess(input);
  const originAudienceEntityIds = normalized.origin_audience_entity_ids;

  if (originAudienceEntityIds.length === 0 && normalized.shared) {
    return publicMemoryDisclosureLabel();
  }

  if (originAudienceEntityIds.length === 0) {
    return unknownMemoryDisclosureLabel();
  }

  return {
    disclosureClass: "relationship_private",
    originAudienceEntityIds,
    privateToEntityIds: originAudienceEntityIds,
    publicToEntityIds: [],
  };
}
