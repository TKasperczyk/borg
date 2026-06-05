import type { BorgRole } from "../memory/commitments/index.js";
import type { MemoryDisclosureLabel } from "../memory/common/disclosure-label.js";
import type { SessionAudienceRole } from "../sessions/index.js";
import type { EntityId, SessionId } from "../util/ids.js";

export {
  MEMORY_DISCLOSURE_CLASSES,
  combineDisclosureLabelForEpisodeIds,
  combineMemoryDisclosureLabels,
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelFromMetadata,
  memoryDisclosureLabelMetadata,
  memoryDisclosureLabelMetadataSchema,
  memoryDisclosureLabelSchema,
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
  resolveDisclosureLabelsByEpisodeId,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureClass,
  type MemoryDisclosureLabel,
  type MemoryDisclosureLabelMetadata,
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
  "Disclosure permission and common-ground status are different: being allowed to say something does not mean the current audience already knows it.",
  "A memory being recallable or discloseable does not make it common ground. Do not frame cross-session memory as already-shared knowledge unless the current audience is in its origin_audience, it is established as common ground for them, or there is evidence they have seen it.",
  "When disclosure is permitted but the current audience likely does not know a memory, introduce it as new information rather than presuming it is shared.",
  "For a group audience, origin_audience means a memory was shared in that venue, not that every current participant saw it; if a participant's presence or prior exposure is uncertain, do not assume it is common ground for them -- introduce it as new or qualify the framing.",
].join("\n");

export type MemoryDisclosureLabelRenderContext = "memory" | "semantic_source";

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
