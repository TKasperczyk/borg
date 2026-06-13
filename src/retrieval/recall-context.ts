import type { BorgRole } from "../memory/commitments/index.js";
import type { MemoryDisclosureLabel } from "../memory/common/disclosure-label.js";
import type { SessionAudienceRole } from "../sessions/index.js";
import type { EntityId, SessionId } from "../util/ids.js";

export {
  MEMORY_DISCLOSURE_INTERNAL_USE_NOTE,
  MEMORY_DISCLOSURE_CLASSES,
  SEMANTIC_SOURCE_DISCLOSURE_INTERNAL_USE_NOTE,
  combineDisclosureLabelForEpisodeIds,
  combineMemoryDisclosureLabels,
  memoryDisclosureInternalUseNote,
  memoryDisclosureLabelFromEpisodeAccess,
  memoryDisclosureLabelFromMetadata,
  memoryDisclosureLabelMetadata,
  memoryDisclosureLabelMetadataSchema,
  memoryDisclosureLabelSchema,
  publicMemoryDisclosureLabel,
  relationshipPrivateMemoryDisclosureLabel,
  renderMemoryDisclosureLabelForModel,
  renderSemanticSourceDisclosureLabelForModel,
  resolveDisclosureLabelsByEpisodeId,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
  type MemoryDisclosureClass,
  type MemoryDisclosureLabel,
  type MemoryDisclosureLabelRenderContext,
  type MemoryDisclosureLabelMetadata,
} from "../memory/common/disclosure-label.js";

/**
 * Memory disclosure label classes for recalled records. These are metadata for render/emission
 * judgment after global cognition recall; they are not predicates for hiding records from the being.
 * `operator_private` and `sensitive` are reserved classes for future per-record disclosure
 * policies and may appear in persisted review/tool metadata.
 */
/**
 * Label attached to recalled memory so render/disclosure can decide what may be said after recall.
 * It must not be used to hide records from the being's cognition.
 */
export const SELF_RECALL_SCOPE = "self" as const;

/**
 * Self-memory recall inversion epoch.
 *
 * Chosen boundary: 774580e at 2026-06-04T14:43:27Z ("global self-memory recall"),
 * because that commit made self-memory recall global for cognition. Nearby candidates
 * were e6c127d (recall/disclosure context split), 1845b28 (episodic ungating),
 * 2eac649 (semantic ungating), da1a90d (retrieval option split lock-in), and 4eabeb2
 * (later static self-architecture prompt edit). This constant labels provenance only;
 * it must not be used to hide, suppress, reorder, or retire memories.
 */
export const SELF_MEMORY_RECALL_INVERSION_EPOCH_MS = Date.parse("2026-06-04T14:43:27Z");

export type SelfMemoryExtractionEpochLabel =
  | "extracted_before_recall_inversion"
  | "extracted_after_recall_inversion"
  | "extraction_epoch_unknown";

export function selfMemoryExtractionEpochLabel(
  createdAtMs: number | null | undefined,
): SelfMemoryExtractionEpochLabel {
  if (createdAtMs === null || createdAtMs === undefined || !Number.isFinite(createdAtMs)) {
    return "extraction_epoch_unknown";
  }

  return createdAtMs < SELF_MEMORY_RECALL_INVERSION_EPOCH_MS
    ? "extracted_before_recall_inversion"
    : "extracted_after_recall_inversion";
}

export function renderSelfMemoryExtractionEpochLabel(
  createdAtMs: number | null | undefined,
): string {
  return `extraction_epoch=${selfMemoryExtractionEpochLabel(createdAtMs)}`;
}

export const MEMORY_DISCLOSURE_GUIDANCE_FOR_MODEL = [
  "Memory disclosure labels are input-side guidance for my reasoning with recalled memory.",
  "Some entries may be labeled relationship_private, operator_private, self_private, sensitive, or unknown, with private-to=<ids> and origin_audience=<ids> metadata.",
  "I use labeled-private memories internally to inform my judgment, empathy, caution, continuity, and uncertainty.",
  "I do not reveal labeled-private content, source details, or the existence of a private memory to the current audience unless the rendered disclosure policy, creator/operator context, or current audience authorization permits it.",
  "Operator or creator context may permit fuller discussion; I use the rendered authority and disclosure context to decide how much I can discuss.",
  "Disclosure permission and common-ground status are different: my being allowed to say something does not mean the current audience already knows it.",
  "A memory I can recall or disclose is not automatically common ground. I do not frame cross-session memory as already-shared knowledge unless the current audience is in its origin_audience, it is established as common ground for them, or there is evidence they have seen it.",
  "When disclosure is permitted but the current audience likely does not know a memory, I introduce it as new information rather than presuming it is shared.",
  "For a group audience, origin_audience means a memory was shared in that venue, not that every current participant saw it; if a participant's presence or prior exposure is uncertain, I do not assume it is common ground for them -- I introduce it as new or qualify the framing.",
].join("\n");

export type CognitionRecallContext = {
  readonly reader: typeof SELF_RECALL_SCOPE;
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
