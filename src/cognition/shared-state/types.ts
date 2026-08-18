import { z } from "zod";

import type {
  SharedStateCanonicalizes,
  SharedStateEntryKind,
  SharedStateSourceTrustRejectionReason,
} from "../../memory/shared-state/types.js";
import type { MemoryDisclosureLabelMetadata } from "../../memory/common/disclosure-label.js";
import type { RelationalSlot } from "../../memory/relational-slots/types.js";
import type {
  ActionId,
  CommitmentId,
  EntityId,
  GoalId,
  OpenQuestionId,
  SharedStateEntryId,
} from "../../util/ids.js";
import type { ParticipantRoster } from "../perception/types.js";
import {
  relationshipClaimSchema,
  type RelationshipClaim,
} from "../../memory/common/relationship-claims.js";
import type { SharedStateCommitmentCanonicalizationType } from "./commitment-canonicalization.js";
import { MAX_OPERATIONS_PER_COMPILE, SHARED_STATE_TOOL_ENTRY_KINDS } from "./constants.js";

const sharedStateToolKindSchema = z.enum(SHARED_STATE_TOOL_ENTRY_KINDS);
const sourceStreamEntryIdsSchema = z
  .array(z.string().trim().min(1))
  .describe("Stream entry ids that support this artifact operation.");
const relationshipClaimsSchema = z
  .array(relationshipClaimSchema)
  .optional()
  .default([])
  .describe("Sensitive interpersonal relationship claims asserted by this operation text.");
const stateKeySchema = z
  .string()
  .trim()
  .min(1)
  .describe(
    "Stable, domain-neutral dotted key for the shared-state dimension this entry belongs to.",
  );
// On an update the key is written, not matched -- the store sets state_key from this field, and a
// key-only change is material enough to survive the empty-update drop. Say so, because the shared
// description reads as "restate where this entry lives" and a wrong key is otherwise permanent.
const updateStateKeySchema = stateKeySchema.describe(
  "Stable, domain-neutral dotted key for the shared-state dimension this entry belongs to. On an update this value is written, not matched: repeating the entry's current key leaves it where it is, and a different key renames the entry in place -- same id, created_at, rank, body and supersede history, under the new name, with last_updated_at moved as for any update.",
);
export const canonicalizesSchema = z
  .object({
    goal_ids: z.array(z.string().trim().min(1)).optional(),
    commitment_ids: z.array(z.string().trim().min(1)).optional(),
    action_ids: z.array(z.string().trim().min(1)).optional(),
    open_question_ids: z.array(z.string().trim().min(1)).optional(),
  })
  .strict()
  .optional()
  .describe("Active shared state ids this locked artifact entry makes canonical.");
const ownerEntityIdSchema = z
  .string()
  .trim()
  .min(1)
  .nullable()
  .optional()
  .describe("Entity id for the owner of the decision, or null when there is no specific owner.");
const newKeyReasonSchema = z
  .string()
  .trim()
  .min(1)
  .optional()
  .describe(
    "For add operations using a never-seen state_key, a brief explanation of what distinct object or thread this new key represents.",
  );

const addOperationSchema = z
  .object({
    type: z.literal("add"),
    state_key: stateKeySchema,
    new_key_reason: newKeyReasonSchema,
    kind: sharedStateToolKindSchema,
    text: z.string().trim().min(1),
    owner_entity_id: ownerEntityIdSchema,
    source_stream_entry_ids: sourceStreamEntryIdsSchema,
    relationship_claims: relationshipClaimsSchema,
    canonicalizes: canonicalizesSchema,
  })
  .strict();

const updateOperationSchema = z
  .object({
    type: z.literal("update"),
    id: z.string().trim().min(1),
    state_key: updateStateKeySchema,
    kind: sharedStateToolKindSchema.optional(),
    text: z.string().trim().min(1).optional(),
    owner_entity_id: ownerEntityIdSchema,
    source_stream_entry_ids: sourceStreamEntryIdsSchema,
    relationship_claims: relationshipClaimsSchema,
    canonicalizes: canonicalizesSchema,
  })
  .strict();

const replacementEntrySchema = z
  .object({
    state_key: stateKeySchema,
    kind: sharedStateToolKindSchema,
    text: z.string().trim().min(1),
    owner_entity_id: ownerEntityIdSchema,
    source_stream_entry_ids: sourceStreamEntryIdsSchema,
    relationship_claims: relationshipClaimsSchema,
  })
  .strict();

const supersedeOperationSchema = z
  .object({
    type: z.literal("supersede"),
    id: z.string().trim().min(1),
    replacement: replacementEntrySchema,
    source_stream_entry_ids: sourceStreamEntryIdsSchema.optional(),
    canonicalizes: canonicalizesSchema,
  })
  .strict();

const pruneOperationSchema = z
  .object({
    type: z.literal("prune"),
    id: z.string().trim().min(1),
    reason: z.string().trim().min(1).optional(),
  })
  .strict();

export const sharedStatePatchSchema = z
  .object({
    operations: z
      .array(
        z.discriminatedUnion("type", [
          addOperationSchema,
          updateOperationSchema,
          supersedeOperationSchema,
          pruneOperationSchema,
        ]),
      )
      .max(MAX_OPERATIONS_PER_COMPILE),
  })
  .strict();

export type EmitSharedStatePatch = z.input<typeof sharedStatePatchSchema>;
export type EmitDecisionArtifactPatch = EmitSharedStatePatch;

export type SharedStateArtifactParticipantContext = {
  entityId: EntityId;
  displayName?: string | null;
};

export type SharedStateArtifactAudienceContext = {
  entityId: EntityId;
  displayName?: string | null;
  kind?: "person" | "group" | "self" | "abstract" | null;
};

export type SharedStateCanonicalizationCandidate = {
  id: string;
  text: string;
  disclosure: string;
  disclosure_label: MemoryDisclosureLabelMetadata;
};

export type SharedStateActionCanonicalizationCandidate = SharedStateCanonicalizationCandidate & {
  actor?: string;
  state?: string;
  session_scope?: string | null;
};

export type SharedStateCommitmentCanonicalizationCandidate =
  SharedStateCanonicalizationCandidate & {
    kind: string;
    type: SharedStateCommitmentCanonicalizationType;
    directive_family: string;
    enforcement_class: string;
  };

export type SharedStateCanonicalizationCandidates = {
  goals?: readonly SharedStateCanonicalizationCandidate[];
  commitments?: readonly SharedStateCommitmentCanonicalizationCandidate[];
  actions?: readonly SharedStateActionCanonicalizationCandidate[];
  openQuestions?: readonly SharedStateCanonicalizationCandidate[];
};

export type SharedStateRelationalSlotContext = Pick<
  RelationalSlot,
  | "id"
  | "subject_entity_id"
  | "slot_key"
  | "value"
  | "state"
  | "evidence_stream_entry_ids"
  | "contradicted_by_stream_entry_ids"
  | "alternate_values"
>;

export type SharedStateCompileDegradedReason =
  | "llm_unavailable"
  | "repository_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload"
  | "invalid_patch"
  | "all_operations_rejected"
  | "repository_failed";

export type SharedStateLifecycleOptions = {
  maxActiveEntries?: number;
  maxLiveEntriesPerKey?: number;
  kindSoftCaps?: Partial<Record<SharedStateEntryKind, number>>;
  newestStateChangeReservedSlots?: number;
  recentTurnThreshold?: number;
  dormantTurnThreshold?: number;
};

export type SharedStateLedgerMode = "delta" | "full_fallback";

export type ParsedPatchOperation = EmitSharedStatePatch["operations"][number];
export type ParsedCanonicalizes = NonNullable<z.infer<typeof canonicalizesSchema>>;

export type PatchRejection = {
  reason:
    | "invalid_entry_id"
    | "unknown_entry_id"
    | "invalid_owner_entity_id"
    | "unsupported_kind"
    | "missing_citation"
    | "invalid_source_stream_entry_id"
    | "disallowed_source_stream_entry_id"
    | "quarantined_source_stream_entry_id"
    | "inactive_source_stream_entry_id"
    | "empty_update"
    | "live_entry_cap_exceeded_for_key"
    | "locked_state_key_collision"
    | "near_duplicate_state_key"
    | "missing_new_key_reason"
    | "relationship_claim_ungrounded";
  operationType: ParsedPatchOperation["type"];
  operationIndex: number;
  // The kind the operation asked for, not the kind anything ended up with. Only the store records
  // kinds, and the store only holds what landed, so without this a refused operation loses the one
  // property that says what it was trying to be.
  entryKind?: SharedStateEntryKind;
  sourceStreamEntryId?: string;
  sourceTrustReason?: SharedStateSourceTrustRejectionReason | "unknown";
  stateKey?: string;
  currentCount?: number;
  proposedCount?: number;
  maxLiveEntriesPerKey?: number;
  targetEntryId?: string;
  lockedEntryIds?: string[];
  similarStateKeys?: string[];
  sharedStateKeyTokens?: string[];
  relationshipClaims?: RelationshipClaim[];
  ungroundedRelationshipClaims?: RelationshipClaim[];
  rejectedRelationshipClaimEvidenceRelationalSlotIds?: string[];
  rejectedRelationshipClaimEvidenceStreamEntryIds?: Array<{ id: string; reason: string }>;
};

export type CanonicalizeIdChannel = "goal" | "commitment" | "action" | "open_question";

export type DroppedCanonicalizeId = {
  channel: CanonicalizeIdChannel;
  id: string;
  reason: "invalid_id" | "unknown_id";
  operationType: ParsedPatchOperation["type"];
  operationIndex: number;
};

export type CanonicalizationDuplicateDrop = {
  artifact_entry_id: SharedStateEntryId;
  kind: SharedStateEntryKind;
  dropped_ids: SharedStateCanonicalizes;
};

export type NonLockedCanonicalizesDrop = {
  operation_index: number;
  kind: SharedStateEntryKind;
  dropped_ids: {
    goal_ids: string[];
    commitment_ids: string[];
    action_ids: string[];
    open_question_ids: string[];
  };
};

export type EmptyUpdateDrop = {
  operationIndex: number;
  operationId: SharedStateEntryId;
  stateKey: string | null;
  fieldPresence: {
    kind: boolean;
    text: boolean;
    owner_entity_id: boolean;
    canonicalizes: boolean;
  };
};

// A prune is an unqualified delete with no tombstone: once it lands, the row is gone
// and the only thing that could say why is the reason the model attached to the
// operation. Normalization drops that reason on the way to the store, so without this
// the trace cannot tell a deliberate retraction from the lifecycle cap's forced
// eviction -- both surface as one fewer entry. Kept as the model wrote it; the reason
// is optional in the tool schema, so null here means "pruned without saying why".
export type ModelPruneRequest = {
  operationIndex: number;
  operationId: SharedStateEntryId;
  stateKey: string | null;
  kind: SharedStateEntryKind;
  lastUpdatedAt: number;
  reason: string | null;
};

export type AllowedCanonicalizationIds = {
  goalIds: ReadonlySet<GoalId>;
  commitmentIds: ReadonlySet<CommitmentId>;
  actionIds: ReadonlySet<ActionId>;
  openQuestionIds: ReadonlySet<OpenQuestionId>;
};

export type CompileSharedStateArtifactInputBase = {
  audienceEntityId: EntityId;
  currentAudience?: SharedStateArtifactAudienceContext | null;
  selfEntityId: EntityId;
  speakerEntityId?: EntityId | null;
  participants: readonly SharedStateArtifactParticipantContext[];
  participantRoster?: ParticipantRoster | null;
  currentUserMessage: string;
};
