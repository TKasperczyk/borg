import { z } from "zod";

import { type LLMClient, type LLMToolDefinition, toToolInputSchema } from "../../llm/index.js";
import {
  SHARED_STATE_ENTRY_KINDS,
  type SharedStateArtifact,
  type SharedStateCanonicalizes,
  type SharedStateEntryKind,
  type SharedStateRepository,
  type SharedStateSourceTrustRejectionReason,
  type SharedStateSourceTrustValidator,
} from "../../memory/decision-artifacts/index.js";
import type { Clock } from "../../util/clock.js";
import type {
  ActionId,
  CommitmentId,
  EntityId,
  GoalId,
  OpenQuestionId,
  SharedStateEntryId,
  StreamEntryId,
} from "../../util/ids.js";
import type { SharedStateRenderOptions } from "./render.js";
import type { SharedStatePromptSummaryOptions } from "./summary.js";
import type { SharedStateCommitmentCanonicalizationType } from "./commitment-canonicalization.js";
import type {
  SharedStateReconciliationRepositories,
  SharedStateSemanticBeliefRevisionDependencies,
} from "./reconciliation.js";
import type { TurnTracer } from "../tracing/tracer.js";

export const SHARED_STATE_TOOL_NAME = "EmitDecisionArtifactPatch";
const MAX_OPERATIONS_PER_COMPILE = 40;
export const MAX_PATCH_OUTPUT_TOKENS = 1536;
export const SHARED_STATE_PROMPT_WARNING_TOKEN_THRESHOLD = 35_000;
const DEFAULT_MAX_ACTIVE_SHARED_STATE_ENTRIES = 40;
const DEFAULT_SHARED_STATE_KIND_SOFT_CAPS = {
  locked: 24,
  live: 10,
  invalidated: 4,
  pending: 4,
  tentative: 2,
} as const satisfies Record<SharedStateEntryKind, number>;
const SHARED_STATE_LIFECYCLE_PRUNE_ORDER = [
  "tentative",
  "locked",
  "invalidated",
  "pending",
  "live",
] as const satisfies readonly SharedStateEntryKind[];
const sharedStateToolKindSchema = z.enum(SHARED_STATE_ENTRY_KINDS);
const sourceStreamEntryIdsSchema = z
  .array(z.string().trim().min(1))
  .describe("Stream entry ids that support this artifact operation.");
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

const addOperationSchema = z
  .object({
    type: z.literal("add"),
    kind: sharedStateToolKindSchema,
    text: z.string().trim().min(1),
    owner_entity_id: ownerEntityIdSchema,
    source_stream_entry_ids: sourceStreamEntryIdsSchema,
    canonicalizes: canonicalizesSchema,
  })
  .strict();

const updateOperationSchema = z
  .object({
    type: z.literal("update"),
    id: z.string().trim().min(1),
    kind: sharedStateToolKindSchema.optional(),
    text: z.string().trim().min(1).optional(),
    owner_entity_id: ownerEntityIdSchema,
    source_stream_entry_ids: sourceStreamEntryIdsSchema,
    canonicalizes: canonicalizesSchema,
  })
  .strict();

const replacementEntrySchema = z
  .object({
    kind: sharedStateToolKindSchema,
    text: z.string().trim().min(1),
    owner_entity_id: ownerEntityIdSchema,
    source_stream_entry_ids: sourceStreamEntryIdsSchema,
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

export const SHARED_STATE_TOOL = {
  name: SHARED_STATE_TOOL_NAME,
  description:
    "Compile durable shared state for this audience into additive, updating, superseding, or pruning operations.",
  inputSchema: toToolInputSchema(sharedStatePatchSchema),
} satisfies LLMToolDefinition;

export type EmitSharedStatePatch = z.infer<typeof sharedStatePatchSchema>;

export type SharedStateArtifactParticipantContext = {
  entityId: EntityId;
  displayName?: string | null;
};

export type SharedStateCanonicalizationCandidate = {
  id: string;
  text: string;
};

export type SharedStateCommitmentCanonicalizationCandidate =
  SharedStateCanonicalizationCandidate & {
    kind: string;
    type: SharedStateCommitmentCanonicalizationType;
    directive_family: string;
  };

export type SharedStateCanonicalizationCandidates = {
  goals?: readonly SharedStateCanonicalizationCandidate[];
  commitments?: readonly SharedStateCommitmentCanonicalizationCandidate[];
  actions?: readonly SharedStateCanonicalizationCandidate[];
  openQuestions?: readonly SharedStateCanonicalizationCandidate[];
};

export type SharedStateCompileDegradedReason =
  | "llm_unavailable"
  | "repository_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload"
  | "invalid_patch"
  | "repository_failed";

export type SharedStateLifecycleOptions = {
  maxActiveEntries?: number;
  kindSoftCaps?: Partial<Record<SharedStateEntryKind, number>>;
};

export type SharedStateLedgerMode = "delta" | "full_fallback";

export type CompileSharedStateArtifactInput = {
  llmClient?: LLMClient;
  model?: string;
  repository?: Pick<SharedStateRepository, "get" | "upsert">;
  audienceEntityId: EntityId;
  selfEntityId: EntityId;
  speakerEntityId?: EntityId | null;
  participants: readonly SharedStateArtifactParticipantContext[];
  currentUserMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  promptVisibleLedger: string;
  previousArtifact?: SharedStateArtifact | null;
  allowedSourceStreamEntryIds?: readonly StreamEntryId[];
  offLimitsSourceStreamEntryIds?: readonly StreamEntryId[];
  sourceTrustValidator?: SharedStateSourceTrustValidator;
  canonicalizationCandidates?: SharedStateCanonicalizationCandidates;
  reconciliation?: SharedStateReconciliationRepositories;
  semanticBeliefRevision?: Omit<SharedStateSemanticBeliefRevisionDependencies, "llmClient">;
  clock?: Clock;
  tracer?: TurnTracer;
  turnId?: string;
  turnCounter?: number;
  lifecycle?: SharedStateLifecycleOptions;
  renderOptions?: SharedStateRenderOptions;
  previousArtifactSummaryOptions?: SharedStatePromptSummaryOptions;
  ledgerMode?: SharedStateLedgerMode;
  onDegraded?: (reason: SharedStateCompileDegradedReason, error?: unknown) => Promise<void> | void;
};

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
    | "empty_update";
  operationType: ParsedPatchOperation["type"];
  operationIndex: number;
  sourceStreamEntryId?: string;
  sourceTrustReason?: SharedStateSourceTrustRejectionReason | "unknown";
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

export type AllowedCanonicalizationIds = {
  goalIds: ReadonlySet<GoalId>;
  commitmentIds: ReadonlySet<CommitmentId>;
  actionIds: ReadonlySet<ActionId>;
  openQuestionIds: ReadonlySet<OpenQuestionId>;
};

export class MissingSharedStateArtifactToolCallError extends Error {}
