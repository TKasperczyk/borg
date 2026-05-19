import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import {
  DECISION_ARTIFACT_ENTRY_KINDS,
  type DecisionArtifact,
  type DecisionArtifactCanonicalizes,
  type DecisionArtifactEntry,
  type DecisionArtifactEntryKind,
  type DecisionArtifactOperation,
  type DecisionArtifactRepository,
  type DecisionArtifactSourceTrustRejectionReason,
  type DecisionArtifactSourceTrustValidator,
} from "../../memory/decision-artifacts/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { JsonValue } from "../../util/json-value.js";
import { estimatePromptTokens } from "../../util/token-estimate.js";
import {
  actionIdHelpers,
  commitmentIdHelpers,
  createDecisionArtifactEntryId,
  decisionArtifactEntryIdHelpers,
  entityIdHelpers,
  goalIdHelpers,
  openQuestionIdHelpers,
  streamEntryIdHelpers,
  type ActionId,
  type CommitmentId,
  type DecisionArtifactEntryId,
  type EntityId,
  type GoalId,
  type OpenQuestionId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  summarizeDecisionStateArtifactRender,
  type DecisionArtifactRenderOptions,
} from "./render.js";
import {
  buildDecisionArtifactPromptSummary,
  type DecisionArtifactPromptSummary,
  type DecisionArtifactPromptSummaryOptions,
} from "./summary.js";
import { buildUsageTraceBlock, toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import {
  findUnsettledDecisionArtifactReconciliation,
  mergeSemanticBeliefRevisionResult,
  reconcileDecisionArtifactCanonicalizations,
  reconcileSemanticBeliefRevision,
  type DecisionArtifactSemanticBeliefRevisionDependencies,
  type DecisionArtifactReconciliationRepositories,
  type DecisionArtifactReconciliationResult,
  type DecisionArtifactUnsettledReconciliationSummary,
} from "./reconciliation.js";
import type { DecisionArtifactCommitmentCanonicalizationType } from "./commitment-canonicalization.js";

const DECISION_ARTIFACT_TOOL_NAME = "EmitDecisionArtifactPatch";
const MAX_OPERATIONS_PER_COMPILE = 40;
const MAX_PATCH_OUTPUT_TOKENS = 1536;
const DECISION_ARTIFACT_PROMPT_WARNING_TOKEN_THRESHOLD = 35_000;
const DEFAULT_MAX_ACTIVE_DECISION_ARTIFACT_ENTRIES = 40;
const DEFAULT_DECISION_ARTIFACT_KIND_SOFT_CAPS = {
  locked: 24,
  live: 10,
  invalidated: 4,
  pending: 4,
  tentative: 2,
} as const satisfies Record<DecisionArtifactEntryKind, number>;
const DECISION_ARTIFACT_LIFECYCLE_PRUNE_ORDER = [
  "tentative",
  "locked",
  "invalidated",
  "pending",
  "live",
] as const satisfies readonly DecisionArtifactEntryKind[];
const decisionArtifactToolKindSchema = z.enum(DECISION_ARTIFACT_ENTRY_KINDS);
const sourceStreamEntryIdsSchema = z
  .array(z.string().trim().min(1))
  .describe("Stream entry ids that support this artifact operation.");
const canonicalizesSchema = z
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
    kind: decisionArtifactToolKindSchema,
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
    kind: decisionArtifactToolKindSchema.optional(),
    text: z.string().trim().min(1).optional(),
    owner_entity_id: ownerEntityIdSchema,
    source_stream_entry_ids: sourceStreamEntryIdsSchema,
    canonicalizes: canonicalizesSchema,
  })
  .strict();

const replacementEntrySchema = z
  .object({
    kind: decisionArtifactToolKindSchema,
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

const decisionArtifactPatchSchema = z
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

const DECISION_ARTIFACT_TOOL = {
  name: DECISION_ARTIFACT_TOOL_NAME,
  description:
    "Compile durable shared state for this audience into additive, updating, superseding, or pruning operations.",
  inputSchema: toToolInputSchema(decisionArtifactPatchSchema),
} satisfies LLMToolDefinition;

const DECISION_ARTIFACT_SYSTEM_PROMPT = [
  "Compile canonical shared audience state from the previous artifact, the current user turn, and the prompt-visible ledger.",
  "Return only the required tool call.",
  "",
  "Scope:",
  "- This is a structure compiler. It does not decide whether Borg's answer is correct.",
  "- It does not rewrite, suppress, approve, or police user-facing text.",
  "- It must not invent facts, owners, or commitments.",
  "- Prefer no operations when uncertain.",
  "",
  "The decision artifact is the shared-state surface for this audience.",
  "It captures durable decisions and constraints for this audience, not tasks, commitments, observations, or social facts.",
  "",
  "What belongs:",
  '- locked: group-agreed durable decisions or constraints, such as "PR merge order: backend first, then frontend", "The party spared the duke", "No surprise visits this week", or "Granada is four nights".',
  "- live: currently-under-discussion shared-state decisions or constraints the audience is actively working on.",
  "- pending: items awaiting verification or external information before they can become durable shared state.",
  "- invalidated: assumptions explicitly overturned by later evidence; kept for context.",
  "- tentative: weak proposals not yet endorsed.",
  "",
  "What does not belong:",
  '- Participant tasks, such as "Alice will write the chapter 3 summary"; those stay as action records.',
  '- Assistant commitments, such as "Borg will send the status note tomorrow"; those stay as commitment records.',
  "- Observations or social facts such as mood, signoffs, or group dynamics; those stay in the semantic graph or stream.",
  "- Stream-level chitchat such as greetings, emoji closures, or thanks.",
  "",
  "Canonicalization handles:",
  "- The input includes active_goals, active_commitments, active_actions, and open_questions as compact id:text lists; active_commitments also include type and directive_family.",
  "- When a locked add, update, or supersede replacement makes listed active state redundant, set canonicalizes with only those exact ids.",
  "- For commitments, canonicalize only supplied active promises or rules that the locked artifact entry fully subsumes; never create an artifact entry merely to restate a commitment.",
  "- Never canonicalize preference or boundary commitments.",
  "- Do not canonicalize ids for live, pending, tentative, or invalidated entries.",
  "- Do not infer ids that are not supplied.",
  "",
  "Identity and ownership:",
  "- Cite stream ids for every add, update, and supersede replacement.",
  "- Preserve stable entry ids when updating or superseding existing entries.",
  "- Use only supplied entity ids. Use null when no owner is explicitly supplied.",
  "- The audience entity for a group chat is the audience itself.",
  "- The speaker entity is the sender of the current turn.",
  "- Borg is the self entity.",
  "",
  "Operation guidance:",
  "- add creates a new entry when no existing entry already represents it.",
  "- update modifies an existing entry while preserving its id.",
  "- supersede replaces an existing entry when the conversation changes or narrows it.",
  "- prune removes stale artifact clutter only when the supplied context makes it clearly obsolete.",
  "- If a similar entry already exists, prefer update or supersede instead of adding a new one.",
  "- Prefer update, supersede, and prune over add whenever the existing artifact already carries the relevant shared state.",
].join("\n");

export type EmitDecisionArtifactPatch = z.infer<typeof decisionArtifactPatchSchema>;

export type DecisionArtifactParticipantContext = {
  entityId: EntityId;
  displayName?: string | null;
};

export type DecisionArtifactCanonicalizationCandidate = {
  id: string;
  text: string;
};

export type DecisionArtifactCommitmentCanonicalizationCandidate =
  DecisionArtifactCanonicalizationCandidate & {
    type: DecisionArtifactCommitmentCanonicalizationType;
    directive_family: string;
  };

export type DecisionArtifactCanonicalizationCandidates = {
  goals?: readonly DecisionArtifactCanonicalizationCandidate[];
  commitments?: readonly DecisionArtifactCommitmentCanonicalizationCandidate[];
  actions?: readonly DecisionArtifactCanonicalizationCandidate[];
  openQuestions?: readonly DecisionArtifactCanonicalizationCandidate[];
};

export type DecisionArtifactCompileDegradedReason =
  | "llm_unavailable"
  | "repository_unavailable"
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload"
  | "invalid_patch"
  | "repository_failed";

export type DecisionArtifactLifecycleOptions = {
  maxActiveEntries?: number;
  kindSoftCaps?: Partial<Record<DecisionArtifactEntryKind, number>>;
};

export type DecisionArtifactLedgerMode = "delta" | "full_fallback";

export type CompileDecisionArtifactInput = {
  llmClient?: LLMClient;
  model?: string;
  repository?: Pick<DecisionArtifactRepository, "get" | "upsert">;
  audienceEntityId: EntityId;
  selfEntityId: EntityId;
  speakerEntityId?: EntityId | null;
  participants: readonly DecisionArtifactParticipantContext[];
  currentUserMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  promptVisibleLedger: string;
  previousArtifact?: DecisionArtifact | null;
  allowedSourceStreamEntryIds?: readonly StreamEntryId[];
  offLimitsSourceStreamEntryIds?: readonly StreamEntryId[];
  sourceTrustValidator?: DecisionArtifactSourceTrustValidator;
  canonicalizationCandidates?: DecisionArtifactCanonicalizationCandidates;
  reconciliation?: DecisionArtifactReconciliationRepositories;
  semanticBeliefRevision?: Omit<DecisionArtifactSemanticBeliefRevisionDependencies, "llmClient">;
  clock?: Clock;
  tracer?: TurnTracer;
  turnId?: string;
  turnCounter?: number;
  lifecycle?: DecisionArtifactLifecycleOptions;
  renderOptions?: DecisionArtifactRenderOptions;
  previousArtifactSummaryOptions?: DecisionArtifactPromptSummaryOptions;
  ledgerMode?: DecisionArtifactLedgerMode;
  onDegraded?: (
    reason: DecisionArtifactCompileDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

type ParsedPatchOperation = EmitDecisionArtifactPatch["operations"][number];
type ParsedCanonicalizes = NonNullable<z.infer<typeof canonicalizesSchema>>;

type PatchRejection = {
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
  sourceTrustReason?: DecisionArtifactSourceTrustRejectionReason | "unknown";
};

type CanonicalizeIdChannel = "goal" | "commitment" | "action" | "open_question";

export type DroppedCanonicalizeId = {
  channel: CanonicalizeIdChannel;
  id: string;
  reason: "invalid_id" | "unknown_id";
  operationType: ParsedPatchOperation["type"];
  operationIndex: number;
};

type CanonicalizationDuplicateDrop = {
  artifact_entry_id: DecisionArtifactEntryId;
  kind: DecisionArtifactEntryKind;
  dropped_ids: DecisionArtifactCanonicalizes;
};

type NonLockedCanonicalizesDrop = {
  operation_index: number;
  kind: DecisionArtifactEntryKind;
  dropped_ids: {
    goal_ids: string[];
    commitment_ids: string[];
    action_ids: string[];
    open_question_ids: string[];
  };
};

type AllowedCanonicalizationIds = {
  goalIds: ReadonlySet<GoalId>;
  commitmentIds: ReadonlySet<CommitmentId>;
  actionIds: ReadonlySet<ActionId>;
  openQuestionIds: ReadonlySet<OpenQuestionId>;
};

class MissingDecisionArtifactToolCallError extends Error {}

function emptyPatch(): EmitDecisionArtifactPatch {
  return { operations: [] };
}

function buildCanonicalizationCandidatePromptPayload(
  candidates: DecisionArtifactCanonicalizationCandidates,
): {
  active_goals: readonly DecisionArtifactCanonicalizationCandidate[];
  active_commitments: readonly DecisionArtifactCommitmentCanonicalizationCandidate[];
  active_actions: readonly DecisionArtifactCanonicalizationCandidate[];
  open_questions: readonly DecisionArtifactCanonicalizationCandidate[];
} {
  return {
    active_goals: candidates.goals ?? [],
    active_commitments: candidates.commitments ?? [],
    active_actions: candidates.actions ?? [],
    open_questions: candidates.openQuestions ?? [],
  };
}

function buildDecisionArtifactMessages(input: {
  audienceEntityId: EntityId;
  selfEntityId: EntityId;
  speakerEntityId: EntityId | null;
  participants: readonly DecisionArtifactParticipantContext[];
  currentUserMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  promptVisibleLedger: string;
  previousArtifactSummary: DecisionArtifactPromptSummary | null;
  canonicalizationCandidates: DecisionArtifactCanonicalizationCandidates;
  allowedSourceStreamEntryIds?: readonly StreamEntryId[];
  offLimitsSourceStreamEntryIds?: readonly StreamEntryId[];
}): LLMMessage[] {
  const canonicalizationCandidates = buildCanonicalizationCandidatePromptPayload(
    input.canonicalizationCandidates,
  );

  return [
    {
      role: "user",
      content: JSON.stringify({
        audience_entity_id: input.audienceEntityId,
        self_entity_id: input.selfEntityId,
        speaker_entity_id: input.speakerEntityId,
        participant_entities: input.participants.map((participant) => ({
          entity_id: participant.entityId,
          display_name: participant.displayName ?? null,
        })),
        current_user_turn: {
          stream_entry_id: input.currentUserStreamEntryId,
          text: input.currentUserMessage,
        },
        source_trust: {
          citation_eligible_source_stream_entry_id_count:
            input.allowedSourceStreamEntryIds?.length ?? null,
          off_limits_source_stream_entry_ids: input.offLimitsSourceStreamEntryIds ?? [],
        },
        previous_artifact_summary: input.previousArtifactSummary,
        canonicalization_candidates: canonicalizationCandidates,
        prompt_visible_ledger: input.promptVisibleLedger,
      }),
    },
  ];
}

type DecisionArtifactPromptBudget = {
  inputTokenEstimate: number;
  breakdown: Record<string, number>;
};

function estimateDecisionArtifactPromptBudget(input: {
  messages: readonly LLMMessage[];
  tools: readonly LLMToolDefinition[];
  previousArtifactSummary: DecisionArtifactPromptSummary | null;
  promptVisibleLedger: string;
  currentUserMessage: string;
  canonicalizationCandidates: DecisionArtifactCanonicalizationCandidates;
}): DecisionArtifactPromptBudget {
  const system = estimatePromptTokens(DECISION_ARTIFACT_SYSTEM_PROMPT);
  const toolSchema = estimatePromptTokens(JSON.stringify(input.tools));
  const previousArtifactSummary = estimatePromptTokens(
    JSON.stringify(input.previousArtifactSummary),
  );
  const promptVisibleLedger = estimatePromptTokens(input.promptVisibleLedger);
  const currentUserTurn = estimatePromptTokens(input.currentUserMessage);
  const canonicalizationCandidates = estimatePromptTokens(
    JSON.stringify(buildCanonicalizationCandidatePromptPayload(input.canonicalizationCandidates)),
  );
  const inputTokenEstimate =
    system +
    toolSchema +
    input.messages.reduce((sum, message) => sum + estimatePromptTokens(message.content), 0);
  const accounted =
    system +
    toolSchema +
    previousArtifactSummary +
    promptVisibleLedger +
    currentUserTurn +
    canonicalizationCandidates;

  return {
    inputTokenEstimate,
    breakdown: {
      system,
      tool_schema: toolSchema,
      previous_artifact_summary: previousArtifactSummary,
      prompt_visible_ledger: promptVisibleLedger,
      current_user_turn: currentUserTurn,
      canonicalization_candidates: canonicalizationCandidates,
      prompt_envelope: Math.max(0, inputTokenEstimate - accounted),
    },
  };
}

function parseResponse(result: LLMCompleteResult): EmitDecisionArtifactPatch {
  const call = result.tool_calls.find((toolCall) => toolCall.name === DECISION_ARTIFACT_TOOL_NAME);

  if (call === undefined) {
    throw new MissingDecisionArtifactToolCallError(
      `Decision artifact compiler did not emit ${DECISION_ARTIFACT_TOOL_NAME}`,
    );
  }

  const parsed = decisionArtifactPatchSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return parsed.data;
}

function summarizeDecisionArtifactResponseShape(response: LLMCompleteResult): JsonValue {
  return {
    textLength: response.text.length,
    toolUseBlocks: response.tool_calls.map((call) => ({
      id: call.id,
      name: call.name,
    })),
  };
}

function countCompletePromptChars(systemPrompt: string, messages: readonly LLMMessage[]): number {
  return (
    systemPrompt.length +
    messages.reduce((sum, message) => sum + message.role.length + message.content.length, 0)
  );
}

function summarizeToolSchemas(tools: readonly LLMToolDefinition[]): JsonValue {
  return tools.map((tool) => ({
    name: tool.name,
    propertyCount:
      tool.inputSchema.properties === undefined
        ? 0
        : Object.keys(tool.inputSchema.properties).length,
    required: Array.isArray(tool.inputSchema.required) ? tool.inputSchema.required.map(String) : [],
  }));
}

function traceLlmCallStarted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  model: string;
  messages: readonly LLMMessage[];
  tools: readonly LLMToolDefinition[];
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_started", {
      turnId: options.turnId,
      label: "decision_artifact_compiler",
      model: options.model,
      promptCharCount: countCompletePromptChars(DECISION_ARTIFACT_SYSTEM_PROMPT, options.messages),
      toolSchemas: summarizeToolSchemas(options.tools),
    });
  }
}

function traceLlmCallResponse(options: {
  tracer?: TurnTracer;
  turnId?: string;
  response: LLMCompleteResult;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_response", {
      turnId: options.turnId,
      label: "decision_artifact_compiler",
      responseShape: summarizeDecisionArtifactResponseShape(options.response),
      stopReason: options.response.stop_reason,
      usage: buildUsageTraceBlock(options.response),
    });
  }
}

function traceLlmCallError(options: {
  tracer?: TurnTracer;
  turnId?: string;
  error: unknown;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_response", {
      turnId: options.turnId,
      label: "decision_artifact_compiler",
      responseShape: {
        error: options.error instanceof Error ? options.error.message : String(options.error),
      },
      stopReason: null,
      usage: null,
    });
  }
}

function traceCompileCompleted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  previousEntryCount: number;
  operationCount: number;
  rejected: readonly PatchRejection[];
  applied: boolean;
  artifact: DecisionArtifact | null;
  renderOptions?: DecisionArtifactRenderOptions;
  prunedEntryCountThisTurn: number;
  supersededEntryCountThisTurn: number;
  ledgerMode: DecisionArtifactLedgerMode;
  promptBudget: DecisionArtifactPromptBudget;
  nonLockedCanonicalizesDrops?: readonly NonLockedCanonicalizesDrop[];
}): void {
  const artifactSummary = summarizeDecisionStateArtifactRender(
    options.artifact,
    options.renderOptions,
  );

  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("decision_artifact_compile_completed", {
      turnId: options.turnId,
      audienceEntityId: options.audienceEntityId,
      previousEntryCount: options.previousEntryCount,
      operationCount: options.operationCount,
      rejectedCount: options.rejected.length,
      rejectionReasons: options.rejected.map((rejection) => rejection.reason),
      source_trust_rejections: toTraceJsonValue(
        options.rejected
          .filter((rejection) => rejection.sourceTrustReason !== undefined)
          .map((rejection) => ({
            operation_index: rejection.operationIndex,
            operation_type: rejection.operationType,
            source_stream_entry_id: rejection.sourceStreamEntryId ?? null,
            source_trust_reason: rejection.sourceTrustReason ?? "unknown",
          })),
      ),
      applied: options.applied,
      recordVersion: options.artifact?.record_version ?? null,
      artifactEntryCount: artifactSummary.renderedEntryCount,
      artifactRenderedTokenEstimate: artifactSummary.estimatedTokens,
      artifact_total_entry_count: artifactSummary.totalEntryCount,
      artifact_active_entry_count: artifactSummary.activeEntryCount,
      artifact_omitted_entry_count: artifactSummary.omittedEntryCount,
      artifact_pruned_entry_count_this_turn: options.prunedEntryCountThisTurn,
      artifact_superseded_count_this_turn: options.supersededEntryCountThisTurn,
      rendered_by_kind: toTraceJsonValue(artifactSummary.renderedByKind),
      ledger_mode: options.ledgerMode,
      input_token_estimate: options.promptBudget.inputTokenEstimate,
      input_token_breakdown: toTraceJsonValue(options.promptBudget.breakdown),
      canonicalizes_rejected_non_locked: toTraceJsonValue(
        options.nonLockedCanonicalizesDrops ?? [],
      ),
    });
  }
}

function traceCompileOverBudget(options: {
  tracer?: TurnTracer;
  turnId?: string;
  audienceEntityId: EntityId;
  ledgerMode: DecisionArtifactLedgerMode;
  promptBudget: DecisionArtifactPromptBudget;
}): void {
  if (
    options.promptBudget.inputTokenEstimate <= DECISION_ARTIFACT_PROMPT_WARNING_TOKEN_THRESHOLD ||
    options.tracer?.enabled !== true ||
    options.turnId === undefined
  ) {
    return;
  }

  options.tracer.emit("decision_artifact_compile_over_budget", {
    turnId: options.turnId,
    audienceEntityId: options.audienceEntityId,
    ledger_mode: options.ledgerMode,
    input_token_estimate: options.promptBudget.inputTokenEstimate,
    input_token_budget: DECISION_ARTIFACT_PROMPT_WARNING_TOKEN_THRESHOLD,
    breakdown: toTraceJsonValue(options.promptBudget.breakdown),
  });
}

function traceReconciliationCompleted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  result: DecisionArtifactReconciliationResult;
  canonicalizationDuplicateDrops?: readonly CanonicalizationDuplicateDrop[];
  currentOperationCanonicalizationCount?: number;
  retriedStrandedCanonicalizationCount?: number;
  retrySummary?: DecisionArtifactUnsettledReconciliationSummary | null;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("decision_artifact_reconciliation_completed", {
      turnId: options.turnId,
      goals_retired: options.result.goals_retired,
      commitments_retired: options.result.commitments_retired,
      actions_retired: options.result.actions_retired,
      open_questions_retired: options.result.open_questions_retired,
      goals_canonicalized_attempted: options.result.goals_canonicalized_attempted,
      goals_canonicalized_succeeded: options.result.goals_canonicalized_succeeded,
      goals_canonicalized_skipped: options.result.goals_canonicalized_skipped,
      commitments_revoked_attempted: options.result.commitments_revoked_attempted,
      commitments_revoked_succeeded: options.result.commitments_revoked_succeeded,
      commitments_revoked_skipped: options.result.commitments_revoked_skipped,
      actions_completed_attempted: options.result.actions_completed_attempted,
      actions_completed_succeeded: options.result.actions_completed_succeeded,
      actions_completed_skipped: options.result.actions_completed_skipped,
      open_questions_resolved_attempted: options.result.open_questions_resolved_attempted,
      open_questions_resolved_succeeded: options.result.open_questions_resolved_succeeded,
      open_questions_resolved_skipped: options.result.open_questions_resolved_skipped,
      semantic_nodes_reviewed_attempted: options.result.semantic_nodes_reviewed_attempted,
      semantic_nodes_marked_superseded: options.result.semantic_nodes_marked_superseded,
      semantic_nodes_marked_contradicted: options.result.semantic_nodes_marked_contradicted,
      semantic_nodes_skipped: options.result.semantic_nodes_skipped,
      unknown_ids: toTraceJsonValue(options.result.unknown_ids),
      canonicalization_duplicates_dropped: toTraceJsonValue(
        options.canonicalizationDuplicateDrops ?? [],
      ),
      current_operation_canonicalization_count: options.currentOperationCanonicalizationCount ?? 0,
      retried_stranded_canonicalization_count: options.retriedStrandedCanonicalizationCount ?? 0,
      retry_unsettled_summary: toTraceJsonValue(options.retrySummary ?? null),
      skipped_commitments: toTraceJsonValue(options.result.skipped_commitments),
      errors: toTraceJsonValue(options.result.errors),
    });
  }
}

function semanticBeliefRevisionDependencies(
  input: CompileDecisionArtifactInput,
): DecisionArtifactSemanticBeliefRevisionDependencies | undefined {
  if (input.semanticBeliefRevision === undefined || input.llmClient === undefined) {
    return undefined;
  }

  return {
    ...input.semanticBeliefRevision,
    llmClient: input.llmClient,
  };
}

async function reconcileSemanticBeliefRevisionFailOpen(
  input: Parameters<typeof reconcileSemanticBeliefRevision>[0],
): ReturnType<typeof reconcileSemanticBeliefRevision> {
  try {
    return await reconcileSemanticBeliefRevision(input);
  } catch {
    return {
      semantic_nodes_reviewed_attempted: 0,
      semantic_nodes_marked_superseded: 0,
      semantic_nodes_marked_contradicted: 0,
      semantic_nodes_skipped: 0,
    };
  }
}

function parseEntryId(value: string): DecisionArtifactEntryId | null {
  try {
    return decisionArtifactEntryIdHelpers.parse(value);
  } catch {
    return null;
  }
}

function parseSourceStreamEntryIds(
  values: readonly string[],
  allowedSourceStreamEntryIds: ReadonlySet<StreamEntryId> | null,
  sourceTrustValidator: DecisionArtifactSourceTrustValidator | undefined,
): {
  streamEntryIds: StreamEntryId[];
  reason: PatchRejection["reason"] | null;
  rejectedStreamEntryId?: string;
  sourceTrustReason?: DecisionArtifactSourceTrustRejectionReason | "unknown";
} {
  if (values.length === 0) {
    return { streamEntryIds: [], reason: "missing_citation" };
  }

  const streamEntryIds: StreamEntryId[] = [];

  for (const value of values) {
    if (!streamEntryIdHelpers.is(value)) {
      return {
        streamEntryIds: [],
        reason: "invalid_source_stream_entry_id",
        rejectedStreamEntryId: value,
      };
    }

    const trust = sourceTrustValidator?.(value);

    if (trust?.allowed === false) {
      return {
        streamEntryIds: [],
        reason:
          trust.reason === "quarantined"
            ? "quarantined_source_stream_entry_id"
            : "inactive_source_stream_entry_id",
        rejectedStreamEntryId: value,
        sourceTrustReason: trust.reason ?? "unknown",
      };
    }

    if (allowedSourceStreamEntryIds !== null && !allowedSourceStreamEntryIds.has(value)) {
      return {
        streamEntryIds: [],
        reason: "disallowed_source_stream_entry_id",
        rejectedStreamEntryId: value,
      };
    }

    if (!streamEntryIds.some((entryId) => entryId === value)) {
      streamEntryIds.push(value);
    }
  }

  return { streamEntryIds, reason: null };
}

function normalizeOwnerEntityId(
  value: string | null | undefined,
  allowedOwnerEntityIds: ReadonlySet<EntityId>,
): EntityId | null | "invalid" {
  if (value === undefined || value === null) {
    return null;
  }

  if (!entityIdHelpers.is(value)) {
    return "invalid";
  }

  return allowedOwnerEntityIds.has(value) ? value : "invalid";
}

function emptyCanonicalizes(): DecisionArtifactCanonicalizes {
  return {
    goal_ids: [],
    commitment_ids: [],
    action_ids: [],
    open_question_ids: [],
  };
}

function hasCanonicalizes(value: DecisionArtifactCanonicalizes | undefined): boolean {
  return (
    value !== undefined &&
    (value.goal_ids.length > 0 ||
      value.commitment_ids.length > 0 ||
      value.action_ids.length > 0 ||
      value.open_question_ids.length > 0)
  );
}

function parsedCanonicalizesTraceIds(
  value: ParsedCanonicalizes | undefined,
): NonLockedCanonicalizesDrop["dropped_ids"] | null {
  if (value === undefined) {
    return null;
  }

  const droppedIds = {
    goal_ids: [...(value.goal_ids ?? [])],
    commitment_ids: [...(value.commitment_ids ?? [])],
    action_ids: [...(value.action_ids ?? [])],
    open_question_ids: [...(value.open_question_ids ?? [])],
  };

  return droppedIds.goal_ids.length > 0 ||
    droppedIds.commitment_ids.length > 0 ||
    droppedIds.action_ids.length > 0 ||
    droppedIds.open_question_ids.length > 0
    ? droppedIds
    : null;
}

function allowedCanonicalizationIds(
  candidates: DecisionArtifactCanonicalizationCandidates | undefined,
): AllowedCanonicalizationIds {
  return {
    goalIds: new Set(
      (candidates?.goals ?? [])
        .map((candidate) => candidate.id)
        .filter((id): id is GoalId => goalIdHelpers.is(id)),
    ),
    commitmentIds: new Set(
      (candidates?.commitments ?? [])
        .map((candidate) => candidate.id)
        .filter((id): id is CommitmentId => commitmentIdHelpers.is(id)),
    ),
    actionIds: new Set(
      (candidates?.actions ?? [])
        .map((candidate) => candidate.id)
        .filter((id): id is ActionId => actionIdHelpers.is(id)),
    ),
    openQuestionIds: new Set(
      (candidates?.openQuestions ?? [])
        .map((candidate) => candidate.id)
        .filter((id): id is OpenQuestionId => openQuestionIdHelpers.is(id)),
    ),
  };
}

function normalizeCanonicalizeIds<TId extends string>(input: {
  values: readonly string[] | undefined;
  channel: CanonicalizeIdChannel;
  isId: (value: string) => value is TId;
  allowedIds: ReadonlySet<TId>;
  operation: ParsedPatchOperation;
  operationIndex: number;
  dropped: DroppedCanonicalizeId[];
}): TId[] {
  const ids: TId[] = [];

  for (const value of input.values ?? []) {
    if (!input.isId(value)) {
      input.dropped.push({
        channel: input.channel,
        id: value,
        reason: "invalid_id",
        operationType: input.operation.type,
        operationIndex: input.operationIndex,
      });
      continue;
    }

    if (!input.allowedIds.has(value)) {
      input.dropped.push({
        channel: input.channel,
        id: value,
        reason: "unknown_id",
        operationType: input.operation.type,
        operationIndex: input.operationIndex,
      });
      continue;
    }

    if (!ids.some((id) => id === value)) {
      ids.push(value);
    }
  }

  return ids;
}

function normalizeCanonicalizes(input: {
  value: ParsedCanonicalizes | undefined;
  kind: DecisionArtifactEntryKind;
  allowedIds: AllowedCanonicalizationIds;
  operation: ParsedPatchOperation;
  operationIndex: number;
  dropped: DroppedCanonicalizeId[];
  nonLockedDrops: NonLockedCanonicalizesDrop[];
}): DecisionArtifactCanonicalizes | undefined {
  if (input.value === undefined) {
    return undefined;
  }

  const nonLockedDroppedIds = parsedCanonicalizesTraceIds(input.value);

  if (input.kind !== "locked" && nonLockedDroppedIds !== null) {
    input.nonLockedDrops.push({
      operation_index: input.operationIndex,
      kind: input.kind,
      dropped_ids: nonLockedDroppedIds,
    });

    return undefined;
  }

  const canonicalizes: DecisionArtifactCanonicalizes = {
    goal_ids: normalizeCanonicalizeIds({
      values: input.value.goal_ids,
      channel: "goal",
      isId: goalIdHelpers.is,
      allowedIds: input.allowedIds.goalIds,
      operation: input.operation,
      operationIndex: input.operationIndex,
      dropped: input.dropped,
    }),
    commitment_ids: normalizeCanonicalizeIds({
      values: input.value.commitment_ids,
      channel: "commitment",
      isId: commitmentIdHelpers.is,
      allowedIds: input.allowedIds.commitmentIds,
      operation: input.operation,
      operationIndex: input.operationIndex,
      dropped: input.dropped,
    }),
    action_ids: normalizeCanonicalizeIds({
      values: input.value.action_ids,
      channel: "action",
      isId: actionIdHelpers.is,
      allowedIds: input.allowedIds.actionIds,
      operation: input.operation,
      operationIndex: input.operationIndex,
      dropped: input.dropped,
    }),
    open_question_ids: normalizeCanonicalizeIds({
      values: input.value.open_question_ids,
      channel: "open_question",
      isId: openQuestionIdHelpers.is,
      allowedIds: input.allowedIds.openQuestionIds,
      operation: input.operation,
      operationIndex: input.operationIndex,
      dropped: input.dropped,
    }),
  };

  return hasCanonicalizes(canonicalizes) ? canonicalizes : emptyCanonicalizes();
}

function previousEntryById(
  previousEntries: ReadonlyMap<DecisionArtifactEntryId, DecisionArtifactEntry>,
  id: string,
): { id: DecisionArtifactEntryId | null; entry: DecisionArtifactEntry | null } {
  const parsedId = parseEntryId(id);

  if (parsedId === null) {
    return { id: null, entry: null };
  }

  return {
    id: parsedId,
    entry: previousEntries.get(parsedId) ?? null,
  };
}

function rejection(
  operation: ParsedPatchOperation,
  operationIndex: number,
  reason: PatchRejection["reason"],
  details: Pick<PatchRejection, "sourceStreamEntryId" | "sourceTrustReason"> = {},
): PatchRejection {
  return {
    reason,
    operationType: operation.type,
    operationIndex,
    ...details,
  };
}

function normalizePatch(input: {
  patch: EmitDecisionArtifactPatch;
  previousArtifact: DecisionArtifact | null;
  audienceEntityId: EntityId;
  selfEntityId: EntityId;
  speakerEntityId: EntityId | null;
  participants: readonly DecisionArtifactParticipantContext[];
  allowedSourceStreamEntryIds: ReadonlySet<StreamEntryId> | null;
  sourceTrustValidator?: DecisionArtifactSourceTrustValidator;
  allowedCanonicalizationIds: AllowedCanonicalizationIds;
}): {
  operations: DecisionArtifactOperation[];
  rejected: PatchRejection[];
  droppedCanonicalizeIds: DroppedCanonicalizeId[];
  nonLockedCanonicalizesDrops: NonLockedCanonicalizesDrop[];
} {
  const allowedOwnerEntityIds = new Set<EntityId>([
    input.audienceEntityId,
    input.selfEntityId,
    ...input.participants.map((participant) => participant.entityId),
  ]);

  if (input.speakerEntityId !== null) {
    allowedOwnerEntityIds.add(input.speakerEntityId);
  }

  const previousEntries = new Map<DecisionArtifactEntryId, DecisionArtifactEntry>(
    (input.previousArtifact?.entries ?? []).map((entry) => [entry.id, entry]),
  );
  const operations: DecisionArtifactOperation[] = [];
  const rejected: PatchRejection[] = [];
  const droppedCanonicalizeIds: DroppedCanonicalizeId[] = [];
  const nonLockedCanonicalizesDrops: NonLockedCanonicalizesDrop[] = [];
  const baseRank = input.previousArtifact?.entries.length ?? 0;

  input.patch.operations.forEach((operation, operationIndex) => {
    switch (operation.type) {
      case "add": {
        const ownerEntityId = normalizeOwnerEntityId(
          operation.owner_entity_id,
          allowedOwnerEntityIds,
        );

        if (ownerEntityId === "invalid") {
          rejected.push(rejection(operation, operationIndex, "invalid_owner_entity_id"));
          return;
        }

        const citations = parseSourceStreamEntryIds(
          operation.source_stream_entry_ids,
          input.allowedSourceStreamEntryIds,
          input.sourceTrustValidator,
        );

        if (citations.reason !== null) {
          rejected.push(
            rejection(operation, operationIndex, citations.reason, {
              sourceStreamEntryId: citations.rejectedStreamEntryId,
              sourceTrustReason: citations.sourceTrustReason,
            }),
          );
          return;
        }

        const canonicalizes = normalizeCanonicalizes({
          value: operation.canonicalizes,
          kind: operation.kind,
          allowedIds: input.allowedCanonicalizationIds,
          operation,
          operationIndex,
          dropped: droppedCanonicalizeIds,
          nonLockedDrops: nonLockedCanonicalizesDrops,
        });

        operations.push({
          type: "add",
          kind: operation.kind,
          text: operation.text,
          owner_entity_id: ownerEntityId,
          provenance_stream_entry_ids: citations.streamEntryIds,
          last_updated_stream_entry_ids: citations.streamEntryIds,
          rank: baseRank + operations.length,
          ...(canonicalizes === undefined ? {} : { canonicalizes }),
        });
        return;
      }

      case "update": {
        const { id, entry } = previousEntryById(previousEntries, operation.id);

        if (id === null) {
          rejected.push(rejection(operation, operationIndex, "invalid_entry_id"));
          return;
        }

        if (entry === null) {
          rejected.push(rejection(operation, operationIndex, "unknown_entry_id"));
          return;
        }

        const nextKind = operation.kind ?? entry.kind;

        if (
          operation.kind === undefined &&
          operation.text === undefined &&
          operation.owner_entity_id === undefined &&
          operation.canonicalizes === undefined
        ) {
          rejected.push(rejection(operation, operationIndex, "empty_update"));
          return;
        }

        const ownerEntityId = normalizeOwnerEntityId(
          operation.owner_entity_id,
          allowedOwnerEntityIds,
        );

        if (ownerEntityId === "invalid") {
          rejected.push(rejection(operation, operationIndex, "invalid_owner_entity_id"));
          return;
        }

        const citations = parseSourceStreamEntryIds(
          operation.source_stream_entry_ids,
          input.allowedSourceStreamEntryIds,
          input.sourceTrustValidator,
        );

        if (citations.reason !== null) {
          rejected.push(
            rejection(operation, operationIndex, citations.reason, {
              sourceStreamEntryId: citations.rejectedStreamEntryId,
              sourceTrustReason: citations.sourceTrustReason,
            }),
          );
          return;
        }

        const canonicalizes = normalizeCanonicalizes({
          value: operation.canonicalizes,
          kind: nextKind,
          allowedIds: input.allowedCanonicalizationIds,
          operation,
          operationIndex,
          dropped: droppedCanonicalizeIds,
          nonLockedDrops: nonLockedCanonicalizesDrops,
        });

        operations.push({
          type: "update",
          id,
          kind: nextKind,
          text: operation.text,
          owner_entity_id: operation.owner_entity_id === undefined ? undefined : ownerEntityId,
          add_provenance_stream_entry_ids: citations.streamEntryIds,
          last_updated_stream_entry_ids: citations.streamEntryIds,
          ...(canonicalizes === undefined ? {} : { canonicalizes }),
        });
        return;
      }

      case "supersede": {
        const { id, entry } = previousEntryById(previousEntries, operation.id);

        if (id === null) {
          rejected.push(rejection(operation, operationIndex, "invalid_entry_id"));
          return;
        }

        if (entry === null) {
          rejected.push(rejection(operation, operationIndex, "unknown_entry_id"));
          return;
        }

        const ownerEntityId = normalizeOwnerEntityId(
          operation.replacement.owner_entity_id,
          allowedOwnerEntityIds,
        );

        if (ownerEntityId === "invalid") {
          rejected.push(rejection(operation, operationIndex, "invalid_owner_entity_id"));
          return;
        }

        const replacementCitations = parseSourceStreamEntryIds(
          operation.replacement.source_stream_entry_ids,
          input.allowedSourceStreamEntryIds,
          input.sourceTrustValidator,
        );

        if (replacementCitations.reason !== null) {
          rejected.push(
            rejection(operation, operationIndex, replacementCitations.reason, {
              sourceStreamEntryId: replacementCitations.rejectedStreamEntryId,
              sourceTrustReason: replacementCitations.sourceTrustReason,
            }),
          );
          return;
        }

        const updateCitationValues =
          operation.source_stream_entry_ids ?? operation.replacement.source_stream_entry_ids;
        const updateCitations = parseSourceStreamEntryIds(
          updateCitationValues,
          input.allowedSourceStreamEntryIds,
          input.sourceTrustValidator,
        );

        if (updateCitations.reason !== null) {
          rejected.push(
            rejection(operation, operationIndex, updateCitations.reason, {
              sourceStreamEntryId: updateCitations.rejectedStreamEntryId,
              sourceTrustReason: updateCitations.sourceTrustReason,
            }),
          );
          return;
        }

        const canonicalizes = normalizeCanonicalizes({
          value: operation.canonicalizes,
          kind: operation.replacement.kind,
          allowedIds: input.allowedCanonicalizationIds,
          operation,
          operationIndex,
          dropped: droppedCanonicalizeIds,
          nonLockedDrops: nonLockedCanonicalizesDrops,
        });

        operations.push({
          type: "supersede",
          id,
          replacement: {
            kind: operation.replacement.kind,
            text: operation.replacement.text,
            owner_entity_id: ownerEntityId,
            provenance_stream_entry_ids: replacementCitations.streamEntryIds,
            last_updated_stream_entry_ids: replacementCitations.streamEntryIds,
            rank: baseRank + operations.length,
            ...(canonicalizes === undefined ? {} : { canonicalizes }),
          },
          last_updated_stream_entry_ids: updateCitations.streamEntryIds,
        });
        return;
      }

      case "prune": {
        const { id, entry } = previousEntryById(previousEntries, operation.id);

        if (id === null) {
          rejected.push(rejection(operation, operationIndex, "invalid_entry_id"));
          return;
        }

        if (entry === null) {
          rejected.push(rejection(operation, operationIndex, "unknown_entry_id"));
          return;
        }

        operations.push({
          type: "prune",
          id,
        });
      }
    }
  });

  return { operations, rejected, droppedCanonicalizeIds, nonLockedCanonicalizesDrops };
}

type LifecycleEntry = Pick<
  DecisionArtifactEntry,
  "id" | "kind" | "created_at" | "last_updated_at" | "superseded_by_id" | "rank"
>;

function normalizeLifecycleKindSoftCaps(
  options: DecisionArtifactLifecycleOptions | undefined,
): Record<DecisionArtifactEntryKind, number> {
  return {
    ...DEFAULT_DECISION_ARTIFACT_KIND_SOFT_CAPS,
    ...(options?.kindSoftCaps ?? {}),
  };
}

function lifecycleMaxActiveEntries(options: DecisionArtifactLifecycleOptions | undefined): number {
  const value = options?.maxActiveEntries ?? DEFAULT_MAX_ACTIVE_DECISION_ARTIFACT_ENTRIES;

  return Number.isFinite(value) && value > 0
    ? Math.floor(value)
    : DEFAULT_MAX_ACTIVE_DECISION_ARTIFACT_ENTRIES;
}

function operationIdsMaterialized(
  operations: readonly DecisionArtifactOperation[],
): DecisionArtifactOperation[] {
  return operations.map((operation) => {
    switch (operation.type) {
      case "add":
        return {
          ...operation,
          id: operation.id ?? createDecisionArtifactEntryId(),
        };
      case "supersede":
        return {
          ...operation,
          replacement: {
            ...operation.replacement,
            id: operation.replacement.id ?? createDecisionArtifactEntryId(),
          },
        };
      case "update":
      case "prune":
        return operation;
    }
  });
}

function canonicalizedEntryIdsFromOperations(
  operations: readonly DecisionArtifactOperation[],
): Set<DecisionArtifactEntryId> {
  const ids = new Set<DecisionArtifactEntryId>();

  for (const operation of operations) {
    switch (operation.type) {
      case "add":
        if (operation.id !== undefined && hasCanonicalizes(operation.canonicalizes)) {
          ids.add(operation.id);
        }
        break;
      case "update":
        if (hasCanonicalizes(operation.canonicalizes)) {
          ids.add(operation.id);
        }
        break;
      case "supersede":
        if (
          operation.replacement.id !== undefined &&
          hasCanonicalizes(operation.replacement.canonicalizes)
        ) {
          ids.add(operation.replacement.id);
        }
        break;
      case "prune":
        break;
    }
  }

  return ids;
}

function canonicalizedEntriesFromOperations(input: {
  artifact: DecisionArtifact;
  operations: readonly DecisionArtifactOperation[];
}): DecisionArtifactEntry[] {
  const ids = canonicalizedEntryIdsFromOperations(input.operations);

  if (ids.size === 0) {
    return [];
  }

  return input.artifact.entries.filter((entry) => ids.has(entry.id));
}

type DecisionArtifactReconciliationWorkSet = {
  entries: DecisionArtifactEntry[];
  currentOperationCanonicalizationCount: number;
  retriedStrandedCanonicalizationCount: number;
  retrySummary: DecisionArtifactUnsettledReconciliationSummary | null;
};

function canonicalizationKey(input: {
  entryId: DecisionArtifactEntryId;
  channel: CanonicalizeIdChannel;
  id: string;
}): string {
  return `${input.entryId}:${input.channel}:${input.id}`;
}

function emptyReconciliationEntry(entry: DecisionArtifactEntry): DecisionArtifactEntry {
  return {
    ...entry,
    canonicalizes: emptyCanonicalizes(),
  };
}

function appendReconciliationEntry(input: {
  entry: DecisionArtifactEntry;
  entriesById: Map<DecisionArtifactEntryId, DecisionArtifactEntry>;
  seen: Set<string>;
}): number {
  const mergedEntry =
    input.entriesById.get(input.entry.id) ?? emptyReconciliationEntry(input.entry);
  let count = 0;

  if (!input.entriesById.has(input.entry.id)) {
    input.entriesById.set(input.entry.id, mergedEntry);
  }

  for (const goalId of input.entry.canonicalizes.goal_ids) {
    const key = canonicalizationKey({
      entryId: input.entry.id,
      channel: "goal",
      id: goalId,
    });

    if (!input.seen.has(key)) {
      input.seen.add(key);
      mergedEntry.canonicalizes.goal_ids.push(goalId);
      count += 1;
    }
  }

  for (const commitmentId of input.entry.canonicalizes.commitment_ids) {
    const key = canonicalizationKey({
      entryId: input.entry.id,
      channel: "commitment",
      id: commitmentId,
    });

    if (!input.seen.has(key)) {
      input.seen.add(key);
      mergedEntry.canonicalizes.commitment_ids.push(commitmentId);
      count += 1;
    }
  }

  for (const actionId of input.entry.canonicalizes.action_ids) {
    const key = canonicalizationKey({
      entryId: input.entry.id,
      channel: "action",
      id: actionId,
    });

    if (!input.seen.has(key)) {
      input.seen.add(key);
      mergedEntry.canonicalizes.action_ids.push(actionId);
      count += 1;
    }
  }

  for (const openQuestionId of input.entry.canonicalizes.open_question_ids) {
    const key = canonicalizationKey({
      entryId: input.entry.id,
      channel: "open_question",
      id: openQuestionId,
    });

    if (!input.seen.has(key)) {
      input.seen.add(key);
      mergedEntry.canonicalizes.open_question_ids.push(openQuestionId);
      count += 1;
    }
  }

  return count;
}

function buildDecisionArtifactReconciliationWorkSet(input: {
  artifact: DecisionArtifact | null;
  operations: readonly DecisionArtifactOperation[];
  repositories?: DecisionArtifactReconciliationRepositories;
  nowMs: number;
}): DecisionArtifactReconciliationWorkSet {
  if (input.artifact === null) {
    return {
      entries: [],
      currentOperationCanonicalizationCount: 0,
      retriedStrandedCanonicalizationCount: 0,
      retrySummary: null,
    };
  }

  const entriesById = new Map<DecisionArtifactEntryId, DecisionArtifactEntry>();
  const seen = new Set<string>();
  let currentOperationCanonicalizationCount = 0;
  let retriedStrandedCanonicalizationCount = 0;

  for (const entry of canonicalizedEntriesFromOperations({
    artifact: input.artifact,
    operations: input.operations,
  })) {
    currentOperationCanonicalizationCount += appendReconciliationEntry({
      entry,
      entriesById,
      seen,
    });
  }

  const retry = findUnsettledDecisionArtifactReconciliation({
    previousArtifact: input.artifact,
    repositories: input.repositories,
    nowMs: input.nowMs,
  });

  for (const entry of retry?.entries ?? []) {
    retriedStrandedCanonicalizationCount += appendReconciliationEntry({
      entry,
      entriesById,
      seen,
    });
  }

  return {
    entries: [...entriesById.values()].filter((entry) => hasCanonicalizes(entry.canonicalizes)),
    currentOperationCanonicalizationCount,
    retriedStrandedCanonicalizationCount,
    retrySummary: retry?.summary ?? null,
  };
}

function dedupeCanonicalizeIds<TId extends string>(
  ids: readonly TId[],
  seen: Set<TId>,
): { kept: TId[]; dropped: TId[] } {
  const kept: TId[] = [];
  const dropped: TId[] = [];

  for (const id of ids) {
    if (seen.has(id)) {
      dropped.push(id);
      continue;
    }

    seen.add(id);
    kept.push(id);
  }

  return { kept, dropped };
}

function dedupeCanonicalizesAcrossOperations(operations: readonly DecisionArtifactOperation[]): {
  operations: DecisionArtifactOperation[];
  duplicateDrops: CanonicalizationDuplicateDrop[];
} {
  const prunedEntryIds = new Set(
    operations
      .filter((operation): operation is Extract<DecisionArtifactOperation, { type: "prune" }> => {
        return operation.type === "prune";
      })
      .map((operation) => operation.id),
  );
  const seenGoalIds = new Set<GoalId>();
  const seenCommitmentIds = new Set<CommitmentId>();
  const seenActionIds = new Set<ActionId>();
  const seenOpenQuestionIds = new Set<OpenQuestionId>();
  const duplicateDrops: CanonicalizationDuplicateDrop[] = [];

  const dedupe = (input: {
    id: DecisionArtifactEntryId | undefined;
    kind: DecisionArtifactEntryKind;
    canonicalizes: DecisionArtifactCanonicalizes | undefined;
  }): DecisionArtifactCanonicalizes | undefined => {
    if (
      input.canonicalizes === undefined ||
      input.id === undefined ||
      prunedEntryIds.has(input.id)
    ) {
      return input.canonicalizes;
    }

    const goals = dedupeCanonicalizeIds(input.canonicalizes.goal_ids, seenGoalIds);
    const commitments = dedupeCanonicalizeIds(
      input.canonicalizes.commitment_ids,
      seenCommitmentIds,
    );
    const actions = dedupeCanonicalizeIds(input.canonicalizes.action_ids, seenActionIds);
    const openQuestions = dedupeCanonicalizeIds(
      input.canonicalizes.open_question_ids,
      seenOpenQuestionIds,
    );
    const dropped: DecisionArtifactCanonicalizes = {
      goal_ids: goals.dropped,
      commitment_ids: commitments.dropped,
      action_ids: actions.dropped,
      open_question_ids: openQuestions.dropped,
    };

    if (hasCanonicalizes(dropped)) {
      duplicateDrops.push({
        artifact_entry_id: input.id,
        kind: input.kind,
        dropped_ids: dropped,
      });
    }

    return {
      goal_ids: goals.kept,
      commitment_ids: commitments.kept,
      action_ids: actions.kept,
      open_question_ids: openQuestions.kept,
    };
  };

  return {
    duplicateDrops,
    operations: operations.map((operation) => {
      switch (operation.type) {
        case "add":
          return {
            ...operation,
            canonicalizes: dedupe({
              id: operation.id,
              kind: operation.kind,
              canonicalizes: operation.canonicalizes,
            }),
          };
        case "update":
          return {
            ...operation,
            canonicalizes: dedupe({
              id: operation.id,
              kind: operation.kind ?? "locked",
              canonicalizes: operation.canonicalizes,
            }),
          };
        case "supersede":
          return {
            ...operation,
            replacement: {
              ...operation.replacement,
              canonicalizes: dedupe({
                id: operation.replacement.id,
                kind: operation.replacement.kind,
                canonicalizes: operation.replacement.canonicalizes,
              }),
            },
          };
        case "prune":
          return operation;
      }
    }),
  };
}

function lifecycleEntryFromDecisionEntry(entry: DecisionArtifactEntry): LifecycleEntry {
  return {
    id: entry.id,
    kind: entry.kind,
    created_at: entry.created_at,
    last_updated_at: entry.last_updated_at,
    superseded_by_id: entry.superseded_by_id,
    rank: entry.rank,
  };
}

function materializePostPatchLifecycleEntries(input: {
  previousArtifact: DecisionArtifact | null;
  operations: readonly DecisionArtifactOperation[];
  nowMs: number;
  applyPrunes?: boolean;
}): LifecycleEntry[] {
  const entries = new Map<DecisionArtifactEntryId, LifecycleEntry>();

  for (const entry of input.previousArtifact?.entries ?? []) {
    entries.set(entry.id, lifecycleEntryFromDecisionEntry(entry));
  }

  for (const operation of input.operations) {
    switch (operation.type) {
      case "add": {
        const id = operation.id ?? createDecisionArtifactEntryId();
        entries.set(id, {
          id,
          kind: operation.kind,
          created_at: operation.created_at ?? input.nowMs,
          last_updated_at: operation.last_updated_at ?? operation.created_at ?? input.nowMs,
          superseded_by_id: null,
          rank: operation.rank ?? entries.size,
        });
        break;
      }
      case "update": {
        const current = entries.get(operation.id);

        if (current === undefined) {
          break;
        }

        entries.set(operation.id, {
          ...current,
          kind: operation.kind ?? current.kind,
          last_updated_at: operation.last_updated_at ?? input.nowMs,
          rank: operation.rank ?? current.rank,
        });
        break;
      }
      case "supersede": {
        const current = entries.get(operation.id);
        const replacementId = operation.replacement.id ?? createDecisionArtifactEntryId();

        if (current !== undefined) {
          entries.set(operation.id, {
            ...current,
            superseded_by_id: replacementId,
            last_updated_at: operation.last_updated_at ?? input.nowMs,
          });
        }

        entries.set(replacementId, {
          id: replacementId,
          kind: operation.replacement.kind,
          created_at: operation.replacement.created_at ?? input.nowMs,
          last_updated_at:
            operation.replacement.last_updated_at ??
            operation.replacement.created_at ??
            input.nowMs,
          superseded_by_id: null,
          rank: operation.replacement.rank ?? entries.size,
        });
        break;
      }
      case "prune":
        if (input.applyPrunes !== false) {
          entries.delete(operation.id);
        }
        break;
    }
  }

  return [...entries.values()];
}

function activeLifecycleEntries(entries: readonly LifecycleEntry[]): LifecycleEntry[] {
  return entries.filter((entry) => entry.superseded_by_id === null);
}

function lifecycleKindCounts(
  entries: readonly LifecycleEntry[],
): Record<DecisionArtifactEntryKind, number> {
  const counts = Object.fromEntries(
    DECISION_ARTIFACT_ENTRY_KINDS.map((kind) => [kind, 0]),
  ) as Record<DecisionArtifactEntryKind, number>;

  for (const entry of entries) {
    counts[entry.kind] += 1;
  }

  return counts;
}

function compareLifecyclePrunePriority(left: LifecycleEntry, right: LifecycleEntry): number {
  return (
    left.last_updated_at - right.last_updated_at ||
    left.rank - right.rank ||
    left.created_at - right.created_at ||
    left.id.localeCompare(right.id)
  );
}

function nextLifecyclePruneCandidate(input: {
  entries: readonly LifecycleEntry[];
  kind: DecisionArtifactEntryKind;
  prunedIds: ReadonlySet<DecisionArtifactEntryId>;
}): LifecycleEntry | null {
  return (
    activeLifecycleEntries(input.entries)
      .filter((entry) => entry.kind === input.kind && !input.prunedIds.has(entry.id))
      .sort(compareLifecyclePrunePriority)[0] ?? null
  );
}

function lifecycleEntriesById(
  entries: readonly LifecycleEntry[],
): Map<DecisionArtifactEntryId, LifecycleEntry> {
  const byId = new Map<DecisionArtifactEntryId, LifecycleEntry>();

  for (const entry of entries) {
    byId.set(entry.id, entry);
  }

  return byId;
}

function lifecycleReferrersByReplacement(
  entries: readonly LifecycleEntry[],
): Map<DecisionArtifactEntryId, LifecycleEntry[]> {
  const byReplacement = new Map<DecisionArtifactEntryId, LifecycleEntry[]>();

  for (const entry of entries) {
    if (entry.superseded_by_id === null) {
      continue;
    }

    const referrers = byReplacement.get(entry.superseded_by_id) ?? [];
    referrers.push(entry);
    byReplacement.set(entry.superseded_by_id, referrers);
  }

  return byReplacement;
}

function appendPruneWithDependencies(input: {
  entryId: DecisionArtifactEntryId;
  entriesById: ReadonlyMap<DecisionArtifactEntryId, LifecycleEntry>;
  referrersByReplacement: ReadonlyMap<DecisionArtifactEntryId, readonly LifecycleEntry[]>;
  prunedIds: Set<DecisionArtifactEntryId>;
  visitingIds: Set<DecisionArtifactEntryId>;
  pruneOperations: DecisionArtifactOperation[];
}): boolean {
  if (input.prunedIds.has(input.entryId)) {
    return true;
  }

  if (input.visitingIds.has(input.entryId)) {
    return false;
  }

  if (!input.entriesById.has(input.entryId)) {
    return true;
  }

  input.visitingIds.add(input.entryId);

  for (const referrer of input.referrersByReplacement.get(input.entryId) ?? []) {
    if (input.prunedIds.has(referrer.id)) {
      continue;
    }

    if (
      !appendPruneWithDependencies({
        ...input,
        entryId: referrer.id,
      })
    ) {
      input.visitingIds.delete(input.entryId);
      return false;
    }
  }

  input.visitingIds.delete(input.entryId);
  input.prunedIds.add(input.entryId);
  input.pruneOperations.push({
    type: "prune",
    id: input.entryId,
  });

  return true;
}

function expandPruneDependencies(input: {
  previousArtifact: DecisionArtifact | null;
  operations: readonly DecisionArtifactOperation[];
  nowMs: number;
}): DecisionArtifactOperation[] {
  const entries = materializePostPatchLifecycleEntries({
    previousArtifact: input.previousArtifact,
    operations: input.operations,
    nowMs: input.nowMs,
    applyPrunes: false,
  });
  const entriesById = lifecycleEntriesById(entries);
  const referrersByReplacement = lifecycleReferrersByReplacement(entries);
  const expandedOperations: DecisionArtifactOperation[] = [];
  const prunedIds = new Set<DecisionArtifactEntryId>();

  for (const operation of input.operations) {
    if (operation.type !== "prune") {
      expandedOperations.push(operation);
      continue;
    }

    const previousPrunedIds = new Set(prunedIds);
    const previousOperationCount = expandedOperations.length;
    const appended = appendPruneWithDependencies({
      entryId: operation.id,
      entriesById,
      referrersByReplacement,
      prunedIds,
      visitingIds: new Set<DecisionArtifactEntryId>(),
      pruneOperations: expandedOperations,
    });

    if (!appended) {
      expandedOperations.splice(previousOperationCount);
      prunedIds.clear();
      for (const prunedId of previousPrunedIds) {
        prunedIds.add(prunedId);
      }
      expandedOperations.push(operation);
    }
  }

  return expandedOperations;
}

function applyDecisionArtifactLifecycleCap(input: {
  previousArtifact: DecisionArtifact | null;
  operations: readonly DecisionArtifactOperation[];
  options?: DecisionArtifactLifecycleOptions;
  nowMs: number;
}): {
  operations: DecisionArtifactOperation[];
  maxActiveEntries: number;
  postPlanActiveEntryCount: number;
  overCapDelta: number;
} {
  const operations = operationIdsMaterialized(input.operations);
  const entries = materializePostPatchLifecycleEntries({
    previousArtifact: input.previousArtifact,
    operations,
    nowMs: input.nowMs,
  });
  const maxActiveEntries = lifecycleMaxActiveEntries(input.options);
  const kindSoftCaps = normalizeLifecycleKindSoftCaps(input.options);
  const prunedIds = new Set<DecisionArtifactEntryId>();
  const pruneOperations: DecisionArtifactOperation[] = [];
  let activeEntries = activeLifecycleEntries(entries);
  let activeCounts = lifecycleKindCounts(activeEntries);

  const selectFromKind = (kind: DecisionArtifactEntryKind): boolean => {
    const candidate = nextLifecyclePruneCandidate({
      entries,
      kind,
      prunedIds,
    });

    if (candidate === null) {
      return false;
    }

    prunedIds.add(candidate.id);
    pruneOperations.push({
      type: "prune",
      id: candidate.id,
    });
    activeCounts[candidate.kind] -= 1;
    activeEntries = activeEntries.filter((entry) => entry.id !== candidate.id);
    return true;
  };

  while (activeEntries.length > maxActiveEntries) {
    let pruned = false;

    for (const kind of DECISION_ARTIFACT_LIFECYCLE_PRUNE_ORDER) {
      if (activeCounts[kind] <= kindSoftCaps[kind]) {
        continue;
      }

      pruned = selectFromKind(kind);

      if (pruned) {
        break;
      }
    }

    if (pruned) {
      continue;
    }

    for (const kind of DECISION_ARTIFACT_LIFECYCLE_PRUNE_ORDER) {
      pruned = selectFromKind(kind);

      if (pruned) {
        break;
      }
    }

    if (!pruned) {
      break;
    }
  }

  const postPlanActiveEntryCount = activeEntries.length;

  return {
    operations: [...operations, ...pruneOperations],
    maxActiveEntries,
    postPlanActiveEntryCount,
    overCapDelta: Math.max(0, postPlanActiveEntryCount - maxActiveEntries),
  };
}

async function degraded(
  input: CompileDecisionArtifactInput,
  reason: DecisionArtifactCompileDegradedReason,
  error?: unknown,
): Promise<EmitDecisionArtifactPatch> {
  try {
    await input.onDegraded?.(reason, error);
  } catch {
    // Best-effort degraded-mode logging only.
  }

  return emptyPatch();
}

export async function compileDecisionArtifact(
  input: CompileDecisionArtifactInput,
): Promise<EmitDecisionArtifactPatch> {
  if (input.llmClient === undefined || input.model === undefined) {
    return degraded(input, "llm_unavailable");
  }

  if (input.repository === undefined) {
    return degraded(input, "repository_unavailable");
  }

  const previousArtifact =
    input.previousArtifact === undefined
      ? input.repository.get(input.audienceEntityId)
      : input.previousArtifact;
  const previousEntryCount = previousArtifact?.entries.length ?? 0;
  const speakerEntityId = input.speakerEntityId ?? null;
  const previousArtifactSummary = buildDecisionArtifactPromptSummary(
    previousArtifact,
    input.previousArtifactSummaryOptions,
  );
  const canonicalizationCandidates = input.canonicalizationCandidates ?? {};
  const messages = buildDecisionArtifactMessages({
    audienceEntityId: input.audienceEntityId,
    selfEntityId: input.selfEntityId,
    speakerEntityId,
    participants: input.participants,
    currentUserMessage: input.currentUserMessage,
    currentUserStreamEntryId: input.currentUserStreamEntryId,
    promptVisibleLedger: input.promptVisibleLedger,
    previousArtifactSummary,
    canonicalizationCandidates,
    allowedSourceStreamEntryIds: input.allowedSourceStreamEntryIds,
    offLimitsSourceStreamEntryIds: input.offLimitsSourceStreamEntryIds,
  });
  const tools = [DECISION_ARTIFACT_TOOL];
  const ledgerMode = input.ledgerMode ?? "full_fallback";
  const promptBudget = estimateDecisionArtifactPromptBudget({
    messages,
    tools,
    previousArtifactSummary,
    promptVisibleLedger: input.promptVisibleLedger,
    currentUserMessage: input.currentUserMessage,
    canonicalizationCandidates,
  });

  traceCompileOverBudget({
    tracer: input.tracer,
    turnId: input.turnId,
    audienceEntityId: input.audienceEntityId,
    ledgerMode,
    promptBudget,
  });
  const compileCompletedTraceBase = {
    tracer: input.tracer,
    turnId: input.turnId,
    audienceEntityId: input.audienceEntityId,
    previousEntryCount,
    renderOptions: input.renderOptions,
    ledgerMode,
    promptBudget,
  };

  traceLlmCallStarted({
    tracer: input.tracer,
    turnId: input.turnId,
    model: input.model,
    messages,
    tools,
  });

  let response: LLMCompleteResult;

  try {
    response = await input.llmClient.complete({
      model: input.model,
      system: DECISION_ARTIFACT_SYSTEM_PROMPT,
      messages,
      tools,
      tool_choice: { type: "tool", name: DECISION_ARTIFACT_TOOL_NAME },
      max_tokens: MAX_PATCH_OUTPUT_TOKENS,
      budget: "decision-artifact-compiler",
    });
  } catch (error) {
    traceLlmCallError({
      tracer: input.tracer,
      turnId: input.turnId,
      error,
    });
    traceCompileCompleted({
      ...compileCompletedTraceBase,
      operationCount: 0,
      rejected: [],
      applied: false,
      artifact: previousArtifact,
      prunedEntryCountThisTurn: 0,
      supersededEntryCountThisTurn: 0,
    });

    return degraded(input, "llm_failed", error);
  }

  traceLlmCallResponse({
    tracer: input.tracer,
    turnId: input.turnId,
    response,
  });

  let parsed: EmitDecisionArtifactPatch;

  try {
    parsed = parseResponse(response);
  } catch (error) {
    traceCompileCompleted({
      ...compileCompletedTraceBase,
      operationCount: 0,
      rejected: [],
      applied: false,
      artifact: previousArtifact,
      prunedEntryCountThisTurn: 0,
      supersededEntryCountThisTurn: 0,
    });

    return degraded(
      input,
      error instanceof MissingDecisionArtifactToolCallError
        ? "missing_tool_call"
        : error instanceof z.ZodError
          ? "invalid_payload"
          : "llm_failed",
      error,
    );
  }

  const allowedSourceStreamEntryIds =
    input.allowedSourceStreamEntryIds === undefined
      ? null
      : new Set(input.allowedSourceStreamEntryIds);
  const normalized = normalizePatch({
    patch: parsed,
    previousArtifact,
    audienceEntityId: input.audienceEntityId,
    selfEntityId: input.selfEntityId,
    speakerEntityId,
    participants: input.participants,
    allowedSourceStreamEntryIds,
    sourceTrustValidator: input.sourceTrustValidator,
    allowedCanonicalizationIds: allowedCanonicalizationIds(input.canonicalizationCandidates),
  });

  if (normalized.operations.length === 0 && normalized.rejected.length > 0) {
    traceCompileCompleted({
      ...compileCompletedTraceBase,
      operationCount: 0,
      rejected: normalized.rejected,
      applied: false,
      artifact: previousArtifact,
      prunedEntryCountThisTurn: 0,
      supersededEntryCountThisTurn: 0,
      nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
    });

    return degraded(input, "invalid_patch");
  }

  const clock = input.clock ?? new SystemClock();
  const nowMs = clock.now();
  const lifecycle = applyDecisionArtifactLifecycleCap({
    previousArtifact,
    operations: normalized.operations,
    options: input.lifecycle,
    nowMs,
  });
  if (lifecycle.overCapDelta > 0 && input.tracer?.enabled === true && input.turnId !== undefined) {
    input.tracer.emit("decision_artifact_lifecycle_unable_to_cap", {
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      maxActiveEntries: lifecycle.maxActiveEntries,
      postPlanActiveEntryCount: lifecycle.postPlanActiveEntryCount,
      overCapDelta: lifecycle.overCapDelta,
    });
  }
  const expandedOperations = expandPruneDependencies({
    previousArtifact,
    operations: lifecycle.operations,
    nowMs,
  });
  const dedupedCanonicalizations = dedupeCanonicalizesAcrossOperations(expandedOperations);
  const operations = dedupedCanonicalizations.operations;
  const prunedEntryCountThisTurn = operations.filter(
    (operation) => operation.type === "prune",
  ).length;
  const supersededEntryCountThisTurn = operations.filter(
    (operation) => operation.type === "supersede",
  ).length;

  if (operations.length === 0) {
    let markedArtifact = previousArtifact;

    try {
      markedArtifact = input.repository.upsert(input.audienceEntityId, [], {
        expectedVersion: previousArtifact?.record_version,
        now: nowMs,
        lastCompiledAt: nowMs,
        lastCompiledStreamEntryId: input.currentUserStreamEntryId,
        sourceTrustValidator: input.sourceTrustValidator,
      });
    } catch (error) {
      traceCompileCompleted({
        ...compileCompletedTraceBase,
        operationCount: 0,
        rejected: normalized.rejected,
        applied: false,
        artifact: previousArtifact,
        prunedEntryCountThisTurn,
        supersededEntryCountThisTurn,
        nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
      });

      return degraded(input, "repository_failed", error);
    }

    const reconciliationWorkSet = buildDecisionArtifactReconciliationWorkSet({
      artifact: markedArtifact,
      operations: [],
      repositories: input.reconciliation,
      nowMs,
    });
    const reconciliationResult = reconcileDecisionArtifactCanonicalizations({
      entries: reconciliationWorkSet.entries,
      repositories: input.reconciliation,
      unknownIds: normalized.droppedCanonicalizeIds,
      nowMs,
      sourceTrustValidator: input.sourceTrustValidator,
      tracer: input.tracer,
      turnId: input.turnId,
    });
    const semanticReconciliationResult = await reconcileSemanticBeliefRevisionFailOpen({
      artifact: markedArtifact,
      operations: [],
      dependencies: semanticBeliefRevisionDependencies(input),
      nowMs,
      sourceTrustValidator: input.sourceTrustValidator,
      tracer: input.tracer,
      turnId: input.turnId,
      turnCounter: input.turnCounter,
    });
    mergeSemanticBeliefRevisionResult(reconciliationResult, semanticReconciliationResult);

    traceReconciliationCompleted({
      tracer: input.tracer,
      turnId: input.turnId,
      result: reconciliationResult,
      canonicalizationDuplicateDrops: dedupedCanonicalizations.duplicateDrops,
      currentOperationCanonicalizationCount:
        reconciliationWorkSet.currentOperationCanonicalizationCount,
      retriedStrandedCanonicalizationCount:
        reconciliationWorkSet.retriedStrandedCanonicalizationCount,
      retrySummary: reconciliationWorkSet.retrySummary,
    });

    traceCompileCompleted({
      ...compileCompletedTraceBase,
      operationCount: 0,
      rejected: normalized.rejected,
      applied: false,
      artifact: markedArtifact,
      prunedEntryCountThisTurn,
      supersededEntryCountThisTurn,
      nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
    });

    return emptyPatch();
  }

  try {
    const nextArtifact = input.repository.upsert(input.audienceEntityId, operations, {
      expectedVersion: previousArtifact?.record_version,
      now: nowMs,
      lastCompiledAt: nowMs,
      lastCompiledStreamEntryId: input.currentUserStreamEntryId,
      sourceTrustValidator: input.sourceTrustValidator,
    });

    const reconciliationWorkSet = buildDecisionArtifactReconciliationWorkSet({
      artifact: nextArtifact,
      operations,
      repositories: input.reconciliation,
      nowMs,
    });
    const reconciliationResult = reconcileDecisionArtifactCanonicalizations({
      entries: reconciliationWorkSet.entries,
      repositories: input.reconciliation,
      unknownIds: normalized.droppedCanonicalizeIds,
      nowMs,
      sourceTrustValidator: input.sourceTrustValidator,
      tracer: input.tracer,
      turnId: input.turnId,
    });
    const semanticReconciliationResult = await reconcileSemanticBeliefRevisionFailOpen({
      artifact: nextArtifact,
      operations,
      dependencies: semanticBeliefRevisionDependencies(input),
      nowMs,
      sourceTrustValidator: input.sourceTrustValidator,
      tracer: input.tracer,
      turnId: input.turnId,
      turnCounter: input.turnCounter,
    });
    mergeSemanticBeliefRevisionResult(reconciliationResult, semanticReconciliationResult);

    traceReconciliationCompleted({
      tracer: input.tracer,
      turnId: input.turnId,
      result: reconciliationResult,
      canonicalizationDuplicateDrops: dedupedCanonicalizations.duplicateDrops,
      currentOperationCanonicalizationCount:
        reconciliationWorkSet.currentOperationCanonicalizationCount,
      retriedStrandedCanonicalizationCount:
        reconciliationWorkSet.retriedStrandedCanonicalizationCount,
      retrySummary: reconciliationWorkSet.retrySummary,
    });

    traceCompileCompleted({
      ...compileCompletedTraceBase,
      operationCount: operations.length,
      rejected: normalized.rejected,
      applied: true,
      artifact: nextArtifact,
      prunedEntryCountThisTurn,
      supersededEntryCountThisTurn,
      nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
    });
  } catch (error) {
    traceCompileCompleted({
      ...compileCompletedTraceBase,
      operationCount: operations.length,
      rejected: normalized.rejected,
      applied: false,
      artifact: previousArtifact,
      prunedEntryCountThisTurn,
      supersededEntryCountThisTurn,
      nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
    });

    return degraded(input, "repository_failed", error);
  }

  return {
    operations: operations.map((operation) => {
      switch (operation.type) {
        case "add":
          return {
            type: "add",
            kind: operation.kind,
            text: operation.text,
            owner_entity_id: operation.owner_entity_id,
            source_stream_entry_ids: [...operation.provenance_stream_entry_ids],
            ...(operation.canonicalizes === undefined
              ? {}
              : { canonicalizes: operation.canonicalizes }),
          };
        case "update":
          return {
            type: "update",
            id: operation.id,
            kind: operation.kind,
            text: operation.text,
            owner_entity_id: operation.owner_entity_id,
            source_stream_entry_ids: [...operation.last_updated_stream_entry_ids],
            ...(operation.canonicalizes === undefined
              ? {}
              : { canonicalizes: operation.canonicalizes }),
          };
        case "supersede":
          return {
            type: "supersede",
            id: operation.id,
            replacement: {
              kind: operation.replacement.kind,
              text: operation.replacement.text,
              owner_entity_id: operation.replacement.owner_entity_id,
              source_stream_entry_ids: [...operation.replacement.provenance_stream_entry_ids],
            },
            source_stream_entry_ids: [...operation.last_updated_stream_entry_ids],
            ...(operation.replacement.canonicalizes === undefined
              ? {}
              : { canonicalizes: operation.replacement.canonicalizes }),
          };
        case "prune":
          return {
            type: "prune",
            id: operation.id,
          };
      }
    }),
  };
}

export { DECISION_ARTIFACT_TOOL_NAME, DECISION_ARTIFACT_SYSTEM_PROMPT, MAX_PATCH_OUTPUT_TOKENS };
