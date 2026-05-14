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
  type DecisionArtifactEntry,
  type DecisionArtifactEntryKind,
  type DecisionArtifactOperation,
  type DecisionArtifactRepository,
} from "../../memory/decision-artifacts/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { JsonValue } from "../../util/json-value.js";
import {
  createDecisionArtifactEntryId,
  decisionArtifactEntryIdHelpers,
  entityIdHelpers,
  streamEntryIdHelpers,
  type DecisionArtifactEntryId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";
import {
  summarizeDecisionStateArtifactRender,
  type DecisionArtifactRenderOptions,
} from "../evidence-ledger/index.js";
import { buildUsageTraceBlock, toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";

const DECISION_ARTIFACT_TOOL_NAME = "EmitDecisionArtifactPatch";
const MAX_OPERATIONS_PER_COMPILE = 40;
const MAX_PATCH_OUTPUT_TOKENS = 1536;
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
    "Compile shared planning decision state into additive, updating, superseding, or pruning operations.",
  inputSchema: toToolInputSchema(decisionArtifactPatchSchema),
} satisfies LLMToolDefinition;

const DECISION_ARTIFACT_SYSTEM_PROMPT = [
  "Compile shared planning state from the previous artifact, the current user turn, and the prompt-visible ledger.",
  "Return only the required tool call.",
  "",
  "Scope:",
  "- This is a structure compiler. It does not decide whether Borg's answer is correct.",
  "- It does not rewrite, suppress, approve, or police user-facing text.",
  "- It must not invent facts, owners, or commitments.",
  "- Prefer no operations when uncertain.",
  "",
  "Artifact kinds:",
  '- locked: canonical decisions the group has agreed to, such as "Granada 3 nights".',
  "- live: under-active-discussion decisions or proposals the group is currently working on.",
  "- tentative: weak proposals not yet endorsed; trial-balloon facts.",
  "- invalidated: assumptions explicitly overturned by later evidence; kept for context, not action.",
  '- pending: items awaiting verification or external information, such as "Ben to check Alhambra booking windows".',
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
  "- Prefer update, supersede, and prune over add whenever the existing artifact already carries the relevant planning state.",
].join("\n");

export type EmitDecisionArtifactPatch = z.infer<typeof decisionArtifactPatchSchema>;

export type DecisionArtifactParticipantContext = {
  entityId: EntityId;
  displayName?: string | null;
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
  clock?: Clock;
  tracer?: TurnTracer;
  turnId?: string;
  lifecycle?: DecisionArtifactLifecycleOptions;
  renderOptions?: DecisionArtifactRenderOptions;
  onDegraded?: (
    reason: DecisionArtifactCompileDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

type ParsedPatchOperation = EmitDecisionArtifactPatch["operations"][number];

type PatchRejection = {
  reason:
    | "invalid_entry_id"
    | "unknown_entry_id"
    | "invalid_owner_entity_id"
    | "unsupported_kind"
    | "missing_citation"
    | "invalid_source_stream_entry_id"
    | "disallowed_source_stream_entry_id"
    | "empty_update";
  operationType: ParsedPatchOperation["type"];
  operationIndex: number;
};

class MissingDecisionArtifactToolCallError extends Error {}

function emptyPatch(): EmitDecisionArtifactPatch {
  return { operations: [] };
}

function buildDecisionArtifactMessages(input: {
  audienceEntityId: EntityId;
  selfEntityId: EntityId;
  speakerEntityId: EntityId | null;
  participants: readonly DecisionArtifactParticipantContext[];
  currentUserMessage: string;
  currentUserStreamEntryId: StreamEntryId;
  promptVisibleLedger: string;
  previousArtifact: DecisionArtifact | null;
}): LLMMessage[] {
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
        previous_artifact: input.previousArtifact,
        prompt_visible_ledger: input.promptVisibleLedger,
      }),
    },
  ];
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
    });
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
): { streamEntryIds: StreamEntryId[]; reason: PatchRejection["reason"] | null } {
  if (values.length === 0) {
    return { streamEntryIds: [], reason: "missing_citation" };
  }

  const streamEntryIds: StreamEntryId[] = [];

  for (const value of values) {
    if (!streamEntryIdHelpers.is(value)) {
      return { streamEntryIds: [], reason: "invalid_source_stream_entry_id" };
    }

    if (allowedSourceStreamEntryIds !== null && !allowedSourceStreamEntryIds.has(value)) {
      return { streamEntryIds: [], reason: "disallowed_source_stream_entry_id" };
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
): PatchRejection {
  return {
    reason,
    operationType: operation.type,
    operationIndex,
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
}): { operations: DecisionArtifactOperation[]; rejected: PatchRejection[] } {
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
        );

        if (citations.reason !== null) {
          rejected.push(rejection(operation, operationIndex, citations.reason));
          return;
        }

        operations.push({
          type: "add",
          kind: operation.kind,
          text: operation.text,
          owner_entity_id: ownerEntityId,
          provenance_stream_entry_ids: citations.streamEntryIds,
          last_updated_stream_entry_ids: citations.streamEntryIds,
          rank: baseRank + operations.length,
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
          operation.owner_entity_id === undefined
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
        );

        if (citations.reason !== null) {
          rejected.push(rejection(operation, operationIndex, citations.reason));
          return;
        }

        operations.push({
          type: "update",
          id,
          kind: nextKind,
          text: operation.text,
          owner_entity_id: operation.owner_entity_id === undefined ? undefined : ownerEntityId,
          add_provenance_stream_entry_ids: citations.streamEntryIds,
          last_updated_stream_entry_ids: citations.streamEntryIds,
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
        );

        if (replacementCitations.reason !== null) {
          rejected.push(rejection(operation, operationIndex, replacementCitations.reason));
          return;
        }

        const updateCitationValues =
          operation.source_stream_entry_ids ?? operation.replacement.source_stream_entry_ids;
        const updateCitations = parseSourceStreamEntryIds(
          updateCitationValues,
          input.allowedSourceStreamEntryIds,
        );

        if (updateCitations.reason !== null) {
          rejected.push(rejection(operation, operationIndex, updateCitations.reason));
          return;
        }

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

  return { operations, rejected };
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
  const messages = buildDecisionArtifactMessages({
    audienceEntityId: input.audienceEntityId,
    selfEntityId: input.selfEntityId,
    speakerEntityId,
    participants: input.participants,
    currentUserMessage: input.currentUserMessage,
    currentUserStreamEntryId: input.currentUserStreamEntryId,
    promptVisibleLedger: input.promptVisibleLedger,
    previousArtifact,
  });
  const tools = [DECISION_ARTIFACT_TOOL];

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
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      previousEntryCount,
      operationCount: 0,
      rejected: [],
      applied: false,
      artifact: previousArtifact,
      renderOptions: input.renderOptions,
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
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      previousEntryCount,
      operationCount: 0,
      rejected: [],
      applied: false,
      artifact: previousArtifact,
      renderOptions: input.renderOptions,
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
  });

  if (normalized.operations.length === 0 && normalized.rejected.length > 0) {
    traceCompileCompleted({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      previousEntryCount,
      operationCount: 0,
      rejected: normalized.rejected,
      applied: false,
      artifact: previousArtifact,
      renderOptions: input.renderOptions,
      prunedEntryCountThisTurn: 0,
      supersededEntryCountThisTurn: 0,
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
  const operations = expandPruneDependencies({
    previousArtifact,
    operations: lifecycle.operations,
    nowMs,
  });
  const prunedEntryCountThisTurn = operations.filter(
    (operation) => operation.type === "prune",
  ).length;
  const supersededEntryCountThisTurn = operations.filter(
    (operation) => operation.type === "supersede",
  ).length;

  if (operations.length === 0) {
    traceCompileCompleted({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      previousEntryCount,
      operationCount: 0,
      rejected: normalized.rejected,
      applied: false,
      artifact: previousArtifact,
      renderOptions: input.renderOptions,
      prunedEntryCountThisTurn,
      supersededEntryCountThisTurn,
    });

    return emptyPatch();
  }

  try {
    const nextArtifact = input.repository.upsert(input.audienceEntityId, operations, {
      expectedVersion: previousArtifact?.record_version,
      now: nowMs,
      lastCompiledAt: nowMs,
      lastCompiledStreamEntryId: input.currentUserStreamEntryId,
    });

    traceCompileCompleted({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      previousEntryCount,
      operationCount: operations.length,
      rejected: normalized.rejected,
      applied: true,
      artifact: nextArtifact,
      renderOptions: input.renderOptions,
      prunedEntryCountThisTurn,
      supersededEntryCountThisTurn,
    });
  } catch (error) {
    traceCompileCompleted({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      previousEntryCount,
      operationCount: operations.length,
      rejected: normalized.rejected,
      applied: false,
      artifact: previousArtifact,
      renderOptions: input.renderOptions,
      prunedEntryCountThisTurn,
      supersededEntryCountThisTurn,
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
          };
        case "update":
          return {
            type: "update",
            id: operation.id,
            kind: operation.kind,
            text: operation.text,
            owner_entity_id: operation.owner_entity_id,
            source_stream_entry_ids: [...operation.last_updated_stream_entry_ids],
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
