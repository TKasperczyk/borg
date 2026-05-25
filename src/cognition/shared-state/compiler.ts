import { z } from "zod";

import type { LLMCompleteResult } from "../../llm/index.js";
import type {
  SharedStateArtifact,
  SharedStateEntry,
  SharedStateEntryKind,
  SharedStateOperation,
} from "../../memory/decision-artifacts/index.js";
import { dedupePreservingOrder } from "../../util/collections.js";
import { SystemClock } from "../../util/clock.js";
import type { StreamEntryId } from "../../util/ids.js";
import { PROTECTED_RELATIONSHIP_LABELS } from "../prompts/relationship-labels.js";
import { SHARED_STATE_SYSTEM_PROMPT } from "../prompts/shared-state.js";
import {
  mergeSemanticBeliefRevisionResult,
  reconcileSharedStateCanonicalizations,
  reconcileSemanticBeliefRevision,
  type SharedStateSemanticBeliefRevisionDependencies,
} from "./reconciliation.js";
import {
  buildSharedStateArtifactMessages,
  estimateSharedStateArtifactPromptBudget,
} from "./compiler-prompt.js";
import {
  MissingSharedStateArtifactToolCallError,
  SHARED_STATE_TOOLS,
  type CompileSharedStateArtifactInput,
} from "./schema.js";
import {
  MAX_PATCH_OUTPUT_TOKENS,
  SHARED_STATE_TOOL_NAME,
} from "./constants.js";
import {
  type EmptyUpdateDrop,
  type EmitSharedStatePatch,
  type PatchRejection,
  type SharedStateCompileDegradedReason,
} from "./types.js";
import {
  parseResponse,
  summarizeSharedStateArtifactResponseShape,
  traceAddRejectedMissingNewKeyReason,
  traceAddRejectedCapExceeded,
  traceAddRejectedNearDuplicateStateKey,
  traceCompileCompleted,
  traceCompileDegraded,
  traceCompileOverBudget,
  traceCompileRepairAttempted,
  traceCompileRepairFailed,
  traceCompileRepairSucceeded,
  traceEmptyUpdateDropped,
  traceLabelUngrounded,
  traceReconciliationCompleted,
} from "./compiler-io.js";
import {
  traceLlmCallError,
  traceLlmCallResponse,
  traceLlmCallStarted,
} from "../tracing/llm-call-trace.js";
import {
  allowedCanonicalizationIds,
  dedupeCanonicalizesAcrossOperations,
  normalizePatch,
} from "./patch-validation.js";
import { errorMessage } from "./reconciliation-summary.js";
import { applySharedStateArtifactLifecycleCap, expandPruneDependencies } from "./lifecycle-cap.js";
import { materializeSharedStateOperationIds } from "./lifecycle-cap.js";
import {
  applyLifecycleAging,
  materializeSharedStateEntriesAfterOperations,
  type SharedStateLifecycleTransition,
} from "./lifecycle-aging.js";
import { buildSharedStateReconciliationWorkSet } from "./canonicalization-candidates.js";
import { buildExistingStateKeyRegistry, buildSharedStateArtifactPromptSummary } from "./summary.js";
import { SHARED_STATE_RECENT_TURN_THRESHOLD } from "./render.js";

function semanticBeliefRevisionDependencies(
  input: CompileSharedStateArtifactInput,
): SharedStateSemanticBeliefRevisionDependencies | undefined {
  if (input.semanticBeliefRevision === undefined || input.llmClient === undefined) {
    return undefined;
  }

  return {
    ...input.semanticBeliefRevision,
    llmClient: input.llmClient,
  };
}

function traceSharedStateSemanticRevisionDegraded(
  input: Parameters<typeof reconcileSemanticBeliefRevision>[0],
  error: unknown,
): void {
  if (input.tracer?.enabled !== true || input.turnId === undefined) {
    return;
  }

  input.tracer.emit("shared_state.semantic_revision.degraded", {
    turnId: input.turnId,
    reason: errorMessage(error),
    skipped_due_to_error: 1,
  });
}

async function reconcileSemanticBeliefRevisionFailOpen(
  input: Parameters<typeof reconcileSemanticBeliefRevision>[0],
): ReturnType<typeof reconcileSemanticBeliefRevision> {
  try {
    return await reconcileSemanticBeliefRevision(input);
  } catch (error) {
    traceSharedStateSemanticRevisionDegraded(input, error);

    return {
      semantic_nodes_reviewed_attempted: 0,
      semantic_nodes_marked_superseded: 0,
      semantic_nodes_marked_contradicted: 0,
      semantic_nodes_skipped: 0,
    };
  }
}

function emptyPatch(): EmitSharedStatePatch {
  return { operations: [] };
}

type SharedStateOperationKind = EmitSharedStatePatch["operations"][number]["type"];
type PublicSharedStateOperation = Exclude<SharedStateOperation, { type: "transition_kind" }>;
type SharedStateToolEntryKind = Extract<
  EmitSharedStatePatch["operations"][number],
  { type: "add" }
>["kind"];

function emptyOperationCountsByKind(): Record<SharedStateOperationKind, number> {
  return {
    add: 0,
    update: 0,
    supersede: 0,
    prune: 0,
  };
}

function operationCountsByKind(
  operations: readonly { type: SharedStateOperationKind }[],
): Record<SharedStateOperationKind, number> {
  const counts = emptyOperationCountsByKind();

  for (const operation of operations) {
    counts[operation.type] += 1;
  }

  return counts;
}

function emptyOperationCountsByStateKey(): Record<SharedStateOperationKind, number> {
  return emptyOperationCountsByKind();
}

function entryStateKeyById(
  artifact: SharedStateArtifact | null,
): Map<SharedStateEntry["id"], string | null> {
  return new Map((artifact?.entries ?? []).map((entry) => [entry.id, entry.state_key]));
}

function operationStateKey(
  operation: SharedStateOperation,
  previousStateKeysById: ReadonlyMap<SharedStateEntry["id"], string | null>,
): string | null {
  switch (operation.type) {
    case "add":
      return operation.state_key ?? null;
    case "update":
      return operation.state_key ?? previousStateKeysById.get(operation.id) ?? null;
    case "supersede":
      return operation.replacement.state_key ?? previousStateKeysById.get(operation.id) ?? null;
    case "prune":
      return previousStateKeysById.get(operation.id) ?? null;
    case "transition_kind":
      return previousStateKeysById.get(operation.id) ?? null;
  }
}

function publicSharedStateOperations(
  operations: readonly SharedStateOperation[],
): PublicSharedStateOperation[] {
  return operations.filter(
    (operation): operation is PublicSharedStateOperation => operation.type !== "transition_kind",
  );
}

function publicKindForOmittedUpdate(
  entryKind: SharedStateEntryKind | undefined,
): SharedStateToolEntryKind | undefined {
  if (entryKind === "low_salience_live" || entryKind === "dormant_live") {
    return "live";
  }

  return entryKind as SharedStateToolEntryKind | undefined;
}

function touchedSharedStateEntryIds(
  operations: readonly SharedStateOperation[],
): Set<SharedStateEntry["id"]> {
  const ids = new Set<SharedStateEntry["id"]>();

  for (const operation of operations) {
    if (operation.type === "update" || operation.type === "supersede") {
      ids.add(operation.id);
    }
  }

  return ids;
}

function lastUpdatedTurnByEntryId(input: {
  entries: readonly SharedStateEntry[];
  currentUserStreamEntryId: StreamEntryId;
  currentTurnCounter?: number;
  lastUpdatedTurnByStreamEntryId?: Readonly<Record<string, number>>;
}): Record<string, number> {
  const turnsByStreamEntryId = {
    ...(input.lastUpdatedTurnByStreamEntryId ?? {}),
    ...(input.currentTurnCounter === undefined
      ? {}
      : { [input.currentUserStreamEntryId]: input.currentTurnCounter }),
  };
  const result: Record<string, number> = {};

  for (const entry of input.entries) {
    for (const streamEntryId of entry.last_updated_stream_entry_ids) {
      const turn = turnsByStreamEntryId[streamEntryId];

      if (turn === undefined || !Number.isFinite(turn)) {
        continue;
      }

      const previous = result[entry.id];
      result[entry.id] = previous === undefined ? turn : Math.max(previous, turn);
    }
  }

  return result;
}

function lastUpdatedTurnForStreamIds(
  streamEntryIds: readonly StreamEntryId[],
  turnsByStreamEntryId: Readonly<Record<string, number>> | undefined,
  fallback: number | null,
): number | null {
  let result: number | null = null;

  for (const streamEntryId of streamEntryIds) {
    const turn = turnsByStreamEntryId?.[streamEntryId];
    if (turn === undefined || !Number.isFinite(turn)) {
      continue;
    }

    result = result === null ? turn : Math.max(result, turn);
  }

  return result ?? fallback;
}

function withOperationLastUpdatedTurns(
  operations: readonly SharedStateOperation[],
  input: CompileSharedStateArtifactInput,
): SharedStateOperation[] {
  const fallback = input.turnCounter ?? null;
  const turnsByStreamEntryId = {
    ...(input.renderOptions?.lastUpdatedTurnByStreamEntryId ?? {}),
    ...(input.turnCounter === undefined
      ? {}
      : { [input.currentUserStreamEntryId]: input.turnCounter }),
  };

  return operations.map((operation) => {
    switch (operation.type) {
      case "add": {
        const streamEntryIds =
          operation.last_updated_stream_entry_ids ?? operation.provenance_stream_entry_ids;
        return {
          ...operation,
          last_updated_turn_global: lastUpdatedTurnForStreamIds(
            streamEntryIds,
            turnsByStreamEntryId,
            fallback,
          ),
        };
      }
      case "update":
        return {
          ...operation,
          last_updated_turn_global: lastUpdatedTurnForStreamIds(
            operation.last_updated_stream_entry_ids,
            turnsByStreamEntryId,
            fallback,
          ),
        };
      case "supersede":
        return {
          ...operation,
          last_updated_turn_global: lastUpdatedTurnForStreamIds(
            operation.last_updated_stream_entry_ids,
            turnsByStreamEntryId,
            fallback,
          ),
          replacement: {
            ...operation.replacement,
            last_updated_turn_global: lastUpdatedTurnForStreamIds(
              operation.replacement.last_updated_stream_entry_ids ??
                operation.replacement.provenance_stream_entry_ids,
              turnsByStreamEntryId,
              fallback,
            ),
          },
        };
      default:
        return operation;
    }
  });
}

function traceSharedStateLifecycleTransitions(input: {
  tracer: CompileSharedStateArtifactInput["tracer"];
  turnId: CompileSharedStateArtifactInput["turnId"];
  audienceEntityId: CompileSharedStateArtifactInput["audienceEntityId"];
  transitions: readonly SharedStateLifecycleTransition[];
}): void {
  if (input.tracer?.enabled !== true || input.turnId === undefined) {
    return;
  }

  for (const transition of input.transitions) {
    input.tracer.emit(
      transition.transition === "reactivated"
        ? "shared_state.lifecycle.reactivated"
        : "shared_state.lifecycle.demoted",
      {
        turnId: input.turnId,
        audienceEntityId: input.audienceEntityId,
        entry_id: transition.entryId,
        from_kind: transition.fromKind,
        to_kind: transition.toKind,
        reason: transition.reason,
      },
    );
  }
}

function operationCountsByStateKey(
  operations: readonly PublicSharedStateOperation[],
  previousArtifact: SharedStateArtifact | null,
): Record<string, Record<SharedStateOperationKind, number>> {
  const previousStateKeysById = entryStateKeyById(previousArtifact);
  const counts: Record<string, Record<SharedStateOperationKind, number>> = {};

  for (const operation of operations) {
    const stateKey = operationStateKey(operation, previousStateKeysById);

    if (stateKey === null) {
      continue;
    }

    counts[stateKey] = counts[stateKey] ?? emptyOperationCountsByStateKey();
    counts[stateKey][operation.type] += 1;
  }

  return Object.fromEntries(
    Object.entries(counts).sort(([left], [right]) => left.localeCompare(right)),
  );
}

function activeStateKeys(artifact: SharedStateArtifact | null): Set<string> {
  return new Set(
    (artifact?.entries ?? []).flatMap((entry) => {
      if (entry.superseded_by_id !== null || entry.state_key === null) {
        return [];
      }

      return [entry.state_key];
    }),
  );
}

function introducedStateKeys(
  operations: readonly SharedStateOperation[],
  previousArtifact: SharedStateArtifact | null,
): string[] {
  const seen = activeStateKeys(previousArtifact);
  const introduced: string[] = [];

  for (const operation of operations) {
    const stateKey =
      operation.type === "add"
        ? operation.state_key
        : operation.type === "update"
          ? operation.state_key
          : operation.type === "supersede"
            ? operation.replacement.state_key
            : null;

    if (stateKey === null || seen.has(stateKey)) {
      continue;
    }

    seen.add(stateKey);
    introduced.push(stateKey);
  }

  return introduced.sort((left, right) => left.localeCompare(right));
}

function repairablePatchRejections(rejections: readonly PatchRejection[]): PatchRejection[] {
  return rejections.filter(
    (rejection) =>
      rejection.reason === "live_entry_cap_exceeded_for_key" ||
      rejection.reason === "locked_state_key_collision" ||
      rejection.reason === "near_duplicate_state_key" ||
      rejection.reason === "missing_new_key_reason" ||
      rejection.reason === "relationship_label_ungrounded",
  );
}

function patchRejectionRepairMessage(rejections: readonly PatchRejection[]): string {
  const details = rejections
    .map((rejection) => {
      if (rejection.reason === "live_entry_cap_exceeded_for_key") {
        return [
          `operation ${rejection.operationIndex} add state_key=${rejection.stateKey ?? "unknown"}`,
          `would create ${rejection.proposedCount ?? "too many"} live entries`,
          `with max ${rejection.maxLiveEntriesPerKey ?? "unknown"}`,
          rejection.targetEntryId === undefined
            ? "use update or supersede on an existing same-key live entry"
            : `use update or supersede around existing entry ${rejection.targetEntryId}`,
        ].join("; ");
      }

      if (rejection.reason === "locked_state_key_collision") {
        const lockedIds = rejection.lockedEntryIds?.join(", ") || "an existing locked entry";

        return [
          `operation ${rejection.operationIndex} add state_key=${rejection.stateKey ?? "unknown"}`,
          `collides with locked entry ${lockedIds}`,
          "update/supersede the locked entry, or mark unsettled material tentative/pending",
        ].join("; ");
      }

      if (rejection.reason === "near_duplicate_state_key") {
        const similarStateKey = rejection.similarStateKeys?.[0] ?? "an active state_key";
        const sharedTokens = rejection.sharedStateKeyTokens?.join(", ") || "unknown";

        return [
          `operation ${rejection.operationIndex} add used state_key=${rejection.stateKey ?? "unknown"}`,
          `but active state_key=${similarStateKey} already exists for the same audience and appears to cover the same thread`,
          `shared tokens: ${sharedTokens}`,
          "reuse the existing key with update or supersede, or emit a genuinely distinct key with new_key_reason explaining what makes it different",
        ].join("; ");
      }

      if (rejection.reason === "missing_new_key_reason") {
        return [
          `operation ${rejection.operationIndex} add used never-seen state_key=${rejection.stateKey ?? "unknown"}`,
          "with no new_key_reason",
          "add a brief new_key_reason explaining what new object/thread this represents, or reuse an existing key from the registry",
        ].join("; ");
      }

      if (rejection.reason === "relationship_label_ungrounded") {
        const strictLabels = PROTECTED_RELATIONSHIP_LABELS.join(", ");
        return [
          `operation ${rejection.operationIndex} ${rejection.operationType}`,
          `uses strict relationship label ${rejection.protectedRelationshipLabels?.join(", ") || "unknown"}`,
          `strict labels are ${strictLabels}`,
          "if the label refers to a person-to-person role assignment, add relationship_evidence_relational_slot_ids citing a supplied relational slot, or relationship_evidence_stream_entry_ids citing a supplied trusted user message",
          "if the strict label appears incidental or as part of a context noun phrase, rewrite the operation to remove or replace the strict term with a generic noun such as family member, partner, or helper",
          "if the meaning belongs in another operation shape, re-emit it as a different operation kind that does not carry the strict assignment",
        ].join("; ");
      }

      return `operation ${rejection.operationIndex} rejected: ${rejection.reason}`;
    })
    .join(" | ");

  return `Your previous patch violated structural shared-state key compaction: ${details}. Emit a corrected patch.`;
}

function traceRepairablePatchRejection(
  input: CompileSharedStateArtifactInput,
  rejection: PatchRejection,
): void {
  if (rejection.reason === "live_entry_cap_exceeded_for_key") {
    traceAddRejectedCapExceeded({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      rejection,
    });
    return;
  }

  if (rejection.reason === "near_duplicate_state_key") {
    traceAddRejectedNearDuplicateStateKey({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      rejection,
    });
    return;
  }

  if (rejection.reason === "missing_new_key_reason") {
    traceAddRejectedMissingNewKeyReason({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      rejection,
    });
    return;
  }

  if (rejection.reason === "relationship_label_ungrounded") {
    traceLabelUngrounded({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      rejection,
    });
  }
}

function traceEmptyUpdateDrops(
  input: CompileSharedStateArtifactInput,
  drops: readonly EmptyUpdateDrop[],
): void {
  for (const drop of drops) {
    traceEmptyUpdateDropped({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      drop,
    });
  }
}

function uniqueStreamEntryIds(ids: readonly StreamEntryId[]): StreamEntryId[] {
  return dedupePreservingOrder(ids);
}

function relationalSlotEvidenceStreamEntryIds(
  input: Pick<CompileSharedStateArtifactInput, "relationalSlotsContext">,
): StreamEntryId[] {
  return uniqueStreamEntryIds(
    (input.relationalSlotsContext ?? []).flatMap((slot) => slot.evidence_stream_entry_ids),
  );
}

function trustedSourceStreamEntryIds(
  streamEntryIds: readonly StreamEntryId[],
  input: Pick<CompileSharedStateArtifactInput, "sourceTrustValidator">,
): StreamEntryId[] {
  return streamEntryIds.filter(
    (streamEntryId) => input.sourceTrustValidator?.(streamEntryId).allowed !== false,
  );
}

function offLimitsSourceStreamEntryIds(
  streamEntryIds: readonly StreamEntryId[],
  input: Pick<CompileSharedStateArtifactInput, "sourceTrustValidator">,
): StreamEntryId[] {
  if (input.sourceTrustValidator === undefined) {
    return [];
  }

  return streamEntryIds.filter(
    (streamEntryId) => input.sourceTrustValidator?.(streamEntryId).allowed === false,
  );
}

async function degraded(
  input: CompileSharedStateArtifactInput,
  reason: SharedStateCompileDegradedReason,
  error?: unknown,
): Promise<EmitSharedStatePatch> {
  traceCompileDegraded({
    tracer: input.tracer,
    turnId: input.turnId,
    audienceEntityId: input.audienceEntityId,
    reason,
    error,
  });

  try {
    await input.onDegraded?.(reason, error);
  } catch {
    // Best-effort degraded-mode logging only.
  }

  return emptyPatch();
}

export async function compileSharedStateArtifact(
  input: CompileSharedStateArtifactInput,
): Promise<EmitSharedStatePatch> {
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
  const previousArtifactSummary = buildSharedStateArtifactPromptSummary(
    previousArtifact,
    input.previousArtifactSummaryOptions,
  );
  const existingStateKeyRegistry = buildExistingStateKeyRegistry(previousArtifact);
  const canonicalizationCandidates = input.canonicalizationCandidates ?? {};
  const relationalSlotSourceStreamEntryIds = relationalSlotEvidenceStreamEntryIds(input);
  const allowedSourceStreamEntryIdsForPrompt =
    input.allowedSourceStreamEntryIds === undefined
      ? undefined
      : uniqueStreamEntryIds([
          ...input.allowedSourceStreamEntryIds,
          ...trustedSourceStreamEntryIds(relationalSlotSourceStreamEntryIds, input),
        ]);
  const offLimitsSourceStreamEntryIdsForPrompt = uniqueStreamEntryIds([
    ...(input.offLimitsSourceStreamEntryIds ?? []),
    ...offLimitsSourceStreamEntryIds(relationalSlotSourceStreamEntryIds, input),
  ]);
  const messageInput = {
    audienceEntityId: input.audienceEntityId,
    selfEntityId: input.selfEntityId,
    speakerEntityId,
    participants: input.participants,
    participantRoster: input.participantRoster,
    currentUserMessage: input.currentUserMessage,
    currentUserStreamEntryId: input.currentUserStreamEntryId,
    promptVisibleLedger: input.promptVisibleLedger,
    existingStateKeyRegistry,
    previousArtifactSummary,
    canonicalizationCandidates,
    relationalSlotsContext: input.relationalSlotsContext,
    allowedSourceStreamEntryIds: allowedSourceStreamEntryIdsForPrompt,
    offLimitsSourceStreamEntryIds: offLimitsSourceStreamEntryIdsForPrompt,
  };
  const messages = buildSharedStateArtifactMessages(messageInput);
  const tools = SHARED_STATE_TOOLS;
  const ledgerMode = input.ledgerMode ?? "full_fallback";
  const promptBudget = estimateSharedStateArtifactPromptBudget({
    messages,
    tools,
    previousArtifactSummary,
    existingStateKeyRegistry,
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
    currentTurnCounter: input.turnCounter,
    currentUserStreamEntryId: input.currentUserStreamEntryId,
    ledgerMode,
    promptBudget,
    emptyUpdateAttemptedCount: 0,
    emptyUpdateDroppedCount: 0,
    emptyUpdateRepairedCount: 0,
    addRejectedCapExceededCount: 0,
  };
  const recordEmptyUpdateValidation = (normalizedPatch: {
    emptyUpdateAttemptedCount: number;
    emptyUpdateDrops: readonly EmptyUpdateDrop[];
  }): void => {
    compileCompletedTraceBase.emptyUpdateAttemptedCount +=
      normalizedPatch.emptyUpdateAttemptedCount;
    compileCompletedTraceBase.emptyUpdateDroppedCount += normalizedPatch.emptyUpdateDrops.length;
    traceEmptyUpdateDrops(input, normalizedPatch.emptyUpdateDrops);
  };

  traceLlmCallStarted({
    tracer: input.tracer,
    turnId: input.turnId,
    label: "decision_artifact_compiler",
    model: input.model,
    systemPrompt: SHARED_STATE_SYSTEM_PROMPT,
    messages,
    tools,
  });

  let response: LLMCompleteResult;

  try {
    response = await input.llmClient.complete({
      model: input.model,
      system: SHARED_STATE_SYSTEM_PROMPT,
      messages,
      tools,
      tool_choice: { type: "tool", name: SHARED_STATE_TOOL_NAME },
      max_tokens: MAX_PATCH_OUTPUT_TOKENS,
      budget: "decision-artifact-compiler",
    });
  } catch (error) {
    traceLlmCallError({
      tracer: input.tracer,
      turnId: input.turnId,
      label: "decision_artifact_compiler",
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
    label: "decision_artifact_compiler",
    response,
    responseShape: summarizeSharedStateArtifactResponseShape(response),
  });

  let parsed: EmitSharedStatePatch | undefined;
  let repairAttempted = false;

  try {
    parsed = parseResponse(response);
  } catch (error) {
    if (error instanceof z.ZodError) {
      if (response.stop_reason === "max_tokens") {
        traceCompileCompleted({
          ...compileCompletedTraceBase,
          operationCount: 0,
          rejected: [],
          applied: false,
          artifact: previousArtifact,
          prunedEntryCountThisTurn: 0,
          supersededEntryCountThisTurn: 0,
        });

        return degraded(input, "invalid_payload", error);
      }

      traceCompileRepairAttempted({
        tracer: input.tracer,
        turnId: input.turnId,
        audienceEntityId: input.audienceEntityId,
        error,
      });
      repairAttempted = true;

      const repairMessages = buildSharedStateArtifactMessages({
        ...messageInput,
        additionalPromptSections: [
          `Your previous patch was invalid: ${errorMessage(error)}. Emit a corrected patch.`,
        ],
      });

      traceLlmCallStarted({
        tracer: input.tracer,
        turnId: input.turnId,
        label: "decision_artifact_compiler",
        model: input.model,
        systemPrompt: SHARED_STATE_SYSTEM_PROMPT,
        messages: repairMessages,
        tools,
      });

      let repairResponse: LLMCompleteResult;

      try {
        repairResponse = await input.llmClient.complete({
          model: input.model,
          system: SHARED_STATE_SYSTEM_PROMPT,
          messages: repairMessages,
          tools,
          tool_choice: { type: "tool", name: SHARED_STATE_TOOL_NAME },
          max_tokens: MAX_PATCH_OUTPUT_TOKENS,
          budget: "decision-artifact-compiler",
        });
      } catch (repairError) {
        traceLlmCallError({
          tracer: input.tracer,
          turnId: input.turnId,
          label: "decision_artifact_compiler",
          error: repairError,
        });
        traceCompileRepairFailed({
          tracer: input.tracer,
          turnId: input.turnId,
          audienceEntityId: input.audienceEntityId,
          error: repairError,
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

        return degraded(input, "llm_failed", repairError);
      }

      traceLlmCallResponse({
        tracer: input.tracer,
        turnId: input.turnId,
        label: "decision_artifact_compiler",
        response: repairResponse,
        responseShape: summarizeSharedStateArtifactResponseShape(repairResponse),
      });

      try {
        parsed = parseResponse(repairResponse);
        traceCompileRepairSucceeded({
          tracer: input.tracer,
          turnId: input.turnId,
          audienceEntityId: input.audienceEntityId,
        });
      } catch (repairError) {
        traceCompileRepairFailed({
          tracer: input.tracer,
          turnId: input.turnId,
          audienceEntityId: input.audienceEntityId,
          error: repairError,
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

        return degraded(
          input,
          repairError instanceof MissingSharedStateArtifactToolCallError
            ? "missing_tool_call"
            : repairError instanceof z.ZodError
              ? "invalid_payload"
              : "llm_failed",
          repairError,
        );
      }
    } else {
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
        error instanceof MissingSharedStateArtifactToolCallError
          ? "missing_tool_call"
          : "llm_failed",
        error,
      );
    }
  }

  if (parsed === undefined) {
    traceCompileCompleted({
      ...compileCompletedTraceBase,
      operationCount: 0,
      rejected: [],
      applied: false,
      artifact: previousArtifact,
      prunedEntryCountThisTurn: 0,
      supersededEntryCountThisTurn: 0,
    });

    return degraded(input, "llm_failed");
  }

  const allowedSourceStreamEntryIds =
    allowedSourceStreamEntryIdsForPrompt === undefined
      ? null
      : new Set(allowedSourceStreamEntryIdsForPrompt);
  let normalized = normalizePatch({
    patch: parsed,
    previousArtifact,
    audienceEntityId: input.audienceEntityId,
    selfEntityId: input.selfEntityId,
    speakerEntityId,
    participants: input.participants,
    allowedSourceStreamEntryIds,
    sourceTrustValidator: input.sourceTrustValidator,
    participantRoster: input.participantRoster,
    relationshipEvidenceStreamEntryTrust: input.relationshipEvidenceStreamEntryTrust,
    allowedCanonicalizationIds: allowedCanonicalizationIds(input.canonicalizationCandidates),
    maxLiveEntriesPerKey: input.lifecycle?.maxLiveEntriesPerKey,
  });
  recordEmptyUpdateValidation(normalized);

  let repairableRejections = repairablePatchRejections(normalized.rejected);
  for (const rejection of repairableRejections) {
    if (rejection.reason === "live_entry_cap_exceeded_for_key") {
      compileCompletedTraceBase.addRejectedCapExceededCount += 1;
    }
    traceRepairablePatchRejection(input, rejection);
  }

  if (repairableRejections.length > 0) {
    const repairError = new Error(patchRejectionRepairMessage(repairableRejections));

    if (repairAttempted) {
      traceCompileRepairFailed({
        tracer: input.tracer,
        turnId: input.turnId,
        audienceEntityId: input.audienceEntityId,
        error: repairError,
      });
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

      return degraded(input, "invalid_patch", repairError);
    }

    traceCompileRepairAttempted({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      error: repairError,
    });
    repairAttempted = true;

    const repairMessages = buildSharedStateArtifactMessages({
      ...messageInput,
      additionalPromptSections: [patchRejectionRepairMessage(repairableRejections)],
    });

    traceLlmCallStarted({
      tracer: input.tracer,
      turnId: input.turnId,
      label: "decision_artifact_compiler",
      model: input.model,
      systemPrompt: SHARED_STATE_SYSTEM_PROMPT,
      messages: repairMessages,
      tools,
    });

    let repairResponse: LLMCompleteResult;

    try {
      repairResponse = await input.llmClient.complete({
        model: input.model,
        system: SHARED_STATE_SYSTEM_PROMPT,
        messages: repairMessages,
        tools,
        tool_choice: { type: "tool", name: SHARED_STATE_TOOL_NAME },
        max_tokens: MAX_PATCH_OUTPUT_TOKENS,
        budget: "decision-artifact-compiler",
      });
    } catch (error) {
      traceLlmCallError({
        tracer: input.tracer,
        turnId: input.turnId,
        label: "decision_artifact_compiler",
        error,
      });
      traceCompileRepairFailed({
        tracer: input.tracer,
        turnId: input.turnId,
        audienceEntityId: input.audienceEntityId,
        error,
      });
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

      return degraded(input, "llm_failed", error);
    }

    traceLlmCallResponse({
      tracer: input.tracer,
      turnId: input.turnId,
      label: "decision_artifact_compiler",
      response: repairResponse,
      responseShape: summarizeSharedStateArtifactResponseShape(repairResponse),
    });

    let repairedParsed: EmitSharedStatePatch;

    try {
      repairedParsed = parseResponse(repairResponse);
    } catch (error) {
      traceCompileRepairFailed({
        tracer: input.tracer,
        turnId: input.turnId,
        audienceEntityId: input.audienceEntityId,
        error,
      });
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

      return degraded(
        input,
        error instanceof MissingSharedStateArtifactToolCallError
          ? "missing_tool_call"
          : error instanceof z.ZodError
            ? "invalid_payload"
            : "llm_failed",
        error,
      );
    }

    normalized = normalizePatch({
      patch: repairedParsed,
      previousArtifact,
      audienceEntityId: input.audienceEntityId,
      selfEntityId: input.selfEntityId,
      speakerEntityId,
      participants: input.participants,
      allowedSourceStreamEntryIds,
      sourceTrustValidator: input.sourceTrustValidator,
      participantRoster: input.participantRoster,
      relationshipEvidenceStreamEntryTrust: input.relationshipEvidenceStreamEntryTrust,
      allowedCanonicalizationIds: allowedCanonicalizationIds(input.canonicalizationCandidates),
      maxLiveEntriesPerKey: input.lifecycle?.maxLiveEntriesPerKey,
    });
    recordEmptyUpdateValidation(normalized);
    repairableRejections = repairablePatchRejections(normalized.rejected);
    for (const rejection of repairableRejections) {
      if (rejection.reason === "live_entry_cap_exceeded_for_key") {
        compileCompletedTraceBase.addRejectedCapExceededCount += 1;
      }
      traceRepairablePatchRejection(input, rejection);
    }

    if (repairableRejections.length > 0) {
      const error = new Error(patchRejectionRepairMessage(repairableRejections));
      traceCompileRepairFailed({
        tracer: input.tracer,
        turnId: input.turnId,
        audienceEntityId: input.audienceEntityId,
        error,
      });
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

      return degraded(input, "invalid_patch", error);
    }

    traceCompileRepairSucceeded({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
    });
  }

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
  const compilerOperations = withOperationLastUpdatedTurns(
    materializeSharedStateOperationIds(normalized.operations),
    input,
  );
  const postCompilerEntries = materializeSharedStateEntriesAfterOperations({
    previousArtifact,
    operations: compilerOperations,
    audienceEntityId: input.audienceEntityId,
    nowMs,
    lastUpdatedTurnGlobal: input.turnCounter ?? null,
  });
  const aging = applyLifecycleAging({
    entries: postCompilerEntries,
    currentTurnCounter: input.turnCounter,
    currentUserStreamEntryId: input.currentUserStreamEntryId,
    ledgerStreamEntryIds: input.renderOptions?.ledgerStreamEntryIds,
    activeOpenQuestionIds: input.renderOptions?.activeOpenQuestionIds,
    activeActionIds: input.renderOptions?.activeActionIds,
    activeGoalIds: input.renderOptions?.activeGoalIds,
    activeCriticalCommitmentIds: input.renderOptions?.activeCriticalCommitmentIds,
    activeOperationalCommitmentIds: input.renderOptions?.activeOperationalCommitmentIds,
    recentlyRetrievedEntryIds: input.renderOptions?.recentlyRetrievedEntryIds,
    touchedEntryIds: touchedSharedStateEntryIds(compilerOperations),
    lastUpdatedTurnByEntryId: lastUpdatedTurnByEntryId({
      entries: postCompilerEntries,
      currentUserStreamEntryId: input.currentUserStreamEntryId,
      currentTurnCounter: input.turnCounter,
      lastUpdatedTurnByStreamEntryId: input.renderOptions?.lastUpdatedTurnByStreamEntryId,
    }),
    recentTurnThreshold:
      input.lifecycle?.recentTurnThreshold ??
      input.renderOptions?.recentTurnThreshold ??
      SHARED_STATE_RECENT_TURN_THRESHOLD,
    dormantTurnThreshold: input.lifecycle?.dormantTurnThreshold,
  });
  const lifecycleTransitionOperations: SharedStateOperation[] = aging.transitions.map(
    (transition) => ({
      type: "transition_kind",
      id: transition.entryId,
      kind: transition.toKind,
    }),
  );
  const lifecycle = applySharedStateArtifactLifecycleCap({
    previousArtifact,
    operations: [...compilerOperations, ...lifecycleTransitionOperations],
    options: input.lifecycle,
    nowMs,
  });
  if (lifecycle.overCapDelta > 0 && input.tracer?.enabled === true && input.turnId !== undefined) {
    input.tracer.emit("shared_state.lifecycle.degraded", {
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      maxActiveEntries: lifecycle.maxActiveEntries,
      postPlanActiveEntryCount: lifecycle.postPlanActiveEntryCount,
      overCapDelta: lifecycle.overCapDelta,
    });
  }
  const compileCompletedTraceWithLifecycle = {
    ...compileCompletedTraceBase,
    maxActiveEntries: lifecycle.maxActiveEntries,
    lifecycleAgingBlockerCountsLiveToLowSalience: aging.blockerCountsLiveToLowSalience,
    lifecycleAgingBlockerCountsLowSalienceToDormant: aging.blockerCountsLowSalienceToDormant,
    lifecycleAgingBlockedSample: aging.blockedSample,
    lifecycleAgingUnknownAgeSample: aging.unknownAgeSample,
  };
  const expandedOperations = expandPruneDependencies({
    previousArtifact,
    operations: lifecycle.operations,
    nowMs,
  });
  const dedupedCanonicalizations = dedupeCanonicalizesAcrossOperations(expandedOperations);
  const operations = dedupedCanonicalizations.operations;
  const publicOperations = publicSharedStateOperations(operations);
  const operationCounts = operationCountsByKind(publicOperations);
  const operationCountsByStateKeyForTrace = operationCountsByStateKey(
    publicOperations,
    previousArtifact,
  );
  const newStateKeysForTrace = introducedStateKeys(publicOperations, previousArtifact);
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
        ...compileCompletedTraceWithLifecycle,
        operationCount: 0,
        rejected: normalized.rejected,
        applied: false,
        artifact: previousArtifact,
        prunedEntryCountThisTurn,
        supersededEntryCountThisTurn,
        operationCountsByKind: operationCounts,
        operationCountsByStateKey: operationCountsByStateKeyForTrace,
        newStateKeys: newStateKeysForTrace,
        nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
        lifecycleTransitions: aging.transitions,
      });

      return degraded(input, "repository_failed", error);
    }

    const reconciliationWorkSet = buildSharedStateReconciliationWorkSet({
      artifact: markedArtifact,
      operations: [],
      repositories: input.reconciliation,
      nowMs,
    });
    const reconciliationResult = reconcileSharedStateCanonicalizations({
      entries: reconciliationWorkSet.entries,
      repositories: input.reconciliation,
      unknownIds: normalized.droppedCanonicalizeIds,
      nowMs,
      turnCounter: input.turnCounter ?? null,
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
      ...compileCompletedTraceWithLifecycle,
      operationCount: 0,
      rejected: normalized.rejected,
      applied: false,
      artifact: markedArtifact,
      prunedEntryCountThisTurn,
      supersededEntryCountThisTurn,
      operationCountsByKind: operationCounts,
      operationCountsByStateKey: operationCountsByStateKeyForTrace,
      newStateKeys: newStateKeysForTrace,
      nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
      lifecycleTransitions: aging.transitions,
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
      lastUpdatedTurnGlobal: input.turnCounter ?? null,
    });

    const reconciliationWorkSet = buildSharedStateReconciliationWorkSet({
      artifact: nextArtifact,
      operations,
      repositories: input.reconciliation,
      nowMs,
    });
    const reconciliationResult = reconcileSharedStateCanonicalizations({
      entries: reconciliationWorkSet.entries,
      repositories: input.reconciliation,
      unknownIds: normalized.droppedCanonicalizeIds,
      nowMs,
      turnCounter: input.turnCounter ?? null,
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
    traceSharedStateLifecycleTransitions({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      transitions: aging.transitions,
    });

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
      ...compileCompletedTraceWithLifecycle,
      operationCount: operations.length,
      rejected: normalized.rejected,
      applied: true,
      artifact: nextArtifact,
      prunedEntryCountThisTurn,
      supersededEntryCountThisTurn,
      operationCountsByKind: operationCounts,
      operationCountsByStateKey: operationCountsByStateKeyForTrace,
      newStateKeys: newStateKeysForTrace,
      nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
      lifecycleTransitions: aging.transitions,
    });
  } catch (error) {
    traceCompileCompleted({
      ...compileCompletedTraceWithLifecycle,
      operationCount: operations.length,
      rejected: normalized.rejected,
      applied: false,
      artifact: previousArtifact,
      prunedEntryCountThisTurn,
      supersededEntryCountThisTurn,
      operationCountsByKind: operationCounts,
      operationCountsByStateKey: operationCountsByStateKeyForTrace,
      newStateKeys: newStateKeysForTrace,
      nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
      lifecycleTransitions: aging.transitions,
    });

    return degraded(input, "repository_failed", error);
  }

  return {
    operations: publicOperations.map((operation) => {
      switch (operation.type) {
        case "add":
          return {
            type: "add",
            state_key: operation.state_key ?? "legacy",
            kind: operation.kind as SharedStateToolEntryKind,
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
            state_key: operation.state_key ?? "legacy",
            kind: publicKindForOmittedUpdate(operation.kind),
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
              state_key: operation.replacement.state_key ?? "legacy",
              kind: operation.replacement.kind as SharedStateToolEntryKind,
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

export {
  DECISION_ARTIFACT_TOOL_NAME,
  SHARED_STATE_ACCEPTED_TOOL_NAMES,
  SHARED_STATE_TOOL_NAME,
  SHARED_STATE_TOOL_NAME_ALIASES,
  MAX_PATCH_OUTPUT_TOKENS,
} from "./constants.js";
export { SHARED_STATE_SYSTEM_PROMPT };
export type { CompileSharedStateArtifactInput } from "./schema.js";
export type {
  DroppedCanonicalizeId,
  EmitDecisionArtifactPatch,
  EmitSharedStatePatch,
  SharedStateArtifactParticipantContext,
  SharedStateActionCanonicalizationCandidate,
  SharedStateCanonicalizationCandidate,
  SharedStateCanonicalizationCandidates,
  SharedStateCommitmentCanonicalizationCandidate,
  SharedStateCompileDegradedReason,
  SharedStateLedgerMode,
  SharedStateLifecycleOptions,
} from "./types.js";
