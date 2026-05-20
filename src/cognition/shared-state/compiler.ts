import { z } from "zod";

import type { LLMCompleteResult } from "../../llm/index.js";
import { SystemClock } from "../../util/clock.js";
import type { StreamEntryId } from "../../util/ids.js";
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
  MAX_PATCH_OUTPUT_TOKENS,
  MissingSharedStateArtifactToolCallError,
  SHARED_STATE_TOOL_NAME,
  SHARED_STATE_TOOLS,
  type CompileSharedStateArtifactInput,
  type EmitSharedStatePatch,
  type SharedStateCompileDegradedReason,
} from "./schema.js";
import {
  parseResponse,
  traceCompileCompleted,
  traceCompileOverBudget,
  traceLlmCallError,
  traceLlmCallResponse,
  traceLlmCallStarted,
  traceReconciliationCompleted,
} from "./compiler-io.js";
import {
  allowedCanonicalizationIds,
  dedupeCanonicalizesAcrossOperations,
  normalizePatch,
} from "./patch-validation.js";
import { applySharedStateArtifactLifecycleCap, expandPruneDependencies } from "./lifecycle-cap.js";
import { buildSharedStateReconciliationWorkSet } from "./canonicalization-candidates.js";
import { buildSharedStateArtifactPromptSummary } from "./summary.js";

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

function emptyPatch(): EmitSharedStatePatch {
  return { operations: [] };
}

function uniqueStreamEntryIds(ids: readonly StreamEntryId[]): StreamEntryId[] {
  const seen = new Set<string>();
  const unique: StreamEntryId[] = [];

  for (const id of ids) {
    if (seen.has(id)) {
      continue;
    }

    seen.add(id);
    unique.push(id);
  }

  return unique;
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
  const messages = buildSharedStateArtifactMessages({
    audienceEntityId: input.audienceEntityId,
    selfEntityId: input.selfEntityId,
    speakerEntityId,
    participants: input.participants,
    currentUserMessage: input.currentUserMessage,
    currentUserStreamEntryId: input.currentUserStreamEntryId,
    promptVisibleLedger: input.promptVisibleLedger,
    previousArtifactSummary,
    canonicalizationCandidates,
    relationalSlotsContext: input.relationalSlotsContext,
    allowedSourceStreamEntryIds: allowedSourceStreamEntryIdsForPrompt,
    offLimitsSourceStreamEntryIds: offLimitsSourceStreamEntryIdsForPrompt,
  });
  const tools = SHARED_STATE_TOOLS;
  const ledgerMode = input.ledgerMode ?? "full_fallback";
  const promptBudget = estimateSharedStateArtifactPromptBudget({
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

  let parsed: EmitSharedStatePatch;

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
      error instanceof MissingSharedStateArtifactToolCallError
        ? "missing_tool_call"
        : error instanceof z.ZodError
          ? "invalid_payload"
          : "llm_failed",
      error,
    );
  }

  const allowedSourceStreamEntryIds =
    allowedSourceStreamEntryIdsForPrompt === undefined
      ? null
      : new Set(allowedSourceStreamEntryIdsForPrompt);
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
  const lifecycle = applySharedStateArtifactLifecycleCap({
    previousArtifact,
    operations: normalized.operations,
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
  };
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
        ...compileCompletedTraceWithLifecycle,
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
      nonLockedCanonicalizesDrops: normalized.nonLockedCanonicalizesDrops,
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

export {
  DECISION_ARTIFACT_TOOL_NAME,
  SHARED_STATE_ACCEPTED_TOOL_NAMES,
  SHARED_STATE_TOOL_NAME,
  SHARED_STATE_TOOL_NAME_ALIASES,
  MAX_PATCH_OUTPUT_TOKENS,
} from "./schema.js";
export { SHARED_STATE_SYSTEM_PROMPT };
export type {
  CompileSharedStateArtifactInput,
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
} from "./schema.js";
