import { z } from "zod";

import type { LLMCompleteResult } from "../../llm/index.js";
import type {
  SharedStateArtifact,
  SharedStateEntry,
  SharedStateOperation,
} from "../../memory/decision-artifacts/index.js";
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
  type PatchRejection,
  type SharedStateCompileDegradedReason,
} from "./schema.js";
import {
  parseResponse,
  traceAddRejectedCapExceeded,
  traceCompileCompleted,
  traceCompileDegraded,
  traceCompileOverBudget,
  traceCompileRepairAttempted,
  traceCompileRepairFailed,
  traceCompileRepairSucceeded,
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
import { errorMessage } from "./reconciliation-summary.js";
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
  }
}

function operationCountsByStateKey(
  operations: readonly SharedStateOperation[],
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

function repairablePatchRejections(rejections: readonly PatchRejection[]): PatchRejection[] {
  return rejections.filter(
    (rejection) =>
      rejection.reason === "live_entry_cap_exceeded_for_key" ||
      rejection.reason === "locked_state_key_collision",
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

      return `operation ${rejection.operationIndex} rejected: ${rejection.reason}`;
    })
    .join(" | ");

  return `Your previous patch violated structural shared-state key compaction: ${details}. Emit a corrected patch.`;
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
        model: input.model,
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
        response: repairResponse,
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
    allowedCanonicalizationIds: allowedCanonicalizationIds(input.canonicalizationCandidates),
    maxLiveEntriesPerKey: input.lifecycle?.maxLiveEntriesPerKey,
  });

  let repairableRejections = repairablePatchRejections(normalized.rejected);
  for (const rejection of repairableRejections) {
    if (rejection.reason !== "live_entry_cap_exceeded_for_key") {
      continue;
    }

    traceAddRejectedCapExceeded({
      tracer: input.tracer,
      turnId: input.turnId,
      audienceEntityId: input.audienceEntityId,
      rejection,
    });
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
      model: input.model,
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
      response: repairResponse,
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
      allowedCanonicalizationIds: allowedCanonicalizationIds(input.canonicalizationCandidates),
      maxLiveEntriesPerKey: input.lifecycle?.maxLiveEntriesPerKey,
    });
    repairableRejections = repairablePatchRejections(normalized.rejected);
    for (const rejection of repairableRejections) {
      if (rejection.reason !== "live_entry_cap_exceeded_for_key") {
        continue;
      }

      traceAddRejectedCapExceeded({
        tracer: input.tracer,
        turnId: input.turnId,
        audienceEntityId: input.audienceEntityId,
        rejection,
      });
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
  const operationCounts = operationCountsByKind(operations);
  const operationCountsByStateKeyForTrace = operationCountsByStateKey(operations, previousArtifact);
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
      operationCountsByKind: operationCounts,
      operationCountsByStateKey: operationCountsByStateKeyForTrace,
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
      operationCountsByKind: operationCounts,
      operationCountsByStateKey: operationCountsByStateKeyForTrace,
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
      operationCountsByKind: operationCounts,
      operationCountsByStateKey: operationCountsByStateKeyForTrace,
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
            state_key: operation.state_key ?? "legacy",
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
            state_key: operation.state_key ?? "legacy",
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
              state_key: operation.replacement.state_key ?? "legacy",
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
