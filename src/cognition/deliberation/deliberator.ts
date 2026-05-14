// Thin deliberation orchestrator: selects S1/S2, calls planner/finalizer, and assembles results.
import { computeRetrievalConfidence, type RetrievedEpisode } from "../../retrieval/index.js";
import type { StreamWriter } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import type { LLMCompleteOptions } from "../../llm/index.js";
import {
  DEFAULT_DELIBERATION_PLAN_MAX_TOKENS,
  DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS,
  DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
  DEFAULT_SEMANTIC_CONTEXT_BUDGET,
  UNTRUSTED_DATA_PREAMBLE,
} from "./constants.js";
import { buildDialogueMessages, toContentBlockMessages } from "./dialogue.js";
import { runFinalizer, type FinalizerResult } from "./finalizer.js";
import { chooseDeliberationPath } from "./path-selector.js";
import { formatTurnPlanForPrompt } from "./prompt/plan-rendering.js";
import { summarizeRetrievedEvidence } from "./prompt/retrieval.js";
import { renderTaggedPromptBlock } from "./prompt/sections.js";
import {
  buildBaseSystemPrompt,
  buildCacheableBaseSystemPromptParts,
} from "./prompt/system-prompt.js";
import { runS2Planner } from "./s2-planner.js";
import { formatTurnPlanForThought, persistDeliberationThoughts } from "./thoughts.js";
import { NOOP_TRACER, toTraceJsonValue, type TurnTracer } from "../tracing/tracer.js";
import { buildCompactPlannerLedgerPrompt } from "../evidence-ledger/index.js";
import type { GenerationSuppressionReason, PendingTurnEmission } from "../generation/types.js";
import type {
  DeliberationContext,
  DeliberationResult,
  DeliberationUsage,
  DeliberatorOptions,
} from "./types.js";

export type {
  DeliberationContext,
  DeliberationResult,
  DeliberationUsage,
  DeliberatorOptions,
  SelfSnapshot,
  TurnStakes,
} from "./types.js";

function sumOptional(current: number | undefined, next: number | undefined): number | undefined {
  if (current === undefined && next === undefined) {
    return undefined;
  }
  return (current ?? 0) + (next ?? 0);
}

function aggregateUsage(
  current: DeliberationUsage,
  next: {
    input_tokens: number;
    output_tokens: number;
    cache_creation_input_tokens?: number;
    cache_read_input_tokens?: number;
    stop_reason: string | null;
  },
): DeliberationUsage {
  // Cache token fields are kept separate from input_tokens (per
  // observability standard: cache_read is ~0.1x input rate and doesn't
  // count against rate limits, summing them inflates totals by 100x+).
  const cacheCreation = sumOptional(
    current.cache_creation_input_tokens,
    next.cache_creation_input_tokens,
  );
  const cacheRead = sumOptional(current.cache_read_input_tokens, next.cache_read_input_tokens);
  return {
    input_tokens: current.input_tokens + next.input_tokens,
    output_tokens: current.output_tokens + next.output_tokens,
    stop_reason: next.stop_reason,
    ...(cacheCreation === undefined ? {} : { cache_creation_input_tokens: cacheCreation }),
    ...(cacheRead === undefined ? {} : { cache_read_input_tokens: cacheRead }),
  };
}

function dedupeRetrievedEpisodes(results: readonly RetrievedEpisode[]): RetrievedEpisode[] {
  const seen = new Set<string>();
  const deduped: RetrievedEpisode[] = [];

  for (const result of results) {
    if (seen.has(result.episode.id)) {
      continue;
    }

    seen.add(result.episode.id);
    deduped.push(result);
  }

  return deduped;
}

type FinalizerEmission = {
  response: string;
  emitted: boolean;
  emission: PendingTurnEmission;
};

function finalizerSuppressionReason(result: FinalizerResult): GenerationSuppressionReason | null {
  switch (result.decision.kind) {
    case "no_output":
      return "finalizer_no_output";
    case "empty":
      return "empty_finalizer";
    case "invalid_tool":
      return "finalizer_failed";
    case "answer":
    case "observe":
    case "self_report":
      return null;
  }
}

function buildFinalizerEmission(result: FinalizerResult): FinalizerEmission {
  const suppressionReason = finalizerSuppressionReason(result);

  if (suppressionReason !== null) {
    return {
      response: "",
      emitted: false,
      emission: {
        kind: "suppressed",
        reason: suppressionReason,
      },
    };
  }

  if (result.decision.kind === "self_report") {
    return {
      response: result.decision.text,
      emitted: true,
      emission: {
        kind: "message",
        content: result.decision.text,
        persistence_class: result.decision.persistence_class,
      },
    };
  }

  if (result.decision.kind === "observe") {
    return {
      response: "",
      emitted: false,
      emission: {
        kind: "observed",
        reason: result.decision.reason,
      },
    };
  }

  if (result.decision.kind !== "answer") {
    return {
      response: "",
      emitted: false,
      emission: {
        kind: "suppressed",
        reason: "finalizer_failed",
      },
    };
  }

  return {
    response: result.decision.text,
    emitted: true,
    emission: {
      kind: "message",
      content: result.decision.text,
      ...(result.decision.reply_target === undefined
        ? {}
        : { reply_target: result.decision.reply_target }),
    },
  };
}

function cognitionThinkingOption(
  options: DeliberatorOptions,
): LLMCompleteOptions["thinking"] | undefined {
  if (options.cognitionThinking?.enabled !== true) {
    return undefined;
  }

  return {
    type: "enabled",
    budget_tokens: options.cognitionThinking.budget_tokens,
  };
}

export class Deliberator {
  private readonly tracer: TurnTracer;
  private readonly clock: Clock;

  constructor(private readonly options: DeliberatorOptions) {
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.clock = options.clock ?? new SystemClock();
  }

  async run(
    context: DeliberationContext,
    streamWriter?: StreamWriter,
  ): Promise<DeliberationResult> {
    const stakes = context.options?.stakes ?? "low";
    const planningMaxTokens =
      context.options?.maxThinkingTokens ?? DEFAULT_DELIBERATION_PLAN_MAX_TOKENS;
    const semanticContextBudget = Math.max(DEFAULT_SEMANTIC_CONTEXT_BUDGET, planningMaxTokens * 4);
    const retrievalContextBudget = DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET;
    const systemOneMaxTokens = DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS;
    const systemTwoMaxTokens = DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS;
    const trace =
      this.tracer.enabled && context.turnId !== undefined
        ? {
            tracer: this.tracer,
            turnId: context.turnId,
          }
        : undefined;
    const retrievalConfidence =
      context.retrievalConfidence ??
      computeRetrievalConfidence({
        episodes: context.retrievalResult,
        contradictionPresent: context.contradictionPresent ?? false,
        nowMs: this.clock.now(),
      });
    const effectiveContext: DeliberationContext = {
      ...context,
      retrievalConfidence,
    };
    const decision = chooseDeliberationPath(
      context.perception.mode,
      stakes,
      context.retrievalResult,
      context.contradictionPresent,
      retrievalConfidence,
      trace,
    );
    const baseSystemPromptOptions = {
      retrievalContextBudget,
      semanticContextBudget,
      nowMs: this.clock.now(),
      ...(this.options.hostCapabilities === undefined
        ? {}
        : { hostCapabilities: this.options.hostCapabilities }),
    };
    const baseSystemPrompt = buildBaseSystemPrompt(effectiveContext, baseSystemPromptOptions);
    const cacheableBaseSystemPrompt = buildCacheableBaseSystemPromptParts(
      effectiveContext,
      baseSystemPromptOptions,
    );
    const evidenceLedgerPromptSections =
      context.evidenceLedgerPromptSection === undefined ||
      context.evidenceLedgerPromptSection === null
        ? null
        : [context.evidenceLedgerPromptSection];

    const dialogueMessages = buildDialogueMessages(context.recencyMessages, context.userMessage);
    const dialogueBlockMessages = toContentBlockMessages(dialogueMessages);
    const thinking = cognitionThinkingOption(this.options);

    if (decision.path === "system_1") {
      const response = await runFinalizer({
        llmClient: this.options.llmClient,
        dispatcher: this.options.toolDispatcher,
        sessionId: context.sessionId,
        audienceEntityId: context.audienceEntityId,
        model: this.options.cognitionModel,
        baseSystemPrompt,
        cacheableSystemPrompt: cacheableBaseSystemPrompt,
        initialMessages: dialogueBlockMessages,
        userEntryId: context.userEntryId,
        maxTokens: systemOneMaxTokens,
        ...(thinking === undefined ? {} : { thinking }),
        path: "system_1",
        ...(evidenceLedgerPromptSections === null
          ? {}
          : { additionalPromptSections: evidenceLedgerPromptSections }),
        tracer: this.tracer,
        turnId: context.turnId,
      });
      const finalized = buildFinalizerEmission(response);

      return {
        path: "system_1",
        response: finalized.response,
        emitted: finalized.emitted,
        emission: finalized.emission,
        emissionRecommendation: "emit",
        thoughtStreamEntryIds: [],
        thoughts: [],
        tool_calls: response.toolCallsMade,
        usage: response.usage,
        decision_reason: decision.reason,
        retrievedEpisodes: [...context.retrievalResult],
        referencedEpisodeIds: null,
        intents: [],
        thoughtsPersisted: false,
      };
    }

    // S2 staged: both calls share the full baseSystemPrompt (identity, voice,
    // tagged memory context, trusted guidance) so voice consistency is
    // guaranteed across the plan and the final response. The planner call
    // emits a structured plan via tool-use; the finalizer consumes that
    // plan as explicit structured context rather than "scratchpad text"
    // jammed into its system prompt.
    const compactPlannerLedger =
      context.evidenceLedger === undefined || context.evidenceLedger === null
        ? null
        : buildCompactPlannerLedgerPrompt(context.evidenceLedger, {
            decisionArtifact: this.options.decisionArtifactRenderOptions,
          });

    if (compactPlannerLedger !== null && this.tracer.enabled && context.turnId !== undefined) {
      this.tracer.emit("planner_compact_ledger_built", {
        turnId: context.turnId,
        entry_counts: toTraceJsonValue(compactPlannerLedger.traceSummary.entryCountsBySection),
        omitted_entry_counts: toTraceJsonValue(
          compactPlannerLedger.traceSummary.omittedEntryCountsBySection,
        ),
        estimated_tokens_by_section: toTraceJsonValue(
          compactPlannerLedger.traceSummary.estimatedTokensBySection,
        ),
        decision_artifact_entry_count: compactPlannerLedger.traceSummary.decisionArtifactEntryCount,
        decision_artifact_rendered_token_estimate:
          compactPlannerLedger.traceSummary.decisionArtifactRenderedTokens,
        decision_artifact_rendered_by_kind: toTraceJsonValue(
          compactPlannerLedger.traceSummary.decisionArtifactRenderedByKind,
        ),
        total_estimated_tokens: compactPlannerLedger.traceSummary.totalEstimatedTokens,
        target_tokens: compactPlannerLedger.traceSummary.targetTokens,
        hard_cap_tokens: compactPlannerLedger.traceSummary.hardCapTokens,
      });
    }

    const planner = await runS2Planner({
      llmClient: this.options.llmClient,
      model: this.options.cognitionModel,
      baseSystemPrompt,
      dialogueMessages,
      selfSnapshot: context.selfSnapshot,
      ...(compactPlannerLedger?.promptSection === undefined ||
      compactPlannerLedger.promptSection === null
        ? {}
        : { additionalPromptSections: [compactPlannerLedger.promptSection] }),
      maxTokens: planningMaxTokens,
      ...(thinking === undefined ? {} : { thinking }),
      tracer: this.tracer,
      turnId: context.turnId,
    });
    const plan = planner.plan;
    const thoughts = plan === null ? [] : [formatTurnPlanForThought(plan)];
    const persistedThoughtEntries = await persistDeliberationThoughts(streamWriter, thoughts, {
      turnId: context.turnId,
    });
    const thoughtsPersisted = persistedThoughtEntries.length > 0;

    if (this.tracer.enabled && context.turnId !== undefined) {
      const persistedEntry = persistedThoughtEntries[0];

      if (persistedEntry !== undefined) {
        this.tracer.emit("plan_persisted", {
          turnId: context.turnId,
          streamEntryId: persistedEntry.id,
        });
      } else {
        this.tracer.emit("plan_persistence_skipped", {
          turnId: context.turnId,
          reason:
            plan === null
              ? "no_plan_extracted"
              : streamWriter === undefined
                ? "stream_writer_unavailable"
                : "empty_thoughts",
        });
      }
    }

    // Verification steps from the plan drive any secondary retrieval. If the
    // plan didn't surface anything to double-check, we skip the re-retrieve
    // call entirely (Phase D removed the regex-on-scratchpad approach).
    const verificationQuery = plan === null ? "" : plan.verification_steps.join("; ").trim();
    const secondaryRetrieval =
      verificationQuery.length > 0 && context.reRetrieve !== undefined
        ? await context.reRetrieve(verificationQuery, { limit: 3 })
        : null;

    const additionalRetrievalBlock = renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
      {
        tag: "borg_additional_retrieval",
        content: summarizeRetrievedEvidence(
          "Additional retrieval",
          {
            evidence: secondaryRetrieval?.evidence ?? [],
            episodes: secondaryRetrieval?.episodes ?? [],
            semantic: secondaryRetrieval?.semantic ?? null,
            openQuestions: secondaryRetrieval?.open_questions ?? [],
          },
          retrievalContextBudget,
        ),
      },
    ]);
    const planSection = plan === null ? null : formatTurnPlanForPrompt(plan);
    const additionalPromptSections =
      evidenceLedgerPromptSections === null
        ? [additionalRetrievalBlock, planSection]
        : [...evidenceLedgerPromptSections, additionalRetrievalBlock, planSection];
    let usage = planner.usage;
    let finalized: FinalizerEmission;
    let finalToolCallsMade: FinalizerResult["toolCallsMade"] = [];

    const finalResponse = await runFinalizer({
      llmClient: this.options.llmClient,
      dispatcher: this.options.toolDispatcher,
      sessionId: context.sessionId,
      audienceEntityId: context.audienceEntityId,
      model: this.options.cognitionModel,
      baseSystemPrompt,
      cacheableSystemPrompt: cacheableBaseSystemPrompt,
      initialMessages: dialogueBlockMessages,
      userEntryId: context.userEntryId,
      maxTokens: systemTwoMaxTokens,
      ...(thinking === undefined ? {} : { thinking }),
      path: "system_2",
      additionalPromptSections,
      tracer: this.tracer,
      turnId: context.turnId,
    });
    usage = aggregateUsage(usage, finalResponse.usage);
    finalized = buildFinalizerEmission(finalResponse);
    finalToolCallsMade = finalResponse.toolCallsMade;

    return {
      path: "system_2",
      response: finalized.response,
      emitted: finalized.emitted,
      emission: finalized.emission,
      emissionRecommendation: "emit",
      thoughtStreamEntryIds: persistedThoughtEntries.map((entry) => entry.id),
      thoughts,
      tool_calls: finalToolCallsMade,
      usage,
      decision_reason: decision.reason,
      retrievedEpisodes: dedupeRetrievedEpisodes([
        ...context.retrievalResult,
        ...(secondaryRetrieval?.episodes ?? []),
      ]),
      referencedEpisodeIds: null,
      intents: plan === null ? [] : [...plan.intents],
      thoughtsPersisted,
    };
  }
}
