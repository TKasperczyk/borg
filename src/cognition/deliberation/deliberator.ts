// Thin deliberation orchestrator: selects S1/S2, calls planner/finalizer, and assembles results.
import { computeRetrievalConfidence, type RetrievedEpisode } from "../../retrieval/index.js";
import type { StreamWriter } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import {
  DEFAULT_DELIBERATION_PLAN_MAX_TOKENS,
  DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS,
  DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET,
  DEFAULT_SEMANTIC_CONTEXT_BUDGET,
  UNTRUSTED_DATA_PREAMBLE,
} from "./constants.js";
import { buildDialogueMessages, toContentBlockMessages } from "./dialogue.js";
import { runFinalizer } from "./finalizer.js";
import { chooseDeliberationPath } from "./path-selector.js";
import { formatTurnPlanForPrompt } from "./prompt/plan-rendering.js";
import { summarizeRetrievedEpisodes } from "./prompt/retrieval.js";
import { renderTaggedPromptBlock } from "./prompt/sections.js";
import { buildBaseSystemPrompt } from "./prompt/system-prompt.js";
import { runS2Planner } from "./s2-planner.js";
import { formatTurnPlanForThought, persistDeliberationThoughts } from "./thoughts.js";
import { NOOP_TRACER, type TurnTracer } from "../tracing/tracer.js";
import { isMinimalUserGenerationInput } from "../generation/generation-gate.js";
import {
  renderOutputValidatorRetrySection,
  validateAssistantOutput,
  type OutputValidationFailure,
} from "../generation/output-validator.js";
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

function aggregateUsage(
  current: DeliberationUsage,
  next: {
    input_tokens: number;
    output_tokens: number;
    stop_reason: string | null;
  },
): DeliberationUsage {
  return {
    input_tokens: current.input_tokens + next.input_tokens,
    output_tokens: current.output_tokens + next.output_tokens,
    stop_reason: next.stop_reason,
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

type FinalizerResult = Awaited<ReturnType<typeof runFinalizer>>;

type ValidatedFinalizerResult = {
  result: FinalizerResult;
  suppressedFailure: OutputValidationFailure | null;
  retryUsed: boolean;
};

function combineFinalizerResults(
  first: FinalizerResult,
  second: FinalizerResult,
): FinalizerResult {
  return {
    text: second.text,
    iterations: first.iterations + second.iterations,
    toolCallsMade: [...first.toolCallsMade, ...second.toolCallsMade],
    stopReason: second.stopReason,
    usage: aggregateUsage(first.usage, second.usage),
  };
}

function allowsOutputValidatorRetry(context: DeliberationContext): boolean {
  const activeStop = context.workingMemory.discourse_state?.stop_until_substantive_content ?? null;

  return activeStop === null && !isMinimalUserGenerationInput(context.userMessage);
}

export class Deliberator {
  private readonly tracer: TurnTracer;
  private readonly clock: Clock;

  constructor(private readonly options: DeliberatorOptions) {
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.clock = options.clock ?? new SystemClock();
  }

  private emitOutputValidatorBlocked(input: {
    context: DeliberationContext;
    path: "system_1" | "system_2";
    failure: OutputValidationFailure;
    retry: boolean;
  }): void {
    if (!this.tracer.enabled || input.context.turnId === undefined) {
      return;
    }

    this.tracer.emit("output_validator_blocked", {
      turnId: input.context.turnId,
      path: input.path,
      reason: input.failure.reason,
      kind: input.failure.kind,
      retry: input.retry,
      ...(input.failure.line === undefined ? {} : { line: input.failure.line }),
      ...(input.failure.label === undefined ? {} : { label: input.failure.label }),
    });
  }

  private async runValidatedFinalizer(input: {
    context: DeliberationContext;
    finalizerOptions: Parameters<typeof runFinalizer>[0];
    baseAdditionalPromptSections?: readonly (string | null)[];
  }): Promise<ValidatedFinalizerResult> {
    const first = await runFinalizer(input.finalizerOptions);
    const firstValidation = validateAssistantOutput(first.text);

    if (firstValidation.ok) {
      return {
        result: first,
        suppressedFailure: null,
        retryUsed: false,
      };
    }

    const allowRetry = allowsOutputValidatorRetry(input.context);
    this.emitOutputValidatorBlocked({
      context: input.context,
      path: input.finalizerOptions.path,
      failure: firstValidation.failure,
      retry: allowRetry,
    });

    if (!allowRetry) {
      return {
        result: {
          ...first,
          text: "",
        },
        suppressedFailure: firstValidation.failure,
        retryUsed: false,
      };
    }

    const retry = await runFinalizer({
      ...input.finalizerOptions,
      additionalPromptSections: [
        ...(input.baseAdditionalPromptSections ?? []),
        renderOutputValidatorRetrySection(firstValidation.failure),
      ],
    });
    const combined = combineFinalizerResults(first, retry);
    const retryValidation = validateAssistantOutput(retry.text);

    if (retryValidation.ok) {
      return {
        result: combined,
        suppressedFailure: null,
        retryUsed: true,
      };
    }

    this.emitOutputValidatorBlocked({
      context: input.context,
      path: input.finalizerOptions.path,
      failure: retryValidation.failure,
      retry: false,
    });

    return {
      result: {
        ...combined,
        text: "",
      },
      suppressedFailure: retryValidation.failure,
      retryUsed: true,
    };
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
    const baseSystemPrompt = buildBaseSystemPrompt(effectiveContext, {
      retrievalContextBudget,
      semanticContextBudget,
      nowMs: this.clock.now(),
    });

    const dialogueMessages = buildDialogueMessages(context.recencyMessages, context.userMessage);
    const dialogueBlockMessages = toContentBlockMessages(dialogueMessages);
    const deliberatorTools = this.options.toolDispatcher.listTools("deliberator");

    if (decision.path === "system_1") {
      const response = await this.runValidatedFinalizer({
        context,
        finalizerOptions: {
          llmClient: this.options.llmClient,
          dispatcher: this.options.toolDispatcher,
          sessionId: context.sessionId,
          audienceEntityId: context.audienceEntityId,
          model: this.options.cognitionModel,
          baseSystemPrompt,
          initialMessages: dialogueBlockMessages,
          tools: deliberatorTools,
          userEntryId: context.userEntryId,
          maxTokens: systemOneMaxTokens,
          path: "system_1",
          tracer: this.tracer,
          turnId: context.turnId,
        },
      });
      const emission =
        response.suppressedFailure === null
          ? ({
              kind: "message",
              content: response.result.text,
            } as const)
          : ({
              kind: "suppressed",
              reason: response.suppressedFailure.reason,
            } as const);

      return {
        path: "system_1",
        response: response.suppressedFailure === null ? response.result.text : "",
        emitted: response.suppressedFailure === null,
        emission,
        emissionRecommendation: "emit",
        thoughtStreamEntryIds: [],
        thoughts: [],
        tool_calls: response.result.toolCallsMade,
        usage: response.result.usage,
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
    const planner = await runS2Planner({
      llmClient: this.options.llmClient,
      model: this.options.cognitionModel,
      baseSystemPrompt,
      dialogueMessages,
      selfSnapshot: context.selfSnapshot,
      maxTokens: planningMaxTokens,
      tracer: this.tracer,
      turnId: context.turnId,
    });
    const plan = planner.plan;

    // Verification steps from the plan drive any secondary retrieval. If the
    // plan didn't surface anything to double-check, we skip the re-retrieve
    // call entirely (Phase D removed the regex-on-scratchpad approach).
    const verificationQuery = plan === null ? "" : plan.verification_steps.join("; ").trim();
    const secondaryRetrieval =
      verificationQuery.length > 0 && context.reRetrieve !== undefined
        ? await context.reRetrieve(verificationQuery, { limit: 3 })
        : [];

    const planSection = plan === null ? null : formatTurnPlanForPrompt(plan);
    const thoughts = plan === null ? [] : [formatTurnPlanForThought(plan)];
    const additionalRetrievalBlock = renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
      {
        tag: "borg_additional_retrieval",
        content: summarizeRetrievedEpisodes(
          "Additional retrieval",
          secondaryRetrieval,
          retrievalContextBudget,
        ),
      },
    ]);
    const finalResponse = await this.runValidatedFinalizer({
      context,
      finalizerOptions: {
        llmClient: this.options.llmClient,
        dispatcher: this.options.toolDispatcher,
        sessionId: context.sessionId,
        audienceEntityId: context.audienceEntityId,
        model: this.options.cognitionModel,
        baseSystemPrompt,
        initialMessages: dialogueBlockMessages,
        tools: deliberatorTools,
        userEntryId: context.userEntryId,
        maxTokens: systemTwoMaxTokens,
        path: "system_2",
        additionalPromptSections: [additionalRetrievalBlock, planSection],
        tracer: this.tracer,
        turnId: context.turnId,
      },
      baseAdditionalPromptSections: [additionalRetrievalBlock, planSection],
    });
    const persistedThoughtEntries = await persistDeliberationThoughts(streamWriter, thoughts);
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
    const usage = aggregateUsage(planner.usage, finalResponse.result.usage);
    const emission =
      finalResponse.suppressedFailure === null
        ? ({
            kind: "message",
            content: finalResponse.result.text,
          } as const)
        : ({
            kind: "suppressed",
            reason: finalResponse.suppressedFailure.reason,
          } as const);

    return {
      path: "system_2",
      response: finalResponse.suppressedFailure === null ? finalResponse.result.text : "",
      emitted: finalResponse.suppressedFailure === null,
      emission,
      emissionRecommendation: plan?.emission_recommendation ?? "emit",
      thoughtStreamEntryIds: persistedThoughtEntries.map((entry) => entry.id),
      thoughts,
      tool_calls: finalResponse.result.toolCallsMade,
      usage,
      decision_reason: decision.reason,
      retrievedEpisodes: dedupeRetrievedEpisodes([
        ...context.retrievalResult,
        ...secondaryRetrieval,
      ]),
      referencedEpisodeIds: plan?.referenced_episode_ids ?? null,
      intents: plan === null ? [] : [...plan.intents],
      thoughtsPersisted,
    };
  }
}
