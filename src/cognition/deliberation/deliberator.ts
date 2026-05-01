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
import { NO_OUTPUT_FINALIZER_TOOL_NAME, runFinalizer } from "./finalizer.js";
import { chooseDeliberationPath } from "./path-selector.js";
import { formatTurnPlanForPrompt } from "./prompt/plan-rendering.js";
import { summarizeRetrievedEvidence } from "./prompt/retrieval.js";
import { renderTaggedPromptBlock } from "./prompt/sections.js";
import { buildBaseSystemPrompt } from "./prompt/system-prompt.js";
import { runS2Planner } from "./s2-planner.js";
import { formatTurnPlanForThought, persistDeliberationThoughts } from "./thoughts.js";
import { NOOP_TRACER, type TurnTracer } from "../tracing/tracer.js";
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

type FinalizerEmission = {
  response: string;
  emitted: boolean;
  emission: PendingTurnEmission;
};

function finalizerSuppressionReason(result: FinalizerResult): GenerationSuppressionReason | null {
  if (result.terminalToolCalls.some((call) => call.name === NO_OUTPUT_FINALIZER_TOOL_NAME)) {
    return "no_output_tool";
  }

  if (result.text.trim().length === 0) {
    return "empty_finalizer";
  }

  return null;
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

  return {
    response: result.text,
    emitted: true,
    emission: {
      kind: "message",
      content: result.text,
    },
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
    const baseSystemPrompt = buildBaseSystemPrompt(effectiveContext, {
      retrievalContextBudget,
      semanticContextBudget,
      nowMs: this.clock.now(),
    });

    const dialogueMessages = buildDialogueMessages(context.recencyMessages, context.userMessage);
    const dialogueBlockMessages = toContentBlockMessages(dialogueMessages);
    const deliberatorTools = this.options.toolDispatcher.listTools("deliberator");

    if (decision.path === "system_1") {
      const response = await runFinalizer({
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
    const thoughts = plan === null ? [] : [formatTurnPlanForThought(plan)];
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

    if (plan?.emission_recommendation === "no_output") {
      return {
        path: "system_2",
        response: "",
        emitted: false,
        emission: {
          kind: "suppressed",
          reason: "s2_planner_no_output",
        },
        emissionRecommendation: "no_output",
        thoughtStreamEntryIds: persistedThoughtEntries.map((entry) => entry.id),
        thoughts,
        tool_calls: [],
        usage: planner.usage,
        decision_reason: decision.reason,
        retrievedEpisodes: [...context.retrievalResult],
        referencedEpisodeIds: plan.referenced_episode_ids,
        intents: [],
        thoughtsPersisted,
      };
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
    const finalResponse = await runFinalizer({
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
    });
    const usage = aggregateUsage(planner.usage, finalResponse.usage);
    const finalized = buildFinalizerEmission(finalResponse);

    return {
      path: "system_2",
      response: finalized.response,
      emitted: finalized.emitted,
      emission: finalized.emission,
      emissionRecommendation: plan?.emission_recommendation ?? "emit",
      thoughtStreamEntryIds: persistedThoughtEntries.map((entry) => entry.id),
      thoughts,
      tool_calls: finalResponse.toolCallsMade,
      usage,
      decision_reason: decision.reason,
      retrievedEpisodes: dedupeRetrievedEpisodes([
        ...context.retrievalResult,
        ...(secondaryRetrieval?.episodes ?? []),
      ]),
      referencedEpisodeIds: plan?.referenced_episode_ids ?? null,
      intents: plan === null ? [] : [...plan.intents],
      thoughtsPersisted,
    };
  }
}
