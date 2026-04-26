// Thin deliberation orchestrator: selects S1/S2, calls planner/finalizer, and assembles results.
import type { RetrievedEpisode } from "../../retrieval/index.js";
import type { StreamWriter } from "../../stream/index.js";
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

export class Deliberator {
  private readonly tracer: TurnTracer;

  constructor(private readonly options: DeliberatorOptions) {
    this.tracer = options.tracer ?? NOOP_TRACER;
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
    const decision = chooseDeliberationPath(
      context.perception.mode,
      stakes,
      context.retrievalResult,
      context.contradictionPresent,
      context.retrievalConfidence ?? null,
      trace,
    );
    const baseSystemPrompt = buildBaseSystemPrompt(context, {
      retrievalContextBudget,
      semanticContextBudget,
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

      return {
        path: "system_1",
        response: response.text,
        thoughts: [],
        tool_calls: response.toolCallsMade,
        usage: response.usage,
        decision_reason: decision.reason,
        retrievedEpisodes: [...context.retrievalResult],
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
    const usage = aggregateUsage(planner.usage, finalResponse.usage);

    return {
      path: "system_2",
      response: finalResponse.text,
      thoughts,
      tool_calls: finalResponse.toolCallsMade,
      usage,
      decision_reason: decision.reason,
      retrievedEpisodes: dedupeRetrievedEpisodes([
        ...context.retrievalResult,
        ...secondaryRetrieval,
      ]),
      thoughtsPersisted,
    };
  }
}
