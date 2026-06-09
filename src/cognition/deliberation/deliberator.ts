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
  THINKING_DELIBERATION_MAX_TOKENS,
} from "./constants.js";
import { UNTRUSTED_DATA_PREAMBLE } from "../prompts/base-identity.js";
import {
  buildDialogueMessages,
  toContentBlockMessages,
  withFinalizerImageBudget,
  withCurrentUserContentBlocks,
  withLedgerImageContentBlocks,
} from "./dialogue.js";
import { traceTurnPhase } from "../lifecycle/turn-phase/phase-trace.js";
import {
  EMIT_ANSWER_FINALIZER_TOOL_NAME,
  EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME,
  EMIT_OBSERVE_FINALIZER_TOOL_NAME,
  EMIT_SELF_REPORT_FINALIZER_TOOL_NAME,
  resolveAvailableEmissionNames,
  runFinalizer,
  type EmissionDecision,
  type EmissionToolName,
  type FinalizerResult,
  type RunFinalizerOptions,
} from "./finalizer.js";
import { chooseDeliberationPath } from "./path-selector.js";
import { formatTurnPlanForPrompt } from "./prompt/plan-rendering.js";
import { summarizeRetrievedEvidence } from "./prompt/retrieval.js";
import { renderTaggedPromptBlock } from "./prompt/sections.js";
import {
  buildBaseSystemPrompt,
  buildCacheableBaseSystemPromptParts,
  type BuildBaseSystemPromptOptions,
} from "./prompt/system-prompt.js";
import { runS2Planner } from "./s2-planner.js";
import { formatTurnPlanForThought, persistDeliberationThoughts } from "./thoughts.js";
import { NOOP_TRACER, toTraceJsonValue, type TurnTracer } from "../../tracing/tracer.js";
import {
  buildCompactPlannerLedgerPrompt,
  renderEvidenceLedger,
  truncateTextForCompactPlannerLedger,
} from "../evidence-ledger/index.js";
import type {
  FinalizerInvalidToolDiagnostic,
  FinalizerNoOutputCategory,
  FinalizerNoOutputSemanticCategory,
  FinalizerNoOutputStructuralCategory,
  FinalizerNoOutputStructuralFlag,
  GenerationSuppressionReason,
  PendingTurnEmission,
} from "../generation/types.js";
import { deriveFinalizerNoOutputPrimaryReason } from "../generation/types.js";
import type {
  DeliberationContext,
  DeliberationResult,
  DeliberationUsage,
  DeliberatorOptions,
} from "./types.js";
import type { SessionParticipationPolicy } from "../../sessions/index.js";
import { isCreatorInOperatorContext } from "../authority.js";
import { exposesOutboundTool, type TurnOrigin } from "../types.js";
import { mergeDeliberationUsage } from "./usage.js";

export type {
  DeliberationContext,
  DeliberationResult,
  DeliberationUsage,
  DeliberatorOptions,
  SelfSnapshot,
  TurnStakes,
} from "./types.js";

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

function renderForcedContradictionOpenQuestionsPrompt(context: DeliberationContext): string | null {
  const routingOverride = context.routingOverride;
  const openQuestions = routingOverride?.openQuestions ?? [];

  if (
    routingOverride?.forceSystem2 !== true ||
    routingOverride.forcedBy !== "open_question_contradiction" ||
    openQuestions.length === 0
  ) {
    return null;
  }

  const contradictionQuestionLines = openQuestions
    .slice(0, 5)
    .map(
      (question, index) =>
        `${index + 1}. ${question.localHandle ?? `contradiction_${index + 1}`} [source=${
          question.source
        }]: ${truncateTextForCompactPlannerLedger(question.question, 75) ?? ""}`,
    );

  return [
    "Planner routing note: An unresolved contradiction is flagged in the open questions above. I either reconcile it in my plan, or explicitly name the conflict in the planning output rather than ignoring it.",
    renderTaggedPromptBlock(UNTRUSTED_DATA_PREAMBLE, [
      {
        tag: "borg_unresolved_contradiction_open_questions",
        content: contradictionQuestionLines.join("\n"),
      },
    ]),
  ].join("\n\n");
}

type FinalizerEmission = {
  response: string;
  emitted: boolean;
  emission: PendingTurnEmission;
};

type InvalidToolDecision = Extract<EmissionDecision, { kind: "invalid_tool" }>;
type FinalizerCallOptionsContext = {
  context: DeliberationContext;
  effectiveContext: DeliberationContext;
  baseSystemPrompt: string;
  cacheableSystemPrompt: RunFinalizerOptions["cacheableSystemPrompt"];
  initialMessages: RunFinalizerOptions["initialMessages"];
  maxTokens: number;
  reasoningCallOptions: Partial<Pick<RunFinalizerOptions, "thinking" | "effort">>;
  allowedEmissions: readonly EmissionToolName[] | undefined;
  outboundToolAvailable: boolean;
  structuralNoOutputFlags: readonly FinalizerNoOutputStructuralFlag[];
  tracer: TurnTracer;
};
type FinalizerCallVariableOptions = {
  path: RunFinalizerOptions["path"];
  additionalPromptSections?: RunFinalizerOptions["additionalPromptSections"];
  finalizerAttempt: NonNullable<RunFinalizerOptions["finalizerAttempt"]>;
};
type FinalizerTriadResultInput = {
  finalized: FinalizerEmission;
  responseForResult: FinalizerResult;
  usage: DeliberationUsage;
};
type RunFinalizerTriadInput = {
  callContext: FinalizerCallOptionsContext;
  path: RunFinalizerOptions["path"];
  additionalPromptSections: readonly (string | null)[] | null;
  availableEmissionNames: readonly EmissionToolName[];
  priorUsage?: DeliberationUsage;
  buildResult: (input: FinalizerTriadResultInput) => DeliberationResult;
};

function buildFinalizerCallOptions(
  context: FinalizerCallOptionsContext,
  options: DeliberatorOptions,
  variable: FinalizerCallVariableOptions,
): RunFinalizerOptions {
  return {
    llmClient: options.llmClient,
    dispatcher: options.toolDispatcher,
    sessionId: context.context.sessionId,
    audienceEntityId: context.context.audienceEntityId,
    model: options.cognitionModel,
    baseSystemPrompt: context.baseSystemPrompt,
    cacheableSystemPrompt: context.cacheableSystemPrompt,
    initialMessages: context.initialMessages,
    userEntryId: context.context.userEntryId,
    maxTokens: context.maxTokens,
    ...context.reasoningCallOptions,
    path: variable.path,
    ...(context.allowedEmissions === undefined
      ? {}
      : { allowedEmissions: context.allowedEmissions }),
    outboundToolAvailable: context.outboundToolAvailable,
    turnOrigin: context.effectiveContext.turnOrigin,
    currentSenderBorgRole: context.effectiveContext.creatorContext?.currentSenderBorgRole ?? null,
    sessionAudienceRole: context.effectiveContext.creatorContext?.sessionAudienceRole,
    ...(variable.additionalPromptSections === undefined
      ? {}
      : { additionalPromptSections: variable.additionalPromptSections }),
    structuralNoOutputFlags: context.structuralNoOutputFlags,
    tracer: context.tracer,
    turnId: context.context.turnId,
    finalizerAttempt: variable.finalizerAttempt,
  };
}

function mergeFinalizerTriadUsage(input: {
  priorUsage?: DeliberationUsage;
  initial: FinalizerResult;
  result: FinalizerResult;
}): DeliberationUsage {
  if (input.priorUsage === undefined) {
    return input.result === input.initial
      ? input.initial.usage
      : mergeDeliberationUsage(input.initial.usage, input.result.usage);
  }

  let usage = mergeDeliberationUsage(input.priorUsage, input.initial.usage);

  if (input.result !== input.initial) {
    usage = mergeDeliberationUsage(usage, input.result.usage);
  }

  return usage;
}

function appendFinalizerPromptSections(
  base: readonly (string | null)[] | null,
  extra: readonly (string | null)[],
): readonly (string | null)[] {
  return base === null ? [...extra] : [...base, ...extra];
}

function attachRegenerator(
  result: DeliberationResult,
  regenerateFinalResponse: NonNullable<DeliberationResult["regenerateFinalResponse"]>,
): DeliberationResult {
  Object.defineProperty(result, "regenerateFinalResponse", {
    value: regenerateFinalResponse,
    enumerable: false,
    configurable: false,
    writable: false,
  });

  return result;
}

function finalizerSuppressionReason(result: FinalizerResult): GenerationSuppressionReason | null {
  switch (result.decision.kind) {
    case "no_output":
      return "finalizer_no_output";
    case "empty":
      return "empty_finalizer";
    case "invalid_tool":
      return "finalizer_failed";
    case "answer":
    case "continue_thought":
    case "observe":
    case "self_report":
      return null;
  }
}

function finalizerInvalidToolDiagnostic(
  decision: InvalidToolDecision,
  attempt: FinalizerInvalidToolDiagnostic["attempt"],
): FinalizerInvalidToolDiagnostic {
  return {
    tool_name: decision.toolName,
    reason: decision.reason,
    attempt,
  };
}

function invalidToolRetryCause(decision: InvalidToolDecision): string {
  if (decision.toolName === "none") {
    return "I emitted 0 terminal emission tool calls; I need to emit exactly one.";
  }

  if (decision.toolName === "multiple") {
    return `I emitted multiple terminal emission tool calls; ${decision.reason}.`;
  }

  if (decision.reason === "unknown terminal emission tool") {
    return `I called an unknown emission tool ${decision.toolName}.`;
  }

  return `${decision.toolName} input was invalid: ${decision.reason}`;
}

export function buildInvalidToolFinalizerRetryPromptSection(
  decision: InvalidToolDecision,
  availableEmissionNames: readonly EmissionToolName[] = resolveAvailableEmissionNames(undefined),
): string {
  return [
    "My previous turn did not emit a valid final response.",
    invalidToolRetryCause(decision),
    `I need to emit exactly one of ${availableEmissionNames.join(" / ")} with valid input.`,
  ].join(" ");
}

async function retryInvalidToolFinalizer(input: {
  initial: FinalizerResult;
  availableEmissionNames: readonly EmissionToolName[];
  runRetry: (retryPromptSection: string) => Promise<FinalizerResult>;
}): Promise<{
  result: FinalizerResult;
  invalidToolAfterRegenerate?: FinalizerInvalidToolDiagnostic;
}> {
  if (input.initial.decision.kind !== "invalid_tool") {
    return { result: input.initial };
  }

  const retryPromptSection = buildInvalidToolFinalizerRetryPromptSection(
    input.initial.decision,
    input.availableEmissionNames,
  );
  const retry = await input.runRetry(retryPromptSection);

  return retry.decision.kind === "invalid_tool"
    ? {
        result: retry,
        invalidToolAfterRegenerate: finalizerInvalidToolDiagnostic(retry.decision, "regenerate"),
      }
    : { result: retry };
}

function structuralNoOutputFlags(
  context: DeliberationContext,
  input: { additionalOpenQuestionsRenderedCount?: number } = {},
): FinalizerNoOutputStructuralFlag[] {
  const flags: FinalizerNoOutputStructuralFlag[] = [];

  if ((context.sharedStateAppliedOperationCount ?? 0) > 0) {
    flags.push("with_state_delta", "current_turn_state_delta");
  }

  const renderedOpenQuestionCount =
    (context.openQuestionsRenderedToFinalizerCount ?? 0) +
    (input.additionalOpenQuestionsRenderedCount ?? 0);

  if (renderedOpenQuestionCount > 0) {
    flags.push("with_open_question", "open_question_rendered");
  }

  return flags;
}

function uniqueNoOutputCategories(
  categories: readonly FinalizerNoOutputCategory[],
): FinalizerNoOutputCategory[] {
  return [...new Set(categories)];
}

function uniqueNoOutputStructuralFlags(
  flags: readonly FinalizerNoOutputStructuralFlag[],
): FinalizerNoOutputStructuralFlag[] {
  return [...new Set(flags)];
}

function legacyStructuralCategoriesFromFlags(
  flags: readonly FinalizerNoOutputStructuralFlag[],
): FinalizerNoOutputStructuralCategory[] {
  const categories: FinalizerNoOutputStructuralCategory[] = [];

  if (flags.includes("with_state_delta")) {
    categories.push("with_state_delta");
  }

  if (flags.includes("with_open_question")) {
    categories.push("with_open_question");
  }

  return categories;
}

function buildFinalizerEmission(
  result: FinalizerResult,
  structuralFlags: readonly FinalizerNoOutputStructuralFlag[] = [],
  options: {
    invalidToolSuppressionReason?: Extract<
      GenerationSuppressionReason,
      "finalizer_failed" | "invalid_tool_after_regenerate"
    >;
    invalidToolDiagnosticAttempt?: FinalizerInvalidToolDiagnostic["attempt"];
  } = {},
): FinalizerEmission {
  const suppressionReason =
    result.decision.kind === "invalid_tool"
      ? (options.invalidToolSuppressionReason ?? finalizerSuppressionReason(result))
      : finalizerSuppressionReason(result);

  if (suppressionReason !== null) {
    const noOutputSemanticCategories =
      result.decision.kind === "no_output" ? result.decision.no_output_categories : undefined;
    const noOutputStructuralFlags =
      noOutputSemanticCategories === undefined
        ? undefined
        : uniqueNoOutputStructuralFlags([
            ...structuralFlags,
            ...(noOutputSemanticCategories.includes("when_borg_addressed")
              ? (["borg_directly_addressed"] as const)
              : []),
          ]);
    const noOutputCategories =
      noOutputSemanticCategories === undefined || noOutputStructuralFlags === undefined
        ? undefined
        : uniqueNoOutputCategories([
            ...noOutputSemanticCategories,
            ...legacyStructuralCategoriesFromFlags(noOutputStructuralFlags),
          ]);
    const primaryNoOutputReason =
      noOutputSemanticCategories === undefined
        ? undefined
        : ((result.decision.kind === "no_output"
            ? result.decision.primary_no_output_reason
            : undefined) ?? deriveFinalizerNoOutputPrimaryReason(noOutputSemanticCategories));

    return {
      response: "",
      emitted: false,
      emission: {
        kind: "suppressed",
        reason: suppressionReason,
        ...(noOutputCategories === undefined ? {} : { no_output_categories: noOutputCategories }),
        ...(primaryNoOutputReason === undefined
          ? {}
          : { primary_no_output_reason: primaryNoOutputReason }),
        ...(noOutputStructuralFlags === undefined
          ? {}
          : { structural_no_output_flags: noOutputStructuralFlags }),
        ...(result.decision.kind === "no_output"
          ? { decision_rationale: result.decision.reason }
          : {}),
        ...(result.decision.kind !== "invalid_tool"
          ? {}
          : {
              finalizer_invalid_tool: finalizerInvalidToolDiagnostic(
                result.decision,
                options.invalidToolDiagnosticAttempt ?? "initial",
              ),
            }),
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
        ...(result.decision.discourse_control === undefined
          ? {}
          : { discourse_control: result.decision.discourse_control }),
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

  if (result.decision.kind === "continue_thought") {
    return {
      response: "",
      emitted: false,
      emission: {
        kind: "continue_thought",
        text: result.decision.text,
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
      ...(result.decision.discourse_control === undefined
        ? {}
        : { discourse_control: result.decision.discourse_control }),
    },
  };
}

function cognitionThinkingOption(
  options: DeliberatorOptions,
): LLMCompleteOptions["thinking"] | undefined {
  if (options.cognitionThinking?.enabled !== true) {
    return undefined;
  }

  // Adaptive is the supported mode on Opus 4.6+/Sonnet 4.6 (the only mode on
  // 4.7/4.8). It pairs with `effort` (see cognitionEffortOption) and uses
  // tool_choice:auto at the call sites so the model may think before emitting.
  if (options.cognitionThinking.mode === "adaptive") {
    return { type: "adaptive" };
  }

  return {
    type: "enabled",
    budget_tokens: options.cognitionThinking.budget_tokens,
  };
}

function cognitionEffortOption(
  options: DeliberatorOptions,
): LLMCompleteOptions["effort"] | undefined {
  if (
    options.cognitionThinking?.enabled !== true ||
    options.cognitionThinking.mode !== "adaptive"
  ) {
    return undefined;
  }

  return options.cognitionThinking.effort;
}

function allowedEmissionsForParticipationPolicy(
  policy: SessionParticipationPolicy | undefined,
  turnOrigin: TurnOrigin | undefined,
): readonly EmissionToolName[] | undefined {
  switch (policy ?? "active") {
    case "active":
      return turnOrigin === "autonomous"
        ? undefined
        : [
            EMIT_ANSWER_FINALIZER_TOOL_NAME,
            EMIT_OBSERVE_FINALIZER_TOOL_NAME,
            EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME,
            EMIT_SELF_REPORT_FINALIZER_TOOL_NAME,
          ];
    case "paused":
    case "muted":
      return [EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME];
    case "observing":
      return [EMIT_OBSERVE_FINALIZER_TOOL_NAME, EMIT_NO_OUTPUT_FINALIZER_TOOL_NAME];
  }
}

export class Deliberator {
  private readonly tracer: TurnTracer;
  private readonly clock: Clock;

  constructor(private readonly options: DeliberatorOptions) {
    this.tracer = options.tracer ?? NOOP_TRACER;
    this.clock = options.clock ?? new SystemClock();
  }

  private async runFinalizerPhase(
    turnId: string | undefined,
    options: RunFinalizerOptions,
  ): Promise<FinalizerResult> {
    return traceTurnPhase({
      tracer: this.tracer,
      clock: this.clock,
      turnId: turnId ?? "unknown",
      sessionId: options.sessionId,
      phase: "final",
      sub: options.path,
      run: () => runFinalizer(options),
      completedSub: (result) =>
        `path=${options.path} decision=${result.decision.kind} stop=${result.usage.stop_reason ?? "none"}`,
    });
  }

  private async runFinalizerTriad(input: RunFinalizerTriadInput): Promise<DeliberationResult> {
    const response = await this.runFinalizerPhase(
      input.callContext.context.turnId,
      buildFinalizerCallOptions(input.callContext, this.options, {
        path: input.path,
        ...(input.additionalPromptSections === null
          ? {}
          : { additionalPromptSections: input.additionalPromptSections }),
        finalizerAttempt: "initial",
      }),
    );
    const finalizerRetry = await retryInvalidToolFinalizer({
      initial: response,
      availableEmissionNames: input.availableEmissionNames,
      runRetry: (retryPromptSection) =>
        this.runFinalizerPhase(
          input.callContext.context.turnId,
          buildFinalizerCallOptions(input.callContext, this.options, {
            path: input.path,
            additionalPromptSections: appendFinalizerPromptSections(
              input.additionalPromptSections,
              [retryPromptSection],
            ),
            finalizerAttempt: "regenerate",
          }),
        ),
    });
    const responseForResult = finalizerRetry.result;
    let usage =
      input.priorUsage === undefined
        ? undefined
        : mergeFinalizerTriadUsage({
            priorUsage: input.priorUsage,
            initial: response,
            result: responseForResult,
          });
    const finalized = buildFinalizerEmission(
      responseForResult,
      input.callContext.structuralNoOutputFlags,
      {
        ...(finalizerRetry.invalidToolAfterRegenerate === undefined
          ? {}
          : {
              invalidToolSuppressionReason: "invalid_tool_after_regenerate",
              invalidToolDiagnosticAttempt: finalizerRetry.invalidToolAfterRegenerate.attempt,
            }),
      },
    );
    usage =
      usage ??
      mergeFinalizerTriadUsage({
        initial: response,
        result: responseForResult,
      });
    const result = input.buildResult({
      finalized,
      responseForResult,
      usage,
    });

    return attachRegenerator(result, async (regeneration) => {
      const regeneratedResponse = await this.runFinalizerPhase(
        input.callContext.context.turnId,
        buildFinalizerCallOptions(input.callContext, this.options, {
          path: input.path,
          additionalPromptSections: appendFinalizerPromptSections(
            input.additionalPromptSections,
            regeneration.additionalPromptSections,
          ),
          finalizerAttempt: "regenerate",
        }),
      );
      const regeneratedFinalized = buildFinalizerEmission(
        regeneratedResponse,
        input.callContext.structuralNoOutputFlags,
        { invalidToolDiagnosticAttempt: "regenerate" },
      );

      return {
        ...result,
        response: regeneratedFinalized.response,
        emitted: regeneratedFinalized.emitted,
        emission: regeneratedFinalized.emission,
        tool_calls: regeneratedResponse.toolCallsMade,
        usage: mergeDeliberationUsage(result.usage, regeneratedResponse.usage),
      };
    });
  }

  async run(
    context: DeliberationContext,
    streamWriter?: StreamWriter,
  ): Promise<DeliberationResult> {
    const stakes = context.options?.stakes ?? "low";
    const cognitionThinkingEnabled = this.options.cognitionThinking?.enabled === true;
    const planningMaxTokens =
      context.options?.maxThinkingTokens ?? DEFAULT_DELIBERATION_PLAN_MAX_TOKENS;
    const semanticContextBudget = Math.max(DEFAULT_SEMANTIC_CONTEXT_BUDGET, planningMaxTokens * 4);
    const retrievalContextBudget = DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET;
    // Adaptive thinking spends output tokens; the per-call budget must hold the
    // thinking AND the emission or the model exhausts max_tokens mid-thought and
    // never emits a tool. Raise the output budget when thinking is on. The context
    // budget above stays keyed to planningMaxTokens (unchanged).
    const responseMaxTokens = cognitionThinkingEnabled
      ? THINKING_DELIBERATION_MAX_TOKENS
      : DEFAULT_DELIBERATION_RESPONSE_MAX_TOKENS;
    const plannerCallMaxTokens = cognitionThinkingEnabled
      ? THINKING_DELIBERATION_MAX_TOKENS
      : planningMaxTokens;
    const systemOneMaxTokens = responseMaxTokens;
    const systemTwoMaxTokens = responseMaxTokens;
    const trace =
      this.tracer.enabled && context.turnId !== undefined
        ? {
            tracer: this.tracer,
            turnId: context.turnId,
            sessionId: context.sessionId,
          }
        : undefined;
    const retrievalConfidence =
      context.retrievalConfidence ??
      computeRetrievalConfidence({
        episodes: context.retrievalResult,
        contradictionPresent: context.contradictionPresent ?? false,
        nowMs: this.clock.now(),
      });
    let effectiveContext: DeliberationContext = {
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
      context.routingOverride,
      {
        routing: context.contradictionRouting ?? null,
        cooldown: context.contradictionRoutingCooldown,
        audienceKey: context.audienceEntityId ?? context.audience ?? context.sessionId,
        currentTurn: context.workingMemory.turn_counter,
        cooldownTurns: context.contradictionRoutingConfig?.cooldownTurns,
        enabled: context.contradictionRoutingConfig?.enabled ?? true,
      },
    );
    effectiveContext = {
      ...effectiveContext,
      contradictionRoutingTier: decision.contradiction_tier,
      deliberationPath: decision.path,
    };
    const baseSystemPromptOptions: BuildBaseSystemPromptOptions = {
      retrievalContextBudget,
      semanticContextBudget,
      nowMs: this.clock.now(),
      participationPolicy: effectiveContext.participationPolicy ?? "active",
      ...(this.options.hostCapabilities === undefined
        ? {}
        : { hostCapabilities: this.options.hostCapabilities }),
      ...(this.options.promptBlocks === undefined
        ? {}
        : { promptBlocks: this.options.promptBlocks }),
    };
    const baseSystemPrompt = buildBaseSystemPrompt(effectiveContext, baseSystemPromptOptions);
    const cacheableBaseSystemPrompt = buildCacheableBaseSystemPromptParts(
      effectiveContext,
      baseSystemPromptOptions,
    );
    const sessionReentryContinuityPromptSections =
      context.sessionReentryContinuityPromptSection === undefined ||
      context.sessionReentryContinuityPromptSection === null
        ? []
        : [context.sessionReentryContinuityPromptSection];
    const dialogueMessages = buildDialogueMessages(context.recencyMessages, context.userMessage);
    const currentUserBlockMessages = withCurrentUserContentBlocks(
      toContentBlockMessages(dialogueMessages),
      context.currentUserContent,
    );
    const finalizerEvidenceLedger = withFinalizerImageBudget(
      currentUserBlockMessages,
      context.evidenceLedger,
      { maxImagesPerLlmCall: this.options.maxImagesPerLlmCall },
    );
    const finalizerEvidenceLedgerPromptSection =
      finalizerEvidenceLedger === undefined || finalizerEvidenceLedger === null
        ? context.evidenceLedgerPromptSection
        : finalizerEvidenceLedger === context.evidenceLedger &&
            context.evidenceLedgerPromptSection !== undefined &&
            context.evidenceLedgerPromptSection !== null
          ? context.evidenceLedgerPromptSection
          : renderEvidenceLedger(finalizerEvidenceLedger, {
              sharedState: this.options.sharedStateRenderOptions,
            });
    const evidenceLedgerPromptSections =
      finalizerEvidenceLedgerPromptSection === undefined ||
      finalizerEvidenceLedgerPromptSection === null
        ? null
        : [finalizerEvidenceLedgerPromptSection];
    const finalizerGroundingPromptSections = [
      ...sessionReentryContinuityPromptSections,
      ...(evidenceLedgerPromptSections ?? []),
    ];
    const dialogueBlockMessages = withLedgerImageContentBlocks(
      currentUserBlockMessages,
      finalizerEvidenceLedger,
      { maxImagesPerLlmCall: this.options.maxImagesPerLlmCall },
    );
    const thinking = cognitionThinkingOption(this.options);
    const effort = cognitionEffortOption(this.options);
    // Spread into every being-cognition LLM call (finalizer + s2 planner). Thinking
    // and effort travel together; those call sites use tool_choice:auto when
    // thinking is active, since the API rejects thinking under forced tool use.
    const reasoningCallOptions = {
      ...(thinking === undefined ? {} : { thinking }),
      ...(effort === undefined ? {} : { effort }),
    };
    const allowedEmissions = allowedEmissionsForParticipationPolicy(
      effectiveContext.participationPolicy,
      effectiveContext.turnOrigin,
    );
    const availableEmissionNames = resolveAvailableEmissionNames(
      allowedEmissions,
      effectiveContext.turnOrigin,
    );
    const manualOutboundAuthorized =
      isCreatorInOperatorContext({
        currentSenderBorgRole: effectiveContext.creatorContext?.currentSenderBorgRole ?? null,
        sessionAudienceRole: effectiveContext.creatorContext?.sessionAudienceRole ?? null,
      }) &&
      (effectiveContext.operatorSessionSnapshot?.sessions.some(
        (session) => session.outbound_targetable,
      ) ??
        false);
    const autonomousOutboundAuthorized =
      effectiveContext.turnOrigin === "autonomous" &&
      (effectiveContext.autonomousOutbound?.targets.length ?? 0) > 0;
    const outboundToolAvailable =
      exposesOutboundTool(effectiveContext.turnOrigin) &&
      (manualOutboundAuthorized || autonomousOutboundAuthorized);
    const baseFinalizerCallContext = {
      context,
      effectiveContext,
      baseSystemPrompt,
      cacheableSystemPrompt: cacheableBaseSystemPrompt,
      initialMessages: dialogueBlockMessages,
      reasoningCallOptions,
      allowedEmissions,
      outboundToolAvailable,
      tracer: this.tracer,
    } satisfies Omit<FinalizerCallOptionsContext, "maxTokens" | "structuralNoOutputFlags">;

    if (decision.path === "system_1") {
      const finalizerStructuralFlags = structuralNoOutputFlags(effectiveContext);

      return this.runFinalizerTriad({
        callContext: {
          ...baseFinalizerCallContext,
          maxTokens: systemOneMaxTokens,
          structuralNoOutputFlags: finalizerStructuralFlags,
        },
        path: "system_1",
        additionalPromptSections:
          finalizerGroundingPromptSections.length === 0 ? null : finalizerGroundingPromptSections,
        availableEmissionNames,
        buildResult: ({ finalized, responseForResult, usage }) => ({
          path: "system_1",
          response: finalized.response,
          emitted: finalized.emitted,
          emission: finalized.emission,
          emissionRecommendation: "emit",
          thoughtStreamEntryIds: [],
          thoughts: [],
          tool_calls: responseForResult.toolCallsMade,
          usage,
          decision_reason: decision.reason,
          retrievedEpisodes: [...context.retrievalResult],
          referencedEpisodeIds: null,
          intents: [],
          thoughtsPersisted: false,
        }),
      });
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
            sharedState: this.options.sharedStateRenderOptions,
          });
    const forcedContradictionOpenQuestionsPrompt =
      renderForcedContradictionOpenQuestionsPrompt(context);
    const plannerAdditionalPromptSections = [
      ...sessionReentryContinuityPromptSections,
      compactPlannerLedger?.promptSection ?? null,
      forcedContradictionOpenQuestionsPrompt,
    ];

    if (compactPlannerLedger !== null && this.tracer.enabled && context.turnId !== undefined) {
      this.tracer.emit("deliberation.planner_ledger.completed", {
        turnId: context.turnId,
        session_id: context.sessionId,
        entry_counts: toTraceJsonValue(compactPlannerLedger.traceSummary.entryCountsBySection),
        omitted_entry_counts: toTraceJsonValue(
          compactPlannerLedger.traceSummary.omittedEntryCountsBySection,
        ),
        estimated_tokens_by_section: toTraceJsonValue(
          compactPlannerLedger.traceSummary.estimatedTokensBySection,
        ),
        decision_artifact_entry_count: compactPlannerLedger.traceSummary.sharedStateEntryCount,
        decision_artifact_rendered_token_estimate:
          compactPlannerLedger.traceSummary.sharedStateRenderedTokens,
        decision_artifact_rendered_by_kind: toTraceJsonValue(
          compactPlannerLedger.traceSummary.sharedStateRenderedByKind,
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
      additionalPromptSections: plannerAdditionalPromptSections,
      maxTokens: plannerCallMaxTokens,
      ...reasoningCallOptions,
      tracer: this.tracer,
      turnId: context.turnId,
      sessionId: context.sessionId,
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
        this.tracer.emit("deliberation.plan_persistence.completed", {
          turnId: context.turnId,
          session_id: context.sessionId,
          streamEntryId: persistedEntry.id,
        });
      } else {
        this.tracer.emit("deliberation.plan_persistence.skipped", {
          turnId: context.turnId,
          session_id: context.sessionId,
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
      finalizerGroundingPromptSections.length === 0
        ? [additionalRetrievalBlock, planSection]
        : [...finalizerGroundingPromptSections, additionalRetrievalBlock, planSection];
    const finalizerStructuralFlags = structuralNoOutputFlags(effectiveContext, {
      additionalOpenQuestionsRenderedCount: secondaryRetrieval?.open_questions.length ?? 0,
    });

    return this.runFinalizerTriad({
      callContext: {
        ...baseFinalizerCallContext,
        maxTokens: systemTwoMaxTokens,
        structuralNoOutputFlags: finalizerStructuralFlags,
      },
      path: "system_2",
      additionalPromptSections,
      availableEmissionNames,
      priorUsage: planner.usage,
      buildResult: ({ finalized, responseForResult, usage }) => {
        return {
          path: "system_2",
          response: finalized.response,
          emitted: finalized.emitted,
          emission: finalized.emission,
          emissionRecommendation: "emit",
          thoughtStreamEntryIds: persistedThoughtEntries.map((entry) => entry.id),
          thoughts,
          tool_calls: responseForResult.toolCallsMade,
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
      },
    });
  }
}
