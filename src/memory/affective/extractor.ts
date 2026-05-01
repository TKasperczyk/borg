import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { LLMError } from "../../util/errors.js";

import {
  createNeutralAffectiveSignal,
  dominantEmotionSchema,
  type AffectiveSignal,
} from "./types.js";

const AFFECTIVE_FALLBACK_TOOL_NAME = "EmitAffectiveSignal";
const affectiveFallbackSchema = z.object({
  valence: z
    .number()
    .min(-1)
    .max(1)
    .describe(
      "Affective valence from the speaker's perspective. -1 = highly unpleasant, 0 = neutral, 1 = highly pleasant. Required. If the text is genuinely flat or you are unsure, emit 0.",
    ),
  arousal: z
    .number()
    .min(0)
    .max(1)
    .describe(
      "Affective arousal from the speaker's perspective. 0 = calm/inert, 1 = highly activated/agitated. Required. If the text is genuinely flat or you are unsure, emit 0.1.",
    ),
  dominant_emotion: dominantEmotionSchema
    .nullable()
    .describe(
      'Best-fit dominant emotion category for the speaker, or null when no single category fits. Allowed: "joy", "sadness", "fear", "anger", "surprise", "curiosity", "neutral", or null.',
    ),
});
const AFFECTIVE_SYSTEM_PROMPT = [
  "Infer the dominant affective signal of the supplied text from the speaker's perspective.",
  "",
  'The speaker is the author of "text". Use "recent_history" (the speaker\'s prior turns) only to disambiguate the current text -- do not echo earlier signals if the current text has shifted.',
  "",
  "You MUST emit valence, arousal, and dominant_emotion. Never omit fields. If the text is genuinely flat or you are uncertain, emit valence=0, arousal=0.1, dominant_emotion=\"neutral\".",
].join("\n");
const DEFAULT_MAX_LLM_FALLBACK_CALLS = 1;
export const AFFECTIVE_FALLBACK_TOOL = {
  name: AFFECTIVE_FALLBACK_TOOL_NAME,
  description: "Emit a grounded affective signal for the input text.",
  inputSchema: toToolInputSchema(affectiveFallbackSchema),
} satisfies LLMToolDefinition;

function parseFallbackResponse(result: LLMCompleteResult): AffectiveSignal {
  const call = result.tool_calls.find((toolCall) => toolCall.name === AFFECTIVE_FALLBACK_TOOL_NAME);

  if (call === undefined) {
    throw new LLMError(`Affective extractor did not emit tool ${AFFECTIVE_FALLBACK_TOOL_NAME}`, {
      code: "AFFECTIVE_OUTPUT_INVALID",
    });
  }

  return affectiveFallbackSchema.parse(call.input);
}

export type AffectiveExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  useLlmFallback?: boolean;
  maxLlmFallbackCalls?: number;
  onDegraded?: (reason: AffectiveExtractorDegradedReason, error?: unknown) => Promise<void> | void;
};

export type AffectiveExtractorDegradedReason =
  | "llm_disabled"
  | "llm_unavailable"
  | "llm_exhausted"
  | "llm_failed";

export class AffectiveExtractor {
  private readonly useLlmFallback: boolean;
  private readonly maxLlmFallbackCalls: number;
  private llmFallbackCalls = 0;

  constructor(private readonly options: AffectiveExtractorOptions = {}) {
    this.useLlmFallback = options.useLlmFallback ?? true;
    this.maxLlmFallbackCalls = options.maxLlmFallbackCalls ?? DEFAULT_MAX_LLM_FALLBACK_CALLS;
  }

  private async degraded(
    reason: AffectiveExtractorDegradedReason,
    error?: unknown,
  ): Promise<AffectiveSignal> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return createNeutralAffectiveSignal();
  }

  async analyze(text: string, recentHistory: readonly string[] = []): Promise<AffectiveSignal> {
    if (!this.useLlmFallback) {
      return this.degraded("llm_disabled");
    }

    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    if (this.llmFallbackCalls >= this.maxLlmFallbackCalls) {
      return this.degraded("llm_exhausted");
    }

    this.llmFallbackCalls += 1;

    try {
      const response = await this.options.llmClient.complete({
        model: this.options.model,
        system: AFFECTIVE_SYSTEM_PROMPT,
        messages: [
          {
            role: "user",
            content: JSON.stringify({
              text,
              recent_history: recentHistory.slice(-3),
            }),
          },
        ],
        tools: [AFFECTIVE_FALLBACK_TOOL],
        tool_choice: { type: "tool", name: AFFECTIVE_FALLBACK_TOOL_NAME },
        max_tokens: 256,
        budget: "perception-affective",
      });

      return parseFallbackResponse(response);
    } catch (error) {
      return this.degraded("llm_failed", error);
    }
  }
}
