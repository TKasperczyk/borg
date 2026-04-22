import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { cognitiveModeSchema, type CognitiveMode } from "../types.js";
import { CognitionError, LLMError } from "../../util/errors.js";

const modeFallbackSchema = z.object({
  mode: cognitiveModeSchema,
});
const MODE_FALLBACK_TOOL_NAME = "EmitModeDetection";
export const MODE_FALLBACK_TOOL = {
  name: MODE_FALLBACK_TOOL_NAME,
  description: "Emit the detected cognitive mode for the message.",
  inputSchema: toToolInputSchema(modeFallbackSchema),
} satisfies LLMToolDefinition;

function parseModeFallback(result: LLMCompleteResult): CognitiveMode {
  const call = result.tool_calls.find((toolCall) => toolCall.name === MODE_FALLBACK_TOOL_NAME);

  if (call === undefined) {
    throw new CognitionError(`Mode fallback did not emit tool ${MODE_FALLBACK_TOOL_NAME}`, {
      code: "MODE_FALLBACK_INVALID",
    });
  }

  const parsed = modeFallbackSchema.safeParse(call.input);

  if (!parsed.success) {
    throw new CognitionError("Mode fallback returned invalid payload", {
      cause: parsed.error,
      code: "MODE_FALLBACK_INVALID",
    });
  }

  return parsed.data.mode;
}

export type ModeDetectorOptions = {
  llmClient?: LLMClient;
  model?: string;
  /**
   * If true (default) and an LLM client + model are configured, the detector
   * classifies every message by asking the LLM. If false, or if no LLM is
   * configured, the detector returns `defaultMode` (which itself defaults
   * to "idle") as a neutral classification.
   *
   * Rationale: a pattern-based heuristic tier existed in earlier revisions.
   * It was brittle by construction -- every novel phrasing required a new
   * regex, and the patch-work drifted into the same class of overfit the
   * rest of the cognition loop was deliberately avoiding. The LLM
   * classification path has always existed as a fallback; this removes the
   * heuristic and lets the LLM do the work on every turn.
   */
  useLlmFallback?: boolean;
  /**
   * Mode returned when no LLM classification happens (either fallback
   * disabled or no client configured). Defaults to "idle". Test harnesses
   * that want a specific mode without wiring a scripted fake LLM can set
   * this -- there is no production reason to override it.
   */
  defaultMode?: CognitiveMode;
};

export class ModeDetector {
  private readonly useLlmFallback: boolean;
  private readonly defaultMode: CognitiveMode;

  constructor(private readonly options: ModeDetectorOptions = {}) {
    this.useLlmFallback = options.useLlmFallback ?? true;
    this.defaultMode = options.defaultMode ?? "idle";
  }

  async detectMode(text: string, recentHistory: readonly string[] = []): Promise<CognitiveMode> {
    if (
      !this.useLlmFallback ||
      this.options.llmClient === undefined ||
      this.options.model === undefined
    ) {
      return this.defaultMode;
    }

    try {
      const response = await this.options.llmClient.complete({
        model: this.options.model,
        system: [
          "Classify the user's message into exactly one cognitive mode. The mode steers retrieval weighting and deliberation depth downstream, so pick the one that fits best, not the safest one.",
          "",
          "- problem_solving: the user is working through a technical or practical problem -- errors, debugging, commands, code, tool output, configuration, troubleshooting a specific thing.",
          "- relational: introductions, greetings by name, talk about specific people, emotional or interpersonal content, anything scoped to the person-to-person dynamic rather than a task.",
          "- reflective: the user is thinking out loud, questioning themselves, asking meta or identity questions, exploring patterns in their own behavior, or asking open-ended questions about the being they are talking to.",
          "- idle: trivial acknowledgments, filler, brief greetings with no topic, nothing substantive to engage with.",
          "",
          'When ambiguous, prefer the more engaged mode ("reflective" over "idle", "problem_solving" over "relational", etc.). "idle" is only for genuinely contentless input like "ok", "thanks", "hmm".',
        ].join("\n"),
        messages: [
          {
            role: "user",
            content: JSON.stringify({
              text,
              recentHistory,
            }),
          },
        ],
        tools: [MODE_FALLBACK_TOOL],
        tool_choice: { type: "tool", name: MODE_FALLBACK_TOOL_NAME },
        max_tokens: 256,
        budget: "perception-mode-fallback",
      });
      return parseModeFallback(response);
    } catch (error) {
      if (error instanceof CognitionError || error instanceof LLMError) {
        throw error;
      }

      throw new CognitionError("Failed to detect cognitive mode", {
        cause: error,
        code: "MODE_DETECTION_FAILED",
      });
    }
  }
}
