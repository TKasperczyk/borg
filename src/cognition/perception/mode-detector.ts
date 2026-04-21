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

const IDLE_PATTERNS = [/^(ok|okay|k|thanks|thank you|cool|got it|sure|yep|nope|hi|hello)[.!?]*$/i];
const PROBLEM_PATTERNS = [
  /\berror\b/i,
  /\bstack\b/i,
  /\btrace\b/i,
  /\bexception\b/i,
  /\bpnpm\b/i,
  /\bnpm\b/i,
  /\btsc\b/i,
  /\btraceback\b/i,
  /command not found/i,
  /```/,
];
const RELATIONAL_PATTERNS = [
  /\bthank(s| you)?\b/i,
  /\bsorry\b/i,
  /\bplease\b/i,
  /\bfeel\b/i,
  /\bhope\b/i,
  /@[a-zA-Z0-9_]+/,
];
const REFLECTIVE_PATTERNS = [
  /\bI feel\b/i,
  /\bwho am I\b/i,
  /\babout myself\b/i,
  /\bwhy do I\b/i,
  /\bmy pattern\b/i,
  /\bwhat kind of person\b/i,
];

export function detectModeHeuristically(
  text: string,
  recentHistory: readonly string[] = [],
): CognitiveMode | null {
  const normalized = text.trim();

  if (
    normalized.length === 0 ||
    normalized.length <= 4 ||
    IDLE_PATTERNS.some((pattern) => pattern.test(normalized))
  ) {
    return "idle";
  }

  const matches = new Set<CognitiveMode>();

  if (PROBLEM_PATTERNS.some((pattern) => pattern.test(text))) {
    matches.add("problem_solving");
  }

  if (REFLECTIVE_PATTERNS.some((pattern) => pattern.test(text))) {
    matches.add("reflective");
  }

  if (
    RELATIONAL_PATTERNS.some((pattern) => pattern.test(text)) &&
    (/\byou\b/i.test(text) || recentHistory.some((entry) => /\byou\b/i.test(entry)))
  ) {
    matches.add("relational");
  }

  if (matches.size === 1) {
    const [mode] = matches;
    return mode ?? null;
  }

  if (matches.size > 1) {
    return null;
  }

  return normalized.endsWith("?") ? "reflective" : null;
}

export type ModeDetectorOptions = {
  llmClient?: LLMClient;
  model?: string;
  useLlmFallback?: boolean;
};

export class ModeDetector {
  private readonly useLlmFallback: boolean;

  constructor(private readonly options: ModeDetectorOptions = {}) {
    this.useLlmFallback = options.useLlmFallback ?? true;
  }

  async detectMode(text: string, recentHistory: readonly string[] = []): Promise<CognitiveMode> {
    const heuristic = detectModeHeuristically(text, recentHistory);

    if (
      heuristic !== null ||
      !this.useLlmFallback ||
      this.options.llmClient === undefined ||
      this.options.model === undefined
    ) {
      return heuristic ?? "idle";
    }

    try {
      const response = await this.options.llmClient.complete({
        model: this.options.model,
        system: "Classify the message into problem_solving, relational, reflective, or idle.",
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
