import { z } from "zod";

import type { EmbeddingClient } from "../../embeddings/index.js";
import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { type RecencyMessage } from "../recency/index.js";
import { cosineSimilarity } from "../../retrieval/embedding-similarity.js";
import type { WorkingMemory } from "../../memory/working/index.js";
import type { GenerationSuppressionReason } from "./types.js";

const GATE_TOOL_NAME = "EmitGenerationGateDecision";
const MINIMAL_TOKEN_LIMIT = 3;
const MINIMAL_CHAR_LIMIT = 32;
const MINIMAL_LOOP_TURN_COUNT = 3;
const REPEATED_SIMILARITY_THRESHOLD = 0.96;
const ROLE_LABEL_LINE_PATTERN = /^\s*(?:human|assistant|user|ai|borg)\s*:/imu;
const ROLE_LABEL_PREFIX_PATTERN = /^\s*(?:human|assistant|user|ai|borg)\s*:\s*/iu;

const generationGateDecisionSchema = z.object({
  decision: z.enum(["proceed", "suppress"]),
  substantive: z.boolean(),
  reason: z.string().min(1),
  confidence: z.number().min(0).max(1),
});

export const GENERATION_GATE_TOOL = {
  name: GATE_TOOL_NAME,
  description:
    "Decide whether the assistant should emit a message for the current user turn.",
  inputSchema: toToolInputSchema(generationGateDecisionSchema),
} satisfies LLMToolDefinition;

export type GenerationGateStructuralSignals = {
  minimalUserInput: boolean;
  roleLabelAtLineStart: boolean;
  emergencyRoleLabelPrefix: boolean;
  activeDiscourseStop: boolean;
  recentMinimalUserRun: number;
  repeatedMinimalSimilarity: number | null;
  repeatedMinimalExchange: boolean;
  hardCapDue: boolean;
  hardCapActiveTurns: number;
};

export type GenerationGateResult = {
  action: "proceed" | "suppress";
  reason?: GenerationSuppressionReason;
  explanation: string;
  clearDiscourseStop: boolean;
  classified: boolean;
  signals: GenerationGateStructuralSignals;
};

export type GenerationGateOptions = {
  llmClient?: LLMClient;
  embeddingClient: EmbeddingClient;
  model?: string;
  hardCapTurns: number;
  onDegraded?: (reason: string, error?: unknown) => void | Promise<void>;
};

export type GenerationGateInput = {
  userMessage: string;
  workingMemory: WorkingMemory;
  recencyMessages: readonly RecencyMessage[];
};

type ParsedGateDecision = z.infer<typeof generationGateDecisionSchema>;

function parseGateDecision(result: LLMCompleteResult): ParsedGateDecision {
  const call = result.tool_calls.find((toolCall) => toolCall.name === GATE_TOOL_NAME);

  if (call === undefined) {
    throw new Error(`Generation gate did not emit tool ${GATE_TOOL_NAME}`);
  }

  const parsed = generationGateDecisionSchema.safeParse(call.input);

  if (!parsed.success) {
    throw parsed.error;
  }

  return parsed.data;
}

function countTokens(text: string): number {
  return text.match(/[\p{L}\p{N}]+/gu)?.length ?? 0;
}

function stripRoleLabelPrefix(text: string): string {
  return text.replace(ROLE_LABEL_PREFIX_PATTERN, "");
}

export function isMinimalUserGenerationInput(text: string): boolean {
  const trimmed = stripRoleLabelPrefix(text).trim();

  if (trimmed.length === 0) {
    return true;
  }

  const tokenCount = countTokens(trimmed);

  if (tokenCount === 0) {
    return true;
  }

  return tokenCount <= MINIMAL_TOKEN_LIMIT && trimmed.length <= MINIMAL_CHAR_LIMIT;
}

function hasRoleLabelLine(text: string): boolean {
  return ROLE_LABEL_LINE_PATTERN.test(text);
}

function hasEmergencyRoleLabelPrefix(text: string): boolean {
  if (!ROLE_LABEL_PREFIX_PATTERN.test(text)) {
    return false;
  }

  return isMinimalUserGenerationInput(text);
}

function recentUserMessages(messages: readonly RecencyMessage[]): string[] {
  return messages.filter((message) => message.role === "user").map((message) => message.content);
}

function countRecentMinimalRun(messages: readonly string[]): number {
  let count = 0;

  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];

    if (message === undefined || !isMinimalUserGenerationInput(message)) {
      break;
    }

    count += 1;
  }

  return count;
}

async function measureRepeatedMinimalSimilarity(input: {
  userMessage: string;
  recentUserMessages: readonly string[];
  embeddingClient: EmbeddingClient;
  onDegraded?: (reason: string, error?: unknown) => void | Promise<void>;
}): Promise<number | null> {
  if (input.recentUserMessages.length === 0) {
    return null;
  }

  const candidates = input.recentUserMessages.slice(-4);

  try {
    const vectors = await input.embeddingClient.embedBatch([input.userMessage, ...candidates]);
    const current = vectors[0];

    if (current === undefined) {
      return null;
    }

    let best = 0;

    for (const vector of vectors.slice(1)) {
      best = Math.max(best, cosineSimilarity(current, vector));
    }

    return Number.isFinite(best) ? Math.max(0, best) : 0;
  } catch (error) {
    await input.onDegraded?.("embedding_failed", error);
    return null;
  }
}

function buildFallbackResult(input: {
  activeDiscourseStop: boolean;
  signals: GenerationGateStructuralSignals;
  explanation: string;
}): GenerationGateResult {
  if (input.activeDiscourseStop) {
    return {
      action: "suppress",
      reason: "active_discourse_stop",
      explanation: input.explanation,
      clearDiscourseStop: false,
      classified: false,
      signals: input.signals,
    };
  }

  return {
    action: "proceed",
    explanation: input.explanation,
    clearDiscourseStop: false,
    classified: false,
    signals: input.signals,
  };
}

export class GenerationGate {
  constructor(private readonly options: GenerationGateOptions) {}

  async evaluate(input: GenerationGateInput): Promise<GenerationGateResult> {
    const activeStop = input.workingMemory.discourse_state?.stop_until_substantive_content ?? null;
    const recentUsers = recentUserMessages(input.recencyMessages);
    const minimalUserInput = isMinimalUserGenerationInput(input.userMessage);
    const recentMinimalUserRun = countRecentMinimalRun(recentUsers);
    const repeatedMinimalSimilarity = activeStop === null && minimalUserInput
      ? await measureRepeatedMinimalSimilarity({
          userMessage: input.userMessage,
          recentUserMessages: recentUsers,
          embeddingClient: this.options.embeddingClient,
          onDegraded: this.options.onDegraded,
        })
      : null;
    const repeatedMinimalExchange =
      minimalUserInput &&
      recentMinimalUserRun > 0 &&
      repeatedMinimalSimilarity !== null &&
      repeatedMinimalSimilarity >= REPEATED_SIMILARITY_THRESHOLD;
    const hardCapActiveTurns =
      activeStop === null
        ? 0
        : Math.max(0, input.workingMemory.turn_counter - activeStop.since_turn);
    const hardCapDue = activeStop !== null && hardCapActiveTurns >= this.options.hardCapTurns;
    const signals: GenerationGateStructuralSignals = {
      minimalUserInput,
      roleLabelAtLineStart: hasRoleLabelLine(input.userMessage),
      emergencyRoleLabelPrefix: hasEmergencyRoleLabelPrefix(input.userMessage),
      activeDiscourseStop: activeStop !== null,
      recentMinimalUserRun,
      repeatedMinimalSimilarity,
      repeatedMinimalExchange,
      hardCapDue,
      hardCapActiveTurns,
    };

    if (signals.emergencyRoleLabelPrefix) {
      return {
        action: "suppress",
        reason: activeStop === null ? "generation_gate" : "active_discourse_stop",
        explanation: "Current user input is a minimal role-label prefix probe.",
        clearDiscourseStop: false,
        classified: false,
        signals,
      };
    }

    const sustainedMinimalLoop =
      minimalUserInput && recentMinimalUserRun + 1 >= MINIMAL_LOOP_TURN_COUNT;
    const shouldClassify =
      activeStop !== null ||
      signals.roleLabelAtLineStart ||
      repeatedMinimalExchange ||
      sustainedMinimalLoop ||
      hardCapDue;

    if (!shouldClassify) {
      return {
        action: "proceed",
        explanation: "No suspicious generation dynamics detected.",
        clearDiscourseStop: false,
        classified: false,
        signals,
      };
    }

    if (this.options.llmClient === undefined || this.options.model === undefined) {
      await this.options.onDegraded?.("llm_unavailable");
      return buildFallbackResult({
        activeDiscourseStop: activeStop !== null,
        signals,
        explanation: "Generation gate classifier unavailable.",
      });
    }

    try {
      const response = await this.options.llmClient.complete({
        model: this.options.model,
        system: [
          "Decide whether the assistant should emit a response to the current user turn.",
          "",
          "Suppress only when the current turn is a loop probe, role-label trap, or non-substantive continuation under an active stop-until-substantive-content state.",
          "Proceed when the user provides a real request, new information, or a legitimate brief reply that should receive a normal assistant response.",
          "If an active stop state is present, clear it only by marking substantive=true for a current user turn with real content.",
          "Do not treat ordinary first-time short replies such as 'yes', 'thanks', or 'no' as suppressible unless the context shows an active stop or sustained loop.",
        ].join("\n"),
        messages: [
          {
            role: "user",
            content: JSON.stringify({
              current_user_message: input.userMessage,
              active_stop: activeStop,
              structural_signals: signals,
              recent_messages: input.recencyMessages.slice(-8).map((message) => ({
                role: message.role,
                content: message.content,
              })),
            }),
          },
        ],
        tools: [GENERATION_GATE_TOOL],
        tool_choice: { type: "tool", name: GATE_TOOL_NAME },
        max_tokens: 512,
        budget: "generation-gate",
      });
      const parsed = parseGateDecision(response);
      const suppressReason: GenerationSuppressionReason =
        activeStop !== null && parsed.substantive !== true
          ? "active_discourse_stop"
          : "generation_gate";
      const action =
        activeStop !== null && parsed.substantive !== true ? "suppress" : parsed.decision;

      return {
        action,
        ...(action === "suppress" ? { reason: suppressReason } : {}),
        explanation: parsed.reason,
        clearDiscourseStop: activeStop !== null && action === "proceed" && parsed.substantive,
        classified: true,
        signals,
      };
    } catch (error) {
      await this.options.onDegraded?.("llm_failed", error);
      return buildFallbackResult({
        activeDiscourseStop: activeStop !== null,
        signals,
        explanation: "Generation gate classifier failed.",
      });
    }
  }
}
