import { z } from "zod";

import { findInErrorCauseChain, LLMError } from "../util/errors.js";

export const llmStopReasonClassificationSchema = z
  .object({
    kind: z.enum([
      "completed",
      "tool_use",
      "max_tokens",
      "pause_turn",
      "refusal",
      "context_window_exceeded",
      "unknown",
    ]),
    raw: z.string().nullable(),
  })
  .strict();

export type LLMStopReasonClassification = z.infer<typeof llmStopReasonClassificationSchema>;

// Provider protocol values only. In particular, content_filter is not inferred
// to be an Anthropic refusal, and no response text participates in this decision.
export function classifyLlmStopReason(
  raw: string | null,
  provider: "anthropic" | "openai" = "anthropic",
): LLMStopReasonClassification {
  let kind: LLMStopReasonClassification["kind"] = "unknown";
  if (provider === "openai") {
    switch (raw) {
      case "stop":
        kind = "completed";
        break;
      case "tool_calls":
      case "function_call":
        kind = "tool_use";
        break;
      case "length":
        kind = "max_tokens";
        break;
    }
  } else {
    switch (raw) {
      case "end_turn":
      case "stop_sequence":
        kind = "completed";
        break;
      case "tool_use":
      case "max_tokens":
      case "pause_turn":
      case "refusal":
        kind = raw;
        break;
      case "model_context_window_exceeded":
        kind = "context_window_exceeded";
        break;
    }
  }
  return { kind, raw };
}

export function getLlmErrorStopReason(error: unknown): LLMStopReasonClassification {
  const carrier = findInErrorCauseChain(
    error,
    (
      candidate,
    ): candidate is LLMError & {
      stopReason?: string;
      stopReasonClassification?: LLMStopReasonClassification;
    } =>
      candidate instanceof LLMError &&
      (candidate.code === "LLM_REFUSAL_EXHAUSTED" ||
        ("stopReason" in candidate && typeof candidate.stopReason === "string")),
  );
  if (carrier?.stopReasonClassification !== undefined) return carrier.stopReasonClassification;
  return classifyLlmStopReason(
    carrier?.code === "LLM_REFUSAL_EXHAUSTED" ? "refusal" : (carrier?.stopReason ?? null),
  );
}
