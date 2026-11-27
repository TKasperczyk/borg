import { z } from "zod";

import { findInErrorCauseChain, LLMError } from "../util/errors.js";
import { utf16SafePrefix } from "../util/utf16-boundary.js";

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

export function normalizeLlmStopReason(
  raw: string | null,
  classification?: LLMStopReasonClassification,
  provider: "anthropic" | "openai" = "anthropic",
): LLMStopReasonClassification {
  // An explicit raw reason decides cut status; keep compatible provider details.
  if (raw === "max_tokens" || raw === "length") {
    return classification?.kind === "max_tokens" ? classification : { kind: "max_tokens", raw };
  }
  if (raw != null && classification?.kind === "max_tokens") {
    return classifyLlmStopReason(raw, provider);
  }
  return classification ?? classifyLlmStopReason(raw ?? null, provider);
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
        ("stopReason" in candidate && typeof candidate.stopReason === "string") ||
        ("stopReasonClassification" in candidate &&
          candidate.stopReasonClassification !== undefined)),
  );
  return normalizeLlmStopReason(
    carrier?.stopReason ?? null,
    carrier?.stopReasonClassification ??
      (carrier?.code === "LLM_REFUSAL_EXHAUSTED" ? classifyLlmStopReason("refusal") : undefined),
  );
}

// What the provider says about a refusal: the policy category that declined the request and, when
// it gives one, a sentence saying why. The category is an open set upstream and the SDK's own union
// lags it, so it is kept as whatever string the provider sent.
export type LLMRefusalStopDetails = {
  category: string | null;
  explanation: string | null;
  // The explanation's full length, present only when it was cut to the bound below.
  explanation_chars?: number;
};

const REFUSAL_EXPLANATION_MAX_CHARS = 2_000;
// A category is a short label. The bound is only so a malformed one cannot fill a trace line.
const REFUSAL_CATEGORY_MAX_CHARS = 200;

// Reads `stop_details` off a provider message or a stream's final message_delta. Undefined means
// the provider sent no details: no object, or one that names neither field. That is kept apart
// from details whose fields are present and null, which is the provider answering "no category".
// Structure only: nothing here looks at what the explanation says.
export function readLlmRefusalStopDetails(carrier: unknown): LLMRefusalStopDetails | undefined {
  if (carrier === null || typeof carrier !== "object") return undefined;
  const details = (carrier as { stop_details?: unknown }).stop_details;
  if (details === null || typeof details !== "object" || Array.isArray(details)) return undefined;
  if (!("category" in details) && !("explanation" in details)) return undefined;
  const { category, explanation } = details as { category?: unknown; explanation?: unknown };
  const text = typeof explanation === "string" ? explanation : null;

  return {
    category:
      typeof category === "string" ? utf16SafePrefix(category, REFUSAL_CATEGORY_MAX_CHARS) : null,
    explanation: text === null ? null : utf16SafePrefix(text, REFUSAL_EXPLANATION_MAX_CHARS),
    ...(text !== null && text.length > REFUSAL_EXPLANATION_MAX_CHARS
      ? { explanation_chars: text.length }
      : {}),
  };
}

function isRefusalStopDetails(value: unknown): value is LLMRefusalStopDetails {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const { category, explanation } = value as { category?: unknown; explanation?: unknown };

  return (
    (category === null || typeof category === "string") &&
    (explanation === null || typeof explanation === "string")
  );
}

// Finds the details of the refusal an error stands for. They ride on the refusal itself, which is
// the error or one of its causes. An error whose stop reason is not a refusal reports none.
export function getLlmErrorStopDetails(error: unknown): LLMRefusalStopDetails | undefined {
  if (getLlmErrorStopReason(error).kind !== "refusal") return undefined;

  return findInErrorCauseChain(
    error,
    (candidate): candidate is LLMError & { stopDetails: LLMRefusalStopDetails } =>
      candidate instanceof LLMError &&
      isRefusalStopDetails((candidate as { stopDetails?: unknown }).stopDetails),
  )?.stopDetails;
}
