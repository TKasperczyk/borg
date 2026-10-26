import { findInErrorCauseChain, LLMToolArgumentsError } from "../util/errors.js";
import type { LLMCompleteResult, TokenUsageSink } from "./index.js";

export type LLMUsage = Pick<
  LLMCompleteResult,
  "input_tokens" | "output_tokens" | "cache_creation_input_tokens" | "cache_read_input_tokens"
>;

export function addLlmUsage(left: LLMUsage, right: LLMUsage): LLMUsage {
  const total = { ...left };
  for (const key of [
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
  ] as const) {
    if (left[key] !== undefined || right[key] !== undefined) {
      total[key] = (left[key] ?? 0) + (right[key] ?? 0);
    }
  }
  return total;
}

// Keep the original error identity/code (including deadline and auth failures).
// Cause-aware metadata also survives the production observation/error wrappers.
const consumedUsage = new WeakMap<object, { usage: LLMUsage; reported: WeakSet<TokenUsageSink> }>();

export function attachLlmErrorUsage(error: unknown, usage: LLMUsage): void {
  if (error !== null && typeof error === "object") {
    consumedUsage.set(error, { usage, reported: new WeakSet() });
  }
}

function errorUsageRecord(error: unknown) {
  const carrier = findInErrorCauseChain(
    error,
    (candidate): candidate is object =>
      candidate !== null && typeof candidate === "object" && consumedUsage.has(candidate),
  );
  if (carrier !== undefined) return consumedUsage.get(carrier);
  const unparseable = findInErrorCauseChain(
    error,
    (candidate): candidate is LLMToolArgumentsError =>
      candidate instanceof LLMToolArgumentsError && candidate.usage !== undefined,
  );
  if (unparseable?.usage === undefined) return undefined;
  attachLlmErrorUsage(unparseable, unparseable.usage);
  return consumedUsage.get(unparseable);
}

export function getLlmErrorUsage(error: unknown): LLMUsage | undefined {
  return errorUsageRecord(error)?.usage;
}

export async function reportLlmErrorUsage(
  error: unknown,
  sink: TokenUsageSink | undefined,
  call: { budget: string; model: string },
): Promise<void> {
  const record = errorUsageRecord(error);
  if (sink === undefined || record === undefined || record.reported.has(sink)) return;
  // A budget sink records consumption before it can throw for exceeding its cap.
  record.reported.add(sink);
  await sink({ ...call, ...record.usage });
}
