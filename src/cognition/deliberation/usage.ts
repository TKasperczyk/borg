import type { LLMCompleteResult } from "../../llm/index.js";
import type { DeliberationUsage } from "./types.js";

function sumOptional(current: number | undefined, next: number | undefined): number | undefined {
  if (current === undefined && next === undefined) {
    return undefined;
  }

  return (current ?? 0) + (next ?? 0);
}

export function mergeDeliberationUsage(
  current: DeliberationUsage,
  next: DeliberationUsage,
): DeliberationUsage {
  // Cache token fields are kept separate from input_tokens (per
  // observability standard: cache_read is ~0.1x input rate and doesn't
  // count against rate limits, summing them inflates totals by 100x+).
  const cacheCreation = sumOptional(
    current.cache_creation_input_tokens,
    next.cache_creation_input_tokens,
  );
  const cacheRead = sumOptional(current.cache_read_input_tokens, next.cache_read_input_tokens);

  return {
    input_tokens: current.input_tokens + next.input_tokens,
    output_tokens: current.output_tokens + next.output_tokens,
    stop_reason: next.stop_reason,
    ...(cacheCreation === undefined ? {} : { cache_creation_input_tokens: cacheCreation }),
    ...(cacheRead === undefined ? {} : { cache_read_input_tokens: cacheRead }),
  };
}

export function usageFromCompleteResult(result: LLMCompleteResult): DeliberationUsage {
  return {
    input_tokens: result.input_tokens,
    output_tokens: result.output_tokens,
    stop_reason: result.stop_reason,
    ...(result.cache_creation_input_tokens === undefined
      ? {}
      : { cache_creation_input_tokens: result.cache_creation_input_tokens }),
    ...(result.cache_read_input_tokens === undefined
      ? {}
      : { cache_read_input_tokens: result.cache_read_input_tokens }),
  };
}
