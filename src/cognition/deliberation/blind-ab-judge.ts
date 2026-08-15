import { randomInt } from "node:crypto";

import {
  callStructuredTool,
  isStructuredToolCallError,
  StructuredToolCallError,
  type CallStructuredToolOptions,
  type CallStructuredToolResult,
  type StructuredToolCallUsage,
} from "../../llm/index.js";
import { headTailPlannerExcerpt } from "./prompt/planner-context.js";

export type BlindBinaryAssignment<Variant extends string> = {
  left: Variant;
  right: Variant;
};

export function cryptographicBlindAssignment<Variant extends string>(
  first: Variant,
  second: Variant,
  random: () => number = () => randomInt(0, 2),
): BlindBinaryAssignment<Variant> {
  const sample = random();
  if (!Number.isFinite(sample) || sample < 0 || sample >= 2) {
    throw new RangeError("Blind A/B random source must return a finite value in [0, 2)");
  }
  return sample < 1 ? { left: first, right: second } : { left: second, right: first };
}

/** Literal source-string substitution only; this never interprets candidate text. */
export function neutralizeKnownPresentationReferences(
  value: string,
  phrases: readonly string[],
  neutralToken: string,
): string {
  let neutral = value;
  for (const phrase of phrases) neutral = neutral.replaceAll(phrase, neutralToken);
  return neutral;
}

export type NeutralJudgeValueResult = { value: unknown; truncations: number };

function recordValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

/** Structural completeness gate for scope-sensitive judging of old captures. */
export function hasCompleteCapturedCreatorDirectiveScope(value: unknown): boolean {
  const directive = recordValue(value);
  const scope = recordValue(directive?.scope);
  if (scope === null) return false;
  return (
    typeof scope.directiveId === "string" &&
    typeof scope.createdByEntityId === "string" &&
    typeof scope.sourceSessionId === "string" &&
    typeof scope.contentScope === "string" &&
    Array.isArray(scope.allowedEntityIds) &&
    Array.isArray(scope.excludedEntityIds) &&
    (scope.subjectMayKnow === null || typeof scope.subjectMayKnow === "boolean") &&
    typeof scope.mentionPolicy === "string" &&
    typeof scope.deniedAudienceBehavior === "string" &&
    typeof scope.activationScope === "string" &&
    Array.isArray(scope.activationAllowedEntityIds) &&
    Array.isArray(scope.activationExcludedEntityIds)
  );
}

/** Bounded, language-agnostic serialization used by both offline judges. */
export function renderNeutralJudgeValue(
  value: unknown,
  options: { phrases: readonly string[]; neutralToken: string; maxStringChars: number },
): NeutralJudgeValueResult {
  if (typeof value === "string") {
    const excerpt = headTailPlannerExcerpt(
      neutralizeKnownPresentationReferences(value, options.phrases, options.neutralToken),
      options.maxStringChars,
    );
    return {
      value: excerpt.truncated
        ? {
            text: excerpt.text,
            harness_excerpt: true,
            rendered_chars: excerpt.renderedChars,
            source_chars: excerpt.totalChars,
          }
        : excerpt.text,
      truncations: excerpt.truncated ? 1 : 0,
    };
  }
  if (Array.isArray(value)) {
    let truncations = 0;
    const projected = value.map((entry) => {
      const rendered = renderNeutralJudgeValue(entry, options);
      truncations += rendered.truncations;
      return rendered.value;
    });
    return { value: projected, truncations };
  }
  if (value !== null && typeof value === "object") {
    let truncations = 0;
    const projected = Object.fromEntries(
      Object.entries(value).map(([key, entry]) => {
        const rendered = renderNeutralJudgeValue(entry, options);
        truncations += rendered.truncations;
        return [key, rendered.value];
      }),
    );
    return { value: projected, truncations };
  }
  return { value, truncations: 0 };
}

function addUsage(
  left: StructuredToolCallUsage,
  right: StructuredToolCallUsage,
): StructuredToolCallUsage {
  return {
    input_tokens: left.input_tokens + right.input_tokens,
    output_tokens: left.output_tokens + right.output_tokens,
    ...(left.cache_creation_input_tokens === undefined &&
    right.cache_creation_input_tokens === undefined
      ? {}
      : {
          cache_creation_input_tokens:
            (left.cache_creation_input_tokens ?? 0) + (right.cache_creation_input_tokens ?? 0),
        }),
    ...(left.cache_read_input_tokens === undefined && right.cache_read_input_tokens === undefined
      ? {}
      : {
          cache_read_input_tokens:
            (left.cache_read_input_tokens ?? 0) + (right.cache_read_input_tokens ?? 0),
        }),
  };
}

/**
 * Invalid-payload repair and fresh retries share one cumulative LLM-call
 * budget. Transport and missing-tool failures are never retried here.
 */
export async function callJudgeStructuredTool<T>(
  options: CallStructuredToolOptions<T>,
  maxTotalCalls = 3,
): Promise<CallStructuredToolResult<T>> {
  if (!Number.isInteger(maxTotalCalls) || maxTotalCalls < 1) {
    throw new RangeError("maxTotalCalls must be a positive integer");
  }
  let attemptCount = 0;
  let usage: StructuredToolCallUsage = { input_tokens: 0, output_tokens: 0 };

  while (attemptCount < maxTotalCalls) {
    const remainingCalls = maxTotalCalls - attemptCount;
    const requestedAttempts = options.maxAttempts ?? 2;
    const invocationMaxAttempts = Math.min(requestedAttempts, remainingCalls) as 1 | 2;
    try {
      const result = await callStructuredTool({
        ...options,
        maxAttempts: invocationMaxAttempts,
      });
      return {
        ...result,
        attemptCount: attemptCount + result.attemptCount,
        usage: addUsage(usage, result.usage),
      };
    } catch (error) {
      if (!isStructuredToolCallError(error)) throw error;
      attemptCount += error.attemptCount;
      usage = addUsage(usage, error.usage);
      if (error.kind !== "invalid_payload" || attemptCount >= maxTotalCalls) {
        throw new StructuredToolCallError(error.message, {
          kind: error.kind,
          toolName: error.toolName,
          acceptedToolNames: error.acceptedToolNames,
          stopReason: error.stopReason,
          cause: error.cause,
          repairFailure: error.repairFailure,
          attemptCount,
          usage,
        });
      }
    }
  }
  throw new Error("unreachable");
}
