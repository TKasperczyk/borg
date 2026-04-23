import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { CognitionError, LLMError } from "../../util/errors.js";

const entityFallbackSchema = z.object({
  entities: z.array(z.string().min(1)),
});
const ENTITY_FALLBACK_TOOL_NAME = "EmitEntityExtraction";
const ENTITY_FALLBACK_MAX_INPUT_CHARS = 2_000;
export const ENTITY_FALLBACK_TOOL = {
  name: ENTITY_FALLBACK_TOOL_NAME,
  description: "Emit named entities, handles, products, and quoted phrases from the input.",
  inputSchema: toToolInputSchema(entityFallbackSchema),
} satisfies LLMToolDefinition;

function parseEntityFallback(result: LLMCompleteResult): string[] {
  const call = result.tool_calls.find((toolCall) => toolCall.name === ENTITY_FALLBACK_TOOL_NAME);

  if (call === undefined) {
    throw new CognitionError(`Entity fallback did not emit tool ${ENTITY_FALLBACK_TOOL_NAME}`, {
      code: "ENTITY_FALLBACK_INVALID",
    });
  }

  const parsed = entityFallbackSchema.safeParse(call.input);

  if (!parsed.success) {
    throw new CognitionError("Entity fallback returned invalid payload", {
      cause: parsed.error,
      code: "ENTITY_FALLBACK_INVALID",
    });
  }

  return dedupe(parsed.data.entities);
}

function dedupe(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const items: string[] = [];

  for (const value of values) {
    const normalized = value.trim();

    if (normalized.length === 0) {
      continue;
    }

    const key = normalized.toLowerCase();

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    items.push(normalized);
  }

  return items;
}

function rangeContains(ranges: readonly [number, number][], index: number): boolean {
  return ranges.some(([start, end]) => index >= start && index < end);
}

export function extractEntitiesHeuristically(text: string): string[] {
  const entities: string[] = [];
  const coveredRanges: Array<[number, number]> = [];

  for (const match of text.matchAll(/@[a-zA-Z0-9_]+/g)) {
    entities.push(match[0]);
  }

  for (const match of text.matchAll(/"([^"\n]+)"/g)) {
    if (match[1] !== undefined && match.index !== undefined) {
      entities.push(match[1]);
      coveredRanges.push([match.index, match.index + match[0].length]);
    }
  }

  for (const match of text.matchAll(/'([^'\n]{3,})'/g)) {
    if (match[1] !== undefined && match.index !== undefined) {
      entities.push(match[1]);
      coveredRanges.push([match.index, match.index + match[0].length]);
    }
  }

  for (const match of text.matchAll(/\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b/g)) {
    if (match.index === undefined) {
      continue;
    }

    entities.push(match[0]);
    coveredRanges.push([match.index, match.index + match[0].length]);
  }

  for (const match of text.matchAll(/\b(?:[A-Z]{2,}|[A-Z][a-z]+)\b/g)) {
    if (match.index === undefined) {
      continue;
    }

    if (rangeContains(coveredRanges, match.index)) {
      continue;
    }

    if (match.index === 0 && match[0] !== match[0].toUpperCase()) {
      continue;
    }

    entities.push(match[0]);
  }

  return dedupe(entities);
}

export type EntityExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  useLlmFallback?: boolean;
  shortTextThreshold?: number;
};

export class EntityExtractor {
  private readonly useLlmFallback: boolean;
  private readonly shortTextThreshold: number;

  constructor(private readonly options: EntityExtractorOptions = {}) {
    this.useLlmFallback = options.useLlmFallback ?? true;
    this.shortTextThreshold = options.shortTextThreshold ?? 160;
  }

  async extractEntities(text: string): Promise<string[]> {
    const heuristicEntities = extractEntitiesHeuristically(text);
    const normalizedText = text.trim();

    if (
      heuristicEntities.length > 0 ||
      !this.useLlmFallback ||
      this.options.llmClient === undefined ||
      normalizedText.length === 0
    ) {
      return heuristicEntities;
    }

    const model = this.options.model;

    if (model === undefined) {
      return heuristicEntities;
    }

    try {
      const fallbackInput =
        normalizedText.length > this.shortTextThreshold
          ? normalizedText.slice(0, ENTITY_FALLBACK_MAX_INPUT_CHARS)
          : normalizedText;
      const response = await this.options.llmClient.complete({
        model,
        system: "Extract named entities, handles, products, and quoted phrases.",
        messages: [
          {
            role: "user",
            content: fallbackInput,
          },
        ],
        tools: [ENTITY_FALLBACK_TOOL],
        tool_choice: { type: "tool", name: ENTITY_FALLBACK_TOOL_NAME },
        max_tokens: 512,
        budget: "perception-entity-fallback",
      });
      return parseEntityFallback(response);
    } catch (error) {
      if (error instanceof CognitionError || error instanceof LLMError) {
        throw error;
      }

      throw new CognitionError("Failed to extract entities", {
        cause: error,
        code: "ENTITY_EXTRACTION_FAILED",
      });
    }
  }
}
