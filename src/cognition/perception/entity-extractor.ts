import { z } from "zod";

import type { LLMClient } from "../../llm/index.js";
import { CognitionError, LLMError } from "../../util/errors.js";

const entityFallbackSchema = z.object({
  entities: z.array(z.string().min(1)),
});

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

    if (
      heuristicEntities.length > 0 ||
      !this.useLlmFallback ||
      this.options.llmClient === undefined ||
      text.trim().length > this.shortTextThreshold
    ) {
      return heuristicEntities;
    }

    const model = this.options.model;

    if (model === undefined) {
      return heuristicEntities;
    }

    try {
      const response = await this.options.llmClient.complete({
        model,
        system:
          'Extract named entities, handles, products, and quoted phrases. Return strict JSON: {"entities":["..."]}.',
        messages: [
          {
            role: "user",
            content: text,
          },
        ],
        max_tokens: 200,
        budget: "perception-entity-fallback",
      });
      const parsed = entityFallbackSchema.safeParse(JSON.parse(response.text) as unknown);

      if (!parsed.success) {
        throw new CognitionError("Entity fallback returned invalid JSON", {
          cause: parsed.error,
          code: "ENTITY_FALLBACK_INVALID",
        });
      }

      return dedupe(parsed.data.entities);
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
