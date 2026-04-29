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
  description:
    "Emit specific named entities from the input. Names of people, places, products, project codenames, organizations, and @-handles only.",
  inputSchema: toToolInputSchema(entityFallbackSchema),
} satisfies LLMToolDefinition;

const ENTITY_LLM_SYSTEM_PROMPT = [
  "Extract specific named entities from the user's text. Examples of valid entities: a person's name (Otto, Tom Kasperczyk), a place name (Sevilla, Granada), a product or codename (Helios, JetStream, Postgres), an organization (Anthropic, OpenAI), a @-handle (@yourname), a project's working title.",
  "",
  "Do NOT extract any of the following:",
  "- Common words, even when capitalized at sentence start (Good, If, The, And, But)",
  "- Stopwords or pronouns (you, me, this, that)",
  "- Generic nouns that are not names (system, project, dog, conversation, message)",
  "- Sentence fragments or quoted spans of dialogue (anything containing punctuation that's not part of a name)",
  "- Chat-format markers ('Human:', 'Assistant:', 'User:', 'AI:', anything ending in ':')",
  "- Bracketed stage directions or scene markers ('[end]', '[Held]', '[.]')",
  "- Verbatim phrases longer than ~6 words",
  "",
  "If the text contains no specific named entities, return an empty list. An empty list is the correct output for most casual text. Do not invent entities to fill the list.",
].join("\n");

// Output sanitization: keep only language-neutral structural checks.
// Natural-language validity belongs to the LLM extraction contract.
const MAX_ENTITY_LENGTH = 64;
const FORBIDDEN_ENTITY_PATTERNS: readonly RegExp[] = [
  /^[\p{P}\p{S}]+$/u,
];

function isAcceptableEntity(value: string): boolean {
  const trimmed = value.trim();

  if (trimmed.length === 0 || trimmed.length > MAX_ENTITY_LENGTH) {
    return false;
  }

  for (const pattern of FORBIDDEN_ENTITY_PATTERNS) {
    if (pattern.test(trimmed)) {
      return false;
    }
  }

  return true;
}

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

  return sanitizeEntities(parsed.data.entities);
}

function sanitizeEntities(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const items: string[] = [];

  for (const value of values) {
    const normalized = value.trim();

    if (!isAcceptableEntity(normalized)) {
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

export type EntityExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  shortTextThreshold?: number;
};

export class EntityExtractor {
  private readonly shortTextThreshold: number;

  constructor(private readonly options: EntityExtractorOptions = {}) {
    this.shortTextThreshold = options.shortTextThreshold ?? 160;
  }

  async extractEntities(text: string): Promise<string[]> {
    const normalizedText = text.trim();

    if (normalizedText.length === 0) {
      return [];
    }

    if (this.options.llmClient === undefined || this.options.model === undefined) {
      // No LLM available. Returning empty entities is the honest
      // answer; the previous regex heuristic produced false-positive
      // entities at high rates ('Good', 'If', '[End.]'), and those
      // entities then poisoned downstream retrieval. Empty is better
      // than wrong.
      return [];
    }

    try {
      const fallbackInput =
        normalizedText.length > this.shortTextThreshold
          ? normalizedText.slice(0, ENTITY_FALLBACK_MAX_INPUT_CHARS)
          : normalizedText;
      const response = await this.options.llmClient.complete({
        model: this.options.model,
        system: ENTITY_LLM_SYSTEM_PROMPT,
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
