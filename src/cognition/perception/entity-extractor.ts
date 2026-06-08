import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { entityKindSchema } from "../../memory/commitments/types.js";
import { CognitionError, LLMError } from "../../util/errors.js";
import { EXTRACTOR_MAX_TOKENS_DEFAULT } from "../prompts/constants.js";
import type { EntityExtractionResult, ExtractedEntity } from "./types.js";

export type { EntityExtractionResult, ExtractedEntity } from "./types.js";

const extractedEntitySchema = z.union([
  z.string().min(1),
  z
    .object({
      name: z.string().min(1),
      kind: entityKindSchema.optional(),
    })
    .strict(),
]);
const entityFallbackSchema = z.object({
  entities: z.array(extractedEntitySchema),
  user_identity_names: z.array(z.string().min(1)).optional(),
});
const ENTITY_FALLBACK_TOOL_NAME = "EmitEntityExtraction";
export const ENTITY_FALLBACK_TOOL = {
  name: ENTITY_FALLBACK_TOOL_NAME,
  description:
    "Emit specific named entities from the input. Names of people, places, products, project codenames, organizations, and @-handles only.",
  inputSchema: toToolInputSchema(entityFallbackSchema),
} satisfies LLMToolDefinition;

const ENTITY_LLM_SYSTEM_PROMPT = [
  "Extract specific named entities from the user's text. Examples of valid entities: a person's canonical name or stable handle, a place name (Sevilla, Granada), a product or codename (Helios, JetStream, Postgres), an organization (Anthropic, OpenAI), a @-handle (@yourname), a project's working title.",
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
  "",
  "For each emitted entity, include kind when known: person for humans, group for channels or rooms containing people, self for Borg, abstract for places, concepts, projects, products, things, and other non-person entities. If omitted, Borg will default kind to person.",
  "",
  "Also populate user_identity_names only when the user explicitly states or confirms that a name is their own name in this message. Do not copy audience labels, metadata names, addressed names, or names merely mentioned as topics.",
].join("\n");

// Output sanitization: keep only language-neutral structural checks.
// Natural-language validity belongs to the LLM extraction contract.
const MAX_ENTITY_LENGTH = 64;
const FORBIDDEN_ENTITY_PATTERNS: readonly RegExp[] = [/^[\p{P}\p{S}]+$/u];

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

function parseEntityFallback(result: LLMCompleteResult): EntityExtractionResult {
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

  return {
    ...sanitizeExtractedEntities(parsed.data.entities),
    userIdentityNames: sanitizeEntities(parsed.data.user_identity_names ?? []),
  };
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

function sanitizeExtractedEntities(values: readonly z.infer<typeof extractedEntitySchema>[]): {
  entities: string[];
  entityMentions: ExtractedEntity[];
} {
  const seen = new Set<string>();
  const entities: string[] = [];
  const entityMentions: ExtractedEntity[] = [];

  for (const value of values) {
    const name = typeof value === "string" ? value.trim() : value.name.trim();
    const kind = typeof value === "string" ? "person" : (value.kind ?? "person");

    if (!isAcceptableEntity(name)) {
      continue;
    }

    const key = name.toLowerCase();

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    entities.push(name);
    entityMentions.push({
      name,
      kind,
    });
  }

  return {
    entities,
    entityMentions,
  };
}

export type EntityExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
};

export class EntityExtractor {
  constructor(private readonly options: EntityExtractorOptions = {}) {}

  async extract(text: string): Promise<EntityExtractionResult> {
    const normalizedText = text.trim();

    if (normalizedText.length === 0) {
      return {
        entities: [],
        entityMentions: [],
        userIdentityNames: [],
      };
    }

    if (this.options.llmClient === undefined || this.options.model === undefined) {
      // No LLM available. Returning empty entities is the honest
      // answer; the previous regex heuristic produced false-positive
      // entities at high rates ('Good', 'If', '[End.]'), and those
      // entities then poisoned downstream retrieval. Empty is better
      // than wrong.
      return {
        entities: [],
        entityMentions: [],
        userIdentityNames: [],
      };
    }

    try {
      const response = await this.options.llmClient.complete({
        model: this.options.model,
        system: ENTITY_LLM_SYSTEM_PROMPT,
        messages: [
          {
            role: "user",
            content: normalizedText,
          },
        ],
        tools: [ENTITY_FALLBACK_TOOL],
        tool_choice: { type: "tool", name: ENTITY_FALLBACK_TOOL_NAME },
        max_tokens: EXTRACTOR_MAX_TOKENS_DEFAULT,
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

  async extractEntities(text: string): Promise<string[]> {
    return (await this.extract(text)).entities;
  }
}
