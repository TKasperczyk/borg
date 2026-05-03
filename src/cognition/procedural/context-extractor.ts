import { z } from "zod";

import {
  deriveProceduralContextKey,
  proceduralContextProblemKindSchema,
  proceduralContextSchema,
  type ProceduralContext,
  type ProceduralContextAudienceScope,
} from "../../memory/procedural/index.js";
import type { SocialProfile } from "../../memory/social/index.js";
import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import { LLMError } from "../../util/errors.js";
import type { EntityId } from "../../util/ids.js";
import type { PerceptionResult } from "../types.js";

const proceduralContextExtractionSchema = z.object({
  problem_kind: proceduralContextProblemKindSchema,
  domain_tags: z.array(z.string().min(1).max(64)).max(8),
  confidence: z.number().min(0).max(1),
});

const PROCEDURAL_CONTEXT_TOOL_NAME = "EmitProceduralContext";
const MIN_PROCEDURAL_CONTEXT_CONFIDENCE = 0.35;

export const PROCEDURAL_CONTEXT_TOOL = {
  name: PROCEDURAL_CONTEXT_TOOL_NAME,
  description: "Emit the procedural problem kind and compact canonical domain slugs.",
  inputSchema: toToolInputSchema(proceduralContextExtractionSchema),
} satisfies LLMToolDefinition;

export type ProceduralContextDegradedReason =
  | "llm_unavailable"
  | "llm_failed"
  | "low_confidence";

export type ProceduralContextExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  onDegraded?: (reason: ProceduralContextDegradedReason, error?: unknown) => Promise<void> | void;
};

export type ExtractProceduralContextInput = {
  userMessage: string;
  recentMessages: readonly { role: "user" | "assistant"; content: string }[];
  perception: Pick<PerceptionResult, "mode" | "entities">;
  isSelfAudience: boolean;
  audienceEntityId: EntityId | null;
  audienceProfile?: SocialProfile | null;
  inputAudience?: string;
};

function deriveAudienceScope(input: {
  isSelfAudience: boolean;
  audienceEntityId: EntityId | null;
  audienceProfile: SocialProfile | null;
}): ProceduralContextAudienceScope {
  if (input.isSelfAudience) {
    return "self";
  }

  if (input.audienceEntityId !== null && input.audienceProfile !== null) {
    return "known_other";
  }

  return "unknown";
}

function parseResponse(result: LLMCompleteResult): z.infer<typeof proceduralContextExtractionSchema> {
  const call = result.tool_calls.find((toolCall) => toolCall.name === PROCEDURAL_CONTEXT_TOOL_NAME);

  if (call === undefined) {
    throw new LLMError(`Procedural context extractor did not emit ${PROCEDURAL_CONTEXT_TOOL_NAME}`, {
      code: "PROCEDURAL_CONTEXT_INVALID",
    });
  }

  return proceduralContextExtractionSchema.parse(call.input);
}

function buildPrompt(input: ExtractProceduralContextInput): string {
  return [
    "Classify the procedural context for this turn.",
    `Emit your result by calling the ${PROCEDURAL_CONTEXT_TOOL_NAME} tool exactly once.`,
    "Use problem_kind='other' and domain_tags=[] when no grounded procedural context exists.",
    "Use recent_messages as prior-turn context to disambiguate terse current messages; do not echo prior procedural contexts if the current turn has shifted topic.",
    "Domain tags must be compact canonical slugs across languages, such as typescript, lancedb, deployment, atlas, rust, sqlite, writing, planning.",
    "Use lowercase ASCII slugs when there is a conventional technology, tool, project, or workflow name. Do not emit translations, sentence fragments, generic words, or user-language descriptive phrases.",
    "Prefer specific technologies, tools, project codenames, domains, and workflows. Return at most 8 tags.",
    "Context:",
    JSON.stringify({
      user_message: input.userMessage,
      recent_messages: input.recentMessages.slice(-10),
      cognitive_mode: input.perception.mode,
      entities: input.perception.entities,
      input_audience: input.inputAudience ?? null,
    }),
  ].join("\n");
}

export class ProceduralContextExtractor {
  constructor(private readonly options: ProceduralContextExtractorOptions = {}) {}

  private async degraded(
    reason: ProceduralContextDegradedReason,
    error?: unknown,
  ): Promise<null> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return null;
  }

  async extract(input: ExtractProceduralContextInput): Promise<ProceduralContext | null> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    const audienceScope = deriveAudienceScope({
      isSelfAudience: input.isSelfAudience,
      audienceEntityId: input.audienceEntityId,
      audienceProfile: input.audienceProfile ?? null,
    });

    try {
      const extracted = parseResponse(
        await this.options.llmClient.complete({
          model: this.options.model,
          system:
            "You extract concise procedural context. Return only grounded structured tool output.",
          messages: [
            {
              role: "user",
              content: buildPrompt(input),
            },
          ],
          tools: [PROCEDURAL_CONTEXT_TOOL],
          tool_choice: { type: "tool", name: PROCEDURAL_CONTEXT_TOOL_NAME },
          max_tokens: 512,
          budget: "procedural-context",
        }),
      );

      if (extracted.confidence < MIN_PROCEDURAL_CONTEXT_CONFIDENCE) {
        return this.degraded("low_confidence");
      }

      const contextInput = {
        problem_kind: extracted.problem_kind,
        domain_tags: extracted.domain_tags,
        audience_scope: audienceScope,
      };
      const context = proceduralContextSchema.parse({
        ...contextInput,
        context_key: deriveProceduralContextKey(contextInput),
      });

      if (
        context.problem_kind === "other" &&
        context.domain_tags.length === 0 &&
        context.audience_scope === "unknown"
      ) {
        return null;
      }

      return context;
    } catch (error) {
      return this.degraded("llm_failed", error);
    }
  }
}
