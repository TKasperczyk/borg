import { z } from "zod";

import {
  callStructuredTool,
  isStructuredToolCallError,
  type LLMClient,
  type LLMMessage,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../llm/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import type { SessionId } from "../util/ids.js";

const recallExpansionFacetKindSchema = z.enum([
  "topic",
  "relationship",
  "commitment",
  "open_question",
]);
const MAX_RECALL_EXPANSION_FACETS = 4;
const MAX_RECALL_EXPANSION_NAMED_TERMS = 16;

const recallExpansionFacetSchema = z.object({
  kind: recallExpansionFacetKindSchema,
  query: z.string().min(1).describe("A focused semantic retrieval query for this facet."),
  priority: z.number().min(0).max(1).describe("Relative priority for this facet."),
});

const recallExpansionToolInputSchema = z.object({
  facets: z
    .array(recallExpansionFacetSchema)
    .min(0)
    .max(MAX_RECALL_EXPANSION_FACETS)
    .describe("Two to four focused semantic facets when useful; fewer is fine for simple turns."),
  named_terms: z
    .array(z.string().min(1))
    .max(MAX_RECALL_EXPANSION_NAMED_TERMS)
    .describe(
      "Up to 16 explicit names, aliases, projects, people, products, or labels worth exact known-term lookup.",
    ),
});
const recallExpansionParserSchema = recallExpansionToolInputSchema.extend({
  facets: z.array(recallExpansionFacetSchema).min(0),
});

export type RecallExpansionResult = z.infer<typeof recallExpansionToolInputSchema>;

export type RecallExpansionOptions = {
  llmClient: LLMClient;
  model: string;
  userMessage: string;
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
};

export const RECALL_EXPANSION_TOOL_NAME = "EmitRecallExpansion";
export const DEFAULT_RECALL_EXPANSION_MODEL = "claude-haiku-4-5-20251001";

const RECALL_EXPANSION_TOOL: LLMToolDefinition = {
  name: RECALL_EXPANSION_TOOL_NAME,
  description:
    "Emit semantic recall facets and explicit named terms for exact memory lookup. This is not an answer to the user.",
  inputSchema: toToolInputSchema(recallExpansionToolInputSchema),
};

const RECALL_EXPANSION_SYSTEM_PROMPT = [
  "You expand one user turn into retrieval intents for Borg memory.",
  "Identify semantic facets that may need memories, and separately list explicit named terms worth exact lookup.",
  "Return no more than 4 facets, ranked by priority.",
  "Return at most 16 named terms.",
  "Do not infer facts beyond the message. Do not answer the user. Use the tool exactly once.",
].join("\n");

export async function expandRecall(
  options: RecallExpansionOptions,
): Promise<RecallExpansionResult> {
  const messages: LLMMessage[] = [
    {
      role: "user",
      content: options.userMessage,
    },
  ];
  const tools = [RECALL_EXPANSION_TOOL];

  let parsed: z.infer<typeof recallExpansionParserSchema>;

  try {
    parsed = (
      await callStructuredTool({
        llmClient: options.llmClient,
        request: {
          model: options.model,
          system: RECALL_EXPANSION_SYSTEM_PROMPT,
          messages,
          tools,
          tool_choice: { type: "tool", name: RECALL_EXPANSION_TOOL_NAME },
          max_tokens: 512,
          budget: "recall-expansion",
        },
        toolName: RECALL_EXPANSION_TOOL_NAME,
        parse: (input) => recallExpansionParserSchema.parse(input),
        trace: {
          tracer: options.tracer,
          turnId: options.turnId,
          sessionId: options.sessionId,
          label: "recall_expansion",
          systemPrompt: RECALL_EXPANSION_SYSTEM_PROMPT,
          messages,
          tools,
        },
      })
    ).parsed;
  } catch (error) {
    if (isStructuredToolCallError(error, "missing_tool_call")) {
      throw new Error("Recall expansion did not emit the required tool call");
    }

    if (
      isStructuredToolCallError(error, "invalid_payload") ||
      isStructuredToolCallError(error, "llm_failed")
    ) {
      throw error.cause ?? error;
    }

    throw error;
  }

  if (parsed.facets.length <= MAX_RECALL_EXPANSION_FACETS) {
    return parsed;
  }

  const orderedFacets = parsed.facets
    .map((facet, index) => ({ facet, index }))
    .sort((left, right) => right.facet.priority - left.facet.priority || left.index - right.index);
  const retainedFacets = orderedFacets.slice(0, MAX_RECALL_EXPANSION_FACETS);
  const droppedFacets = orderedFacets.slice(MAX_RECALL_EXPANSION_FACETS);

  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("recall_expansion.completed", {
      turnId: options.turnId,
      ...(options.sessionId === undefined ? {} : { session_id: options.sessionId }),
      clipped: true,
      original_count: parsed.facets.length,
      retained_count: MAX_RECALL_EXPANSION_FACETS,
      ...(options.tracer.includePayloads === true
        ? {
            dropped_facets: droppedFacets.map((item) => ({
              priority: item.facet.priority,
              query: item.facet.query,
            })),
          }
        : {}),
    });
  }

  return {
    ...parsed,
    facets: retainedFacets.map((item) => item.facet),
  };
}
