import { z } from "zod";

import {
  type LLMClient,
  type LLMToolCall,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../llm/index.js";

const recallExpansionFacetKindSchema = z.enum([
  "topic",
  "relationship",
  "commitment",
  "open_question",
]);

const recallExpansionToolInputSchema = z.object({
  facets: z
    .array(
      z.object({
        kind: recallExpansionFacetKindSchema,
        query: z.string().min(1).describe("A focused semantic retrieval query for this facet."),
        priority: z.number().min(0).max(1).describe("Relative priority for this facet."),
      }),
    )
    .min(0)
    .max(4)
    .describe("Two to four focused semantic facets when useful; fewer is fine for simple turns."),
  named_terms: z
    .array(z.string().min(1))
    .max(8)
    .describe(
      "Explicit names, aliases, projects, people, products, or labels worth exact known-term lookup.",
    ),
});

export type RecallExpansionResult = z.infer<typeof recallExpansionToolInputSchema>;

export type RecallExpansionOptions = {
  llmClient: LLMClient;
  model: string;
  userMessage: string;
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
  "Do not infer facts beyond the message. Do not answer the user. Use the tool exactly once.",
].join("\n");

export async function expandRecall(
  options: RecallExpansionOptions,
): Promise<RecallExpansionResult> {
  const response = await options.llmClient.complete({
    model: options.model,
    system: RECALL_EXPANSION_SYSTEM_PROMPT,
    messages: [
      {
        role: "user",
        content: options.userMessage,
      },
    ],
    tools: [RECALL_EXPANSION_TOOL],
    tool_choice: { type: "tool", name: RECALL_EXPANSION_TOOL_NAME },
    max_tokens: 512,
    budget: "recall-expansion",
  });
  const toolCall = response.tool_calls.find(isRecallExpansionToolCall);

  if (toolCall === undefined) {
    throw new Error("Recall expansion did not emit the required tool call");
  }

  return recallExpansionToolInputSchema.parse(toolCall.input);
}

function isRecallExpansionToolCall(call: LLMToolCall): boolean {
  return call.name === RECALL_EXPANSION_TOOL_NAME;
}
