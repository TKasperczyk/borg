import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMMessage,
  type LLMToolCall,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../llm/index.js";
import type { TurnTracer } from "../cognition/tracing/tracer.js";
import type { JsonValue } from "../util/json-value.js";

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
    .max(16)
    .describe(
      "Up to 16 explicit names, aliases, projects, people, products, or labels worth exact known-term lookup.",
    ),
});

export type RecallExpansionResult = z.infer<typeof recallExpansionToolInputSchema>;

export type RecallExpansionOptions = {
  llmClient: LLMClient;
  model: string;
  userMessage: string;
  tracer?: TurnTracer;
  turnId?: string;
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

  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_started", {
      turnId: options.turnId,
      label: "recall_expansion",
      model: options.model,
      promptCharCount: countCompletePromptChars(RECALL_EXPANSION_SYSTEM_PROMPT, messages),
      toolSchemas: summarizeToolSchemas(tools),
    });
  }

  let response: LLMCompleteResult;

  try {
    response = await options.llmClient.complete({
      model: options.model,
      system: RECALL_EXPANSION_SYSTEM_PROMPT,
      messages,
      tools,
      tool_choice: { type: "tool", name: RECALL_EXPANSION_TOOL_NAME },
      max_tokens: 512,
      budget: "recall-expansion",
    });
  } catch (error) {
    if (options.tracer?.enabled === true && options.turnId !== undefined) {
      options.tracer.emit("llm_call_response", {
        turnId: options.turnId,
        label: "recall_expansion",
        responseShape: {
          error: error instanceof Error ? error.message : String(error),
        },
        stopReason: null,
        usage: null,
      });
    }

    throw error;
  }

  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call_response", {
      turnId: options.turnId,
      label: "recall_expansion",
      responseShape: summarizeRecallExpansionResponseShape(response),
      stopReason: response.stop_reason,
      usage: {
        inputTokens: response.input_tokens,
        outputTokens: response.output_tokens,
      },
    });
  }

  const toolCall = response.tool_calls.find(isRecallExpansionToolCall);

  if (toolCall === undefined) {
    throw new Error("Recall expansion did not emit the required tool call");
  }

  return recallExpansionToolInputSchema.parse(toolCall.input);
}

function isRecallExpansionToolCall(call: LLMToolCall): boolean {
  return call.name === RECALL_EXPANSION_TOOL_NAME;
}

function summarizeRecallExpansionResponseShape(response: LLMCompleteResult): JsonValue {
  return {
    textLength: response.text.length,
    toolUseBlocks: response.tool_calls.map((call) => ({
      id: call.id,
      name: call.name,
    })),
  };
}

function countCompletePromptChars(systemPrompt: string, messages: readonly LLMMessage[]): number {
  return (
    systemPrompt.length +
    messages.reduce((sum, message) => sum + message.role.length + message.content.length, 0)
  );
}

function summarizeToolSchemas(tools: readonly LLMToolDefinition[]): JsonValue {
  return tools.map((tool) => ({
    name: tool.name,
    propertyCount:
      tool.inputSchema.properties === undefined
        ? 0
        : Object.keys(tool.inputSchema.properties).length,
    required: Array.isArray(tool.inputSchema.required) ? tool.inputSchema.required.map(String) : [],
  }));
}
