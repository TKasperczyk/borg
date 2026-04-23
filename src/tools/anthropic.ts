import { toToolInputSchema, type LLMToolDefinition } from "../llm/index.js";

import type { ToolDefinition } from "./dispatcher.js";

export function toAnthropicToolDefinitions(
  tools: readonly ToolDefinition[],
): LLMToolDefinition[] {
  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    inputSchema: toToolInputSchema(tool.inputSchema),
  }));
}
