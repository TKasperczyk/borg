import { toToolInputSchema, type LLMToolDefinition } from "../llm/index.js";

import type { ToolDefinition } from "./dispatcher.js";

export const SHORT_HANDLE_MENU_GUIDANCE =
  "Internal-tool input fields naming stored records accept unique short handles printed on the page; this does not apply to structured reflection or action outputs.";

export function toAnthropicToolDefinitions(tools: readonly ToolDefinition[]): LLMToolDefinition[] {
  // One menu-wide statement, attached to the first applicable tool, never to
  // individual schema fields or the structured emission tools in the same menu.
  const guidanceIndex = tools.findIndex((tool) => (tool.recordIdInputs?.length ?? 0) > 0);
  return tools.map((tool, index) => ({
    name: tool.name,
    description:
      index === guidanceIndex
        ? `${tool.description} ${SHORT_HANDLE_MENU_GUIDANCE}`
        : tool.description,
    inputSchema: toToolInputSchema(tool.inputSchema) as LLMToolDefinition["inputSchema"],
  }));
}
