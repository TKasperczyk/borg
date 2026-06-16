import type { ToolDefinition, ToolDispatcher } from "../../tools/dispatcher.js";
import { OUTBOUND_POST_TOOL_NAME } from "../../tools/internal/outbound-post-name.js";
import type { TurnOrigin } from "../types.js";

export const AUTONOMOUS_INTERIOR_FINALIZER_TOOL_NAMES = [
  "tool.journal.append",
  "tool.openQuestions.create",
  "tool.openQuestions.resolve",
  "tool.episodic.recent",
  "tool.episodic.search",
  "tool.semantic.walk",
  "tool.promptSurface.changes",
  "tool.scheduledWakes.create",
  "tool.scheduledWakes.list",
  "tool.scheduledWakes.cancel",
] as const;

export type AutonomousFinalizerToolMenuItem = {
  name: string;
  menuSummary: string;
};

export function resolveFinalizerNonTerminalTools(input: {
  dispatcher: ToolDispatcher;
  turnOrigin?: TurnOrigin;
  outboundToolAvailable?: boolean;
}): ToolDefinition[] {
  const outboundTool =
    input.outboundToolAvailable === true
      ? input.dispatcher.getDefinition(OUTBOUND_POST_TOOL_NAME)
      : null;

  if (input.turnOrigin !== "autonomous") {
    return outboundTool === null ? [] : [outboundTool];
  }

  const autonomousToolsByName = new Map(
    input.dispatcher.listTools("autonomous").map((tool) => [tool.name, tool]),
  );
  const interiorTools = AUTONOMOUS_INTERIOR_FINALIZER_TOOL_NAMES.flatMap((name) => {
    const tool = autonomousToolsByName.get(name);

    return tool === undefined ? [] : [tool];
  });

  return outboundTool === null ? interiorTools : [...interiorTools, outboundTool];
}

export function buildFinalizerToolMenuItems(
  tools: readonly Pick<ToolDefinition, "name" | "description" | "menuSummary">[],
): AutonomousFinalizerToolMenuItem[] {
  return tools.map((tool) => ({
    name: tool.name,
    menuSummary: tool.menuSummary ?? tool.description,
  }));
}
