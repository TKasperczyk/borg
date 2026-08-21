import type { ToolDefinition, ToolDispatcher } from "../../tools/dispatcher.js";
import { OUTBOUND_POST_TOOL_NAME } from "../../tools/internal/outbound-post-name.js";
import type { TurnOrigin } from "../types.js";

export const LIVE_TURN_READ_FINALIZER_TOOL_NAMES = ["tool.ownRecords.list"] as const;

export const LIVE_TURN_READ_FINALIZER_TOOL_MENU = [
  "<borg_live_turn_read_tools>",
  "Read tools available inside every live turn, including ordinary user turns:",
  "- tool.ownRecords.list: Browse my own durable thoughts and journal globally by inclusive origin-time range, with an optional explicit session filter.",
  "I may use this at my own initiative when I need to check what I thought or noticed at an earlier time. It is a structural time-range browse, not text search, so I choose the relevant dates, kinds, and pages.",
  "</borg_live_turn_read_tools>",
].join("\n");

export const AUTONOMOUS_INTERIOR_FINALIZER_TOOL_NAMES = [
  "tool.journal.append",
  "tool.openQuestions.create",
  "tool.openQuestions.resolve",
  "tool.goals.retire",
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
  const toolOrigin = input.turnOrigin === "autonomous" ? "autonomous" : "deliberator";
  const availableToolsByName = new Map(
    input.dispatcher.listTools(toolOrigin).map((tool) => [tool.name, tool]),
  );
  const liveTurnReadTools = LIVE_TURN_READ_FINALIZER_TOOL_NAMES.flatMap((name) => {
    const tool = availableToolsByName.get(name);

    return tool === undefined ? [] : [tool];
  });

  if (input.turnOrigin !== "autonomous") {
    return outboundTool === null ? liveTurnReadTools : [...liveTurnReadTools, outboundTool];
  }

  const interiorTools = AUTONOMOUS_INTERIOR_FINALIZER_TOOL_NAMES.flatMap((name) => {
    const tool = availableToolsByName.get(name);

    return tool === undefined ? [] : [tool];
  });

  const tools = [...liveTurnReadTools, ...interiorTools];

  return outboundTool === null ? tools : [...tools, outboundTool];
}

export function buildFinalizerToolMenuItems(
  tools: readonly Pick<ToolDefinition, "name" | "description" | "menuSummary">[],
): AutonomousFinalizerToolMenuItem[] {
  return tools.map((tool) => ({
    name: tool.name,
    menuSummary: tool.menuSummary ?? tool.description,
  }));
}
