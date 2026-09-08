import { transformToolNameForOAuth } from "../../llm/index.js";
import type { ToolDefinition, ToolDispatcher } from "../../tools/dispatcher.js";
import { OUTBOUND_POST_TOOL_NAME } from "../../tools/internal/outbound-post-name.js";
import { OWN_RECORDS_PAGE_END_CLAIM } from "../../tools/internal/own-records-page-end-claim.js";
import { exposesOutboundTool, type TurnOrigin } from "../types.js";

export const LIVE_TURN_READ_FINALIZER_TOOL_NAMES = [
  "tool.ownRecords.list",
  "tool.openQuestions.ruminations",
] as const;

export const LIVE_TURN_READ_FINALIZER_TOOL_MENU = [
  "<borg_live_turn_read_tools>",
  "Read tools available inside every live turn, including ordinary user turns:",
  `- tool.ownRecords.list: Browse my own durable thoughts and journal globally by inclusive origin-time range, with an optional explicit session filter. ${OWN_RECORDS_PAGE_END_CLAIM}`,
  "- tool.openQuestions.ruminations: Browse the rumination notes my offline mind-maintenance wrote against my open questions, by inclusive created-at range and optionally one question id. A note outlives the question it was written against, so this reaches questions I later resolved and questions the loop abandoned for me -- which it does without asking me, when a question's still-open passes reach the no-traction threshold and no episode created after it cites it and no action against it is active; nothing else in my turn carries those notes.",
  "I may use these at my own initiative when I need to check what I thought or noticed at an earlier time. They are structural time-range browses, not text search, so I choose the relevant dates, kinds, and pages.",
  "</borg_live_turn_read_tools>",
].join("\n");

export const AUTONOMOUS_INTERIOR_FINALIZER_TOOL_NAMES = [
  "tool.journal.append",
  "tool.openQuestions.create",
  "tool.openQuestions.resolve",
  "tool.openQuestions.ruminations",
  "tool.goals.retire",
  "tool.goals.block",
  "tool.goals.unblock",
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

function dedupeToolsByWireName(tools: readonly ToolDefinition[]): ToolDefinition[] {
  const seen = new Set<string>();

  return tools.filter((tool) => {
    const wireName = transformToolNameForOAuth(tool.name);

    if (seen.has(wireName)) {
      return false;
    }

    seen.add(wireName);

    return true;
  });
}

export function resolveFinalizerNonTerminalTools(input: {
  dispatcher: ToolDispatcher;
  turnOrigin?: TurnOrigin;
}): ToolDefinition[] {
  const toolOrigin = input.turnOrigin === "autonomous" ? "autonomous" : "deliberator";
  const availableToolsByName = new Map(
    input.dispatcher.listTools(toolOrigin).map((tool) => [tool.name, tool]),
  );
  const outboundTool = exposesOutboundTool(input.turnOrigin)
    ? availableToolsByName.get(OUTBOUND_POST_TOOL_NAME)
    : undefined;
  const liveTurnReadTools = LIVE_TURN_READ_FINALIZER_TOOL_NAMES.flatMap((name) => {
    const tool = availableToolsByName.get(name);

    return tool === undefined ? [] : [tool];
  });

  if (input.turnOrigin !== "autonomous") {
    return outboundTool === undefined ? liveTurnReadTools : [...liveTurnReadTools, outboundTool];
  }

  const interiorTools = AUTONOMOUS_INTERIOR_FINALIZER_TOOL_NAMES.flatMap((name) => {
    const tool = availableToolsByName.get(name);

    return tool === undefined ? [] : [tool];
  });

  // The two lists overlap by design -- a read tool offered in every live turn is also part of
  // the autonomous interior menu -- and the API rejects a request whose tool names are not
  // unique, taking the whole turn with it. Dedupe on the name the wire will actually see:
  // the OAuth transport folds every non-alphanumeric to `_`, so two distinct source names can
  // still collide there even when they differ here.
  const tools = dedupeToolsByWireName([...liveTurnReadTools, ...interiorTools]);

  return outboundTool === undefined ? tools : dedupeToolsByWireName([...tools, outboundTool]);
}

export function buildFinalizerToolMenuItems(
  tools: readonly Pick<ToolDefinition, "name" | "description" | "menuSummary">[],
): AutonomousFinalizerToolMenuItem[] {
  return tools.map((tool) => ({
    name: tool.name,
    menuSummary: tool.menuSummary ?? tool.description,
  }));
}
