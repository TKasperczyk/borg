// Runs the System 2 structured planner call and parses its EmitTurnPlan output.
import { z } from "zod";

import {
  type LLMClient,
  type LLMMessage,
  type LLMToolCall,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { DeliberationUsage, SelfSnapshot } from "./types.js";
import { renderTaggedPromptSection } from "./prompt/sections.js";
import { summarizeVoiceAnchors } from "./prompt/voice-anchors.js";

const turnPlanSchema = z.object({
  uncertainty: z
    .string()
    .describe(
      "What's unclear about the user's current turn that matters for the answer? Empty string if nothing.",
    ),
  verification_steps: z
    .array(z.string())
    .describe(
      "Short phrases describing what you should double-check or re-retrieve before answering. Empty array if nothing.",
    ),
  tensions: z
    .array(z.string())
    .describe(
      "Conflicts or contradictions in what you already know that need to be reconciled in the response. Empty array if none.",
    ),
  voice_note: z
    .string()
    .describe(
      "How the voice and posture should land for this specific turn. Empty string if default voice fits.",
    ),
});

export type TurnPlan = z.infer<typeof turnPlanSchema>;

export const TURN_PLAN_TOOL_NAME = "EmitTurnPlan";

const TURN_PLAN_TOOL: LLMToolDefinition = {
  name: TURN_PLAN_TOOL_NAME,
  description:
    "Emit a structured plan for this reflective/high-stakes turn before the final response. The plan is passed back to you in the final-response call so you can execute against it.",
  inputSchema: toToolInputSchema(turnPlanSchema),
};

export type RunS2PlannerOptions = {
  llmClient: LLMClient;
  model: string;
  baseSystemPrompt: string;
  dialogueMessages: readonly LLMMessage[];
  selfSnapshot: SelfSnapshot;
  maxTokens: number;
};

export type S2PlannerResult = {
  plan: TurnPlan | null;
  reasoning: string;
  usage: DeliberationUsage;
};

export async function runS2Planner(options: RunS2PlannerOptions): Promise<S2PlannerResult> {
  const plannerVoiceAnchors = renderTaggedPromptSection(
    "borg_voice_anchors",
    summarizeVoiceAnchors(options.selfSnapshot),
  );
  const planner = await options.llmClient.complete({
    model: options.model,
    system: [
      options.baseSystemPrompt,
      plannerVoiceAnchors,
      [
        "You are about to answer a reflective, high-stakes, or contradictory turn.",
        `Emit a structured plan by calling the ${TURN_PLAN_TOOL_NAME} tool exactly once.`,
        "The plan is passed back to you in the next call so you can execute it. Keep it short and grounded in the current turn -- do NOT try to draft the answer itself here.",
      ].join("\n"),
    ]
      .filter((section): section is string => section !== null)
      .join("\n\n"),
    messages: options.dialogueMessages,
    tools: [TURN_PLAN_TOOL],
    tool_choice: { type: "tool", name: TURN_PLAN_TOOL_NAME },
    max_tokens: options.maxTokens,
    budget: "cognition-plan",
  });

  return {
    plan: extractTurnPlan(planner.tool_calls),
    reasoning: planner.text,
    usage: {
      input_tokens: planner.input_tokens,
      output_tokens: planner.output_tokens,
      stop_reason: planner.stop_reason,
    },
  };
}

function extractTurnPlan(toolCalls: readonly LLMToolCall[]): TurnPlan | null {
  const call = toolCalls.find((entry) => entry.name === TURN_PLAN_TOOL_NAME);

  if (call === undefined) {
    return null;
  }

  const parsed = turnPlanSchema.safeParse(call.input);
  return parsed.success ? parsed.data : null;
}
