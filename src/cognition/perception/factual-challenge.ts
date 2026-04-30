import { z } from "zod";

import { toToolInputSchema, type LLMClient, type LLMToolDefinition } from "../../llm/index.js";
import { CognitionError } from "../../util/errors.js";
import { factualChallengeSignalSchema, type FactualChallengeSignal } from "../types.js";

const factualChallengeJudgeSchema = z.object({
  factual_challenge: factualChallengeSignalSchema.nullable(),
});

const FACTUAL_CHALLENGE_TOOL_NAME = "EmitFactualChallenge";
export const FACTUAL_CHALLENGE_TOOL = {
  name: FACTUAL_CHALLENGE_TOOL_NAME,
  description:
    "Detect whether the user is challenging or correcting a factual claim from memory, then emit the disputed entity/property and the user's current position. Emit null when there is no factual challenge.",
  inputSchema: toToolInputSchema(factualChallengeJudgeSchema),
} satisfies LLMToolDefinition;

export type FactualChallengeDetectorDegradedReason = "llm_unavailable" | "llm_failed";

export type FactualChallengeDetectorOptions = {
  llmClient?: LLMClient;
  model?: string;
  onDegraded?: (
    reason: FactualChallengeDetectorDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

const FACTUAL_CHALLENGE_SYSTEM_PROMPT = [
  "Detect whether the user's message is a factual challenge to something the assistant appears to remember, claimed, or implied from prior conversation.",
  "",
  "Return factual_challenge=null when the message is ordinary disagreement, preference, debate, task feedback, or a correction that does not dispute a remembered/stored fact.",
  "",
  "Return a factual_challenge object when the user says or implies that the assistant's remembered fact is wrong. Examples include:",
  "- 'you said X, but I never said that'",
  "- 'small correction: my partner is not Maya'",
  "- 'I don't think I ever told you that'",
  "- 'that's not what happened; I said it was Friday'",
  "",
  "Fields:",
  "- disputed_entity: the name/entity under dispute, or null if none is explicit",
  "- disputed_property: the property or remembered claim being contradicted, or null",
  "- user_position: a short statement of what the user now claims",
  "",
  "Do not decide who is right. Only classify whether the current user message is asking the memory system to reconcile a factual conflict.",
].join("\n");

export async function detectFactualChallenge(
  text: string,
  recentHistory: readonly string[] = [],
  options: FactualChallengeDetectorOptions = {},
): Promise<FactualChallengeSignal | null> {
  if (options.llmClient === undefined || options.model === undefined) {
    await options.onDegraded?.("llm_unavailable");
    return null;
  }

  try {
    const response = await options.llmClient.complete({
      model: options.model,
      system: FACTUAL_CHALLENGE_SYSTEM_PROMPT,
      messages: [
        {
          role: "user",
          content: JSON.stringify({
            text,
            recentHistory,
          }),
        },
      ],
      tools: [FACTUAL_CHALLENGE_TOOL],
      tool_choice: { type: "tool", name: FACTUAL_CHALLENGE_TOOL_NAME },
      max_tokens: 512,
      budget: "perception-factual-challenge",
    });
    const call = response.tool_calls.find(
      (toolCall) => toolCall.name === FACTUAL_CHALLENGE_TOOL_NAME,
    );

    if (call === undefined) {
      throw new CognitionError(
        `Factual challenge detector did not emit tool ${FACTUAL_CHALLENGE_TOOL_NAME}`,
        {
          code: "FACTUAL_CHALLENGE_DETECTOR_INVALID",
        },
      );
    }

    const parsed = factualChallengeJudgeSchema.safeParse(call.input);

    if (!parsed.success) {
      throw new CognitionError("Factual challenge detector returned invalid payload", {
        cause: parsed.error,
        code: "FACTUAL_CHALLENGE_DETECTOR_INVALID",
      });
    }

    return parsed.data.factual_challenge;
  } catch (error) {
    await options.onDegraded?.("llm_failed", error);
    return null;
  }
}
