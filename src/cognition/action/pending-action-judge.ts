import { z } from "zod";

import {
  type LLMClient,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";
import type { IntentRecord } from "../types.js";

const pendingActionJudgmentSchema = z.object({
  classification: z.enum(["action", "non_action"]),
  reason: z.string().min(1),
  confidence: z.number().min(0).max(1),
});

const PENDING_ACTION_JUDGE_TOOL_NAME = "ClassifyPendingAction";
const PENDING_ACTION_JUDGE_TOOL = {
  name: PENDING_ACTION_JUDGE_TOOL_NAME,
  description:
    "Classify whether a planner follow-up item is an unresolved operational action suitable for working memory.",
  inputSchema: toToolInputSchema(pendingActionJudgmentSchema),
} satisfies LLMToolDefinition;

export type PendingActionJudgment = {
  accepted: boolean;
  reason: string;
  confidence: number;
  degraded: boolean;
};

export type PendingActionJudge = {
  judge(record: IntentRecord): Promise<PendingActionJudgment>;
};

export type LLMPendingActionJudgeOptions = {
  llmClient?: LLMClient;
  model?: string;
};

function rejectDegraded(reason: string): PendingActionJudgment {
  return {
    accepted: false,
    reason,
    confidence: 0,
    degraded: true,
  };
}

export class LLMPendingActionJudge implements PendingActionJudge {
  constructor(private readonly options: LLMPendingActionJudgeOptions) {}

  async judge(record: IntentRecord): Promise<PendingActionJudgment> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return rejectDegraded("pending_action_judge_unavailable");
    }

    try {
      const result = await this.options.llmClient.complete({
        model: this.options.model,
        system: [
          "You classify Borg planner follow-up items before they enter working memory.",
          "Accept only unresolved operational follow-ups: concrete future tasks, reminders, checks, questions to ask, or actions to perform.",
          "Reject factual claims, identity claims, relationship claims, biographical updates, interpretations of the user, and memory facts.",
          "When uncertain, reject. Return only the required tool call.",
        ].join("\n"),
        messages: [
          {
            role: "user",
            content: JSON.stringify({
              description: record.description,
              next_action: record.next_action,
            }),
          },
        ],
        tools: [PENDING_ACTION_JUDGE_TOOL],
        tool_choice: { type: "tool", name: PENDING_ACTION_JUDGE_TOOL_NAME },
        max_tokens: 256,
        budget: "pending-action-judge",
      });
      const call = result.tool_calls.find(
        (toolCall) => toolCall.name === PENDING_ACTION_JUDGE_TOOL_NAME,
      );

      if (call === undefined) {
        return rejectDegraded("pending_action_judge_missing_tool_call");
      }

      const parsed = pendingActionJudgmentSchema.safeParse(call.input);

      if (!parsed.success) {
        return rejectDegraded("pending_action_judge_invalid_payload");
      }

      return {
        accepted: parsed.data.classification === "action" && parsed.data.confidence >= 0.5,
        reason: parsed.data.reason,
        confidence: parsed.data.confidence,
        degraded: false,
      };
    } catch (error) {
      return rejectDegraded(error instanceof Error ? error.message : String(error));
    }
  }
}
