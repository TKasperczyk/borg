import { z } from "zod";

import {
  type LLMClient,
  type LLMCompleteResult,
  type LLMToolDefinition,
  toToolInputSchema,
} from "../../llm/index.js";

const stopCommitmentSchema = z.object({
  classification: z.enum(["stop_until_substantive_content", "none"]),
  reason: z.string().min(1),
  confidence: z.number().min(0).max(1),
});

const STOP_COMMITMENT_TOOL_NAME = "EmitStopCommitmentClassification";
const STOP_COMMITMENT_TOOL = {
  name: STOP_COMMITMENT_TOOL_NAME,
  description:
    "Classify whether an assistant response commits to emit no assistant messages until the user provides substantive new content.",
  inputSchema: toToolInputSchema(stopCommitmentSchema),
} satisfies LLMToolDefinition;

export type StopCommitmentExtractorDegradedReason = "llm_unavailable" | "llm_failed";

export type StopCommitmentExtractorOptions = {
  llmClient?: LLMClient;
  model?: string;
  onDegraded?: (
    reason: StopCommitmentExtractorDegradedReason,
    error?: unknown,
  ) => Promise<void> | void;
};

export type ExtractStopCommitmentInput = {
  userMessage: string;
  agentResponse: string;
};

export type StopCommitmentExtraction = {
  reason: string;
  confidence: number;
};

function parseResponse(result: LLMCompleteResult): StopCommitmentExtraction | null {
  const call = result.tool_calls.find((toolCall) => toolCall.name === STOP_COMMITMENT_TOOL_NAME);

  if (call === undefined) {
    throw new Error(`Stop commitment extractor did not emit ${STOP_COMMITMENT_TOOL_NAME}`);
  }

  const parsed = stopCommitmentSchema.parse(call.input);

  if (parsed.classification !== "stop_until_substantive_content") {
    return null;
  }

  return {
    reason: parsed.reason,
    confidence: parsed.confidence,
  };
}

export class StopCommitmentExtractor {
  constructor(private readonly options: StopCommitmentExtractorOptions = {}) {}

  private async degraded(
    reason: StopCommitmentExtractorDegradedReason,
    error?: unknown,
  ): Promise<null> {
    try {
      await this.options.onDegraded?.(reason, error);
    } catch {
      // Best-effort degraded-mode logging only.
    }

    return null;
  }

  async extract(input: ExtractStopCommitmentInput): Promise<StopCommitmentExtraction | null> {
    if (this.options.llmClient === undefined || this.options.model === undefined) {
      return this.degraded("llm_unavailable");
    }

    try {
      return parseResponse(
        await this.options.llmClient.complete({
          model: this.options.model,
          system: [
            "Classify whether the assistant response is an operational commitment to stop emitting assistant messages until the user provides substantive new content.",
            "Return stop_until_substantive_content only for direct, future-facing commitments to emit no assistant messages until substantive user content appears.",
            "Return none for local style, topic, or explanation-boundary commitments that do not imply future no-output behavior.",
          ].join("\n"),
          messages: [
            {
              role: "user",
              content: JSON.stringify({
                user_message: input.userMessage,
                agent_response: input.agentResponse,
              }),
            },
          ],
          tools: [STOP_COMMITMENT_TOOL],
          tool_choice: { type: "tool", name: STOP_COMMITMENT_TOOL_NAME },
          max_tokens: 512,
          budget: "generation-stop-commitment",
        }),
      );
    } catch (error) {
      return this.degraded("llm_failed", error);
    }
  }
}
