import type { LLMCompleteResult, LLMToolCall } from "../../llm/index.js";

export function makeToolUseCompleteResult(input: {
  toolName: string;
  toolInput: unknown;
  toolId?: string;
  text?: string;
  inputTokens?: number;
  outputTokens?: number;
  stopReason?: string | null;
  extraToolCalls?: readonly LLMToolCall[];
}): LLMCompleteResult {
  return {
    text: input.text ?? "",
    input_tokens: input.inputTokens ?? 4,
    output_tokens: input.outputTokens ?? 2,
    stop_reason: input.stopReason ?? "tool_use",
    tool_calls: [
      {
        id: input.toolId ?? `toolu_${input.toolName}`,
        name: input.toolName,
        input: input.toolInput,
      },
      ...(input.extraToolCalls ?? []),
    ],
  };
}
