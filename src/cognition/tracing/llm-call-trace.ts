import type { LLMCompleteResult, LLMMessage, LLMToolDefinition } from "../../llm/index.js";
import type { JsonValue } from "../../util/json-value.js";
import { buildUsageTraceBlock, type TurnTraceData, type TurnTracer } from "./tracer.js";

type TracePayloadExtension = Omit<TurnTraceData, "turnId">;

export function countCompletePromptChars(
  systemPrompt: string,
  messages: readonly LLMMessage[],
): number {
  return (
    systemPrompt.length +
    messages.reduce((sum, message) => sum + message.role.length + message.content.length, 0)
  );
}

export function summarizeToolSchemas(tools: readonly LLMToolDefinition[]): JsonValue {
  return tools.map((tool) => ({
    name: tool.name,
    propertyCount:
      tool.inputSchema.properties === undefined
        ? 0
        : Object.keys(tool.inputSchema.properties).length,
    required: Array.isArray(tool.inputSchema.required) ? tool.inputSchema.required.map(String) : [],
  }));
}

export function traceLlmCallStarted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  label: string;
  model: string;
  systemPrompt: string;
  messages: readonly LLMMessage[];
  tools?: readonly LLMToolDefinition[];
  extra?: TracePayloadExtension;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call.started", {
      turnId: options.turnId,
      label: options.label,
      model: options.model,
      promptCharCount: countCompletePromptChars(options.systemPrompt, options.messages),
      ...(options.tools === undefined ? {} : { toolSchemas: summarizeToolSchemas(options.tools) }),
      ...options.extra,
    });
  }
}

export function traceLlmCallResponse(options: {
  tracer?: TurnTracer;
  turnId?: string;
  label: string;
  response: LLMCompleteResult;
  responseShape: JsonValue;
  extra?: TracePayloadExtension;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call.completed", {
      turnId: options.turnId,
      label: options.label,
      responseShape: options.responseShape,
      stopReason: options.response.stop_reason,
      usage: buildUsageTraceBlock(options.response),
      ...options.extra,
    });
  }
}

export function traceLlmCallError(options: {
  tracer?: TurnTracer;
  turnId?: string;
  label: string;
  error: unknown;
  extra?: TracePayloadExtension;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call.completed", {
      turnId: options.turnId,
      label: options.label,
      responseShape: {
        error: options.error instanceof Error ? options.error.message : String(options.error),
      },
      stopReason: null,
      usage: null,
      ...options.extra,
    });
  }
}
