import type {
  LLMCompleteResult,
  LLMMessage,
  LLMToolDefinition,
  LLMTransportRetryEvent,
} from "../llm/index.js";
import type { SessionId } from "../util/ids.js";
import type { JsonValue } from "../util/json-value.js";
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

export function summarizeToolResponseShape(response: LLMCompleteResult): JsonValue {
  return {
    textLength: response.text.length,
    toolUseBlocks: response.tool_calls.map((call) => ({
      id: call.id,
      name: call.name,
    })),
  };
}

export function traceLlmCallStarted(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  label: string;
  model: string;
  systemPrompt: string;
  messages: readonly LLMMessage[];
  tools?: readonly LLMToolDefinition[];
  extra?: TracePayloadExtension;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call.started", {
      ...options.extra,
      turnId: options.turnId,
      ...(options.sessionId === undefined ? {} : { session_id: options.sessionId }),
      label: options.label,
      model: options.model,
      promptCharCount: countCompletePromptChars(options.systemPrompt, options.messages),
      ...(options.tools === undefined ? {} : { toolSchemas: summarizeToolSchemas(options.tools) }),
    });
  }
}

// Builds the per-call onTransportRetry hook that surfaces in-place transport
// retries (stalled stream rescued non-streaming, connection blips) as
// llm_call.retried trace events. Returns undefined when tracing is off so
// callers can omit the option entirely.
export function traceLlmCallRetryHook(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  label: string;
}): ((event: LLMTransportRetryEvent) => void) | undefined {
  const { tracer, turnId } = options;

  if (tracer?.enabled !== true || turnId === undefined) {
    return undefined;
  }

  return (event) => {
    tracer.emit("llm_call.retried", {
      turnId,
      ...(options.sessionId === undefined ? {} : { session_id: options.sessionId }),
      label: options.label,
      attempt: event.attempt,
      kind: event.kind,
      ...(event.code === undefined ? {} : { code: event.code }),
      retry_transport: event.retry_transport,
    });
  };
}

export type LlmSchemaRepairTraceStatus = "attempted" | "succeeded" | "failed";

export function traceLlmSchemaRepair(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  label: string;
  status: LlmSchemaRepairTraceStatus;
  attempt: number;
  repairOfAttempt: number;
  failureKind?: string;
  error?: unknown;
}): void {
  if (options.tracer?.enabled !== true || options.turnId === undefined) {
    return;
  }

  const event =
    options.status === "attempted"
      ? "llm_call.schema_repair.attempted"
      : options.status === "succeeded"
        ? "llm_call.schema_repair.succeeded"
        : "llm_call.schema_repair.failed";

  options.tracer.emit(event, {
    turnId: options.turnId,
    ...(options.sessionId === undefined ? {} : { session_id: options.sessionId }),
    label: options.label,
    attempt: options.attempt,
    repair_of_attempt: options.repairOfAttempt,
    ...(options.failureKind === undefined ? {} : { failure_kind: options.failureKind }),
    ...(options.error === undefined
      ? {}
      : { error: options.error instanceof Error ? options.error.message : String(options.error) }),
  });
}

export function traceLlmCallResponse(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  label: string;
  response: LLMCompleteResult;
  responseShape: JsonValue;
  extra?: TracePayloadExtension;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call.completed", {
      ...options.extra,
      turnId: options.turnId,
      ...(options.sessionId === undefined ? {} : { session_id: options.sessionId }),
      label: options.label,
      responseShape: options.responseShape,
      stopReason: options.response.stop_reason,
      usage: buildUsageTraceBlock(options.response),
    });
  }
}

export function traceLlmCallError(options: {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  label: string;
  error: unknown;
  extra?: TracePayloadExtension;
}): void {
  if (options.tracer?.enabled === true && options.turnId !== undefined) {
    options.tracer.emit("llm_call.completed", {
      ...options.extra,
      turnId: options.turnId,
      ...(options.sessionId === undefined ? {} : { session_id: options.sessionId }),
      label: options.label,
      responseShape: {
        error: options.error instanceof Error ? options.error.message : String(options.error),
      },
      stopReason: null,
      usage: null,
    });
  }
}
