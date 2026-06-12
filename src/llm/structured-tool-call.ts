import type {
  LLMClient,
  LLMCompleteOptions,
  LLMCompleteResult,
  LLMToolCall,
  LLMToolDefinition,
} from "./index.js";
import {
  summarizeToolResponseShape,
  traceLlmCallError,
  traceLlmCallResponse,
  traceLlmCallRetryHook,
  traceLlmCallStarted,
} from "../tracing/llm-call-trace.js";
import type { TurnTraceData, TurnTracer } from "../tracing/tracer.js";
import { BudgetExceededError, LLMError } from "../util/errors.js";
import type { SessionId } from "../util/ids.js";
import type { JsonValue } from "../util/json-value.js";

export type StructuredToolCallErrorKind =
  | "llm_failed"
  | "missing_tool_call"
  | "invalid_payload";

export type StructuredToolCallErrorOptions = {
  kind: StructuredToolCallErrorKind;
  toolName: string;
  acceptedToolNames: readonly string[];
  stopReason: string | null;
  cause?: unknown;
};

export class StructuredToolCallError extends LLMError {
  readonly kind: StructuredToolCallErrorKind;
  readonly toolName: string;
  readonly acceptedToolNames: readonly string[];
  readonly stopReason: string | null;

  constructor(message: string, options: StructuredToolCallErrorOptions) {
    super(message, {
      cause: options.cause,
      code: "LLM_STRUCTURED_TOOL_CALL_FAILED",
    });
    this.kind = options.kind;
    this.toolName = options.toolName;
    this.acceptedToolNames = options.acceptedToolNames;
    this.stopReason = options.stopReason;
  }
}

export function isStructuredToolCallError(
  error: unknown,
  kind?: StructuredToolCallErrorKind,
): error is StructuredToolCallError {
  return error instanceof StructuredToolCallError && (kind === undefined || error.kind === kind);
}

type TracePayloadExtension = Omit<TurnTraceData, "turnId">;

export type StructuredToolCallTraceOptions = {
  tracer?: TurnTracer;
  turnId?: string;
  sessionId?: SessionId;
  label: string;
  systemPrompt?: string;
  messages?: LLMCompleteOptions["messages"];
  tools?: readonly LLMToolDefinition[];
  includeToolSchemas?: boolean;
  responseShape?: (response: LLMCompleteResult) => JsonValue;
  startedExtra?: TracePayloadExtension;
  responseExtra?: TracePayloadExtension;
  errorExtra?: TracePayloadExtension;
};

export type CallStructuredToolOptions<T> = {
  llmClient: LLMClient;
  request: LLMCompleteOptions;
  toolName: string;
  acceptedToolNames?: readonly string[];
  parse: (
    input: unknown,
    context: {
      toolCall: LLMToolCall;
      response: LLMCompleteResult;
    },
  ) => T;
  trace?: StructuredToolCallTraceOptions;
};

export type CallStructuredToolResult<T> = {
  response: LLMCompleteResult;
  toolCall: LLMToolCall;
  parsed: T;
};

function acceptedToolNames<T>(options: CallStructuredToolOptions<T>): readonly string[] {
  return options.acceptedToolNames ?? [options.toolName];
}

function findAcceptedToolCall(
  response: LLMCompleteResult,
  names: readonly string[],
): LLMToolCall | undefined {
  return response.tool_calls.find((call) => names.includes(call.name));
}

function traceStarted<T>(options: CallStructuredToolOptions<T>): void {
  const trace = options.trace;

  if (trace === undefined) {
    return;
  }

  traceLlmCallStarted({
    tracer: trace.tracer,
    turnId: trace.turnId,
    sessionId: trace.sessionId,
    label: trace.label,
    model: options.request.model,
    systemPrompt:
      trace.systemPrompt ??
      (typeof options.request.system === "string" ? options.request.system : ""),
    messages: trace.messages ?? options.request.messages,
    ...(trace.includeToolSchemas === false
      ? {}
      : { tools: trace.tools ?? options.request.tools }),
    extra: trace.startedExtra,
  });
}

function traceResponse<T>(options: CallStructuredToolOptions<T>, response: LLMCompleteResult): void {
  const trace = options.trace;

  if (trace === undefined) {
    return;
  }

  traceLlmCallResponse({
    tracer: trace.tracer,
    turnId: trace.turnId,
    sessionId: trace.sessionId,
    label: trace.label,
    response,
    responseShape: trace.responseShape?.(response) ?? summarizeToolResponseShape(response),
    extra: trace.responseExtra,
  });
}

function traceError<T>(options: CallStructuredToolOptions<T>, error: unknown): void {
  const trace = options.trace;

  if (trace === undefined) {
    return;
  }

  traceLlmCallError({
    tracer: trace.tracer,
    turnId: trace.turnId,
    sessionId: trace.sessionId,
    label: trace.label,
    error,
    extra: trace.errorExtra,
  });
}

export async function callStructuredTool<T>(
  options: CallStructuredToolOptions<T>,
): Promise<CallStructuredToolResult<T>> {
  const names = acceptedToolNames(options);
  traceStarted(options);

  const onTransportRetry =
    options.trace === undefined
      ? undefined
      : traceLlmCallRetryHook({
          tracer: options.trace.tracer,
          turnId: options.trace.turnId,
          sessionId: options.trace.sessionId,
          label: options.trace.label,
        });
  const request =
    onTransportRetry === undefined ? options.request : { ...options.request, onTransportRetry };

  let response: LLMCompleteResult;

  try {
    response = await options.llmClient.complete(request);
  } catch (error) {
    traceError(options, error);

    if (error instanceof BudgetExceededError) {
      throw error;
    }

    throw new StructuredToolCallError(
      `Structured tool call ${options.toolName} failed before a response was available`,
      {
        kind: "llm_failed",
        toolName: options.toolName,
        acceptedToolNames: names,
        stopReason: null,
        cause: error,
      },
    );
  }

  traceResponse(options, response);

  const toolCall = findAcceptedToolCall(response, names);

  if (toolCall === undefined) {
    throw new StructuredToolCallError(
      `Structured tool call ${options.toolName} was missing from the LLM response`,
      {
        kind: "missing_tool_call",
        toolName: options.toolName,
        acceptedToolNames: names,
        stopReason: response.stop_reason,
      },
    );
  }

  let parsed: T;

  try {
    parsed = options.parse(toolCall.input, { toolCall, response });
  } catch (error) {
    throw new StructuredToolCallError(
      `Structured tool call ${toolCall.name} returned an invalid payload`,
      {
        kind: "invalid_payload",
        toolName: toolCall.name,
        acceptedToolNames: names,
        stopReason: response.stop_reason,
        cause: error,
      },
    );
  }

  return {
    response,
    toolCall,
    parsed,
  };
}
