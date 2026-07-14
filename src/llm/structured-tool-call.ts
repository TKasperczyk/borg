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
  traceLlmSchemaRepair,
  traceLlmCallStarted,
} from "../tracing/llm-call-trace.js";
import type { TurnTraceData, TurnTracer } from "../tracing/tracer.js";
import { BudgetExceededError, LLMError } from "../util/errors.js";
import type { SessionId } from "../util/ids.js";
import type { JsonValue } from "../util/json-value.js";
import { parseErrorMessage } from "../util/zod-errors.js";

const MAX_REPAIR_ARGUMENT_CHARS = 4_000;
const MAX_REPAIR_ISSUE_CHARS = 1_500;
const MAX_REPAIR_ISSUES = 5;

export type StructuredToolCallErrorKind = "llm_failed" | "missing_tool_call" | "invalid_payload";

export type StructuredToolCallErrorOptions = {
  kind: StructuredToolCallErrorKind;
  toolName: string;
  acceptedToolNames: readonly string[];
  stopReason: string | null;
  cause?: unknown;
  repairFailure?: unknown;
  attemptCount?: number;
  usage?: StructuredToolCallUsage;
};

export type StructuredToolCallUsage = Pick<
  LLMCompleteResult,
  "input_tokens" | "output_tokens" | "cache_creation_input_tokens" | "cache_read_input_tokens"
>;

export class StructuredToolCallError extends LLMError {
  readonly kind: StructuredToolCallErrorKind;
  readonly toolName: string;
  readonly acceptedToolNames: readonly string[];
  readonly stopReason: string | null;
  readonly repairFailure: unknown;
  readonly attemptCount: number;
  readonly usage: StructuredToolCallUsage;

  constructor(message: string, options: StructuredToolCallErrorOptions) {
    super(message, {
      cause: options.cause,
      code: "LLM_STRUCTURED_TOOL_CALL_FAILED",
    });
    this.kind = options.kind;
    this.toolName = options.toolName;
    this.acceptedToolNames = options.acceptedToolNames;
    this.stopReason = options.stopReason;
    this.repairFailure = options.repairFailure;
    this.attemptCount = options.attemptCount ?? 0;
    this.usage = options.usage ?? { input_tokens: 0, output_tokens: 0 };
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
  maxAttempts?: 1 | 2;
  parse: (
    input: unknown,
    context: {
      toolCall: LLMToolCall;
      response: LLMCompleteResult;
    },
  ) => T;
  onSchemaRepair?: (event: StructuredToolSchemaRepairEvent) => void;
  trace?: StructuredToolCallTraceOptions;
};

export type StructuredToolSchemaRepairEvent =
  | { status: "attempted"; attempt: 2; error: unknown }
  | { status: "succeeded"; attempt: 2 }
  | { status: "failed"; attempt: 2; error: unknown };

export type CallStructuredToolResult<T> = {
  response: LLMCompleteResult;
  toolCall: LLMToolCall;
  parsed: T;
  attemptCount: number;
  usage: StructuredToolCallUsage;
};

function emptyUsage(): StructuredToolCallUsage {
  return { input_tokens: 0, output_tokens: 0 };
}

function addResponseUsage(
  usage: StructuredToolCallUsage,
  response: LLMCompleteResult,
): StructuredToolCallUsage {
  return {
    input_tokens: usage.input_tokens + response.input_tokens,
    output_tokens: usage.output_tokens + response.output_tokens,
    ...(usage.cache_creation_input_tokens === undefined &&
    response.cache_creation_input_tokens === undefined
      ? {}
      : {
          cache_creation_input_tokens:
            (usage.cache_creation_input_tokens ?? 0) + (response.cache_creation_input_tokens ?? 0),
        }),
    ...(usage.cache_read_input_tokens === undefined &&
    response.cache_read_input_tokens === undefined
      ? {}
      : {
          cache_read_input_tokens:
            (usage.cache_read_input_tokens ?? 0) + (response.cache_read_input_tokens ?? 0),
        }),
  };
}

function acceptedToolNames<T>(options: CallStructuredToolOptions<T>): readonly string[] {
  return options.acceptedToolNames ?? [options.toolName];
}

function findAcceptedToolCall(
  response: LLMCompleteResult,
  names: readonly string[],
): LLMToolCall | undefined {
  return response.tool_calls.find((call) => names.includes(call.name));
}

function serializeInvalidToolArguments(input: unknown): {
  text: string;
  truncated: boolean;
} {
  let serialized: string;

  try {
    serialized = JSON.stringify(input) ?? String(input);
  } catch {
    serialized = String(input);
  }

  if (serialized.length <= MAX_REPAIR_ARGUMENT_CHARS) {
    return { text: serialized, truncated: false };
  }

  return {
    text: `${serialized.slice(0, MAX_REPAIR_ARGUMENT_CHARS)}…`,
    truncated: true,
  };
}

function validationRepairRequest<T>(
  options: CallStructuredToolOptions<T>,
  request: LLMCompleteOptions,
  toolCall: LLMToolCall,
  error: unknown,
): LLMCompleteOptions {
  const invalidArguments = serializeInvalidToolArguments(toolCall.input);
  const validationIssues = parseErrorMessage(error, {
    maxIssues: MAX_REPAIR_ISSUES,
    maxCharacters: MAX_REPAIR_ISSUE_CHARS,
  });

  return {
    ...request,
    messages: [
      ...request.messages,
      {
        role: "user",
        content: [
          `Your previous ${toolCall.name} tool arguments failed client-side schema validation.`,
          `Validation issues (paths and messages): ${validationIssues}`,
          `Previous invalid arguments${invalidArguments.truncated ? " (truncated)" : ""}: ${invalidArguments.text}`,
          `Re-emit exactly one corrected ${options.toolName} tool call that satisfies the supplied schema.`,
        ].join("\n\n"),
      },
    ],
  };
}

function attemptTraceExtra(
  extra: TracePayloadExtension | undefined,
  attempt: number,
  schemaRepair: boolean,
): TracePayloadExtension {
  return {
    ...extra,
    attempt,
    schema_repair: schemaRepair,
    ...(schemaRepair ? { repair_of_attempt: 1 } : {}),
  };
}

function traceStarted<T>(
  options: CallStructuredToolOptions<T>,
  request: LLMCompleteOptions,
  attempt: number,
  schemaRepair: boolean,
): void {
  const trace = options.trace;

  if (trace === undefined) {
    return;
  }

  traceLlmCallStarted({
    tracer: trace.tracer,
    turnId: trace.turnId,
    sessionId: trace.sessionId,
    label: trace.label,
    model: request.model,
    systemPrompt: trace.systemPrompt ?? (typeof request.system === "string" ? request.system : ""),
    messages: schemaRepair ? request.messages : (trace.messages ?? request.messages),
    ...(trace.includeToolSchemas === false ? {} : { tools: trace.tools ?? request.tools }),
    extra: attemptTraceExtra(trace.startedExtra, attempt, schemaRepair),
  });
}

function traceResponse<T>(
  options: CallStructuredToolOptions<T>,
  response: LLMCompleteResult,
  attempt: number,
  schemaRepair: boolean,
): void {
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
    extra: attemptTraceExtra(trace.responseExtra, attempt, schemaRepair),
  });
}

function traceError<T>(
  options: CallStructuredToolOptions<T>,
  error: unknown,
  attempt: number,
  schemaRepair: boolean,
): void {
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
    extra: attemptTraceExtra(trace.errorExtra, attempt, schemaRepair),
  });
}

function repairFailureKind(error: unknown): string {
  if (isStructuredToolCallError(error)) {
    return error.kind;
  }

  return error instanceof BudgetExceededError ? "budget_exceeded" : "invalid_payload";
}

function notifySchemaRepair<T>(
  options: CallStructuredToolOptions<T>,
  event: StructuredToolSchemaRepairEvent,
): void {
  options.onSchemaRepair?.(event);
  traceLlmSchemaRepair({
    tracer: options.trace?.tracer,
    turnId: options.trace?.turnId,
    sessionId: options.trace?.sessionId,
    label: options.trace?.label ?? options.toolName,
    status: event.status,
    attempt: event.attempt,
    repairOfAttempt: 1,
    ...(event.status !== "failed"
      ? {}
      : {
          failureKind: repairFailureKind(event.error),
          error: event.error,
        }),
  });
}

export async function callStructuredTool<T>(
  options: CallStructuredToolOptions<T>,
): Promise<CallStructuredToolResult<T>> {
  const names = acceptedToolNames(options);
  const maxAttempts = options.maxAttempts ?? 2;
  let attemptCount = 0;
  let usage = emptyUsage();

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

  const errorAccounting = () => ({ attemptCount, usage: { ...usage } });
  const complete = async (
    nextRequest: LLMCompleteOptions,
    schemaRepair: boolean,
  ): Promise<LLMCompleteResult> => {
    attemptCount += 1;
    traceStarted(options, nextRequest, attemptCount, schemaRepair);

    let response: LLMCompleteResult;

    try {
      response = await options.llmClient.complete(nextRequest);
    } catch (error) {
      traceError(options, error, attemptCount, schemaRepair);

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
          ...errorAccounting(),
        },
      );
    }

    usage = addResponseUsage(usage, response);
    traceResponse(options, response, attemptCount, schemaRepair);
    return response;
  };

  const acceptedCall = (response: LLMCompleteResult): LLMToolCall => {
    const toolCall = findAcceptedToolCall(response, names);

    if (toolCall === undefined) {
      throw new StructuredToolCallError(
        `Structured tool call ${options.toolName} was missing from the LLM response`,
        {
          kind: "missing_tool_call",
          toolName: options.toolName,
          acceptedToolNames: names,
          stopReason: response.stop_reason,
          ...errorAccounting(),
        },
      );
    }

    return toolCall;
  };

  let response = await complete(request, false);
  let toolCall = acceptedCall(response);

  try {
    const parsed = options.parse(toolCall.input, { toolCall, response });

    return {
      response,
      toolCall,
      parsed,
      ...errorAccounting(),
    };
  } catch (error) {
    const initialResponse = response;
    const initialToolCall = toolCall;
    const originalInvalidPayload = (repairFailure?: unknown) =>
      new StructuredToolCallError(
        `Structured tool call ${initialToolCall.name} returned an invalid payload`,
        {
          kind: "invalid_payload",
          toolName: initialToolCall.name,
          acceptedToolNames: names,
          stopReason: initialResponse.stop_reason,
          cause: error,
          repairFailure,
          ...errorAccounting(),
        },
      );

    if (maxAttempts < 2) {
      throw originalInvalidPayload();
    }

    const repairRequest = validationRepairRequest(options, request, toolCall, error);
    notifySchemaRepair(options, { status: "attempted", attempt: 2, error });

    try {
      response = await complete(repairRequest, true);
      toolCall = acceptedCall(response);
      const parsed = options.parse(toolCall.input, { toolCall, response });
      notifySchemaRepair(options, { status: "succeeded", attempt: 2 });

      return {
        response,
        toolCall,
        parsed,
        ...errorAccounting(),
      };
    } catch (repairError) {
      notifySchemaRepair(options, { status: "failed", attempt: 2, error: repairError });

      if (repairError instanceof BudgetExceededError) {
        throw repairError;
      }

      throw originalInvalidPayload(repairError);
    }
  }
}
