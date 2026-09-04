import type {
  LLMClient,
  LLMContentBlock,
  LLMContentBlockMessage,
  LLMConverseOptions,
  LLMConverseResult,
  LLMStreamTextHandler,
  LLMToolDefinition,
  LLMToolUseBlock,
} from "../../llm/index.js";
import { toAnthropicToolDefinitions } from "../../tools/anthropic.js";
import type {
  ToolDefinition,
  ToolDispatchResult,
  ToolDispatcher,
  ToolOrigin,
} from "../../tools/dispatcher.js";
import type { BorgRole } from "../../memory/commitments/index.js";
import type { SessionAudienceRole } from "../../sessions/index.js";
import type { TurnTracer } from "../../tracing/tracer.js";
import type { TurnOrigin } from "../types.js";
import { DEFAULT_DELIBERATION_PLAN_CALL_TIMEOUT_MS } from "../deliberation/constants.js";
import { buildUsageTraceBlock, toTraceJsonValue } from "../../tracing/tracer.js";
import { summarizeToolSchemas, traceLlmCallRetryHook } from "../../tracing/llm-call-trace.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { LLMError } from "../../util/errors.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { JsonValue } from "../../util/json-value.js";
import { serializeJsonValue } from "../../util/json-value.js";

const DEFAULT_MAX_ITERATIONS = 5;
const DEFAULT_MAX_TOOL_CALLS_PER_ITERATION = 3;
const AUTONOMOUS_MAX_ITERATIONS = 8;
const AUTONOMOUS_MAX_TOOL_CALLS_PER_ITERATION = 5;
export const AUTONOMOUS_TOOL_LOOP_WALL_CLOCK_BUDGET_MS =
  2 * DEFAULT_DELIBERATION_PLAN_CALL_TIMEOUT_MS;
export const AUTONOMOUS_TOOL_LOOP_TOOL_ROUND_BUDGET_MS =
  AUTONOMOUS_TOOL_LOOP_WALL_CLOCK_BUDGET_MS - DEFAULT_DELIBERATION_PLAN_CALL_TIMEOUT_MS;

export type ToolLoopUsage = {
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
  stop_reason: string | null;
};

export type ToolLoopCallRecord = {
  callId: string;
  name: string;
  input: unknown;
  output?: unknown;
  ok: boolean;
  durationMs: number;
};

export type ToolLoopResultObservation = {
  ordinal: number;
  iteration: number;
  batchPosition: number;
  callId: string;
  toolName: string;
  rawArguments: unknown;
  disposition: "dispatched" | "skipped_unavailable" | "skipped_iteration_cap";
  result: { ok: true; output: unknown } | { ok: false; error: string };
  durationMs: number;
};

export type ToolLoopResultObserver = {
  observe(observation: ToolLoopResultObservation): void;
  markIncomplete(error: unknown): void;
};

export type ExecuteToolLoopOptions = {
  llmClient: LLMClient;
  dispatcher: ToolDispatcher;
  sessionId: SessionId;
  model: string;
  systemPrompt?: LLMConverseOptions["system"];
  initialMessages: readonly LLMContentBlockMessage[];
  tools: readonly ToolDefinition[];
  origin: ToolOrigin;
  turnOrigin?: TurnOrigin;
  audienceEntityId?: EntityId | null;
  currentSenderBorgRole?: BorgRole | null;
  sessionAudienceRole?: SessionAudienceRole;
  provenance?: unknown;
  budget: string;
  maxTokens?: number;
  temperature?: number;
  thinking?: LLMConverseOptions["thinking"];
  effort?: LLMConverseOptions["effort"];
  suppressRawTextStream?: LLMConverseOptions["suppressRawTextStream"];
  toolChoice?: LLMConverseOptions["tool_choice"];
  maxIterations?: number;
  maxToolCallsPerIteration?: number;
  clock?: Pick<Clock, "now">;
  terminalToolNames?: readonly string[];
  /** Advertised schemas that are structurally unavailable on this turn. */
  unavailableToolNames?: readonly string[];
  stream?: boolean;
  onTextDelta?: LLMStreamTextHandler;
  tracer?: TurnTracer;
  turnId?: string;
  traceLabel?: string;
  /** Exact transport request snapshot, synchronously exposed before each LLM attempt. */
  onRequestPrepared?: (request: LLMConverseOptions, attempt: number) => void;
  /** Best-effort capture hook. Observer failures never alter the live tool loop. */
  toolResultObserver?: ToolLoopResultObserver;
};

export type ToolLoopResult = {
  text: string;
  iterations: number;
  toolCallsMade: ToolLoopCallRecord[];
  terminalToolCalls: LLMToolUseBlock[];
  stopReason: "text" | "max_iterations" | "terminal_tool";
  usage: ToolLoopUsage;
};

function sumOptional(current: number | undefined, next: number | undefined): number | undefined {
  if (current === undefined && next === undefined) {
    return undefined;
  }

  return (current ?? 0) + (next ?? 0);
}

function aggregateUsage(current: ToolLoopUsage, next: ToolLoopUsage): ToolLoopUsage {
  const cacheCreation = sumOptional(
    current.cache_creation_input_tokens,
    next.cache_creation_input_tokens,
  );
  const cacheRead = sumOptional(current.cache_read_input_tokens, next.cache_read_input_tokens);

  return {
    input_tokens: current.input_tokens + next.input_tokens,
    output_tokens: current.output_tokens + next.output_tokens,
    stop_reason: next.stop_reason,
    ...(cacheCreation === undefined ? {} : { cache_creation_input_tokens: cacheCreation }),
    ...(cacheRead === undefined ? {} : { cache_read_input_tokens: cacheRead }),
  };
}

function cloneMessage(message: LLMContentBlockMessage): LLMContentBlockMessage {
  return {
    role: message.role,
    content: [...message.content],
  };
}

function isToolUseBlock(block: LLMContentBlock): block is LLMToolUseBlock {
  return block.type === "tool_use";
}

function extractText(blocks: readonly LLMContentBlock[]): string {
  return blocks
    .filter((block): block is Extract<LLMContentBlock, { type: "text" }> => block.type === "text")
    .map((block) => block.text)
    .join("");
}

function buildToolResultBlock(
  result: ToolDispatchResult,
): Extract<LLMContentBlock, { type: "tool_result" }> {
  if (result.ok) {
    return {
      type: "tool_result",
      tool_use_id: result.callId,
      content: serializeJsonValue(result.output),
    };
  }

  return {
    type: "tool_result",
    tool_use_id: result.callId,
    content: result.error,
    is_error: true,
  };
}

function buildDroppedToolResultBlock(
  block: LLMToolUseBlock,
  maxToolCallsPerIteration: number,
): Extract<LLMContentBlock, { type: "tool_result" }> {
  return {
    type: "tool_result",
    tool_use_id: block.id,
    content: `Skipped because this turn allows at most ${maxToolCallsPerIteration} tool calls per iteration.`,
    is_error: true,
  };
}

function buildUnavailableToolResultBlock(
  block: LLMToolUseBlock,
): Extract<LLMContentBlock, { type: "tool_result" }> {
  return {
    type: "tool_result",
    tool_use_id: block.id,
    content: `tool ${block.name} not available in this context`,
    is_error: true,
  };
}

function buildDeferredTerminalToolResultBlock(
  block: LLMToolUseBlock,
): Extract<LLMContentBlock, { type: "tool_result" }> {
  return {
    type: "tool_result",
    tool_use_id: block.id,
    content:
      "Terminal emission was not accepted because a sibling tool call was unavailable. Handle the tool error, then call exactly one enabled terminal emission tool again.",
    is_error: true,
  };
}

function toCallRecord(block: LLMToolUseBlock, result: ToolDispatchResult): ToolLoopCallRecord {
  return {
    callId: result.callId,
    name: result.toolName,
    input: block.input,
    ...(result.ok ? { output: result.output } : {}),
    ok: result.ok,
    durationMs: result.durationMs,
  };
}

function observeToolResult(
  observer: ToolLoopResultObserver | undefined,
  observation: ToolLoopResultObservation,
): void {
  if (observer === undefined) return;
  try {
    observer.observe(observation);
  } catch (error) {
    try {
      observer.markIncomplete(error);
    } catch {
      // Capture is observational. Even a broken failure callback must not
      // reach or alter the live tool loop.
    }
  }
}

async function dispatchToolUseBlock(
  dispatcher: ToolDispatcher,
  options: Pick<
    ExecuteToolLoopOptions,
    | "sessionId"
    | "origin"
    | "turnOrigin"
    | "turnId"
    | "audienceEntityId"
    | "currentSenderBorgRole"
    | "sessionAudienceRole"
    | "provenance"
  >,
  block: LLMToolUseBlock,
): Promise<ToolDispatchResult> {
  try {
    return await dispatcher.dispatch({
      callId: block.id,
      toolName: block.name,
      input: block.input,
      sessionId: options.sessionId,
      turnId: options.turnId,
      origin: options.origin,
      turnOrigin: options.turnOrigin,
      audienceEntityId: options.audienceEntityId,
      currentSenderBorgRole: options.currentSenderBorgRole,
      sessionAudienceRole: options.sessionAudienceRole,
      provenance: options.provenance,
    });
  } catch (error) {
    return {
      callId: block.id,
      toolName: block.name,
      ok: false,
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
      durationMs: 0,
    };
  }
}

/**
 * Execute a tool-use conversation loop for a normal turn.
 *
 * Tool calls within a single iteration run sequentially in the exact order
 * the model emitted them. That keeps dispatcher-written `tool_call` /
 * `tool_result` stream entries deterministic, and the returned `tool_result`
 * blocks stay aligned with the model's emission order. `writeScope` on tool
 * definitions remains metadata for future policy work, such as explicit user
 * confirmation before writes; current enforcement is the explicit tool list
 * passed into this loop plus the dispatcher's origin checks.
 */
export async function executeToolLoop(options: ExecuteToolLoopOptions): Promise<ToolLoopResult> {
  const maxIterations =
    options.maxIterations ??
    (options.origin === "autonomous" ? AUTONOMOUS_MAX_ITERATIONS : DEFAULT_MAX_ITERATIONS);
  const maxToolCallsPerIteration =
    options.maxToolCallsPerIteration ??
    (options.origin === "autonomous"
      ? AUTONOMOUS_MAX_TOOL_CALLS_PER_ITERATION
      : DEFAULT_MAX_TOOL_CALLS_PER_ITERATION);
  const wallClock = options.clock ?? new SystemClock();
  const loopStartedAt = wallClock.now();
  const autonomousToolRoundDeadline = loopStartedAt + AUTONOMOUS_TOOL_LOOP_TOOL_ROUND_BUDGET_MS;
  const autonomousLoopDeadline = loopStartedAt + AUTONOMOUS_TOOL_LOOP_WALL_CLOCK_BUDGET_MS;
  const messages = options.initialMessages.map((message) => cloneMessage(message));
  const anthropicTools = toAnthropicToolDefinitions(options.tools);
  const allowedToolNames = new Set(options.tools.map((tool) => tool.name));
  const unavailableToolNames = new Set(options.unavailableToolNames ?? []);
  const terminalToolNames = new Set(options.terminalToolNames ?? []);
  const toolCallsMade: ToolLoopCallRecord[] = [];
  let iterations = 0;
  let toolsEnabled = anthropicTools.length > 0;
  let forcedTextOnly = false;
  let lastResponseText = "";
  let usage: ToolLoopUsage = {
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: null,
  };
  const traceEnabled = options.tracer?.enabled === true && options.turnId !== undefined;
  const traceLabel = options.traceLabel ?? options.budget;
  const onTransportRetry = traceLlmCallRetryHook({
    tracer: options.tracer,
    turnId: options.turnId,
    sessionId: options.sessionId,
    label: traceLabel,
  });

  while (true) {
    let autonomousRequestDeadline: number | null = null;

    if (options.origin === "autonomous") {
      const nowMs = wallClock.now();

      if (toolsEnabled && nowMs >= autonomousToolRoundDeadline) {
        toolsEnabled = false;
        forcedTextOnly = true;
      }

      if (nowMs >= autonomousLoopDeadline) {
        return {
          text: lastResponseText,
          iterations,
          toolCallsMade,
          terminalToolCalls: [],
          stopReason: "max_iterations",
          usage,
        };
      }

      autonomousRequestDeadline = toolsEnabled
        ? autonomousToolRoundDeadline
        : autonomousLoopDeadline;
    }

    if (traceEnabled && options.turnId !== undefined) {
      options.tracer?.emit("llm_call.started", {
        turnId: options.turnId,
        session_id: options.sessionId,
        label: traceLabel,
        iteration: iterations + 1,
        model: options.model,
        promptCharCount: countConversePromptChars(options.systemPrompt, messages),
        toolSchemas: summarizeToolSchemas(anthropicTools),
        ...(options.tracer.includePayloads
          ? {
              prompt: toTraceJsonValue({
                system: options.systemPrompt ?? null,
                messages,
                tools: anthropicTools,
              }),
            }
          : {}),
      });
    }

    const converseOptions = {
      model: options.model,
      system: options.systemPrompt,
      messages,
      ...(toolsEnabled
        ? {
            tools: anthropicTools,
            ...(options.toolChoice === undefined ? {} : { tool_choice: options.toolChoice }),
          }
        : {}),
      max_tokens: options.maxTokens,
      ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
      ...(options.thinking === undefined ? {} : { thinking: options.thinking }),
      ...(options.effort === undefined ? {} : { effort: options.effort }),
      ...(options.suppressRawTextStream === undefined
        ? {}
        : { suppressRawTextStream: options.suppressRawTextStream }),
      ...(onTransportRetry === undefined ? {} : { onTransportRetry }),
      ...(autonomousRequestDeadline === null
        ? {}
        : {
            timeoutMs: Math.max(
              1,
              Math.min(
                DEFAULT_DELIBERATION_PLAN_CALL_TIMEOUT_MS,
                Math.floor(autonomousRequestDeadline - wallClock.now()),
              ),
            ),
          }),
      budget: options.budget,
    } satisfies LLMConverseOptions;
    options.onRequestPrepared?.(converseOptions, iterations + 1);
    let response: LLMConverseResult;

    try {
      response =
        options.stream === true && options.llmClient.streamConverse !== undefined
          ? await options.llmClient.streamConverse({
              ...converseOptions,
              ...(options.onTextDelta === undefined ? {} : { onTextDelta: options.onTextDelta }),
            })
          : await options.llmClient.converse(converseOptions);
    } catch (error) {
      const autonomousBudgetExpired =
        autonomousRequestDeadline !== null &&
        wallClock.now() >= autonomousRequestDeadline &&
        error instanceof LLMError &&
        error.code === "LLM_CALL_TIMED_OUT";

      if (!autonomousBudgetExpired) {
        throw error;
      }

      if (toolsEnabled) {
        toolsEnabled = false;
        forcedTextOnly = true;
        continue;
      }

      return {
        text: lastResponseText,
        iterations,
        toolCallsMade,
        terminalToolCalls: [],
        stopReason: "max_iterations",
        usage,
      };
    }
    usage = aggregateUsage(usage, response);
    lastResponseText = extractText(response.messageBlocks) || lastResponseText;

    const toolUseBlocks = response.messageBlocks.filter(isToolUseBlock);
    const terminalToolCalls = toolUseBlocks.filter(
      (block) =>
        allowedToolNames.has(block.name) &&
        !unavailableToolNames.has(block.name) &&
        terminalToolNames.has(block.name),
    );
    const unavailableToolCalls = toolUseBlocks.filter(
      (block) => !allowedToolNames.has(block.name) || unavailableToolNames.has(block.name),
    );
    const deferTerminalAcceptance =
      toolsEnabled && terminalToolCalls.length > 0 && unavailableToolCalls.length > 0;

    if (traceEnabled && options.turnId !== undefined) {
      options.tracer?.emit("llm_call.completed", {
        turnId: options.turnId,
        session_id: options.sessionId,
        label: traceLabel,
        iteration: iterations + 1,
        responseShape: summarizeResponseShape(response.messageBlocks),
        stopReason: response.stop_reason,
        usage: buildUsageTraceBlock(response),
        ...(options.tracer.includePayloads
          ? {
              response: toTraceJsonValue({
                messageBlocks: response.messageBlocks,
              }),
            }
          : {}),
      });
    }

    if (toolsEnabled && terminalToolCalls.length > 0 && !deferTerminalAcceptance) {
      return {
        text: extractText(response.messageBlocks),
        iterations,
        toolCallsMade,
        terminalToolCalls,
        stopReason: "terminal_tool",
        usage,
      };
    }

    if (!toolsEnabled || toolUseBlocks.length === 0) {
      return {
        text: extractText(response.messageBlocks),
        iterations,
        toolCallsMade,
        terminalToolCalls: [],
        stopReason: forcedTextOnly ? "max_iterations" : "text",
        usage,
      };
    }

    messages.push({
      role: "assistant",
      content: [...response.messageBlocks],
    });

    const runnableBlocks = toolUseBlocks.slice(0, maxToolCallsPerIteration);
    const droppedBlocks = toolUseBlocks.slice(maxToolCallsPerIteration);
    const toolResultBlocks: Array<Extract<LLMContentBlock, { type: "tool_result" }>> = [];

    for (const block of runnableBlocks) {
      if (traceEnabled && options.turnId !== undefined) {
        options.tracer?.emit("tool_call.started", {
          turnId: options.turnId,
          session_id: options.sessionId,
          callId: block.id,
          toolName: block.name,
        });
      }

      if (!allowedToolNames.has(block.name) || unavailableToolNames.has(block.name)) {
        const skippedResult = await options.dispatcher.recordSkippedCall({
          callId: block.id,
          toolName: block.name,
          input: block.input,
          sessionId: options.sessionId,
          turnId: options.turnId,
          origin: options.origin,
          turnOrigin: options.turnOrigin,
          currentSenderBorgRole: options.currentSenderBorgRole,
          sessionAudienceRole: options.sessionAudienceRole,
          provenance: options.provenance,
          skipReason: "tool_not_available_in_context",
        });
        toolCallsMade.push(toCallRecord(block, skippedResult));
        const toolResultBlock = buildUnavailableToolResultBlock(block);
        toolResultBlocks.push(toolResultBlock);
        if (options.toolResultObserver !== undefined) {
          observeToolResult(options.toolResultObserver, {
            ordinal: toolCallsMade.length,
            iteration: iterations + 1,
            batchPosition: toolResultBlocks.length,
            callId: skippedResult.callId,
            toolName: skippedResult.toolName,
            rawArguments: block.input,
            disposition: "skipped_unavailable",
            result: { ok: false, error: String(toolResultBlock.content) },
            durationMs: skippedResult.durationMs,
          });
        }
        if (traceEnabled && options.turnId !== undefined) {
          options.tracer?.emit("tool_call.completed", {
            turnId: options.turnId,
            session_id: options.sessionId,
            callId: skippedResult.callId,
            toolName: skippedResult.toolName,
            success: skippedResult.ok,
            ms: skippedResult.durationMs,
          });
        }
        continue;
      }

      if (deferTerminalAcceptance && terminalToolNames.has(block.name)) {
        const toolResultBlock = buildDeferredTerminalToolResultBlock(block);
        const skippedResult = await options.dispatcher.recordSkippedCall({
          callId: block.id,
          toolName: block.name,
          input: block.input,
          sessionId: options.sessionId,
          turnId: options.turnId,
          origin: options.origin,
          turnOrigin: options.turnOrigin,
          currentSenderBorgRole: options.currentSenderBorgRole,
          sessionAudienceRole: options.sessionAudienceRole,
          provenance: options.provenance,
          skipReason: "terminal_deferred_for_unavailable_sibling",
          error: String(toolResultBlock.content),
        });
        toolCallsMade.push(toCallRecord(block, skippedResult));
        toolResultBlocks.push(toolResultBlock);
        if (options.toolResultObserver !== undefined) {
          observeToolResult(options.toolResultObserver, {
            ordinal: toolCallsMade.length,
            iteration: iterations + 1,
            batchPosition: toolResultBlocks.length,
            callId: skippedResult.callId,
            toolName: skippedResult.toolName,
            rawArguments: block.input,
            disposition: "skipped_unavailable",
            result: { ok: false, error: String(toolResultBlock.content) },
            durationMs: skippedResult.durationMs,
          });
        }
        if (traceEnabled && options.turnId !== undefined) {
          options.tracer?.emit("tool_call.completed", {
            turnId: options.turnId,
            session_id: options.sessionId,
            callId: skippedResult.callId,
            toolName: skippedResult.toolName,
            success: skippedResult.ok,
            ms: skippedResult.durationMs,
          });
        }
        continue;
      }

      const dispatchResult = await dispatchToolUseBlock(
        options.dispatcher,
        {
          sessionId: options.sessionId,
          origin: options.origin,
          turnOrigin: options.turnOrigin,
          turnId: options.turnId,
          audienceEntityId: options.audienceEntityId,
          currentSenderBorgRole: options.currentSenderBorgRole,
          sessionAudienceRole: options.sessionAudienceRole,
          provenance: options.provenance,
        },
        block,
      );
      toolCallsMade.push(toCallRecord(block, dispatchResult));
      toolResultBlocks.push(buildToolResultBlock(dispatchResult));
      if (options.toolResultObserver !== undefined) {
        observeToolResult(options.toolResultObserver, {
          ordinal: toolCallsMade.length,
          iteration: iterations + 1,
          batchPosition: toolResultBlocks.length,
          callId: dispatchResult.callId,
          toolName: dispatchResult.toolName,
          rawArguments: block.input,
          disposition: "dispatched",
          result: dispatchResult.ok
            ? { ok: true, output: dispatchResult.output }
            : { ok: false, error: dispatchResult.error },
          durationMs: dispatchResult.durationMs,
        });
      }
      if (traceEnabled && options.turnId !== undefined) {
        options.tracer?.emit("tool_call.completed", {
          turnId: options.turnId,
          session_id: options.sessionId,
          callId: dispatchResult.callId,
          toolName: dispatchResult.toolName,
          success: dispatchResult.ok,
          ms: dispatchResult.durationMs,
        });
      }
    }

    for (const block of droppedBlocks) {
      const droppedBecauseUnavailable =
        !allowedToolNames.has(block.name) || unavailableToolNames.has(block.name);
      if (traceEnabled && options.turnId !== undefined) {
        options.tracer?.emit("tool_call.started", {
          turnId: options.turnId,
          session_id: options.sessionId,
          callId: block.id,
          toolName: block.name,
          skipped: true,
          reason: droppedBecauseUnavailable
            ? "tool_not_available_in_context"
            : "max_tool_calls_per_iteration",
        });
      }

      if (droppedBecauseUnavailable) {
        const skippedResult = await options.dispatcher.recordSkippedCall({
          callId: block.id,
          toolName: block.name,
          input: block.input,
          sessionId: options.sessionId,
          origin: options.origin,
          turnOrigin: options.turnOrigin,
          currentSenderBorgRole: options.currentSenderBorgRole,
          sessionAudienceRole: options.sessionAudienceRole,
          turnId: options.turnId,
          provenance: options.provenance,
          skipReason: "tool_not_available_in_context",
        });
        toolCallsMade.push(toCallRecord(block, skippedResult));
        const toolResultBlock = buildUnavailableToolResultBlock(block);
        toolResultBlocks.push(toolResultBlock);
        if (options.toolResultObserver !== undefined) {
          observeToolResult(options.toolResultObserver, {
            ordinal: toolCallsMade.length,
            iteration: iterations + 1,
            batchPosition: toolResultBlocks.length,
            callId: skippedResult.callId,
            toolName: skippedResult.toolName,
            rawArguments: block.input,
            disposition: "skipped_unavailable",
            result: { ok: false, error: String(toolResultBlock.content) },
            durationMs: skippedResult.durationMs,
          });
        }
        if (traceEnabled && options.turnId !== undefined) {
          options.tracer?.emit("tool_call.completed", {
            turnId: options.turnId,
            session_id: options.sessionId,
            callId: skippedResult.callId,
            toolName: skippedResult.toolName,
            success: skippedResult.ok,
            ms: skippedResult.durationMs,
          });
        }
        continue;
      }

      const skippedResult = await options.dispatcher.recordSkippedCall({
        callId: block.id,
        toolName: block.name,
        input: block.input,
        sessionId: options.sessionId,
        origin: options.origin,
        turnOrigin: options.turnOrigin,
        currentSenderBorgRole: options.currentSenderBorgRole,
        sessionAudienceRole: options.sessionAudienceRole,
        turnId: options.turnId,
        provenance: options.provenance,
        skipReason: "max_tool_calls_per_iteration",
      });
      toolCallsMade.push(toCallRecord(block, skippedResult));
      const toolResultBlock = buildDroppedToolResultBlock(block, maxToolCallsPerIteration);
      toolResultBlocks.push(toolResultBlock);
      if (options.toolResultObserver !== undefined) {
        observeToolResult(options.toolResultObserver, {
          ordinal: toolCallsMade.length,
          iteration: iterations + 1,
          batchPosition: toolResultBlocks.length,
          callId: skippedResult.callId,
          toolName: skippedResult.toolName,
          rawArguments: block.input,
          disposition: "skipped_iteration_cap",
          result: { ok: false, error: String(toolResultBlock.content) },
          durationMs: skippedResult.durationMs,
        });
      }
      if (traceEnabled && options.turnId !== undefined) {
        options.tracer?.emit("tool_call.completed", {
          turnId: options.turnId,
          session_id: options.sessionId,
          callId: skippedResult.callId,
          toolName: skippedResult.toolName,
          success: skippedResult.ok,
          ms: skippedResult.durationMs,
        });
      }
    }

    messages.push({
      role: "user",
      content: toolResultBlocks,
    });

    iterations += 1;

    if (iterations >= maxIterations) {
      toolsEnabled = false;
      forcedTextOnly = true;
    }
  }
}

function countSystemPromptChars(system: LLMConverseOptions["system"]): number {
  if (system === undefined) {
    return 0;
  }

  if (typeof system === "string") {
    return system.length;
  }

  return system.reduce((sum, block) => sum + block.text.length, 0);
}

function countBlockChars(block: LLMContentBlock): number {
  if (block.type === "text") {
    return block.text.length;
  }

  if (block.type === "tool_use") {
    return block.name.length + (JSON.stringify(block.input) ?? "").length;
  }

  if (block.type === "image_ref") {
    return block.attachment_id.length;
  }

  if (block.type === "thinking") {
    return block.thinking.length + block.signature.length;
  }

  if (block.type === "redacted_thinking") {
    return block.data.length;
  }

  const content =
    typeof block.content === "string"
      ? block.content
      : block.content.map((textBlock) => textBlock.text).join("");
  return block.tool_use_id.length + content.length;
}

function countConversePromptChars(
  system: LLMConverseOptions["system"],
  messages: readonly LLMContentBlockMessage[],
): number {
  return (
    countSystemPromptChars(system) +
    messages.reduce(
      (sum, message) =>
        sum +
        message.role.length +
        message.content.reduce((blockSum, block) => blockSum + countBlockChars(block), 0),
      0,
    )
  );
}

function summarizeResponseShape(blocks: readonly LLMContentBlock[]): JsonValue {
  const textLength = blocks.reduce(
    (sum, block) => (block.type === "text" ? sum + block.text.length : sum),
    0,
  );
  const toolUseBlocks = blocks
    .filter((block): block is LLMToolUseBlock => block.type === "tool_use")
    .map((block) => ({
      id: block.id,
      name: block.name,
    }));

  return {
    textLength,
    toolUseBlocks,
  };
}
