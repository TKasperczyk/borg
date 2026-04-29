import type {
  LLMClient,
  LLMContentBlock,
  LLMContentBlockMessage,
  LLMConverseOptions,
  LLMToolDefinition,
  LLMToolUseBlock,
} from "../../llm/index.js";
import { toAnthropicToolDefinitions } from "../../tools/anthropic.js";
import type {
  ToolDefinition,
  ToolDispatchResult,
  ToolDispatcher,
  ToolOrigin,
} from "../../tools/index.js";
import type { TurnTracer } from "../tracing/tracer.js";
import { toTraceJsonValue } from "../tracing/tracer.js";
import type { EntityId, SessionId } from "../../util/ids.js";
import type { JsonValue } from "../../util/json-value.js";
import { serializeJsonValue } from "../../util/json-value.js";

const DEFAULT_MAX_ITERATIONS = 5;
const DEFAULT_MAX_TOOL_CALLS_PER_ITERATION = 3;

export type ToolLoopUsage = {
  input_tokens: number;
  output_tokens: number;
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

export type ExecuteToolLoopOptions = {
  llmClient: LLMClient;
  dispatcher: ToolDispatcher;
  sessionId: SessionId;
  model: string;
  systemPrompt?: LLMConverseOptions["system"];
  initialMessages: readonly LLMContentBlockMessage[];
  tools: readonly ToolDefinition[];
  origin: ToolOrigin;
  audienceEntityId?: EntityId | null;
  provenance?: unknown;
  budget: string;
  maxTokens?: number;
  temperature?: number;
  thinking?: LLMConverseOptions["thinking"];
  maxIterations?: number;
  maxToolCallsPerIteration?: number;
  terminalToolNames?: readonly string[];
  tracer?: TurnTracer;
  turnId?: string;
  traceLabel?: string;
};

export type ToolLoopResult = {
  text: string;
  iterations: number;
  toolCallsMade: ToolLoopCallRecord[];
  terminalToolCalls: LLMToolUseBlock[];
  stopReason: "text" | "max_iterations" | "terminal_tool";
  usage: ToolLoopUsage;
};

function aggregateUsage(current: ToolLoopUsage, next: ToolLoopUsage): ToolLoopUsage {
  return {
    input_tokens: current.input_tokens + next.input_tokens,
    output_tokens: current.output_tokens + next.output_tokens,
    stop_reason: next.stop_reason,
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

async function dispatchToolUseBlock(
  dispatcher: ToolDispatcher,
  options: Pick<ExecuteToolLoopOptions, "sessionId" | "origin" | "audienceEntityId" | "provenance">,
  block: LLMToolUseBlock,
): Promise<ToolDispatchResult> {
  try {
    return await dispatcher.dispatch({
      callId: block.id,
      toolName: block.name,
      input: block.input,
      sessionId: options.sessionId,
      origin: options.origin,
      audienceEntityId: options.audienceEntityId,
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
  const maxIterations = options.maxIterations ?? DEFAULT_MAX_ITERATIONS;
  const maxToolCallsPerIteration =
    options.maxToolCallsPerIteration ?? DEFAULT_MAX_TOOL_CALLS_PER_ITERATION;
  const messages = options.initialMessages.map((message) => cloneMessage(message));
  const anthropicTools = toAnthropicToolDefinitions(options.tools);
  const allowedToolNames = new Set(options.tools.map((tool) => tool.name));
  const terminalToolNames = new Set(options.terminalToolNames ?? []);
  const toolCallsMade: ToolLoopCallRecord[] = [];
  let iterations = 0;
  let toolsEnabled = anthropicTools.length > 0;
  let forcedTextOnly = false;
  let usage: ToolLoopUsage = {
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: null,
  };
  const traceEnabled = options.tracer?.enabled === true && options.turnId !== undefined;
  const traceLabel = options.traceLabel ?? options.budget;

  while (true) {
    if (traceEnabled && options.turnId !== undefined) {
      options.tracer?.emit("llm_call_started", {
        turnId: options.turnId,
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

    const response = await options.llmClient.converse({
      model: options.model,
      system: options.systemPrompt,
      messages,
      ...(toolsEnabled ? { tools: anthropicTools } : {}),
      max_tokens: options.maxTokens,
      ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
      ...(options.thinking === undefined ? {} : { thinking: options.thinking }),
      budget: options.budget,
    });
    usage = aggregateUsage(usage, response);

    const toolUseBlocks = response.messageBlocks.filter(isToolUseBlock);
    const terminalToolCalls = toolUseBlocks.filter(
      (block) => allowedToolNames.has(block.name) && terminalToolNames.has(block.name),
    );

    if (traceEnabled && options.turnId !== undefined) {
      options.tracer?.emit("llm_call_response", {
        turnId: options.turnId,
        label: traceLabel,
        iteration: iterations + 1,
        responseShape: summarizeResponseShape(response.messageBlocks),
        stopReason: response.stop_reason,
        usage: {
          inputTokens: response.input_tokens,
          outputTokens: response.output_tokens,
        },
        ...(options.tracer.includePayloads
          ? {
              response: toTraceJsonValue({
                messageBlocks: response.messageBlocks,
              }),
            }
          : {}),
      });
    }

    if (toolsEnabled && terminalToolCalls.length > 0) {
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
        options.tracer?.emit("tool_call_dispatched", {
          turnId: options.turnId,
          callId: block.id,
          toolName: block.name,
        });
      }

      if (!allowedToolNames.has(block.name)) {
        const skippedResult = await options.dispatcher.recordSkippedCall({
          callId: block.id,
          toolName: block.name,
          input: block.input,
          sessionId: options.sessionId,
          origin: options.origin,
          provenance: options.provenance,
          skipReason: "tool_not_available_in_context",
        });
        toolCallsMade.push(toCallRecord(block, skippedResult));
        toolResultBlocks.push(buildUnavailableToolResultBlock(block));
        if (traceEnabled && options.turnId !== undefined) {
          options.tracer?.emit("tool_call_completed", {
            turnId: options.turnId,
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
          audienceEntityId: options.audienceEntityId,
          provenance: options.provenance,
        },
        block,
      );
      toolCallsMade.push(toCallRecord(block, dispatchResult));
      toolResultBlocks.push(buildToolResultBlock(dispatchResult));
      if (traceEnabled && options.turnId !== undefined) {
        options.tracer?.emit("tool_call_completed", {
          turnId: options.turnId,
          callId: dispatchResult.callId,
          toolName: dispatchResult.toolName,
          success: dispatchResult.ok,
          ms: dispatchResult.durationMs,
        });
      }
    }

    for (const block of droppedBlocks) {
      if (traceEnabled && options.turnId !== undefined) {
        options.tracer?.emit("tool_call_dispatched", {
          turnId: options.turnId,
          callId: block.id,
          toolName: block.name,
          skipped: true,
          reason: "max_tool_calls_per_iteration",
        });
      }

      const skippedResult = await options.dispatcher.recordSkippedCall({
        callId: block.id,
        toolName: block.name,
        input: block.input,
        sessionId: options.sessionId,
        origin: options.origin,
        provenance: options.provenance,
        skipReason: "max_tool_calls_per_iteration",
      });
      toolCallsMade.push(toCallRecord(block, skippedResult));
      toolResultBlocks.push(buildDroppedToolResultBlock(block, maxToolCallsPerIteration));
      if (traceEnabled && options.turnId !== undefined) {
        options.tracer?.emit("tool_call_completed", {
          turnId: options.turnId,
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

function summarizeToolSchemas(tools: readonly LLMToolDefinition[]): JsonValue {
  return tools.map((tool) => ({
    name: tool.name,
    propertyCount:
      tool.inputSchema.properties === undefined
        ? 0
        : Object.keys(tool.inputSchema.properties).length,
    required: Array.isArray(tool.inputSchema.required) ? tool.inputSchema.required.map(String) : [],
  }));
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
