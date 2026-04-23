import type {
  LLMClient,
  LLMContentBlock,
  LLMContentBlockMessage,
  LLMConverseOptions,
  LLMToolUseBlock,
} from "../../llm/index.js";
import { toAnthropicToolDefinitions } from "../../tools/anthropic.js";
import type { ToolDefinition, ToolDispatchResult, ToolDispatcher, ToolOrigin } from "../../tools/index.js";
import type { SessionId } from "../../util/ids.js";
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
  provenance?: unknown;
  budget: string;
  maxTokens?: number;
  temperature?: number;
  thinking?: LLMConverseOptions["thinking"];
  maxIterations?: number;
  maxToolCallsPerIteration?: number;
};

export type ToolLoopResult = {
  text: string;
  iterations: number;
  toolCallsMade: ToolLoopCallRecord[];
  stopReason: "text" | "max_iterations";
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

function buildToolResultBlock(result: ToolDispatchResult): Extract<LLMContentBlock, { type: "tool_result" }> {
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
  options: Pick<ExecuteToolLoopOptions, "sessionId" | "origin" | "provenance">,
  block: LLMToolUseBlock,
): Promise<ToolDispatchResult> {
  try {
    return await dispatcher.dispatch({
      callId: block.id,
      toolName: block.name,
      input: block.input,
      sessionId: options.sessionId,
      origin: options.origin,
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
  const toolCallsMade: ToolLoopCallRecord[] = [];
  let iterations = 0;
  let toolsEnabled = anthropicTools.length > 0;
  let forcedTextOnly = false;
  let usage: ToolLoopUsage = {
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: null,
  };

  while (true) {
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

    if (!toolsEnabled || toolUseBlocks.length === 0) {
      return {
        text: extractText(response.messageBlocks),
        iterations,
        toolCallsMade,
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
        continue;
      }

      const dispatchResult = await dispatchToolUseBlock(
        options.dispatcher,
        {
          sessionId: options.sessionId,
          origin: options.origin,
          provenance: options.provenance,
        },
        block,
      );
      toolCallsMade.push(toCallRecord(block, dispatchResult));
      toolResultBlocks.push(buildToolResultBlock(dispatchResult));
    }

    for (const block of droppedBlocks) {
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
