// An OpenAI-Chat-Completions LLMClient, for self-hosted OpenAI-compatible
// gateways (vLLM, kratos, LM Studio, ...). It is a sibling of AnthropicLLMClient
// and implements the same LLMClient contract, so it is reached ONLY by injecting
// it through BorgOpenOptions.llmClient -- the Anthropic path stays the default
// and is untouched.
//
// Scope note: the memory write/recall paths use complete() with a forced
// tool_choice for structured extraction; converse() is implemented for contract
// completeness (multi-turn text + tool loop) but image_ref blocks are rejected
// because resolving attachments requires Anthropic-specific plumbing this adapter
// does not carry.
//
// TLS: like OpenAICompatibleEmbeddingClient, this does NOT configure a custom CA
// in code. Trust for a private endpoint (e.g. https://inference.hades.p4.int) is
// established at the process level via NODE_EXTRA_CA_CERTS=/path/to/ca.pem, which
// is the right knob for a dedicated sidecar process. The hades gateway uses
// server-CA trust only (no client certificate). Pass `client` to inject a fully
// pre-configured OpenAI instance when finer control is needed.

import OpenAI from "openai";

import { ConfigError, LLMError } from "../util/errors.js";
import { clampMaxOutputTokens, getModelMaxOutputTokens } from "./max-tokens.js";
import type {
  LLMClient,
  LLMCompleteOptions,
  LLMCompleteResult,
  LLMContentBlock,
  LLMContentBlockMessage,
  LLMConverseOptions,
  LLMConverseResult,
  LLMMessage,
  LLMSystemBlock,
  LLMTextBlock,
  LLMToolCall,
  LLMToolDefinition,
  TokenUsageSink,
} from "./index.js";

// Minimal structural view of the bits of the OpenAI client we use. Keeping it
// narrow (rather than importing the SDK's request/response types) makes the
// `client` escape hatch trivially fakeable in tests, mirroring the embeddings
// client's OpenAIEmbeddingsClient shape.
type OpenAIChatMessageParam = {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: ReadonlyArray<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string;
};

type OpenAIChatCompletionResponse = {
  choices: ReadonlyArray<{
    message: {
      content: string | null;
      tool_calls?: ReadonlyArray<{
        id: string;
        type: string;
        function: { name: string; arguments: string };
      }> | null;
    };
    finish_reason: string | null;
  }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number } | null;
};

export type OpenAIChatCompletionsClient = {
  chat: {
    completions: {
      create(
        params: Record<string, unknown>,
        options?: { signal?: AbortSignal | null; timeout?: number },
      ): Promise<OpenAIChatCompletionResponse>;
    };
  };
};

export type OpenAICompatibleLLMClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  // Inject a pre-built client (tests, or a custom-configured OpenAI instance).
  // When present, baseUrl/apiKey are ignored.
  client?: OpenAIChatCompletionsClient;
  // vLLM/OpenAI both accept "max_tokens"; some newer gateways require
  // "max_completion_tokens". Switchable without code changes; defaults to the
  // broadly-compatible "max_tokens".
  maxTokensField?: "max_tokens" | "max_completion_tokens";
  // Per-request timeout forwarded to the OpenAI SDK call.
  requestTimeoutMs?: number;
  // Optional push-based usage reporting (the per-call result already carries
  // token counts; this mirrors the sink other clients expose).
  usageSink?: TokenUsageSink;
};

function flattenSystem(system: string | readonly LLMSystemBlock[] | undefined): string | null {
  if (system === undefined) {
    return null;
  }
  if (typeof system === "string") {
    return system;
  }
  // cache_control is an Anthropic prompt-caching hint with no OpenAI analogue;
  // dropping it is lossless for correctness.
  return system.map((block) => block.text).join("\n\n");
}

function toOpenAITools(tools: readonly LLMToolDefinition[]): Record<string, unknown>[] {
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      ...(tool.description === undefined ? {} : { description: tool.description }),
      parameters: tool.inputSchema,
    },
  }));
}

function toOpenAIToolChoice(
  choice: NonNullable<LLMCompleteOptions["tool_choice"]>,
): "auto" | "required" | Record<string, unknown> {
  switch (choice.type) {
    case "tool":
      // True single-tool forcing.
      return { type: "function", function: { name: choice.name } };
    case "any":
      return "required";
    case "auto":
      return "auto";
  }
}

function mapFinishReason(finishReason: string | null): string | null {
  switch (finishReason) {
    case "tool_calls":
    case "function_call":
      return "tool_use";
    case "stop":
      return "end_turn";
    case "length":
      return "max_tokens";
    default:
      // Pass through anything else (e.g. "content_filter") unchanged.
      return finishReason;
  }
}

function parseToolArguments(name: string, rawArguments: string | undefined): unknown {
  if (rawArguments === undefined || rawArguments.trim() === "") {
    return {};
  }
  try {
    return JSON.parse(rawArguments);
  } catch (error) {
    // Raised inside complete()/converse(), so callStructuredTool classifies it
    // as "llm_failed": the model produced syntactically broken tool arguments,
    // which is a failed call rather than a schema mismatch.
    throw new LLMError(`Tool call "${name}" returned unparseable JSON arguments`, {
      cause: error,
    });
  }
}

function decodeToolCalls(
  rawToolCalls:
    | ReadonlyArray<{ id: string; type: string; function: { name: string; arguments: string } }>
    | null
    | undefined,
): LLMToolCall[] {
  if (!rawToolCalls) {
    return [];
  }
  return rawToolCalls
    .filter((call) => call.type === "function")
    .map((call) => ({
      id: call.id,
      name: call.function.name,
      input: parseToolArguments(call.function.name, call.function.arguments),
    }));
}

function stringifyToolResultContent(content: string | readonly LLMTextBlock[]): string {
  if (typeof content === "string") {
    return content;
  }
  return content.map((block) => block.text).join("\n");
}

// Translate Borg's Anthropic-shaped content-block messages into OpenAI chat
// messages. A single Anthropic message can fan out into several OpenAI messages:
// tool_result blocks become standalone role:"tool" messages, while text and
// (assistant) tool_use blocks collapse into one role message.
function toOpenAIConversationMessages(
  messages: readonly LLMContentBlockMessage[],
): OpenAIChatMessageParam[] {
  const out: OpenAIChatMessageParam[] = [];

  for (const message of messages) {
    let text = "";
    const toolCalls: Array<NonNullable<OpenAIChatMessageParam["tool_calls"]>[number]> = [];
    const toolResults: OpenAIChatMessageParam[] = [];

    for (const block of message.content) {
      switch (block.type) {
        case "text":
          text += block.text;
          break;
        case "tool_use":
          toolCalls.push({
            id: block.id,
            type: "function",
            function: { name: block.name, arguments: JSON.stringify(block.input ?? {}) },
          });
          break;
        case "tool_result":
          toolResults.push({
            role: "tool",
            tool_call_id: block.tool_use_id,
            content: stringifyToolResultContent(block.content),
          });
          break;
        case "thinking":
        case "redacted_thinking":
          // No OpenAI analogue; reasoning is not round-tripped.
          break;
        case "image_ref":
          throw new LLMError(
            "OpenAICompatibleLLMClient.converse does not support image_ref blocks (no attachment resolver)",
          );
      }
    }

    // Tool results respond to a prior assistant tool call; emit them first.
    out.push(...toolResults);

    if (toolCalls.length > 0) {
      out.push({ role: "assistant", content: text === "" ? null : text, tool_calls: toolCalls });
    } else if (text !== "") {
      out.push({ role: message.role, content: text });
    }
  }

  return out;
}

export class OpenAICompatibleLLMClient implements LLMClient {
  private readonly client: OpenAIChatCompletionsClient;
  private readonly maxTokensField: "max_tokens" | "max_completion_tokens";
  private readonly requestTimeoutMs: number | undefined;
  private readonly usageSink: TokenUsageSink | undefined;

  constructor(options: OpenAICompatibleLLMClientOptions) {
    if (options.client !== undefined) {
      this.client = options.client;
    } else {
      if (!options.baseUrl?.trim()) {
        throw new ConfigError("OpenAI-compatible LLM base URL must be configured");
      }
      if (!options.apiKey?.trim()) {
        throw new ConfigError("OpenAI-compatible LLM API key must be configured");
      }
      // baseURL is the ".../v1" root; the SDK appends "/chat/completions". Do
      // NOT pass a full ".../v1/chat/completions" URL here.
      this.client = new OpenAI({
        apiKey: options.apiKey,
        baseURL: options.baseUrl,
      }) as unknown as OpenAIChatCompletionsClient;
    }
    this.maxTokensField = options.maxTokensField ?? "max_tokens";
    this.requestTimeoutMs = options.requestTimeoutMs;
    this.usageSink = options.usageSink;
  }

  async complete(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    const system = flattenSystem(options.system);
    const messages: OpenAIChatMessageParam[] = [
      ...(system === null ? [] : [{ role: "system" as const, content: system }]),
      ...options.messages.map((message: LLMMessage) => ({
        role: message.role,
        content: message.content,
      })),
    ];

    const response = await this.createCompletion(options, messages);
    const choice = response.choices[0];
    if (choice === undefined) {
      throw new LLMError("OpenAI-compatible completion returned no choices");
    }

    await this.reportUsage(options.budget, options.model, response.usage);

    return {
      text: choice.message.content ?? "",
      input_tokens: response.usage?.prompt_tokens ?? 0,
      output_tokens: response.usage?.completion_tokens ?? 0,
      stop_reason: mapFinishReason(choice.finish_reason),
      tool_calls: decodeToolCalls(choice.message.tool_calls),
    };
  }

  async converse(options: LLMConverseOptions): Promise<LLMConverseResult> {
    const system = flattenSystem(options.system);
    const messages: OpenAIChatMessageParam[] = [
      ...(system === null ? [] : [{ role: "system" as const, content: system }]),
      ...toOpenAIConversationMessages(options.messages),
    ];

    const response = await this.createCompletion(options, messages);
    const choice = response.choices[0];
    if (choice === undefined) {
      throw new LLMError("OpenAI-compatible completion returned no choices");
    }

    await this.reportUsage(options.budget, options.model, response.usage);

    const messageBlocks: LLMContentBlock[] = [];
    if (choice.message.content) {
      const textBlock: LLMTextBlock = { type: "text", text: choice.message.content };
      messageBlocks.push(textBlock);
    }
    for (const toolCall of decodeToolCalls(choice.message.tool_calls)) {
      messageBlocks.push({
        type: "tool_use",
        id: toolCall.id,
        name: toolCall.name,
        input: toolCall.input,
      });
    }

    return {
      messageBlocks,
      input_tokens: response.usage?.prompt_tokens ?? 0,
      output_tokens: response.usage?.completion_tokens ?? 0,
      stop_reason: mapFinishReason(choice.finish_reason),
    };
  }

  private async createCompletion(
    options: LLMCompleteOptions | LLMConverseOptions,
    messages: OpenAIChatMessageParam[],
  ): Promise<OpenAIChatCompletionResponse> {
    // output_config (JSON-schema structured output) has no faithful translation
    // here, and silently dropping it would return an empty structured_output --
    // a silent-wrong-result trap. Fail fast: borg's structured paths use a forced
    // tool_choice instead (which IS supported). No borg path currently sets it.
    if (options.output_config !== undefined) {
      throw new LLMError(
        "OpenAICompatibleLLMClient does not support output_config structured outputs; use a forced tool_choice instead",
      );
    }

    const hasTools = options.tools !== undefined && options.tools.length > 0;
    const requestedMaxTokens = options.max_tokens ?? getModelMaxOutputTokens(options.model);

    const params: Record<string, unknown> = {
      model: options.model,
      messages,
      [this.maxTokensField]: clampMaxOutputTokens(options.model, requestedMaxTokens),
    };
    if (options.temperature !== undefined) {
      params.temperature = options.temperature;
    }
    if (hasTools) {
      params.tools = toOpenAITools(options.tools as readonly LLMToolDefinition[]);
      // Deterministic structured extraction: at most one tool call per turn.
      params.parallel_tool_calls = false;
      if (options.tool_choice !== undefined) {
        params.tool_choice = toOpenAIToolChoice(options.tool_choice);
      }
    }
    // thinking / effort are advisory Anthropic extended-thinking hints with no
    // OpenAI chat-completions equivalent; they are intentionally dropped. Callers
    // needing guaranteed structured output must set a forced tool_choice (every
    // borg structured-tool path does); without one, behaviour falls back to
    // standard OpenAI auto tool-calling.

    const requestOptions =
      this.requestTimeoutMs === undefined ? undefined : { timeout: this.requestTimeoutMs };

    try {
      return await this.client.chat.completions.create(params, requestOptions);
    } catch (error) {
      if (error instanceof LLMError) {
        throw error;
      }
      throw new LLMError("OpenAI-compatible completion request failed", { cause: error });
    }
  }

  private async reportUsage(
    budget: string,
    model: string,
    usage: { prompt_tokens?: number; completion_tokens?: number } | null | undefined,
  ): Promise<void> {
    if (this.usageSink === undefined) {
      return;
    }
    await this.usageSink({
      budget,
      model,
      input_tokens: usage?.prompt_tokens ?? 0,
      output_tokens: usage?.completion_tokens ?? 0,
    });
  }
}
