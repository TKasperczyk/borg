import Anthropic from "@anthropic-ai/sdk";
import type {
  Message,
  MessageParam,
  TextBlock,
  Tool,
  ToolUseBlock,
} from "@anthropic-ai/sdk/resources/messages/messages.js";

import { ConfigError, LLMError } from "../util/errors.js";

export type LLMMessage = {
  role: "user" | "assistant";
  content: string;
};

export type LLMToolDefinition = {
  name: string;
  description?: string;
  inputSchema: {
    type: "object";
    properties?: Record<string, unknown>;
    required?: string[];
    [key: string]: unknown;
  };
};

export type LLMToolCall = {
  id: string;
  name: string;
  input: unknown;
};

export type LLMCompleteOptions = {
  model: string;
  system?: string;
  messages: readonly LLMMessage[];
  tools?: readonly LLMToolDefinition[];
  max_tokens: number;
  budget: string;
};

export type LLMCompleteResult = {
  text: string;
  input_tokens: number;
  output_tokens: number;
  stop_reason: string | null;
  tool_calls: LLMToolCall[];
};

export type TokenUsageEvent = {
  budget: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
};

export type TokenUsageSink = (event: TokenUsageEvent) => void | Promise<void>;

export type LLMClient = {
  complete(options: LLMCompleteOptions): Promise<LLMCompleteResult>;
};

type AnthropicClientLike = {
  messages: {
    create(params: {
      model: string;
      system?: string;
      messages: MessageParam[];
      tools?: Tool[];
      max_tokens: number;
    }): Promise<Message>;
  };
};

function toAnthropicMessages(messages: readonly LLMMessage[]): MessageParam[] {
  return messages.map((message) => ({
    role: message.role,
    content: message.content,
  }));
}

function toAnthropicTools(tools: readonly LLMToolDefinition[] | undefined): Tool[] | undefined {
  return tools?.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema,
  }));
}

function isToolUseBlock(block: Message["content"][number]): block is ToolUseBlock {
  return block.type === "tool_use";
}

function isTextBlock(block: Message["content"][number]): block is TextBlock {
  return block.type === "text";
}

function extractToolCalls(message: Message): LLMToolCall[] {
  return message.content.filter(isToolUseBlock).map((block) => ({
    id: block.id,
    name: block.name,
    input: block.input,
  }));
}

function extractText(message: Message): string {
  return message.content
    .filter(isTextBlock)
    .map((block) => block.text)
    .join("");
}

export type AnthropicLLMClientOptions = {
  apiKey?: string;
  client?: AnthropicClientLike;
  usageSink?: TokenUsageSink;
};

export class AnthropicLLMClient implements LLMClient {
  private readonly client: AnthropicClientLike;
  private readonly usageSink?: TokenUsageSink;

  constructor(options: AnthropicLLMClientOptions = {}) {
    if (options.client !== undefined) {
      this.client = options.client;
    } else {
      if (!options.apiKey?.trim()) {
        throw new ConfigError("Anthropic API key must be configured");
      }

      this.client = new Anthropic({
        apiKey: options.apiKey,
      });
    }

    this.usageSink = options.usageSink;
  }

  async complete(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    try {
      const response = await this.client.messages.create({
        model: options.model,
        system: options.system,
        messages: toAnthropicMessages(options.messages),
        tools: toAnthropicTools(options.tools),
        max_tokens: options.max_tokens,
      });

      const result = {
        text: extractText(response),
        input_tokens: response.usage.input_tokens,
        output_tokens: response.usage.output_tokens,
        stop_reason: response.stop_reason,
        tool_calls: extractToolCalls(response),
      } satisfies LLMCompleteResult;

      if (this.usageSink !== undefined) {
        await this.usageSink({
          budget: options.budget,
          model: options.model,
          input_tokens: result.input_tokens,
          output_tokens: result.output_tokens,
        });
      }

      return result;
    } catch (error) {
      if (error instanceof ConfigError) {
        throw error;
      }

      throw new LLMError("Failed to complete Anthropic request", {
        cause: error,
      });
    }
  }
}

export type FakeLLMResponse =
  | LLMCompleteResult
  | ((options: LLMCompleteOptions) => LLMCompleteResult | Promise<LLMCompleteResult>);

export type FakeLLMClientOptions = {
  responses?: FakeLLMResponse[];
  usageSink?: TokenUsageSink;
};

export class FakeLLMClient implements LLMClient {
  private readonly usageSink?: TokenUsageSink;
  readonly requests: LLMCompleteOptions[] = [];
  private readonly responses: FakeLLMResponse[];

  constructor(options: FakeLLMClientOptions = {}) {
    this.responses = [...(options.responses ?? [])];
    this.usageSink = options.usageSink;
  }

  pushResponse(response: FakeLLMResponse): void {
    this.responses.push(response);
  }

  async complete(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    this.requests.push(options);
    const response = this.responses.shift();

    if (response === undefined) {
      throw new LLMError("FakeLLMClient has no scripted response available");
    }

    const resolved = typeof response === "function" ? await response(options) : response;

    if (this.usageSink !== undefined) {
      await this.usageSink({
        budget: options.budget,
        model: options.model,
        input_tokens: resolved.input_tokens,
        output_tokens: resolved.output_tokens,
      });
    }

    return resolved;
  }
}
