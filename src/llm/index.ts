import Anthropic from "@anthropic-ai/sdk";
import type {
  Message,
  MessageParam,
  TextBlock,
  TextBlockParam,
  Tool,
  ToolChoice,
  ToolUseBlock,
} from "@anthropic-ai/sdk/resources/messages/messages.js";
import { z } from "zod";

import { getFreshCredentials, type GetFreshCredentialsOptions } from "../auth/claude-oauth.js";
import type { Clock } from "../util/clock.js";
import { AuthError, ConfigError, LLMError } from "../util/errors.js";

const OAUTH_BETAS = "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14";
const OAUTH_USER_AGENT = "claude-cli/2.1.2 (external, cli)";

export const CLAUDE_CODE_IDENTITY_BLOCK_TEXT =
  "You are Claude Code, Anthropic's official CLI for Claude.";

export type LLMMessage = {
  role: "user" | "assistant";
  content: string;
};

export type LLMSystemBlock = {
  type: "text";
  text: string;
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

export function toToolInputSchema(schema: z.ZodType): LLMToolDefinition["inputSchema"] {
  const jsonSchema = z.toJSONSchema(schema);

  if (jsonSchema.type !== "object") {
    throw new TypeError("Tool input schema must serialize to a top-level object schema");
  }

  return jsonSchema as LLMToolDefinition["inputSchema"];
}

export type LLMCompleteOptions = {
  model: string;
  system?: string | readonly LLMSystemBlock[];
  messages: readonly LLMMessage[];
  tools?: readonly LLMToolDefinition[];
  tool_choice?: { type: "tool"; name: string } | { type: "any" } | { type: "auto" };
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
      system?: string | TextBlockParam[];
      messages: MessageParam[];
      tools?: Tool[];
      tool_choice?: ToolChoice;
      max_tokens: number;
    }): Promise<Message>;
  };
};

type OAuthAuthKind = {
  kind: "oauth";
  authToken: string;
  source: "env" | "credentials-file";
};

type ResolvedAnthropicAuth = OAuthAuthKind | { kind: "api-key"; apiKey: string };

export type AnthropicAuthMode = "auto" | "oauth" | "api-key";

export type AnthropicLLMClientOptions = {
  apiKey?: string;
  authToken?: string;
  authMode?: AnthropicAuthMode;
  env?: NodeJS.ProcessEnv;
  client?: AnthropicClientLike;
  usageSink?: TokenUsageSink;
  clock?: Clock;
};

export type FakeLLMResponse =
  | LLMCompleteResult
  | ((options: LLMCompleteOptions) => LLMCompleteResult | Promise<LLMCompleteResult>);

export type FakeLLMClientOptions = {
  responses?: FakeLLMResponse[];
  usageSink?: TokenUsageSink;
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

function toAnthropicToolChoice(
  toolChoice: LLMCompleteOptions["tool_choice"],
): ToolChoice | undefined {
  return toolChoice;
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

function transformToolNameForOAuth(name: string): string {
  if (!name) {
    return name;
  }

  if (name.startsWith("mcp__")) {
    return name;
  }

  if (name.charAt(0) === name.charAt(0).toUpperCase() && /[A-Z]/.test(name.charAt(0))) {
    return name;
  }

  return name.charAt(0).toUpperCase() + name.slice(1);
}

function mutateToolUseNames(
  value: unknown,
  originalNamesByTransformed: ReadonlyMap<string, string>,
): boolean {
  if (value === null || typeof value !== "object") {
    return false;
  }

  let changed = false;

  if (Array.isArray(value)) {
    for (const entry of value) {
      if (mutateToolUseNames(entry, originalNamesByTransformed)) {
        changed = true;
      }
    }

    return changed;
  }

  const record = value as Record<string, unknown>;

  if (record.type === "tool_use" && typeof record.name === "string") {
    const original = originalNamesByTransformed.get(record.name);

    if (original !== undefined && original !== record.name) {
      record.name = original;
      changed = true;
    }
  }

  for (const key of Object.keys(record)) {
    if (mutateToolUseNames(record[key], originalNamesByTransformed)) {
      changed = true;
    }
  }

  return changed;
}

function transformSseEvent(
  event: string,
  originalNamesByTransformed: ReadonlyMap<string, string>,
): string {
  if (!event.includes("data:")) {
    return event;
  }

  const lines = event.split("\n");

  return lines
    .map((line) => {
      if (!line.startsWith("data:")) {
        return line;
      }

      const prefixMatch = line.match(/^data:\s*/);
      const prefix = prefixMatch ? prefixMatch[0] : "data: ";
      const data = line.slice(prefix.length);

      if (!data || data === "[DONE]") {
        return line;
      }

      try {
        const parsed = JSON.parse(data) as unknown;

        if (mutateToolUseNames(parsed, originalNamesByTransformed)) {
          return `${prefix}${JSON.stringify(parsed)}`;
        }

        return line;
      } catch {
        return line;
      }
    })
    .join("\n");
}

export function createOAuthFetch(): typeof fetch {
  return async (
    input: Parameters<typeof fetch>[0],
    init?: Parameters<typeof fetch>[1],
  ): Promise<Response> => {
    let requestUrl: URL;

    if (typeof input === "string") {
      requestUrl = new URL(input);
    } else if (input instanceof URL) {
      requestUrl = new URL(input.toString());
    } else {
      requestUrl = new URL(input.url);
    }

    const isMessagesRequest = requestUrl.pathname === "/v1/messages";

    if (isMessagesRequest && !requestUrl.searchParams.has("beta")) {
      requestUrl.searchParams.set("beta", "true");
    }

    let modifiedInit = init;
    const originalNamesByTransformed = new Map<string, string>();

    if (isMessagesRequest && init?.body && typeof init.body === "string") {
      try {
        const parsed = JSON.parse(init.body) as Record<string, unknown>;
        let modified = false;

        if (Array.isArray(parsed.tools)) {
          parsed.tools = parsed.tools.map((tool) => {
            if (tool === null || typeof tool !== "object") {
              return tool;
            }

            const record = tool as Record<string, unknown>;

            if (typeof record.name !== "string") {
              return tool;
            }

            const transformedName = transformToolNameForOAuth(record.name);

            if (transformedName !== record.name) {
              originalNamesByTransformed.set(transformedName, record.name);
              modified = true;
              return {
                ...record,
                name: transformedName,
              };
            }

            return tool;
          });
        }

        if (
          parsed.tool_choice !== null &&
          typeof parsed.tool_choice === "object" &&
          typeof (parsed.tool_choice as { name?: unknown }).name === "string"
        ) {
          const toolChoice = parsed.tool_choice as Record<string, unknown>;
          const transformedName = transformToolNameForOAuth(toolChoice.name as string);

          if (transformedName !== toolChoice.name) {
            originalNamesByTransformed.set(transformedName, toolChoice.name as string);
            parsed.tool_choice = {
              ...toolChoice,
              name: transformedName,
            };
            modified = true;
          }
        }

        if (modified) {
          modifiedInit = {
            ...init,
            body: JSON.stringify(parsed),
          };
        }
      } catch {
        // Leave non-JSON bodies unchanged.
      }
    }

    const response = await globalThis.fetch(requestUrl.toString(), modifiedInit);

    if (!isMessagesRequest) {
      return response;
    }

    const contentType = response.headers.get("content-type") ?? "";

    if (contentType.includes("application/json") && !contentType.includes("stream")) {
      try {
        const text = await response.clone().text();
        const parsed = JSON.parse(text) as unknown;

        if (mutateToolUseNames(parsed, originalNamesByTransformed)) {
          return new Response(JSON.stringify(parsed), {
            status: response.status,
            statusText: response.statusText,
            headers: new Headers(response.headers),
          });
        }

        return new Response(text, {
          status: response.status,
          statusText: response.statusText,
          headers: new Headers(response.headers),
        });
      } catch {
        return response;
      }
    }

    if (
      response.body &&
      (contentType.includes("text/event-stream") || contentType.includes("stream"))
    ) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const encoder = new TextEncoder();
      let buffer = "";

      const stream = new ReadableStream<Uint8Array>({
        async pull(controller) {
          const { done, value } = await reader.read();

          if (done) {
            if (buffer.length > 0) {
              controller.enqueue(
                encoder.encode(transformSseEvent(buffer, originalNamesByTransformed)),
              );
              buffer = "";
            }

            controller.close();
            return;
          }

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split(/\r?\n\r?\n/);
          buffer = events.pop() ?? "";

          if (events.length > 0) {
            controller.enqueue(
              encoder.encode(
                `${events
                  .map((event) => transformSseEvent(event, originalNamesByTransformed))
                  .join("\n\n")}\n\n`,
              ),
            );
          }
        },
      });

      return new Response(stream, {
        status: response.status,
        statusText: response.statusText,
        headers: new Headers(response.headers),
      });
    }

    return response;
  };
}

function normalizeSystemBlocks(
  system: string | readonly LLMSystemBlock[] | undefined,
): TextBlockParam[] {
  if (system === undefined) {
    return [];
  }

  if (typeof system === "string") {
    return [
      {
        type: "text",
        text: system,
      },
    ];
  }

  return system.map((block) => ({
    type: "text",
    text: block.text,
  }));
}

function isAuthenticationFailure(error: unknown): boolean {
  return (
    error instanceof Error &&
    "status" in error &&
    typeof (error as { status?: unknown }).status === "number" &&
    (error as { status: number }).status === 401
  );
}

async function resolveAnthropicAuth(
  options: Pick<AnthropicLLMClientOptions, "apiKey" | "authToken" | "authMode" | "env" | "clock">,
): Promise<ResolvedAnthropicAuth> {
  const authMode = options.authMode ?? "auto";
  const env = options.env ?? process.env;
  const apiKey = options.apiKey?.trim() || env.ANTHROPIC_API_KEY?.trim();

  if (authMode !== "oauth" && apiKey) {
    return {
      kind: "api-key",
      apiKey,
    };
  }

  if (authMode !== "api-key") {
    const authToken = options.authToken?.trim() || env.ANTHROPIC_AUTH_TOKEN?.trim();

    if (authToken) {
      return {
        kind: "oauth",
        authToken,
        source: "env",
      };
    }

    const credentials = await getFreshCredentials({
      env,
      clock: options.clock,
    });

    if (credentials !== null) {
      return {
        kind: "oauth",
        authToken: credentials.accessToken,
        source: "credentials-file",
      };
    }
  }

  throw new AuthError("No Anthropic credentials detected", {
    code: "AUTH_NO_CREDENTIALS",
  });
}

function buildAnthropicClient(auth: ResolvedAnthropicAuth): AnthropicClientLike {
  if (auth.kind === "api-key") {
    return new Anthropic({
      apiKey: auth.apiKey,
    });
  }

  return new Anthropic({
    authToken: auth.authToken,
    defaultHeaders: {
      "anthropic-beta": OAUTH_BETAS,
      "user-agent": OAUTH_USER_AGENT,
    },
    fetch: createOAuthFetch(),
  });
}

export class AnthropicLLMClient implements LLMClient {
  private client?: AnthropicClientLike;
  private auth?: ResolvedAnthropicAuth;
  private initialization?: Promise<void>;
  private readonly usageSink?: TokenUsageSink;
  private readonly options: AnthropicLLMClientOptions;

  constructor(options: AnthropicLLMClientOptions = {}) {
    this.options = options;
    this.client = options.client;
    this.usageSink = options.usageSink;
  }

  private async ensureInitialized(): Promise<void> {
    if (this.client !== undefined) {
      return;
    }

    this.initialization ??= (async () => {
      this.auth = await resolveAnthropicAuth(this.options);
      this.client = buildAnthropicClient(this.auth);
    })();

    await this.initialization;
  }

  private resolveSystemPrompt(
    system: string | readonly LLMSystemBlock[] | undefined,
  ): string | TextBlockParam[] | undefined {
    if (this.auth?.kind !== "oauth") {
      return system === undefined ? undefined : typeof system === "string" ? system : [...system];
    }

    return [
      {
        type: "text",
        text: CLAUDE_CODE_IDENTITY_BLOCK_TEXT,
      },
      ...normalizeSystemBlocks(system),
    ];
  }

  private async refreshOauthClient(): Promise<void> {
    const credentials = await getFreshCredentials({
      env: this.options.env,
      clock: this.options.clock,
      forceRefresh: true,
    } satisfies GetFreshCredentialsOptions);

    if (credentials === null) {
      throw new AuthError("Failed to refresh Claude OAuth credentials", {
        code: "AUTH_REFRESH_FAILED",
      });
    }

    this.auth = {
      kind: "oauth",
      authToken: credentials.accessToken,
      source: "credentials-file",
    };
    this.client = buildAnthropicClient(this.auth);
    this.initialization = Promise.resolve();
  }

  private async createMessage(
    options: LLMCompleteOptions,
    retrying = false,
  ): Promise<LLMCompleteResult> {
    await this.ensureInitialized();

    const client = this.client;

    if (client === undefined) {
      throw new LLMError("Anthropic client failed to initialize");
    }

    try {
      const response = await client.messages.create({
        model: options.model,
        system: this.resolveSystemPrompt(options.system),
        messages: toAnthropicMessages(options.messages),
        tools: toAnthropicTools(options.tools),
        tool_choice: toAnthropicToolChoice(options.tool_choice),
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
      if (!retrying && this.auth?.kind === "oauth" && isAuthenticationFailure(error)) {
        try {
          await this.refreshOauthClient();
        } catch (authError) {
          throw new LLMError("Failed to complete Anthropic request", {
            cause:
              authError instanceof AuthError
                ? authError
                : new AuthError("Failed to refresh Claude OAuth credentials", {
                    code: "AUTH_REFRESH_FAILED",
                    cause: authError,
                  }),
          });
        }

        return this.createMessage(options, true);
      }

      if (isAuthenticationFailure(error) && this.auth?.kind === "oauth") {
        throw new LLMError("Failed to complete Anthropic request", {
          cause: new AuthError("Claude OAuth authentication failed", {
            code: "AUTH_REFRESH_FAILED",
            cause: error,
          }),
        });
      }

      if (error instanceof ConfigError || error instanceof AuthError) {
        throw error;
      }

      throw new LLMError("Failed to complete Anthropic request", {
        cause: error,
      });
    }
  }

  complete(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    return this.createMessage(options);
  }
}

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
