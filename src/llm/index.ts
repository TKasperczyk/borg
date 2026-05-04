import Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlockParam,
  Message,
  MessageParam,
  TextBlock,
  TextBlockParam,
  ThinkingConfigParam,
  Tool,
  ToolChoice,
  ToolResultBlockParam,
  ToolUseBlock,
  ToolUseBlockParam,
} from "@anthropic-ai/sdk/resources/messages/messages.js";
import { z } from "zod";

import { getFreshCredentials, type GetFreshCredentialsOptions } from "../auth/claude-oauth.js";
import type { Clock } from "../util/clock.js";
import { AuthError, ConfigError, LLMError } from "../util/errors.js";
import { getModelMaxOutputTokens } from "./max-tokens.js";

const OAUTH_BETAS = "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14";
const OAUTH_USER_AGENT = "claude-cli/2.1.2 (external, cli)";

// Required as the first system block in OAuth mode. The Anthropic OAuth beta
// endpoint validates identity before serving responses; anything short of
// this exact string (ASCII apostrophe U+0027) trips the validator.
export const CLAUDE_CODE_IDENTITY_BLOCK_TEXT =
  "You are Claude Code, Anthropic's official CLI for Claude.";

export type LLMMessage = {
  role: "user" | "assistant";
  content: string;
};

export type LLMTextBlock = {
  type: "text";
  text: string;
};

export type LLMToolUseBlock = {
  type: "tool_use";
  id: string;
  name: string;
  input: unknown;
};

export type LLMToolResultBlock = {
  type: "tool_result";
  tool_use_id: string;
  content: string | readonly LLMTextBlock[];
  is_error?: boolean;
};

export type LLMContentBlock = LLMTextBlock | LLMToolUseBlock | LLMToolResultBlock;

export type LLMContentBlockMessage = {
  role: "user" | "assistant";
  content: readonly LLMContentBlock[];
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
  const jsonSchema = z.toJSONSchema(schema, {
    io: "input",
    unrepresentable: "any",
  });

  if (jsonSchema.type !== "object") {
    throw new TypeError("Tool input schema must serialize to a top-level object schema");
  }

  return jsonSchema as LLMToolDefinition["inputSchema"];
}

type LLMCallOptions = {
  model: string;
  // If callers embed retrieved memory or other user-derived records into
  // `system`, delimit those blocks explicitly and label them as untrusted
  // data rather than concatenating free-form text that looks like policy.
  system?: string | readonly LLMSystemBlock[];
  tools?: readonly LLMToolDefinition[];
  tool_choice?: { type: "tool"; name: string } | { type: "any" } | { type: "auto" };
  max_tokens?: number;
  temperature?: number;
  thinking?: ThinkingConfigParam;
  budget: string;
};

export type LLMCompleteOptions = LLMCallOptions & {
  messages: readonly LLMMessage[];
};

export type LLMCompleteResult = {
  text: string;
  input_tokens: number;
  output_tokens: number;
  stop_reason: string | null;
  tool_calls: LLMToolCall[];
};

export type LLMConverseOptions = LLMCallOptions & {
  messages: readonly LLMContentBlockMessage[];
};

export type LLMConverseResult = {
  messageBlocks: LLMContentBlock[];
  input_tokens: number;
  output_tokens: number;
  stop_reason: string | null;
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
  converse(options: LLMConverseOptions): Promise<LLMConverseResult>;
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
      temperature?: number;
      thinking?: ThinkingConfigParam;
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

type FakeLLMResponseValue =
  | string
  | readonly LLMContentBlock[]
  | LLMCompleteResult
  | LLMConverseResult;

export type FakeLLMResponse =
  | FakeLLMResponseValue
  | ((options: LLMCompleteOptions) => FakeLLMResponseValue | Promise<FakeLLMResponseValue>)
  | ((options: LLMConverseOptions) => FakeLLMResponseValue | Promise<FakeLLMResponseValue>);

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

function toAnthropicToolResultContent(
  content: LLMToolResultBlock["content"],
): ToolResultBlockParam["content"] {
  if (typeof content === "string") {
    return content;
  }

  return content.map((block) => ({
    type: "text",
    text: block.text,
  }));
}

function toAnthropicContentBlock(block: LLMContentBlock): ContentBlockParam {
  if (block.type === "text") {
    return {
      type: "text",
      text: block.text,
    } satisfies TextBlockParam;
  }

  if (block.type === "tool_use") {
    return {
      type: "tool_use",
      id: block.id,
      name: block.name,
      input: block.input,
    } satisfies ToolUseBlockParam;
  }

  return {
    type: "tool_result",
    tool_use_id: block.tool_use_id,
    content: toAnthropicToolResultContent(block.content),
    ...(block.is_error === undefined ? {} : { is_error: block.is_error }),
  } satisfies ToolResultBlockParam;
}

function toAnthropicConversationMessages(
  messages: readonly LLMContentBlockMessage[],
): MessageParam[] {
  return messages.map((message) => ({
    role: message.role,
    content: message.content.map((block) => toAnthropicContentBlock(block)),
  }));
}

function toAnthropicTools(tools: readonly LLMToolDefinition[] | undefined): Tool[] | undefined {
  return tools?.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema,
  }));
}

function toAnthropicToolChoice(toolChoice: LLMCallOptions["tool_choice"]): ToolChoice | undefined {
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

function extractMessageBlocks(message: Message): LLMContentBlock[] {
  const blocks: LLMContentBlock[] = [];

  for (const block of message.content) {
    if (isTextBlock(block)) {
      blocks.push({
        type: "text",
        text: block.text,
      });
      continue;
    }

    if (isToolUseBlock(block)) {
      blocks.push({
        type: "tool_use",
        id: block.id,
        name: block.name,
        input: block.input,
      });
    }
  }

  return blocks;
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

  const normalized = name.replace(/[^A-Za-z0-9_]/g, "_");

  return normalized.charAt(0).toUpperCase() + normalized.slice(1);
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

type RequestBodyInit = NonNullable<RequestInit["body"]>;

function requestHasBody(request: Request): boolean {
  const method = request.method.toUpperCase();
  return method !== "GET" && method !== "HEAD" && request.body !== null;
}

async function requestToInit(
  request: Request,
  bodyOverride?: RequestBodyInit,
): Promise<RequestInit> {
  return {
    method: request.method,
    headers: new Headers(request.headers),
    body:
      bodyOverride ?? (requestHasBody(request) ? await request.clone().arrayBuffer() : undefined),
    credentials: request.credentials,
    cache: request.cache,
    redirect: request.redirect,
    referrer: request.referrer,
    referrerPolicy: request.referrerPolicy,
    integrity: request.integrity,
    keepalive: request.keepalive,
    mode: request.mode,
    signal: request.signal,
  };
}

function withBodyAndFreshLength(init: RequestInit, body: RequestBodyInit): RequestInit {
  const headers = new Headers(init.headers);
  headers.delete("content-length");

  return {
    ...init,
    headers,
    body,
  };
}

export function createOAuthFetch(): typeof fetch {
  return async (
    input: Parameters<typeof fetch>[0],
    init?: Parameters<typeof fetch>[1],
  ): Promise<Response> => {
    const inputRequest = input instanceof Request ? new Request(input, init) : null;
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

    let modifiedInit = inputRequest === null ? init : await requestToInit(inputRequest);
    const originalNamesByTransformed = new Map<string, string>();
    const requestBody =
      inputRequest !== null && isMessagesRequest && requestHasBody(inputRequest)
        ? await inputRequest.clone().text()
        : undefined;
    const bodyToTransform =
      requestBody ?? (typeof modifiedInit?.body === "string" ? modifiedInit.body : undefined);

    if (isMessagesRequest && bodyToTransform !== undefined && bodyToTransform.length > 0) {
      try {
        const parsed = JSON.parse(bodyToTransform) as Record<string, unknown>;
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
          modifiedInit = withBodyAndFreshLength(modifiedInit ?? {}, JSON.stringify(parsed));
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

function isOpusModel(model: string): boolean {
  return /^claude-opus-4(?:[-._].+)?$/i.test(model.trim());
}

function resolveMaxTokens(options: Pick<LLMCallOptions, "max_tokens" | "model">): number {
  return options.max_tokens ?? getModelMaxOutputTokens(options.model);
}

function shouldOmitTemperature(model: string): boolean {
  return isOpusModel(model);
}

function shouldOmitThinking(
  auth: ResolvedAnthropicAuth | undefined,
  options: Pick<LLMCallOptions, "model" | "tool_choice">,
): boolean {
  if (isOpusModel(options.model)) {
    return true;
  }

  return auth?.kind === "oauth" && options.tool_choice?.type === "tool";
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

    if (this.initialization === undefined) {
      const initialization = (async () => {
        this.auth = await resolveAnthropicAuth(this.options);
        this.client = buildAnthropicClient(this.auth);
      })();
      this.initialization = initialization;
    }

    const initialization = this.initialization;

    try {
      await initialization;
    } catch (error) {
      if (this.initialization === initialization) {
        this.initialization = undefined;
      }
      throw error;
    }
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

  private async createRawMessage(
    options: LLMCallOptions,
    messages: MessageParam[],
    retrying = false,
  ): Promise<Message> {
    await this.ensureInitialized();

    const client = this.client;

    if (client === undefined) {
      throw new LLMError("Anthropic client failed to initialize");
    }

    try {
      return await client.messages.create({
        model: options.model,
        system: this.resolveSystemPrompt(options.system),
        messages,
        tools: toAnthropicTools(options.tools),
        tool_choice: toAnthropicToolChoice(options.tool_choice),
        max_tokens: resolveMaxTokens(options),
        ...(options.temperature !== undefined && !shouldOmitTemperature(options.model)
          ? { temperature: options.temperature }
          : {}),
        ...(options.thinking !== undefined && !shouldOmitThinking(this.auth, options)
          ? { thinking: options.thinking }
          : {}),
      });
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

        return this.createRawMessage(options, messages, true);
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

  private async emitUsage(
    options: Pick<LLMCallOptions, "budget" | "model">,
    result: Pick<LLMCompleteResult, "input_tokens" | "output_tokens">,
  ): Promise<void> {
    if (this.usageSink === undefined) {
      return;
    }

    await this.usageSink({
      budget: options.budget,
      model: options.model,
      input_tokens: result.input_tokens,
      output_tokens: result.output_tokens,
    });
  }

  private async createMessage(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    const response = await this.createRawMessage(options, toAnthropicMessages(options.messages));
    const result = {
      text: extractText(response),
      input_tokens: response.usage.input_tokens,
      output_tokens: response.usage.output_tokens,
      stop_reason: response.stop_reason,
      tool_calls: extractToolCalls(response),
    } satisfies LLMCompleteResult;
    await this.emitUsage(options, result);
    return result;
  }

  private async createConversation(options: LLMConverseOptions): Promise<LLMConverseResult> {
    const response = await this.createRawMessage(
      options,
      toAnthropicConversationMessages(options.messages),
    );
    const result = {
      messageBlocks: extractMessageBlocks(response),
      input_tokens: response.usage.input_tokens,
      output_tokens: response.usage.output_tokens,
      stop_reason: response.stop_reason,
    } satisfies LLMConverseResult;
    await this.emitUsage(options, result);
    return result;
  }

  complete(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    return this.createMessage(options);
  }

  converse(options: LLMConverseOptions): Promise<LLMConverseResult> {
    return this.createConversation(options);
  }
}

function inferStopReasonFromBlocks(blocks: readonly LLMContentBlock[]): string | null {
  return blocks.some((block) => block.type === "tool_use") ? "tool_use" : "end_turn";
}

function blocksFromCompleteResult(result: LLMCompleteResult): LLMContentBlock[] {
  const blocks: LLMContentBlock[] = [];

  if (result.text.length > 0) {
    blocks.push({
      type: "text",
      text: result.text,
    });
  }

  for (const call of result.tool_calls) {
    blocks.push({
      type: "tool_use",
      id: call.id,
      name: call.name,
      input: call.input,
    });
  }

  return blocks;
}

function isFakeBlockArray(response: FakeLLMResponseValue): response is readonly LLMContentBlock[] {
  return Array.isArray(response);
}

function isFakeConverseResult(response: FakeLLMResponseValue): response is LLMConverseResult {
  return typeof response === "object" && response !== null && "messageBlocks" in response;
}

function normalizeFakeConverseResponse(response: FakeLLMResponseValue): LLMConverseResult {
  if (typeof response === "string") {
    return {
      messageBlocks: [
        {
          type: "text",
          text: response,
        },
      ],
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "end_turn",
    };
  }

  if (isFakeBlockArray(response)) {
    return {
      messageBlocks: [...response],
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: inferStopReasonFromBlocks(response),
    };
  }

  if (isFakeConverseResult(response)) {
    return response;
  }

  return {
    messageBlocks: blocksFromCompleteResult(response),
    input_tokens: response.input_tokens,
    output_tokens: response.output_tokens,
    stop_reason: response.stop_reason,
  };
}

function normalizeFakeCompleteResponse(response: FakeLLMResponseValue): LLMCompleteResult {
  if (typeof response === "string") {
    return {
      text: response,
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: "end_turn",
      tool_calls: [],
    };
  }

  if (isFakeBlockArray(response)) {
    return {
      text: response
        .filter((block): block is LLMTextBlock => block.type === "text")
        .map((block) => block.text)
        .join(""),
      input_tokens: 0,
      output_tokens: 0,
      stop_reason: inferStopReasonFromBlocks(response),
      tool_calls: response
        .filter((block): block is LLMToolUseBlock => block.type === "tool_use")
        .map((block) => ({
          id: block.id,
          name: block.name,
          input: block.input,
        })),
    };
  }

  if (isFakeConverseResult(response)) {
    return normalizeFakeCompleteResponse(response.messageBlocks);
  }

  return response;
}

function flattenBlockContentForCompatibility(content: LLMToolResultBlock["content"]): string {
  if (typeof content === "string") {
    return content;
  }

  return content.map((block) => block.text).join("");
}

function flattenMessageBlocksForCompatibility(blocks: readonly LLMContentBlock[]): string {
  return blocks
    .map((block) => {
      if (block.type === "text") {
        return block.text;
      }

      if (block.type === "tool_use") {
        return `[tool_use ${block.name}]`;
      }

      return flattenBlockContentForCompatibility(block.content);
    })
    .join("");
}

function toCompleteCompatibleRequest(options: LLMConverseOptions): LLMCompleteOptions {
  return {
    ...options,
    messages: options.messages.map((message) => ({
      role: message.role,
      content: flattenMessageBlocksForCompatibility(message.content),
    })),
  };
}

function isProceduralContextFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "procedural-context";
}

function isStopCommitmentFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "generation-stop-commitment";
}

function isPendingActionJudgeFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "pending-action-judge";
}

function isCorrectivePreferenceFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "corrective-preference-extractor";
}

function isGoalPromotionFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "goal-promotion-extractor";
}

function isActionStateExtractorFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "action-state-extractor";
}

function isRelationalClaimAuditorFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "relational-claim-auditor";
}

function isRecallExpansionFallbackRequest(options: LLMCompleteOptions): boolean {
  return options.budget === "recall-expansion";
}

function scriptedResponseBudget(response: FakeLLMResponse | undefined): string | undefined {
  if (typeof response !== "function") {
    return undefined;
  }

  const budget = (response as { budget?: unknown }).budget;

  return typeof budget === "string" ? budget : undefined;
}

function isProceduralContextResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitProceduralContext");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitProceduralContext",
    );
  }

  return false;
}

function isStopCommitmentResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some(
      (toolCall) => toolCall.name === "EmitStopCommitmentClassification",
    );
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitStopCommitmentClassification",
    );
  }

  return false;
}

function isPendingActionJudgeResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "ClassifyPendingAction");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "ClassifyPendingAction",
    );
  }

  return false;
}

function isCorrectivePreferenceResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitCorrectivePreference");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitCorrectivePreference",
    );
  }

  return false;
}

function isGoalPromotionResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitGoalPromotion");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitGoalPromotion",
    );
  }

  return false;
}

function isActionStateResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitActionStates");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitActionStates",
    );
  }

  return false;
}

function isRelationalClaimAuditResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitClaimAudit");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitClaimAudit",
    );
  }

  return false;
}

function isRecallExpansionResponse(response: FakeLLMResponse | undefined): boolean {
  if (response === undefined || typeof response === "function" || typeof response !== "object") {
    return false;
  }

  if ("tool_calls" in response) {
    return response.tool_calls.some((toolCall) => toolCall.name === "EmitRecallExpansion");
  }

  if ("messageBlocks" in response) {
    return response.messageBlocks.some(
      (block) => block.type === "tool_use" && block.name === "EmitRecallExpansion",
    );
  }

  return false;
}

function defaultProceduralContextResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_procedural_context",
        name: "EmitProceduralContext",
        input: {
          problem_kind: "other",
          domain_tags: [],
          confidence: 0,
        },
      },
    ],
  };
}

function defaultStopCommitmentResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_stop_commitment",
        name: "EmitStopCommitmentClassification",
        input: {
          classification: "none",
          directive_family: null,
          reason: "No operational no-output commitment.",
          confidence: 0,
        },
      },
    ],
  };
}

function defaultPendingActionJudgeResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_pending_action_judge",
        name: "ClassifyPendingAction",
        input: {
          classification: "action",
          reason: "Accepted by test fallback.",
          confidence: 1,
        },
      },
    ],
  };
}

function defaultCorrectivePreferenceResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_corrective_preference",
        name: "EmitCorrectivePreference",
        input: {
          classification: "none",
          type: null,
          directive: null,
          directive_family: null,
          priority: null,
          reason: "No durable correction detected.",
          confidence: 0,
          supersedes_commitment_id: null,
          slot_negations: [],
        },
      },
    ],
  };
}

function defaultGoalPromotionResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_goal_promotion",
        name: "EmitGoalPromotion",
        input: {
          promotions: [],
        },
      },
    ],
  };
}

function defaultActionStateResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_action_state",
        name: "EmitActionStates",
        input: {
          action_states: [],
        },
      },
    ],
  };
}

function defaultRelationalClaimAuditResponse(): LLMCompleteResult {
  return {
    text: "",
    input_tokens: 0,
    output_tokens: 0,
    stop_reason: "tool_use",
    tool_calls: [
      {
        id: "toolu_default_relational_claim_audit",
        name: "EmitClaimAudit",
        input: {
          claims: [],
        },
      },
    ],
  };
}

export class FakeLLMClient implements LLMClient {
  private readonly usageSink?: TokenUsageSink;
  readonly requests: LLMCompleteOptions[] = [];
  readonly converseRequests: LLMConverseOptions[] = [];
  private readonly responses: FakeLLMResponse[];

  constructor(options: FakeLLMClientOptions = {}) {
    this.responses = [...(options.responses ?? [])];
    this.usageSink = options.usageSink;
  }

  pushResponse(response: FakeLLMResponse): void {
    this.responses.push(response);
  }

  async complete(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    const response = this.responses[0];

    if (
      isRecallExpansionFallbackRequest(options) &&
      typeof response !== "function" &&
      !isRecallExpansionResponse(response)
    ) {
      throw new LLMError("FakeLLMClient has no scripted recall expansion response available");
    }

    this.requests.push(options);

    if (
      isStopCommitmentFallbackRequest(options) &&
      typeof response !== "function" &&
      !isStopCommitmentResponse(response)
    ) {
      return defaultStopCommitmentResponse();
    }

    if (isProceduralContextFallbackRequest(options) && !isProceduralContextResponse(response)) {
      return defaultProceduralContextResponse();
    }

    if (
      isPendingActionJudgeFallbackRequest(options) &&
      typeof response !== "function" &&
      !isPendingActionJudgeResponse(response)
    ) {
      return defaultPendingActionJudgeResponse();
    }

    if (
      isCorrectivePreferenceFallbackRequest(options) &&
      !isCorrectivePreferenceResponse(response)
    ) {
      return defaultCorrectivePreferenceResponse();
    }

    if (
      isGoalPromotionFallbackRequest(options) &&
      typeof response !== "function" &&
      !isGoalPromotionResponse(response)
    ) {
      return defaultGoalPromotionResponse();
    }

    if (
      isActionStateExtractorFallbackRequest(options) &&
      scriptedResponseBudget(response) !== "action-state-extractor" &&
      !isActionStateResponse(response)
    ) {
      return defaultActionStateResponse();
    }

    if (
      isRelationalClaimAuditorFallbackRequest(options) &&
      !isRelationalClaimAuditResponse(response)
    ) {
      return defaultRelationalClaimAuditResponse();
    }

    this.responses.shift();

    if (response === undefined) {
      throw new LLMError("FakeLLMClient has no scripted response available");
    }

    const resolved =
      typeof response === "function"
        ? await (
            response as (
              options: LLMCompleteOptions,
            ) => FakeLLMResponseValue | Promise<FakeLLMResponseValue>
          )(options)
        : response;
    const normalized = normalizeFakeCompleteResponse(resolved);

    if (this.usageSink !== undefined) {
      await this.usageSink({
        budget: options.budget,
        model: options.model,
        input_tokens: normalized.input_tokens,
        output_tokens: normalized.output_tokens,
      });
    }

    return normalized;
  }

  async converse(options: LLMConverseOptions): Promise<LLMConverseResult> {
    this.converseRequests.push(options);
    this.requests.push(toCompleteCompatibleRequest(options));
    const response = this.responses.shift();

    if (response === undefined) {
      throw new LLMError("FakeLLMClient has no scripted response available");
    }

    const resolved =
      typeof response === "function"
        ? await (
            response as (
              options: LLMConverseOptions,
            ) => FakeLLMResponseValue | Promise<FakeLLMResponseValue>
          )(options)
        : response;
    const normalized = normalizeFakeConverseResponse(resolved);

    if (this.usageSink !== undefined) {
      await this.usageSink({
        budget: options.budget,
        model: options.model,
        input_tokens: normalized.input_tokens,
        output_tokens: normalized.output_tokens,
      });
    }

    return normalized;
  }
}
