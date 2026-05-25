import Anthropic from "@anthropic-ai/sdk";
import type {
  ContentBlockParam,
  JSONOutputFormat,
  Message,
  MessageParam,
  OutputConfig,
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
import type { AttachmentId } from "../util/ids.js";
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

export type LLMImageRefBlock = {
  type: "image_ref";
  attachment_id: AttachmentId;
};

export type LLMContentBlock =
  | LLMTextBlock
  | LLMImageRefBlock
  | LLMToolUseBlock
  | LLMToolResultBlock;

export type LLMContentBlockMessage = {
  role: "user" | "assistant";
  content: readonly LLMContentBlock[];
};

// Anthropic prompt caching: a content block carrying cache_control marks the
// end of a cacheable prefix that includes that block. Sprint 8d.6.4 adds the
// plumbing; Sprint 8d.6.5 places the breakpoints.
export type LLMCacheControl = {
  type: "ephemeral";
  ttl?: "5m" | "1h";
};

export type LLMSystemBlock = {
  type: "text";
  text: string;
  cache_control?: LLMCacheControl;
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
  cache_control?: LLMCacheControl;
};

export type LLMToolCall = {
  id: string;
  name: string;
  input: unknown;
};

export class LLMStructuredOutputParseError extends LLMError {
  readonly rawText: string;

  constructor(rawText: string, cause: unknown) {
    super("Failed to parse Anthropic structured output", {
      cause,
      code: "LLM_STRUCTURED_OUTPUT_PARSE_FAILED",
    });
    this.rawText = rawText;
  }
}

export type LLMOutputConfig = {
  format: JSONOutputFormat;
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

function isJsonSchemaObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function normalizeStructuredJsonSchema(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeStructuredJsonSchema(entry));
  }

  if (!isJsonSchemaObject(value)) {
    return value;
  }

  const normalized: Record<string, unknown> = {};

  for (const [key, entry] of Object.entries(value)) {
    if (key === "$schema") {
      continue;
    }

    normalized[key === "oneOf" ? "anyOf" : key] = normalizeStructuredJsonSchema(entry);
  }

  return normalized;
}

// Keep value constraints machine-enforced. The SDK's Zod helper currently
// rewrites some literal/enum constraints into descriptions while preparing a
// stricter schema shape, so Borg performs only the small normalizations needed
// for Anthropic structured outputs and leaves const/enum intact.
export function toStructuredOutputFormat(schema: z.ZodType): JSONOutputFormat {
  const jsonSchema = z.toJSONSchema(schema, {
    io: "output",
    unrepresentable: "any",
  });

  if (jsonSchema.type !== "object") {
    throw new TypeError("Structured output schema must serialize to a top-level object schema");
  }

  return {
    type: "json_schema",
    schema: normalizeStructuredJsonSchema(jsonSchema) as JSONOutputFormat["schema"],
  };
}

type LLMCallOptions = {
  model: string;
  // If callers embed retrieved memory or other user-derived records into
  // `system`, delimit those blocks explicitly and label them as untrusted
  // data rather than concatenating free-form text that looks like policy.
  system?: string | readonly LLMSystemBlock[];
  tools?: readonly LLMToolDefinition[];
  tool_choice?: { type: "tool"; name: string } | { type: "any" } | { type: "auto" };
  output_config?: LLMOutputConfig;
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
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
  stop_reason: string | null;
  tool_calls: LLMToolCall[];
  structured_output?: unknown;
};

export type LLMConverseOptions = LLMCallOptions & {
  messages: readonly LLMContentBlockMessage[];
};

export type LLMConverseResult = {
  messageBlocks: LLMContentBlock[];
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
  stop_reason: string | null;
  structured_output?: unknown;
};

export type TokenUsageEvent = {
  budget: string;
  model: string;
  input_tokens: number;
  output_tokens: number;
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
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
      output_config?: OutputConfig;
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
  attachmentResolver?: (attachmentId: AttachmentId) => {
    mediaType: string;
    bytes: Buffer | Uint8Array;
  };
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

function toAnthropicContentBlock(
  block: LLMContentBlock,
  attachmentResolver: AnthropicLLMClientOptions["attachmentResolver"] | undefined,
): ContentBlockParam {
  if (block.type === "text") {
    return {
      type: "text",
      text: block.text,
    } satisfies TextBlockParam;
  }

  if (block.type === "image_ref") {
    if (attachmentResolver === undefined) {
      throw new LLMError(`No attachment resolver configured for ${block.attachment_id}`, {
        code: "LLM_ATTACHMENT_RESOLVER_MISSING",
      });
    }

    const image = attachmentResolver(block.attachment_id);

    return {
      type: "image",
      source: {
        type: "base64",
        media_type: image.mediaType,
        data: Buffer.from(image.bytes).toString("base64"),
      },
    } as ContentBlockParam;
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
  attachmentResolver: AnthropicLLMClientOptions["attachmentResolver"] | undefined,
): MessageParam[] {
  return messages.map((message) => ({
    role: message.role,
    content: message.content.map((block) => toAnthropicContentBlock(block, attachmentResolver)),
  }));
}

function toAnthropicTools(tools: readonly LLMToolDefinition[] | undefined): Tool[] | undefined {
  return tools?.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: tool.inputSchema,
    ...(tool.cache_control === undefined ? {} : { cache_control: tool.cache_control }),
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

function extractCacheUsage(message: Message): {
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
} {
  // Anthropic surfaces prompt-cache accounting in usage. Both fields are
  // optional in the SDK type and absent when caching is unused.
  const usage = message.usage as {
    cache_creation_input_tokens?: number | null;
    cache_read_input_tokens?: number | null;
  };
  const out: { cache_creation_input_tokens?: number; cache_read_input_tokens?: number } = {};
  if (typeof usage.cache_creation_input_tokens === "number") {
    out.cache_creation_input_tokens = usage.cache_creation_input_tokens;
  }
  if (typeof usage.cache_read_input_tokens === "number") {
    out.cache_read_input_tokens = usage.cache_read_input_tokens;
  }
  return out;
}

function extractText(message: Message): string {
  return message.content
    .filter(isTextBlock)
    .map((block) => block.text)
    .join("");
}

function parseStructuredOutputText(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch (error) {
    throw new LLMStructuredOutputParseError(text, error);
  }
}

function extractStructuredOutput(
  message: Message,
  outputConfig: LLMCallOptions["output_config"],
): unknown {
  if (outputConfig?.format === undefined) {
    return undefined;
  }

  const parsedOutput = (message as Message & { parsed_output?: unknown }).parsed_output;

  if (parsedOutput !== undefined && parsedOutput !== null) {
    return parsedOutput;
  }

  return parseStructuredOutputText(extractText(message));
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

function transformToolNameForOAuthWithMap(
  name: string,
  originalNamesByTransformed: Map<string, string>,
): string {
  const transformed = transformToolNameForOAuth(name);

  if (transformed !== name) {
    originalNamesByTransformed.set(transformed, name);
  }

  return transformed;
}

function mutateOutboundMessageToolUseNames(
  messages: unknown,
  originalNamesByTransformed: Map<string, string>,
): boolean {
  if (!Array.isArray(messages)) {
    return false;
  }

  let changed = false;

  for (const message of messages) {
    if (message === null || typeof message !== "object") {
      continue;
    }

    const content = (message as { content?: unknown }).content;

    if (!Array.isArray(content)) {
      continue;
    }

    for (const block of content) {
      if (block === null || typeof block !== "object") {
        continue;
      }

      const record = block as Record<string, unknown>;

      if (record.type !== "tool_use" || typeof record.name !== "string") {
        continue;
      }

      const transformedName = transformToolNameForOAuthWithMap(
        record.name,
        originalNamesByTransformed,
      );

      if (transformedName !== record.name) {
        record.name = transformedName;
        changed = true;
      }
    }
  }

  return changed;
}

function oauthTransportToolDefinitions(
  tools: readonly LLMToolDefinition[] | undefined,
): readonly LLMToolDefinition[] | undefined {
  return tools?.map((tool) => {
    const transformedName = transformToolNameForOAuth(tool.name);

    if (transformedName === tool.name) {
      return tool;
    }

    return {
      ...tool,
      name: transformedName,
    };
  });
}

function oauthTransportToolChoice(
  toolChoice: LLMCallOptions["tool_choice"],
): LLMCallOptions["tool_choice"] {
  if (toolChoice?.type !== "tool") {
    return toolChoice;
  }

  const transformedName = transformToolNameForOAuth(toolChoice.name);

  if (transformedName === toolChoice.name) {
    return toolChoice;
  }

  return {
    ...toolChoice,
    name: transformedName,
  };
}

function oauthTransportContentBlock(block: LLMContentBlock): LLMContentBlock {
  if (block.type !== "tool_use") {
    return block;
  }

  const transformedName = transformToolNameForOAuth(block.name);

  if (transformedName === block.name) {
    return block;
  }

  return {
    ...block,
    name: transformedName,
  };
}

function oauthTransportConverseOptions(options: LLMConverseOptions): LLMConverseOptions {
  return {
    ...options,
    tools: oauthTransportToolDefinitions(options.tools),
    tool_choice: oauthTransportToolChoice(options.tool_choice),
    messages: options.messages.map((message) => ({
      role: message.role,
      content: message.content.map((block) => oauthTransportContentBlock(block)),
    })),
  };
}

function oauthTransportCompleteOptions(options: LLMCompleteOptions): LLMCompleteOptions {
  return {
    ...options,
    tools: oauthTransportToolDefinitions(options.tools),
    tool_choice: oauthTransportToolChoice(options.tool_choice),
  };
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

            const transformedName = transformToolNameForOAuthWithMap(
              record.name,
              originalNamesByTransformed,
            );

            if (transformedName !== record.name) {
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
          const transformedName = transformToolNameForOAuthWithMap(
            toolChoice.name as string,
            originalNamesByTransformed,
          );

          if (transformedName !== toolChoice.name) {
            parsed.tool_choice = {
              ...toolChoice,
              name: transformedName,
            };
            modified = true;
          }
        }

        if (mutateOutboundMessageToolUseNames(parsed.messages, originalNamesByTransformed)) {
          modified = true;
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
    ...(block.cache_control === undefined ? {} : { cache_control: block.cache_control }),
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

function buildAnthropicClient(
  auth: ResolvedAnthropicAuth,
  env: NodeJS.ProcessEnv = process.env,
): AnthropicClientLike {
  const baseURL = env.ANTHROPIC_BASE_URL?.trim() || undefined;

  if (auth.kind === "api-key") {
    return new Anthropic({
      apiKey: auth.apiKey,
      ...(baseURL ? { baseURL } : {}),
    });
  }

  return new Anthropic({
    authToken: auth.authToken,
    defaultHeaders: {
      "anthropic-beta": OAUTH_BETAS,
      "user-agent": OAUTH_USER_AGENT,
    },
    fetch: createOAuthFetch(),
    ...(baseURL ? { baseURL } : {}),
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
        this.client = buildAnthropicClient(this.auth, this.options.env);
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
    this.client = buildAnthropicClient(this.auth, this.options.env);
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
        ...(options.output_config === undefined ? {} : { output_config: options.output_config }),
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
    result: Pick<
      LLMCompleteResult,
      "input_tokens" | "output_tokens" | "cache_creation_input_tokens" | "cache_read_input_tokens"
    >,
  ): Promise<void> {
    if (this.usageSink === undefined) {
      return;
    }

    await this.usageSink({
      budget: options.budget,
      model: options.model,
      input_tokens: result.input_tokens,
      output_tokens: result.output_tokens,
      ...(result.cache_creation_input_tokens === undefined
        ? {}
        : { cache_creation_input_tokens: result.cache_creation_input_tokens }),
      ...(result.cache_read_input_tokens === undefined
        ? {}
        : { cache_read_input_tokens: result.cache_read_input_tokens }),
    });
  }

  private async createMessage(options: LLMCompleteOptions): Promise<LLMCompleteResult> {
    const response = await this.createRawMessage(options, toAnthropicMessages(options.messages));
    let structuredOutput: unknown;

    try {
      structuredOutput = extractStructuredOutput(response, options.output_config);
    } catch (error) {
      if (error instanceof LLMStructuredOutputParseError) {
        throw error;
      }

      throw new LLMError("Failed to parse Anthropic structured output", {
        cause: error,
        code: "LLM_STRUCTURED_OUTPUT_PARSE_FAILED",
      });
    }

    const result = {
      text: extractText(response),
      input_tokens: response.usage.input_tokens,
      output_tokens: response.usage.output_tokens,
      ...extractCacheUsage(response),
      stop_reason: response.stop_reason,
      tool_calls: extractToolCalls(response),
      ...(options.output_config === undefined ? {} : { structured_output: structuredOutput }),
    } satisfies LLMCompleteResult;
    await this.emitUsage(options, result);
    return result;
  }

  private async createConversation(options: LLMConverseOptions): Promise<LLMConverseResult> {
    const response = await this.createRawMessage(
      options,
      toAnthropicConversationMessages(options.messages, this.options.attachmentResolver),
    );
    let structuredOutput: unknown;

    try {
      structuredOutput = extractStructuredOutput(response, options.output_config);
    } catch (error) {
      if (error instanceof LLMStructuredOutputParseError) {
        throw error;
      }

      throw new LLMError("Failed to parse Anthropic structured output", {
        cause: error,
        code: "LLM_STRUCTURED_OUTPUT_PARSE_FAILED",
      });
    }

    const result = {
      messageBlocks: extractMessageBlocks(response),
      input_tokens: response.usage.input_tokens,
      output_tokens: response.usage.output_tokens,
      ...extractCacheUsage(response),
      stop_reason: response.stop_reason,
      ...(options.output_config === undefined ? {} : { structured_output: structuredOutput }),
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
