import Anthropic from "@anthropic-ai/sdk";
import type {
  JSONOutputFormat,
  Message,
  MessageParam,
  MessageStreamEvent,
  OutputConfig,
  TextBlock,
  TextBlockParam,
  ThinkingConfigParam,
  Tool,
  ToolChoice,
  ToolUseBlock,
} from "@anthropic-ai/sdk/resources/messages/messages.js";
import { z } from "zod";

import { getFreshCredentials, type GetFreshCredentialsOptions } from "../auth/claude-oauth.js";
import type { Clock } from "../util/clock.js";
import { AuthError, ConfigError, findInErrorCauseChain, LLMError } from "../util/errors.js";
import type { AttachmentId } from "../util/ids.js";
import { toAnthropicContentBlockMessages } from "./anthropic-content-blocks.js";
import { clampMaxOutputTokens, getModelMaxOutputTokens } from "./max-tokens.js";

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

// Extended/adaptive thinking blocks. We carry them verbatim (text may be empty
// under display:"omitted"; the signature holds the encrypted full reasoning) so
// they can be round-tripped unchanged in a multi-iteration tool loop -- the API
// requires the preceding thinking block to accompany a tool_use turn when
// thinking is active.
export type LLMThinkingBlock = {
  type: "thinking";
  thinking: string;
  signature: string;
};

export type LLMRedactedThinkingBlock = {
  type: "redacted_thinking";
  data: string;
};

export type LLMContentBlock =
  | LLMTextBlock
  | LLMImageRefBlock
  | LLMToolUseBlock
  | LLMToolResultBlock
  | LLMThinkingBlock
  | LLMRedactedThinkingBlock;

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

function isJsonSchemaObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

const UNSUPPORTED_GUIDED_GENERATION_SCHEMA_KEYS = new Set([
  "propertyNames",
  "patternProperties",
  "if",
  "then",
  "else",
  "dependentSchemas",
  "dependentRequired",
]);

const JSON_SCHEMA_MAP_KEYS = new Set(["properties", "$defs", "definitions"]);
const JSON_SCHEMA_CHILD_KEYS = new Set([
  "additionalItems",
  "additionalProperties",
  "contains",
  "contentSchema",
  "items",
  "not",
  "unevaluatedItems",
  "unevaluatedProperties",
]);
const JSON_SCHEMA_CHILD_ARRAY_KEYS = new Set(["allOf", "anyOf", "oneOf", "prefixItems"]);

// Grammar-guided OpenAI-compatible backends commonly accept a smaller JSON
// Schema subset than Zod emits. The client-side Zod parse remains authoritative,
// so omitting these wire-only constraints is a safe compatibility relaxation.
function sanitizeToolInputJsonSchema(value: unknown): unknown {
  if (!isJsonSchemaObject(value)) {
    return value;
  }

  const sanitized: Record<string, unknown> = {};

  for (const [key, entry] of Object.entries(value)) {
    if (UNSUPPORTED_GUIDED_GENERATION_SCHEMA_KEYS.has(key)) {
      continue;
    }

    if (JSON_SCHEMA_MAP_KEYS.has(key) && isJsonSchemaObject(entry)) {
      sanitized[key] = Object.fromEntries(
        Object.entries(entry).map(([name, schema]) => [name, sanitizeToolInputJsonSchema(schema)]),
      );
      continue;
    }

    if (JSON_SCHEMA_CHILD_KEYS.has(key)) {
      sanitized[key] = sanitizeToolInputJsonSchema(entry);
      continue;
    }

    if (JSON_SCHEMA_CHILD_ARRAY_KEYS.has(key) && Array.isArray(entry)) {
      sanitized[key] = entry.map((schema) => sanitizeToolInputJsonSchema(schema));
      continue;
    }

    sanitized[key] = entry;
  }

  return sanitized;
}

export function toToolInputSchema(schema: z.ZodType): LLMToolDefinition["inputSchema"] {
  const jsonSchema = z.toJSONSchema(schema, {
    io: "input",
    unrepresentable: "any",
  });

  if (jsonSchema.type !== "object") {
    throw new TypeError("Tool input schema must serialize to a top-level object schema");
  }

  return sanitizeToolInputJsonSchema(jsonSchema) as LLMToolDefinition["inputSchema"];
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
  signal?: AbortSignal | null;
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
  // Adaptive-thinking effort guidance (output_config.effort). Only meaningful
  // when thinking is sent; ignored otherwise.
  effort?: OutputConfig["effort"];
  // When streaming, suppress forwarding of visible text deltas to onTextDelta.
  // Used by the emission-tool protocol (finalizer): the user-facing content lives
  // in the terminal tool's input (streamed via the tool-field extraction), so any
  // loose text the model emits alongside the tool must not reach the live stream.
  suppressRawTextStream?: boolean;
  // Invoked when the transport layer retries this call in place (stalled stream,
  // fetch-layer deadline, connection blip). Observability only -- the call's
  // result is unaffected. Callers use this to trace rescued-vs-failed attempts.
  onTransportRetry?: (event: LLMTransportRetryEvent) => void;
  budget: string;
};

export type LLMTransportRetryEvent = {
  // The attempt number about to run (2 = first retry).
  attempt: number;
  kind: "stall" | "connection";
  // Typed LLM error code that triggered the retry (stall class only).
  code?: string;
  // Transport of the upcoming retry attempt. Streaming calls retry unary.
  retry_transport: "streaming" | "unary";
};

export type LLMStreamTextHandler = (text: string) => void;

// Tools whose primary user-content lives in a specific string field. When
// streaming these tools' inputs, we extract that field's incremental value so
// the UI shows clean answer text rather than raw `{"text":"..."}` JSON. Tools
// not in this map have their entire partial_json forwarded as-is (good for
// surfacing structured CoT-shaped tools like EmitTurnPlan).
const TOOL_STREAM_TEXT_FIELDS: Record<string, string> = {
  EmitAnswer: "text",
  EmitObserve: "reason",
  EmitNoOutput: "reason",
  EmitSelfReport: "text",
};

// Walk a (possibly unclosed) JSON string looking for `"fieldName": "..."` and
// return the value's current contents. Handles \" and \n / \t / \\ escapes.
// Returns the partial string when the closing quote hasn't arrived yet.
function extractPartialStringField(json: string, fieldName: string): string | null {
  const key = `"${fieldName}"`;
  const keyAt = json.indexOf(key);
  if (keyAt === -1) {
    return null;
  }

  let cursor = keyAt + key.length;
  while (cursor < json.length && (json[cursor] === " " || json[cursor] === "\t")) {
    cursor += 1;
  }
  if (json[cursor] !== ":") {
    return null;
  }
  cursor += 1;
  while (
    cursor < json.length &&
    (json[cursor] === " " || json[cursor] === "\t" || json[cursor] === "\n")
  ) {
    cursor += 1;
  }
  if (json[cursor] !== '"') {
    return null;
  }
  cursor += 1;

  let out = "";
  while (cursor < json.length) {
    const c = json[cursor];
    if (c === "\\" && cursor + 1 < json.length) {
      const next = json[cursor + 1];
      if (next === "n") out += "\n";
      else if (next === "t") out += "\t";
      else if (next === "r") out += "\r";
      else if (next === '"') out += '"';
      else if (next === "\\") out += "\\";
      else out += next;
      cursor += 2;
    } else if (c === '"') {
      return out;
    } else {
      out += c;
      cursor += 1;
    }
  }
  return out;
}

export type LLMCompleteOptions = LLMCallOptions & {
  messages: readonly LLMMessage[];
};

export type LLMCompleteStreamOptions = LLMCompleteOptions & {
  onTextDelta?: LLMStreamTextHandler;
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

export type LLMConverseStreamOptions = LLMConverseOptions & {
  onTextDelta?: LLMStreamTextHandler;
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
  streamComplete?(options: LLMCompleteStreamOptions): Promise<LLMCompleteResult>;
  streamConverse?(options: LLMConverseStreamOptions): Promise<LLMConverseResult>;
};

export {
  toAnthropicContentBlock,
  toAnthropicContentBlocks,
  toAnthropicContentBlockMessages,
  type AnthropicAttachmentResolver,
  type AnthropicContentBlockOptions,
} from "./anthropic-content-blocks.js";
export * from "./structured-tool-call.js";
export {
  OpenAICompatibleLLMClient,
  type OpenAICompatibleLLMClientOptions,
  type OpenAIChatCompletionsClient,
} from "./openai-compatible.js";

type AnthropicClientLike = {
  messages: {
    create(
      params: {
        model: string;
        system?: string | TextBlockParam[];
        messages: MessageParam[];
        tools?: Tool[];
        tool_choice?: ToolChoice;
        output_config?: OutputConfig;
        max_tokens: number;
        temperature?: number;
        thinking?: ThinkingConfigParam;
      },
      options?: AnthropicRequestOptions,
    ): Promise<Message>;
    stream?(
      params: {
        model: string;
        system?: string | TextBlockParam[];
        messages: MessageParam[];
        tools?: Tool[];
        tool_choice?: ToolChoice;
        output_config?: OutputConfig;
        max_tokens: number;
        temperature?: number;
        thinking?: ThinkingConfigParam;
      },
      options?: AnthropicRequestOptions,
    ): AsyncIterable<MessageStreamEvent> & {
      finalMessage(): Promise<Message>;
    };
  };
};

type AnthropicRequestOptions = {
  signal?: AbortSignal | null;
  timeout?: number;
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
  oauthSseInactivityTimeoutMs?: number;
  oauthSseFirstMessageEventTimeoutMs?: number;
  oauthSseMessageEventGapTimeoutMs?: number;
  oauthFetchHeadersTimeoutMs?: number;
  oauthUnaryBodyTimeoutMs?: number;
  unaryCallTimeoutMs?: number;
  streamingCallTimeoutMs?: number;
  transportStallMaxRetries?: number;
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
    // Thinking/redacted_thinking are preserved verbatim (incl. signature/data)
    // so a tool-loop iteration can pass the assistant turn back unchanged, as
    // the API requires when thinking is active and the turn contains tool_use.
    if (block.type === "thinking") {
      blocks.push({
        type: "thinking",
        thinking: block.thinking,
        signature: block.signature,
      });
      continue;
    }

    if (block.type === "redacted_thinking") {
      blocks.push({
        type: "redacted_thinking",
        data: block.data,
      });
      continue;
    }

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

// Healthy streams deliver chunks/pings continuously; a fully silent stream is
// the documented Anthropic-side stall signature.
const SSE_INACTIVITY_TIMEOUT_MS = 120_000;
const SSE_FIRST_MESSAGE_EVENT_TIMEOUT_MS = 240_000;
const SSE_MESSAGE_EVENT_GAP_TIMEOUT_MS = 180_000;
const OAUTH_FETCH_HEADERS_TIMEOUT_MS = 120_000;
const OAUTH_UNARY_BODY_TIMEOUT_MS = 120_000;
const LLM_UNARY_CALL_TIMEOUT_MS = 6 * 60_000;
const LLM_STREAMING_CALL_TIMEOUT_MS = 12 * 60_000;
const ANTHROPIC_CONNECTION_MAX_RETRIES = 2;
// Mid-call stalls (watchdog-killed streams, fetch-layer deadlines) are usually
// transient upstream degradation; one fresh attempt rescues most of them.
const ANTHROPIC_STALL_MAX_RETRIES = 1;

type OAuthFetchOptions = {
  sseInactivityTimeoutMs?: number;
  sseFirstMessageEventTimeoutMs?: number;
  sseMessageEventGapTimeoutMs?: number;
  fetchHeadersTimeoutMs?: number;
  unaryBodyTimeoutMs?: number;
};

type SseReadResult = Awaited<ReturnType<ReadableStreamDefaultReader<Uint8Array>["read"]>>;
const LLM_STREAM_STALLED_CODE = "LLM_STREAM_STALLED";
const LLM_STREAM_EVENT_STALLED_CODE = "LLM_STREAM_EVENT_STALLED";
const LLM_CALL_TIMED_OUT_CODE = "LLM_CALL_TIMED_OUT";
const LLM_CONNECTION_FAILED_CODE = "LLM_CONNECTION_FAILED";
const RETRYABLE_LLM_TRANSPORT_ERROR_CODES = new Set<string>([
  LLM_STREAM_STALLED_CODE,
  LLM_STREAM_EVENT_STALLED_CODE,
  LLM_CALL_TIMED_OUT_CODE,
  LLM_CONNECTION_FAILED_CODE,
]);
// The subset of typed transport errors worth a fresh in-call attempt: the call
// died mid-flight (stalled stream or fetch-layer deadline) rather than being a
// settled outcome like LLM_CONNECTION_FAILED, which is itself a retry verdict.
const STALL_CLASS_LLM_ERROR_CODES = new Set<string>([
  LLM_STREAM_STALLED_CODE,
  LLM_STREAM_EVENT_STALLED_CODE,
  LLM_CALL_TIMED_OUT_CODE,
]);

function unrefTimer(timer: ReturnType<typeof setTimeout>): void {
  (timer as ReturnType<typeof setTimeout> & { unref?: () => void }).unref?.();
}

function createSseStalledError(timeoutMs: number): LLMError {
  return new LLMError(`Anthropic SSE stream stalled after ${timeoutMs}ms without a chunk`, {
    code: LLM_STREAM_STALLED_CODE,
  });
}

function createSseEventStalledError(
  bound: "first-message-event" | "message-event-gap",
  timeoutMs: number,
): LLMError {
  const detail =
    bound === "first-message-event" ? "before the first message event" : "between message events";

  return new LLMError(`Anthropic SSE stream stalled for ${timeoutMs}ms ${detail}`, {
    code: LLM_STREAM_EVENT_STALLED_CODE,
  });
}

function createLlmCallTimedOutError(kind: "unary" | "streaming", timeoutMs: number): LLMError {
  return new LLMError(`Anthropic ${kind} LLM call timed out after ${timeoutMs}ms`, {
    code: LLM_CALL_TIMED_OUT_CODE,
  });
}

function createFetchLayerLlmDeadlineError(
  kind: "headers" | "body-read",
  timeoutMs: number,
): LLMError {
  return new LLMError(`Anthropic ${kind} LLM deadline exceeded after ${timeoutMs}ms`, {
    code: LLM_CALL_TIMED_OUT_CODE,
  });
}

function createLlmConnectionFailedError(attempts: number, cause: unknown): LLMError {
  return new LLMError(`Anthropic connection failed after ${attempts} attempts`, {
    code: LLM_CONNECTION_FAILED_CODE,
    cause,
  });
}

function errorConstructorName(error: unknown): string | undefined {
  if ((typeof error !== "object" && typeof error !== "function") || error === null) {
    return undefined;
  }

  return (error as { constructor?: { name?: string } }).constructor?.name;
}

function isRetryableLlmTransportError(error: unknown): error is LLMError {
  return error instanceof LLMError && RETRYABLE_LLM_TRANSPORT_ERROR_CODES.has(error.code);
}

function findRetryableLlmTransportError(error: unknown): LLMError | undefined {
  return findInErrorCauseChain(error, isRetryableLlmTransportError);
}

function isAnthropicConnectionError(error: unknown): error is Error {
  const name = errorConstructorName(error);
  return name === "APIConnectionError" || name === "APIConnectionTimeoutError";
}

function findAnthropicConnectionError(error: unknown): Error | undefined {
  return findInErrorCauseChain(error, isAnthropicConnectionError);
}

function rethrowRetryableLlmTransportErrorAtTopLevel(error: unknown): void {
  const retryableError = findRetryableLlmTransportError(error);

  if (retryableError !== undefined) {
    throw new LLMError(retryableError.message, {
      code: retryableError.code,
      cause: error,
    });
  }
}

type AnthropicTransportRetryOptions = {
  connectionMaxRetries?: number;
  stallMaxRetries?: number;
  signal?: AbortSignal;
  onRetry?: (event: { attempt: number; kind: "stall" | "connection"; code?: string }) => void;
};

async function runWithAnthropicTransportRetries<T>(
  run: (attempt: number) => Promise<T>,
  options: AnthropicTransportRetryOptions = {},
): Promise<T> {
  const connectionMaxRetries = options.connectionMaxRetries ?? ANTHROPIC_CONNECTION_MAX_RETRIES;
  const stallMaxRetries = options.stallMaxRetries ?? ANTHROPIC_STALL_MAX_RETRIES;
  let attempt = 0;
  let connectionFailures = 0;
  let stallFailures = 0;

  while (true) {
    attempt += 1;

    try {
      return await run(attempt);
    } catch (error) {
      // The SHALLOWEST typed transport verdict in the cause chain decides.
      // LLM_CONNECTION_FAILED is itself a retry verdict and must never
      // re-enter retries, even when a deeper cause carries a stall-class code.
      const typedError = findRetryableLlmTransportError(error);

      if (typedError !== undefined) {
        if (!STALL_CLASS_LLM_ERROR_CODES.has(typedError.code)) {
          throw error;
        }

        stallFailures += 1;

        // An aborted signal means the outer per-call deadline already settled
        // the race -- a fresh attempt could never be observed by the caller.
        if (stallFailures > stallMaxRetries || options.signal?.aborted === true) {
          throw error;
        }

        options.onRetry?.({ attempt: attempt + 1, kind: "stall", code: typedError.code });
        continue;
      }

      const connectionError = findAnthropicConnectionError(error);

      if (connectionError === undefined) {
        throw error;
      }

      connectionFailures += 1;

      if (connectionFailures > connectionMaxRetries) {
        throw createLlmConnectionFailedError(attempt, error);
      }

      options.onRetry?.({ attempt: attempt + 1, kind: "connection" });
    }
  }
}

function composeAbortSignals(
  signals: readonly (AbortSignal | null | undefined)[],
): AbortSignal | undefined {
  const activeSignals = signals.filter(
    (signal): signal is AbortSignal => signal !== undefined && signal !== null,
  );

  if (activeSignals.length === 0) {
    return undefined;
  }

  if (activeSignals.length === 1) {
    return activeSignals[0];
  }

  return AbortSignal.any(activeSignals);
}

async function readSseChunkWithInactivityTimeout(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  timeoutMs: number,
): Promise<SseReadResult> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let timeoutError: LLMError | undefined;
  const readPromise = reader.read();
  readPromise.catch(() => undefined);

  try {
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        timeoutError = createSseStalledError(timeoutMs);
        reject(timeoutError);
        void reader.cancel(timeoutError).catch(() => undefined);
      }, timeoutMs);
      unrefTimer(timer);
    });

    const result = await Promise.race([readPromise, timeoutPromise]);

    if (timeoutError !== undefined) {
      throw timeoutError;
    }

    return result;
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
}

async function runWithLlmCallTimeout<T>(input: {
  kind: "unary" | "streaming";
  timeoutMs: number;
  run: (signal: AbortSignal) => Promise<T>;
  signal?: AbortSignal | null;
}): Promise<T> {
  const timeoutController = new AbortController();
  const signal = composeAbortSignals([input.signal, timeoutController.signal]);
  let timer: ReturnType<typeof setTimeout> | undefined;
  let timeoutError: LLMError | undefined;

  const operationPromise = Promise.resolve().then(() =>
    input.run(signal ?? timeoutController.signal),
  );
  operationPromise.catch(() => undefined);

  try {
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        timeoutError = createLlmCallTimedOutError(input.kind, input.timeoutMs);
        reject(timeoutError);
        timeoutController.abort(timeoutError);
      }, input.timeoutMs);
      unrefTimer(timer);
    });

    const result = await Promise.race([operationPromise, timeoutPromise]);

    if (timeoutError !== undefined) {
      throw timeoutError;
    }

    return result;
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
}

async function readResponseTextWithTimeout(response: Response, timeoutMs: number): Promise<string> {
  if (response.body === null) {
    return "";
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let text = "";
  let timer: ReturnType<typeof setTimeout> | undefined;
  let timeoutError: LLMError | undefined;

  const textPromise = (async () => {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        text += decoder.decode();
        return text;
      }

      text += decoder.decode(value, { stream: true });
    }
  })();
  textPromise.catch(() => undefined);

  try {
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        timeoutError = createFetchLayerLlmDeadlineError("body-read", timeoutMs);
        reject(timeoutError);
        void reader.cancel(timeoutError).catch(() => undefined);
      }, timeoutMs);
      unrefTimer(timer);
    });

    const text = await Promise.race([textPromise, timeoutPromise]);

    if (timeoutError !== undefined) {
      throw timeoutError;
    }

    return text;
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
}

async function fetchWithHeadersTimeout(
  input: Parameters<typeof fetch>[0],
  init: RequestInit | undefined,
  timeoutMs: number,
): Promise<Response> {
  const timeoutController = new AbortController();
  const signal = composeAbortSignals([init?.signal, timeoutController.signal]);
  let timer: ReturnType<typeof setTimeout> | undefined;

  try {
    timer = setTimeout(() => {
      timeoutController.abort(createFetchLayerLlmDeadlineError("headers", timeoutMs));
    }, timeoutMs);
    unrefTimer(timer);

    return await globalThis.fetch(input, {
      ...init,
      ...(signal === undefined ? {} : { signal }),
    });
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
}

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

type SseFrameKind = "comment" | "ping" | "message";

function parseSseField(line: string): { field: string; value: string } {
  const delimiterIndex = line.indexOf(":");

  if (delimiterIndex === -1) {
    return { field: line, value: "" };
  }

  let value = line.slice(delimiterIndex + 1);

  if (value.startsWith(" ")) {
    value = value.slice(1);
  }

  return {
    field: line.slice(0, delimiterIndex),
    value,
  };
}

function classifySseFrame(frame: string): SseFrameKind {
  const lines = frame.split(/\r?\n/).filter((line) => line.length > 0);

  if (lines.length > 0 && lines.every((line) => line.startsWith(":"))) {
    return "comment";
  }

  let eventName: string | undefined;
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith(":")) {
      continue;
    }

    const field = parseSseField(line);

    if (field.field === "event") {
      eventName = field.value;
      continue;
    }

    if (field.field === "data") {
      dataLines.push(field.value);
    }
  }

  if (eventName === "ping") {
    return "ping";
  }

  if (dataLines.length > 0) {
    try {
      const parsed = JSON.parse(dataLines.join("\n")) as unknown;

      if (
        parsed !== null &&
        typeof parsed === "object" &&
        (parsed as { type?: unknown }).type === "ping"
      ) {
        return "ping";
      }
    } catch {
      // Non-JSON data is still a message event for watchdog purposes.
    }
  }

  return "message";
}

function createSseMessageEventWatchdog(input: {
  reader: ReadableStreamDefaultReader<Uint8Array>;
  firstMessageEventTimeoutMs: number;
  messageEventGapTimeoutMs: number;
  onError: (error: LLMError) => void;
}): {
  arm(controller: ReadableStreamDefaultController<Uint8Array>): void;
  recordFrame(frame: string): void;
  close(): void;
  error(error: unknown): void;
  readonly errored: boolean;
} {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let controller: ReadableStreamDefaultController<Uint8Array> | undefined;
  let sawMessageEvent = false;
  let closed = false;
  let errored = false;

  const clear = () => {
    if (timer !== undefined) {
      clearTimeout(timer);
      timer = undefined;
    }
  };

  const fail = (error: LLMError) => {
    if (closed || errored) {
      return;
    }

    errored = true;
    clear();
    input.onError(error);

    try {
      controller?.error(error);
    } catch {
      // The stream may already have been errored by another watchdog.
    }

    void input.reader.cancel(error).catch(() => undefined);
  };

  const armTimer = () => {
    if (closed || errored || timer !== undefined) {
      return;
    }

    const bound = sawMessageEvent ? "message-event-gap" : "first-message-event";
    const timeoutMs = sawMessageEvent
      ? input.messageEventGapTimeoutMs
      : input.firstMessageEventTimeoutMs;

    timer = setTimeout(() => {
      timer = undefined;
      fail(createSseEventStalledError(bound, timeoutMs));
    }, timeoutMs);
    unrefTimer(timer);
  };

  return {
    arm(nextController) {
      controller = nextController;
      armTimer();
    },
    recordFrame(frame) {
      if (closed || errored || classifySseFrame(frame) !== "message") {
        return;
      }

      sawMessageEvent = true;
      clear();
      armTimer();
    },
    close() {
      closed = true;
      clear();
    },
    error() {
      errored = true;
      clear();
    },
    get errored() {
      return errored;
    },
  };
}

export function createOAuthFetch(options: OAuthFetchOptions = {}): typeof fetch {
  const sseInactivityTimeoutMs = options.sseInactivityTimeoutMs ?? SSE_INACTIVITY_TIMEOUT_MS;
  const sseFirstMessageEventTimeoutMs =
    options.sseFirstMessageEventTimeoutMs ?? SSE_FIRST_MESSAGE_EVENT_TIMEOUT_MS;
  const sseMessageEventGapTimeoutMs =
    options.sseMessageEventGapTimeoutMs ?? SSE_MESSAGE_EVENT_GAP_TIMEOUT_MS;
  const fetchHeadersTimeoutMs = options.fetchHeadersTimeoutMs ?? OAUTH_FETCH_HEADERS_TIMEOUT_MS;
  const unaryBodyTimeoutMs = options.unaryBodyTimeoutMs ?? OAUTH_UNARY_BODY_TIMEOUT_MS;

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

    const response = await fetchWithHeadersTimeout(
      requestUrl.toString(),
      modifiedInit,
      fetchHeadersTimeoutMs,
    );

    if (!isMessagesRequest) {
      return response;
    }

    const contentType = response.headers.get("content-type") ?? "";

    if (contentType.includes("application/json") && !contentType.includes("stream")) {
      try {
        const text = await readResponseTextWithTimeout(response, unaryBodyTimeoutMs);
        let parsed: unknown;

        try {
          parsed = JSON.parse(text) as unknown;
        } catch {
          return new Response(text, {
            status: response.status,
            statusText: response.statusText,
            headers: new Headers(response.headers),
          });
        }

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
      } catch (error) {
        if (isRetryableLlmTransportError(error)) {
          throw error;
        }

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
      const eventWatchdog = createSseMessageEventWatchdog({
        reader,
        firstMessageEventTimeoutMs: sseFirstMessageEventTimeoutMs,
        messageEventGapTimeoutMs: sseMessageEventGapTimeoutMs,
        onError: () => {
          buffer = "";
        },
      });

      const stream = new ReadableStream<Uint8Array>(
        {
          async pull(controller) {
            eventWatchdog.arm(controller);

            let result: SseReadResult;

            try {
              result = await readSseChunkWithInactivityTimeout(reader, sseInactivityTimeoutMs);
            } catch (error) {
              buffer = "";
              eventWatchdog.error(error);
              controller.error(error);
              return;
            }

            if (eventWatchdog.errored) {
              return;
            }

            const { done, value } = result;

            if (done) {
              if (buffer.length > 0) {
                eventWatchdog.recordFrame(buffer);
                if (eventWatchdog.errored) {
                  return;
                }
                controller.enqueue(
                  encoder.encode(transformSseEvent(buffer, originalNamesByTransformed)),
                );
                buffer = "";
              }

              eventWatchdog.close();
              controller.close();
              return;
            }

            buffer += decoder.decode(value, { stream: true });
            const events = buffer.split(/\r?\n\r?\n/);
            buffer = events.pop() ?? "";

            if (events.length > 0) {
              const transformedEvents: string[] = [];

              for (const event of events) {
                eventWatchdog.recordFrame(event);
                if (eventWatchdog.errored) {
                  return;
                }

                transformedEvents.push(transformSseEvent(event, originalNamesByTransformed));
              }

              controller.enqueue(encoder.encode(`${transformedEvents.join("\n\n")}\n\n`));
            }
          },
          async cancel(reason) {
            buffer = "";
            eventWatchdog.close();
            await reader.cancel(reason);
          },
        },
        { highWaterMark: 0 },
      );

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
  const requested = options.max_tokens ?? getModelMaxOutputTokens(options.model);
  return clampMaxOutputTokens(options.model, requested);
}

function shouldOmitTemperature(model: string): boolean {
  return isOpusModel(model);
}

function shouldOmitThinking(
  options: Pick<LLMCallOptions, "model" | "tool_choice" | "thinking">,
): boolean {
  if (options.thinking === undefined) {
    return false;
  }

  // The API rejects thinking when tool_choice forces tool use ("Thinking may not
  // be enabled when tool_choice forces tool use") -- holds for any auth and for
  // both forced shapes. Thinking-active calls must use auto/none tool_choice.
  if (options.tool_choice?.type === "tool" || options.tool_choice?.type === "any") {
    return true;
  }

  // Manual (budget_tokens) thinking is rejected on Opus 4.7/4.8 and deprecated on
  // older Opus; adaptive thinking is the supported mode on Opus 4.6+. Omit only
  // the manual shape on Opus -- adaptive flows through.
  if (options.thinking.type === "enabled" && isOpusModel(options.model)) {
    return true;
  }

  return false;
}

// Whether thinking will actually be sent when a call uses auto/none tool_choice.
// Call sites use this to choose tool_choice: when thinking WILL be sent they must
// use auto (the API rejects forced tool use with thinking active); when it will
// NOT (no thinking / disabled / manual-on-Opus, all of which the client omits),
// they should force the emission tool so a structured emission stays guaranteed.
export function willSendThinkingUnderAutoToolChoice(
  model: string,
  thinking: ThinkingConfigParam | undefined,
): boolean {
  if (thinking === undefined || thinking.type === "disabled") {
    return false;
  }

  if (thinking.type === "enabled" && isOpusModel(model)) {
    return false;
  }

  return true;
}

function buildOutputConfig(
  base: LLMOutputConfig | undefined,
  effort: OutputConfig["effort"] | undefined,
): OutputConfig | undefined {
  if (base === undefined && (effort === undefined || effort === null)) {
    return undefined;
  }

  return {
    ...(base ?? {}),
    ...(effort === undefined || effort === null ? {} : { effort }),
  };
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
  oauthFetchOptions: OAuthFetchOptions = {},
): AnthropicClientLike {
  const baseURL = env.ANTHROPIC_BASE_URL?.trim() || undefined;

  if (auth.kind === "api-key") {
    return new Anthropic({
      apiKey: auth.apiKey,
      maxRetries: 0,
      ...(baseURL ? { baseURL } : {}),
    });
  }

  return new Anthropic({
    authToken: auth.authToken,
    defaultHeaders: {
      "anthropic-beta": OAUTH_BETAS,
      "user-agent": OAUTH_USER_AGENT,
    },
    fetch: createOAuthFetch(oauthFetchOptions),
    maxRetries: 0,
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

  private oauthFetchOptions(): OAuthFetchOptions {
    return {
      sseInactivityTimeoutMs: this.options.oauthSseInactivityTimeoutMs,
      sseFirstMessageEventTimeoutMs: this.options.oauthSseFirstMessageEventTimeoutMs,
      sseMessageEventGapTimeoutMs: this.options.oauthSseMessageEventGapTimeoutMs,
      fetchHeadersTimeoutMs: this.options.oauthFetchHeadersTimeoutMs,
      unaryBodyTimeoutMs: this.options.oauthUnaryBodyTimeoutMs,
    };
  }

  private unaryCallTimeoutMs(): number {
    return this.options.unaryCallTimeoutMs ?? LLM_UNARY_CALL_TIMEOUT_MS;
  }

  private streamingCallTimeoutMs(): number {
    return this.options.streamingCallTimeoutMs ?? LLM_STREAMING_CALL_TIMEOUT_MS;
  }

  private transportStallMaxRetries(): number {
    return this.options.transportStallMaxRetries ?? ANTHROPIC_STALL_MAX_RETRIES;
  }

  // Streaming stalls live in the SSE delivery path; the same request bodies
  // complete via non-streaming create (verified against the Jun 8-12 stall
  // incidents), so stall retries of streaming calls go unary. A buffered
  // non-streaming response sends headers only when generation finishes, so the
  // retry client relaxes the fetch-layer headers guard to the streaming call
  // budget; the outer per-call deadline signal still bounds the whole call.
  private stallRetryUnaryClient(): AnthropicClientLike | undefined {
    if (this.auth === undefined || this.options.client !== undefined) {
      return this.client;
    }

    return buildAnthropicClient(this.auth, this.options.env, {
      ...this.oauthFetchOptions(),
      fetchHeadersTimeoutMs: this.streamingCallTimeoutMs(),
    });
  }

  private async ensureInitialized(): Promise<void> {
    if (this.client !== undefined) {
      return;
    }

    if (this.initialization === undefined) {
      const initialization = (async () => {
        this.auth = await resolveAnthropicAuth(this.options);
        this.client = buildAnthropicClient(this.auth, this.options.env, this.oauthFetchOptions());
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

  private rawMessageParams(
    options: LLMCallOptions,
    messages: MessageParam[],
  ): {
    model: string;
    system?: string | TextBlockParam[];
    messages: MessageParam[];
    tools?: Tool[];
    tool_choice?: ToolChoice;
    output_config?: OutputConfig;
    max_tokens: number;
    temperature?: number;
    thinking?: ThinkingConfigParam;
  } {
    const omitThinking = shouldOmitThinking(options);
    // effort guides adaptive-thinking depth; it only rides when thinking is
    // actually sent, so drop it whenever thinking is absent or omitted.
    const sendThinking = options.thinking !== undefined && !omitThinking;
    const outputConfig = buildOutputConfig(
      options.output_config,
      sendThinking ? options.effort : undefined,
    );

    return {
      model: options.model,
      system: this.resolveSystemPrompt(options.system),
      messages,
      tools: toAnthropicTools(options.tools),
      tool_choice: toAnthropicToolChoice(options.tool_choice),
      ...(outputConfig === undefined ? {} : { output_config: outputConfig }),
      max_tokens: resolveMaxTokens(options),
      ...(options.temperature !== undefined && !shouldOmitTemperature(options.model)
        ? { temperature: options.temperature }
        : {}),
      ...(sendThinking ? { thinking: options.thinking } : {}),
    };
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
    this.client = buildAnthropicClient(this.auth, this.options.env, this.oauthFetchOptions());
    this.initialization = Promise.resolve();
  }

  private async createRawMessage(
    options: LLMCallOptions,
    messages: MessageParam[],
    retrying = false,
    callTimeoutMs = this.unaryCallTimeoutMs(),
  ): Promise<Message> {
    await this.ensureInitialized();

    const client = this.client;

    if (client === undefined) {
      throw new LLMError("Anthropic client failed to initialize");
    }

    try {
      return await runWithLlmCallTimeout({
        kind: "unary",
        timeoutMs: callTimeoutMs,
        signal: options.signal,
        run: (signal) =>
          runWithAnthropicTransportRetries(
            () => client.messages.create(this.rawMessageParams(options, messages), { signal }),
            {
              stallMaxRetries: this.transportStallMaxRetries(),
              signal,
              onRetry: (event) =>
                options.onTransportRetry?.({ ...event, retry_transport: "unary" }),
            },
          ),
      });
    } catch (error) {
      rethrowRetryableLlmTransportErrorAtTopLevel(error);

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

        return this.createRawMessage(options, messages, true, callTimeoutMs);
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

  private async streamRawMessage(
    options: LLMCallOptions,
    messages: MessageParam[],
    onTextDelta: LLMStreamTextHandler | undefined,
    retrying = false,
  ): Promise<Message> {
    await this.ensureInitialized();

    const client = this.client;

    if (client === undefined) {
      throw new LLMError("Anthropic client failed to initialize");
    }

    const streamFactory = client.messages.stream?.bind(client.messages);

    if (streamFactory === undefined) {
      return this.createRawMessage(options, messages, retrying, this.streamingCallTimeoutMs());
    }

    try {
      return await runWithLlmCallTimeout({
        kind: "streaming",
        timeoutMs: this.streamingCallTimeoutMs(),
        signal: options.signal,
        run: (signal) =>
          runWithAnthropicTransportRetries(
            async (attempt) => {
            if (attempt > 1) {
              // Retry attempts switch to non-streaming: the stall mode lives in
              // SSE delivery, and a buffered response rides through it. This
              // also means a retry can never re-emit deltas already forwarded
              // by the stalled first attempt.
              const retryClient = this.stallRetryUnaryClient() ?? client;
              return await retryClient.messages.create(
                this.rawMessageParams(options, messages),
                { signal, timeout: this.streamingCallTimeoutMs() },
              );
            }

            const stream = streamFactory(this.rawMessageParams(options, messages), {
              signal,
            });

            // Track per-content-block tool state so input_json_delta events can be
            // forwarded as token chunks. For tools whose primary user-content lives in
            // a known field (EmitAnswer.text, EmitObserve.reason, EmitNoOutput.reason)
            // we extract that field's incremental value so the UI sees clean answer
            // text. For other tools (S2's EmitTurnPlan, etc.) we forward the raw
            // partial JSON so the structured thinking is visible as it forms.
            const toolBlocks = new Map<
              number,
              { name: string; partial: string; lastExtracted: string; textFieldName?: string }
            >();

            for await (const event of stream) {
              if (event.type === "content_block_start") {
                if (event.content_block.type === "tool_use") {
                  toolBlocks.set(event.index, {
                    name: event.content_block.name,
                    partial: "",
                    lastExtracted: "",
                    textFieldName: TOOL_STREAM_TEXT_FIELDS[event.content_block.name],
                  });
                }
                continue;
              }

              if (event.type === "content_block_delta") {
                if (event.delta.type === "text_delta") {
                  // Under the emission-tool protocol, loose text emitted alongside the
                  // tool is not user-facing (the answer rides in the tool input); skip it
                  // so it never reaches the live stream.
                  if (options.suppressRawTextStream !== true) {
                    onTextDelta?.(event.delta.text);
                  }
                  continue;
                }

                if (event.delta.type === "input_json_delta") {
                  const block = toolBlocks.get(event.index);
                  if (block === undefined) {
                    continue;
                  }

                  block.partial += event.delta.partial_json;

                  if (block.textFieldName === undefined) {
                    // Unknown tool — forward raw partial JSON so the user sees
                    // structured thinking accumulating.
                    if (event.delta.partial_json.length > 0) {
                      onTextDelta?.(event.delta.partial_json);
                    }
                    continue;
                  }

                  const extracted = extractPartialStringField(block.partial, block.textFieldName);
                  if (extracted !== null && extracted.length > block.lastExtracted.length) {
                    onTextDelta?.(extracted.slice(block.lastExtracted.length));
                    block.lastExtracted = extracted;
                  }
                  continue;
                }
              }

              if (event.type === "content_block_stop") {
                toolBlocks.delete(event.index);
              }
            }

            return await stream.finalMessage();
            },
            {
              stallMaxRetries: this.transportStallMaxRetries(),
              signal,
              // Every retry of a streaming call runs unary (see the attempt > 1
              // branch above).
              onRetry: (event) =>
                options.onTransportRetry?.({ ...event, retry_transport: "unary" }),
            },
          ),
      });
    } catch (error) {
      rethrowRetryableLlmTransportErrorAtTopLevel(error);

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

        return this.streamRawMessage(options, messages, onTextDelta, true);
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

  private async streamMessage(options: LLMCompleteStreamOptions): Promise<LLMCompleteResult> {
    const response = await this.streamRawMessage(
      options,
      toAnthropicMessages(options.messages),
      options.onTextDelta,
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
      toAnthropicContentBlockMessages(options.messages, {
        attachmentResolver: this.options.attachmentResolver,
      }),
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

  private async streamConversation(options: LLMConverseStreamOptions): Promise<LLMConverseResult> {
    const response = await this.streamRawMessage(
      options,
      toAnthropicContentBlockMessages(options.messages, {
        attachmentResolver: this.options.attachmentResolver,
      }),
      options.onTextDelta,
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

  streamComplete(options: LLMCompleteStreamOptions): Promise<LLMCompleteResult> {
    return this.streamMessage(options);
  }

  streamConverse(options: LLMConverseStreamOptions): Promise<LLMConverseResult> {
    return this.streamConversation(options);
  }
}
