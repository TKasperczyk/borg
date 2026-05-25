import type {
  ContentBlockParam,
  MessageParam,
  TextBlockParam,
  ToolResultBlockParam,
  ToolUseBlockParam,
} from "@anthropic-ai/sdk/resources/messages/messages.js";

import { LLMError } from "../util/errors.js";
import type { AttachmentId } from "../util/ids.js";
import type { LLMContentBlock, LLMContentBlockMessage, LLMToolResultBlock } from "./index.js";

export type AnthropicAttachmentResolver = (attachmentId: AttachmentId) => {
  mediaType: string;
  bytes: Buffer | Uint8Array;
};

export type AnthropicContentBlockOptions = {
  attachmentResolver?: AnthropicAttachmentResolver;
};

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

export function toAnthropicContentBlock(
  block: LLMContentBlock,
  options: AnthropicContentBlockOptions = {},
): ContentBlockParam {
  if (block.type === "text") {
    return {
      type: "text",
      text: block.text,
    } satisfies TextBlockParam;
  }

  if (block.type === "image_ref") {
    if (options.attachmentResolver === undefined) {
      throw new LLMError(`No attachment resolver configured for ${block.attachment_id}`, {
        code: "LLM_ATTACHMENT_RESOLVER_MISSING",
      });
    }

    const image = options.attachmentResolver(block.attachment_id);

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

export function toAnthropicContentBlocks(
  blocks: readonly LLMContentBlock[],
  options: AnthropicContentBlockOptions = {},
): ContentBlockParam[] {
  return blocks.map((block) => toAnthropicContentBlock(block, options));
}

export function toAnthropicContentBlockMessages(
  messages: readonly LLMContentBlockMessage[],
  options: AnthropicContentBlockOptions = {},
): MessageParam[] {
  return messages.map((message) => ({
    role: message.role,
    content: toAnthropicContentBlocks(message.content, options),
  }));
}
