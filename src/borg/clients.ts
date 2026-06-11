// Client factories for Borg's embedding and LLM dependencies.

import type { Config } from "../config/index.js";
import { createCachingEmbeddingClient } from "../embeddings/cache.js";
import { OpenAICompatibleEmbeddingClient, type EmbeddingClient } from "../embeddings/index.js";
import { AnthropicLLMClient, type LLMClient } from "../llm/index.js";
import type { Clock } from "../util/clock.js";
import type { AttachmentService } from "../attachments/index.js";

export function createEmbeddingClient(config: Config): EmbeddingClient {
  const inner = new OpenAICompatibleEmbeddingClient({
    baseUrl: config.embedding.baseUrl,
    apiKey: config.embedding.apiKey,
    model: config.embedding.model,
    dims: config.embedding.dims,
  });

  return createCachingEmbeddingClient(inner, {
    model: config.embedding.model,
    dims: config.embedding.dims,
  });
}

export function createLlmFactory(
  config: Config,
  llmClient: LLMClient | undefined,
  env: NodeJS.ProcessEnv | undefined,
  clock: Clock,
  attachmentService?: AttachmentService,
): () => LLMClient {
  if (llmClient !== undefined) {
    return () => llmClient;
  }

  let cached: LLMClient | undefined;

  return () => {
    cached ??= new AnthropicLLMClient({
      authMode: config.anthropic.auth,
      apiKey: config.anthropic.apiKey,
      env,
      clock,
      oauthSseInactivityTimeoutMs: config.anthropic.oauthSseInactivityTimeoutMs,
      oauthSseFirstMessageEventTimeoutMs: config.anthropic.oauthSseFirstMessageEventTimeoutMs,
      oauthSseMessageEventGapTimeoutMs: config.anthropic.oauthSseMessageEventGapTimeoutMs,
      oauthFetchHeadersTimeoutMs: config.anthropic.oauthFetchHeadersTimeoutMs,
      oauthUnaryBodyTimeoutMs: config.anthropic.oauthUnaryBodyTimeoutMs,
      unaryCallTimeoutMs: config.anthropic.unaryCallTimeoutMs,
      streamingCallTimeoutMs: config.anthropic.streamingCallTimeoutMs,
      ...(attachmentService === undefined
        ? {}
        : {
            attachmentResolver: (attachmentId) => attachmentService.fetchImageForLlm(attachmentId),
          }),
    });
    return cached;
  };
}

export function createLazyLlmClient(factory: () => LLMClient): LLMClient {
  return {
    complete(options) {
      return factory().complete(options);
    },
    converse(options) {
      return factory().converse(options);
    },
  };
}
