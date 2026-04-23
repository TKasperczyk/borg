// Client factories for Borg's embedding and LLM dependencies.

import type { Config } from "../config/index.js";
import { OpenAICompatibleEmbeddingClient, type EmbeddingClient } from "../embeddings/index.js";
import { AnthropicLLMClient, type LLMClient } from "../llm/index.js";
import type { Clock } from "../util/clock.js";

export function createEmbeddingClient(config: Config): EmbeddingClient {
  return new OpenAICompatibleEmbeddingClient({
    baseUrl: config.embedding.baseUrl,
    apiKey: config.embedding.apiKey,
    model: config.embedding.model,
    dims: config.embedding.dims,
  });
}

export function createLlmFactory(
  config: Config,
  llmClient: LLMClient | undefined,
  env: NodeJS.ProcessEnv | undefined,
  clock: Clock,
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
