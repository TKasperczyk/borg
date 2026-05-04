import OpenAI from "openai";

import { sleep } from "../util/clock.js";
import { ConfigError, EmbeddingError } from "../util/errors.js";

type OpenAIEmbeddingsClient = {
  embeddings: {
    create(params: {
      input: string | string[];
      model: string;
      encoding_format?: "float" | "base64";
    }): Promise<{
      data: Array<{
        embedding: number[];
        index: number;
      }>;
    }>;
  };
};

export type EmbeddingClient = {
  embed(text: string): Promise<Float32Array>;
  embedBatch(texts: readonly string[]): Promise<Float32Array[]>;
};

export type OpenAICompatibleEmbeddingClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  model: string;
  dims: number;
  client?: OpenAIEmbeddingsClient;
  modelReloadRetryDelaysMs?: readonly number[];
};

const DEFAULT_MODEL_RELOAD_RETRY_DELAYS_MS: readonly number[] = [1000, 4000, 10_000];

function isModelNotLoadedError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) {
    return false;
  }

  const candidate = error as { status?: unknown; code?: unknown };

  return candidate.status === 404 && candidate.code === "model_not_found";
}

function validateDimensions(embedding: number[], dims: number, model: string): Float32Array {
  if (embedding.length !== dims) {
    throw new EmbeddingError(
      `Embedding dimension mismatch for model "${model}": configured ${dims}, received ${embedding.length}. Check BORG_EMBEDDING_DIMS matches the loaded model's output dimension (the model itself may be correct -- some providers like LM Studio silently truncate to zeros when the OpenAI SDK's default base64 encoding is mis-decoded; borg requests encoding_format:"float" to avoid this).`,
    );
  }

  return Float32Array.from(embedding);
}

export class OpenAICompatibleEmbeddingClient implements EmbeddingClient {
  private readonly client: OpenAIEmbeddingsClient;
  private readonly model: string;
  private readonly dims: number;
  private readonly modelReloadRetryDelaysMs: readonly number[];

  constructor(options: OpenAICompatibleEmbeddingClientOptions) {
    if (!Number.isInteger(options.dims) || options.dims <= 0) {
      throw new ConfigError("Embedding dimensions must be a positive integer");
    }

    if (!options.model.trim()) {
      throw new ConfigError("Embedding model must be configured");
    }

    if (options.client !== undefined) {
      this.client = options.client;
    } else {
      if (!options.baseUrl?.trim()) {
        throw new ConfigError("Embedding base URL must be configured");
      }

      if (!options.apiKey?.trim()) {
        throw new ConfigError("Embedding API key must be configured");
      }

      this.client = new OpenAI({
        apiKey: options.apiKey,
        baseURL: options.baseUrl,
      });
    }

    this.model = options.model;
    this.dims = options.dims;
    this.modelReloadRetryDelaysMs =
      options.modelReloadRetryDelaysMs ?? DEFAULT_MODEL_RELOAD_RETRY_DELAYS_MS;
  }

  async embed(text: string): Promise<Float32Array> {
    const [embedding] = await this.embedBatch([text]);

    if (embedding === undefined) {
      throw new EmbeddingError("Embedding response was empty");
    }

    return embedding;
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    if (texts.length === 0) {
      return [];
    }

    try {
      const response = await this.createWithModelReloadRetry(texts);

      if (response.data.length !== texts.length) {
        throw new EmbeddingError(
          `Embedding response size mismatch: expected ${texts.length}, received ${response.data.length}`,
        );
      }

      return response.data
        .slice()
        .sort((left, right) => left.index - right.index)
        .map((item) => validateDimensions(item.embedding, this.dims, this.model));
    } catch (error) {
      if (error instanceof EmbeddingError || error instanceof ConfigError) {
        throw error;
      }

      throw new EmbeddingError("Failed to generate embeddings", {
        cause: error,
      });
    }
  }

  // JIT-loading inference servers (LM Studio, llama.cpp) evict the model when
  // a different one is requested, returning 404 model_not_found until reload.
  private async createWithModelReloadRetry(
    texts: readonly string[],
  ): ReturnType<OpenAIEmbeddingsClient["embeddings"]["create"]> {
    const singleInput = texts[0];
    const params = {
      input: texts.length === 1 && singleInput !== undefined ? singleInput : [...texts],
      model: this.model,
      // Explicit -- the OpenAI SDK defaults to "base64", which we then have
      // to decode ourselves. Many OpenAI-compatible providers (LM Studio,
      // llama.cpp, vLLM) ignore encoding_format and always return a float
      // array; the SDK then mis-interprets it as base64-encoded bytes and
      // silently returns a truncated, all-zero Float32Array. Asking for
      // "float" explicitly bypasses the client-side decode.
      encoding_format: "float" as const,
    };

    const maxAttempts = this.modelReloadRetryDelaysMs.length;
    let lastError: unknown;

    for (let attempt = 0; attempt <= maxAttempts; attempt += 1) {
      try {
        return await this.client.embeddings.create(params);
      } catch (error) {
        lastError = error;

        if (!isModelNotLoadedError(error) || attempt === maxAttempts) {
          throw error;
        }

        const delayMs = this.modelReloadRetryDelaysMs[attempt] ?? 0;

        if (delayMs > 0) {
          await sleep(delayMs);
        }
      }
    }

    throw lastError;
  }
}

function hashString(input: string): number {
  let hash = 2_166_136_261;

  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16_777_619);
  }

  return hash >>> 0;
}

function nextPseudoRandom(seed: number): number {
  let value = seed >>> 0;
  value ^= value << 13;
  value ^= value >>> 17;
  value ^= value << 5;
  return value >>> 0;
}

export class FakeEmbeddingClient implements EmbeddingClient {
  constructor(private readonly dims = 32) {
    if (!Number.isInteger(dims) || dims <= 0) {
      throw new ConfigError("FakeEmbeddingClient dims must be a positive integer");
    }
  }

  async embed(text: string): Promise<Float32Array> {
    const [embedding] = await this.embedBatch([text]);

    if (embedding === undefined) {
      throw new EmbeddingError("FakeEmbeddingClient produced no embedding");
    }

    return embedding;
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => {
      const vector = new Float32Array(this.dims);
      let seed = hashString(text);

      for (let index = 0; index < vector.length; index += 1) {
        seed = nextPseudoRandom(seed);
        vector[index] = (seed / 0xffffffff) * 2 - 1;
      }

      return vector;
    });
  }
}
