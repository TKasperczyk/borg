import type { EmbeddingProfile } from "./bank-profile.js";
import type { EmbeddingProvider } from "../config/embedding-provider.js";
import { DEFAULT_EMBEDDING_MAX_BATCH_BYTES } from "../config/embedding-input.js";
import OpenAI from "openai";

import { sleep, SystemClock, type Clock } from "../util/clock.js";
import { runWithDeadline } from "../util/deadline.js";
import {
  EmbeddingBackoffError,
  embeddingHttpError,
  embeddingRetryAt,
  type EmbeddingRequestState,
} from "./request-retry.js";
export { EmbeddingBackoffError, type EmbeddingRequestState } from "./request-retry.js";
import { ConfigError, EmbeddingError, findInErrorCauseChain } from "../util/errors.js";
import { toWellFormedUtf16 } from "../util/utf16-boundary.js";
import { utf8SafePrefix } from "../util/utf8-boundary.js";

type OpenAIEmbeddingsClient = {
  embeddings: {
    create(
      params: {
        input: string | string[];
        model: string;
        encoding_format?: "float" | "base64";
        provider?: EmbeddingProvider;
      },
      options?: { signal?: AbortSignal; maxRetries?: number; timeout?: number },
    ): Promise<{
      data: Array<{
        embedding: number[];
        index: number;
      }>;
    }>;
  };
};

export type EmbeddingClientStats = {
  /** Inputs shortened for transport; in-place retries reuse the prepared input. */
  input_truncations: number;
};

export type EmbeddingCallOptions = {
  signal?: AbortSignal;
};
export type EmbeddingBatchOptions = EmbeddingCallOptions & {
  /** Validated request results, indexed into this call's original input array. */
  onChunk?: (startIndex: number, embeddings: readonly Float32Array[]) => void;
};

export type EmbeddingClient = {
  /** Effective transport cap; wrappers must preserve it for bounded scoring text. */
  readonly maxInputBytes?: number | undefined;
  /** Required when injecting into Borg.open; wrappers must preserve this identity. */
  readonly profile?: EmbeddingProfile;
  stats?(): EmbeddingClientStats;
  requestState?(): EmbeddingRequestState;
  embed(text: string, options?: EmbeddingCallOptions): Promise<Float32Array>;
  embedBatch(texts: readonly string[], options?: EmbeddingBatchOptions): Promise<Float32Array[]>;
};

export type OpenAICompatibleEmbeddingClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  model: string;
  dims: number;
  client?: OpenAIEmbeddingsClient;
  modelReloadRetryDelaysMs?: readonly number[];
  maxBatchSize?: number;
  maxBatchBytes?: number;
  maxInputBytes?: number;
  provider?: EmbeddingProvider;
  requestTimeoutMs?: number;
  rateLimitMaxRetries?: number;
  rateLimitBaseDelayMs?: number;
  clock?: Clock;
};

const DEFAULT_MODEL_RELOAD_RETRY_DELAYS_MS: readonly number[] = [1000, 4000, 10_000];

// Callers pass unbounded arrays (every action record missing an embedding,
// every reflector tag, every retrieval intent), and a local inference server
// does not degrade gracefully on a large one: measured against LM Studio with
// an 8B embedding model, 16 inputs took 7s, 64 took 29s, and 128 returned
// `400 "Model has unloaded or crashed."` -- taking the model down for every
// other consumer on the host, not just this one. Chunking here covers every
// call site at once, which is why it belongs in the client rather than in the
// callers. 32 sits comfortably under the observed cliff.
const DEFAULT_MAX_EMBED_BATCH_SIZE = 32;

// A JIT-loading server reports an evicted model in more than one way, and the
// reload retry below is useless unless every shape is recognised. LM Studio
// answers 404/model_not_found when the model was never loaded, 400 "Model
// has unloaded or crashed." once it has evicted an idle one, and 400 "Failed
// to load model ... Model does not exist." for requests that race into the
// window between an eviction and the JIT reload starting -- observed live when
// the host's internal RAG engine loads a different embedding model, evicting
// ours: the request that triggers the reload rides it out and succeeds, while
// concurrent siblings fail instantly with that third shape. Each unrecognised
// shape was a hard turn failure.
//
// "does not exist" also matches a permanently misconfigured model id; the
// retry ladder makes that cost one bounded delay cycle before surfacing the
// same error, which is acceptable against silently dropping the recoverable
// race case.
//
// This inspects a transport-layer error object, not model output: provider
// error signatures are the one thing that legitimately has to be matched by
// shape. Kept narrow on purpose -- a bare 400 is an ordinary bad request.
function isModelNotLoadedError(error: unknown): boolean {
  if (typeof error !== "object" || error === null) {
    return false;
  }

  const candidate = error as {
    status?: unknown;
    code?: unknown;
    message?: unknown;
    error?: { message?: unknown };
  };

  if (candidate.status === 404 && candidate.code === "model_not_found") {
    return true;
  }

  if (candidate.status !== 400) {
    return false;
  }

  const message = [candidate.message, candidate.error?.message]
    .filter((value): value is string => typeof value === "string")
    .join(" ")
    .toLowerCase();

  // Require "model" plus a load-failure half so an unrelated 400 mentioning a
  // model in passing does not get retried forever.
  return (
    message.includes("model") &&
    (message.includes("unloaded") ||
      message.includes("crashed") ||
      message.includes("does not exist"))
  );
}

// The wrapped cause carries the only information that identifies WHY a request
// failed (transport vs 4xx vs abort vs provider quirk), and it does not survive
// into the trace or the journal -- every failure reads identically as "Failed to
// generate embeddings". That cost two wrong diagnoses of a live incident, so the
// distinguishing fields go in the message itself. Bounded, and no request
// content: status/code/name are provider metadata, and the message is truncated.
function describeEmbeddingCause(error: unknown): string {
  if (typeof error !== "object" || error === null) {
    return String(error);
  }

  const candidate = error as {
    name?: unknown;
    status?: unknown;
    code?: unknown;
    type?: unknown;
    message?: unknown;
    error?: { message?: unknown; type?: unknown };
    cause?: { code?: unknown; message?: unknown };
  };

  const parts: string[] = [];
  const push = (label: string, value: unknown): void => {
    if (value !== undefined && value !== null && value !== "") {
      parts.push(`${label}=${String(value)}`);
    }
  };

  push("name", candidate.name);
  push("status", candidate.status);
  push("code", candidate.code ?? candidate.cause?.code);
  push("type", candidate.type ?? candidate.error?.type);

  const message = candidate.message ?? candidate.error?.message ?? candidate.cause?.message;
  if (typeof message === "string" && message !== "") {
    parts.push(`message="${message.slice(0, 300)}"`);
  }

  return parts.length > 0 ? parts.join(" ") : "unknown cause";
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
  private readonly maxBatchSize: number;
  private readonly maxBatchBytes: number;
  readonly maxInputBytes: number | undefined;
  private inputTruncations = 0;
  private readonly provider: EmbeddingProvider | undefined;
  private readonly requestTimeoutMs: number;
  private readonly rateLimitMaxRetries: number;
  private readonly rateLimitBaseDelayMs: number;
  private readonly clock: Clock;
  private retryAt: number | null = null;
  private readonly activeRequests = new Map<object, number>();

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
    this.provider = options.provider;
    this.modelReloadRetryDelaysMs =
      options.modelReloadRetryDelaysMs ?? DEFAULT_MODEL_RELOAD_RETRY_DELAYS_MS;

    const requestedBatchSize = options.maxBatchSize ?? DEFAULT_MAX_EMBED_BATCH_SIZE;
    if (!Number.isInteger(requestedBatchSize) || requestedBatchSize <= 0) {
      throw new ConfigError("Embedding max batch size must be a positive integer");
    }
    this.maxBatchSize = requestedBatchSize;

    const requestedBatchBytes = options.maxBatchBytes ?? DEFAULT_EMBEDDING_MAX_BATCH_BYTES;
    if (!Number.isSafeInteger(requestedBatchBytes) || requestedBatchBytes <= 0) {
      throw new ConfigError("Embedding max batch bytes must be a positive integer");
    }
    this.maxBatchBytes = requestedBatchBytes;

    if (
      options.maxInputBytes !== undefined &&
      (!Number.isSafeInteger(options.maxInputBytes) || options.maxInputBytes <= 0)
    ) {
      throw new ConfigError("Embedding max input bytes must be a positive integer");
    }
    this.maxInputBytes = options.maxInputBytes;
    this.requestTimeoutMs = options.requestTimeoutMs ?? 120_000;
    this.rateLimitMaxRetries = options.rateLimitMaxRetries ?? 2;
    this.rateLimitBaseDelayMs = options.rateLimitBaseDelayMs ?? 1_000;
    for (const value of [this.requestTimeoutMs, this.rateLimitBaseDelayMs]) {
      if (!Number.isSafeInteger(value) || value <= 0)
        throw new ConfigError("Embedding request budgets must be positive integers");
    }
    if (!Number.isSafeInteger(this.rateLimitMaxRetries) || this.rateLimitMaxRetries < 0) {
      throw new ConfigError("Embedding rateLimitMaxRetries must be a non-negative integer");
    }
    this.clock = options.clock ?? new SystemClock();
  }

  get profile(): EmbeddingProfile {
    return { model: this.model, dimensions: this.dims };
  }

  stats(): EmbeddingClientStats {
    return { input_truncations: this.inputTruncations };
  }

  requestState(): EmbeddingRequestState {
    return {
      retryAt: this.retryAt !== null && this.retryAt > this.clock.now() ? this.retryAt : null,
      inFlight: this.activeRequests.size,
      oldestRequestStartedAt:
        this.activeRequests.size === 0 ? null : Math.min(...this.activeRequests.values()),
    };
  }

  private prepareInput(text: string): string {
    // A lone surrogate makes the gateway reject the whole batch as "No model
    // field." This is the last line of defence for transport input hygiene.
    // Repair first: a replacement character occupies three UTF-8 bytes. Only
    // the transport copy is shortened; callers retain their original text.
    const wellFormed = toWellFormedUtf16(text);
    if (
      this.maxInputBytes === undefined ||
      Buffer.byteLength(wellFormed, "utf8") <= this.maxInputBytes
    ) {
      return wellFormed;
    }

    this.inputTruncations += 1;
    return utf8SafePrefix(wellFormed, this.maxInputBytes);
  }

  async embed(text: string, options?: EmbeddingCallOptions): Promise<Float32Array> {
    const [embedding] = await this.embedBatch([text], options);

    if (embedding === undefined) {
      throw new EmbeddingError("Embedding response was empty");
    }

    return embedding;
  }

  async embedBatch(
    texts: readonly string[],
    options: EmbeddingBatchOptions = {},
  ): Promise<Float32Array[]> {
    if (texts.length === 0) {
      return [];
    }

    // Sequential, not parallel: the point is to keep concurrent load off a
    // shared local inference server, so firing the chunks at once would
    // reintroduce the pressure this exists to avoid.
    const out: Float32Array[] = [];
    // Retain one prepared look-ahead input across a byte boundary so truncation
    // happens once, without preparing the entire caller array in advance.
    let nextInput: string | undefined;
    for (let start = 0; start < texts.length; ) {
      options.signal?.throwIfAborted();
      const inputs: string[] = [];
      let bytes = 0;
      while (inputs.length < this.maxBatchSize && start + inputs.length < texts.length) {
        nextInput ??= this.prepareInput(texts[start + inputs.length]!);
        const inputBytes = Buffer.byteLength(nextInput, "utf8");
        // An oversized input must still make progress, alone in its request.
        if (inputs.length > 0 && bytes + inputBytes > this.maxBatchBytes) break;
        inputs.push(nextInput);
        bytes += inputBytes;
        nextInput = undefined;
      }
      const chunk = await this.embedChunk(inputs, options);
      options.onChunk?.(
        start,
        chunk.map((vector) => new Float32Array(vector)),
      );
      out.push(...chunk);
      start += inputs.length;
    }
    return out;
  }

  private async embedChunk(
    texts: readonly string[],
    options: EmbeddingCallOptions,
  ): Promise<Float32Array[]> {
    try {
      const response = await this.createWithModelReloadRetry(texts, options);

      if (response.data.length !== texts.length) {
        throw new EmbeddingError(
          `Embedding response size mismatch: expected ${texts.length}, received ${response.data.length}`,
        );
      }

      return response.data
        .slice()
        .sort((left, right) => left.index - right.index)
        .map((item, index) => {
          if (item.index !== index) {
            throw new EmbeddingError(
              "Embedding response indices do not uniquely cover the input batch",
            );
          }
          return validateDimensions(item.embedding, this.dims, this.model);
        });
    } catch (error) {
      if (error instanceof EmbeddingError || error instanceof ConfigError) {
        throw error;
      }

      throw new EmbeddingError(`Failed to generate embeddings (${describeEmbeddingCause(error)})`, {
        cause: error,
      });
    }
  }

  // JIT-loading inference servers (LM Studio, llama.cpp) evict the model both
  // when a different one is requested AND when it sits idle -- see
  // isModelNotLoadedError for the two error shapes that means. A long turn
  // (a multi-minute planner call) is enough idle time to lose it, so this
  // retry is on the normal path, not an edge case.
  private async createWithModelReloadRetry(
    texts: readonly string[],
    options: EmbeddingCallOptions,
  ): ReturnType<OpenAIEmbeddingsClient["embeddings"]["create"]> {
    // Inputs are already repaired and capped before byte-budget chunking.
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
      ...(this.provider === undefined ? {} : { provider: this.provider }),
    };

    const requestId = {};
    const startedAt = this.clock.now();
    const deadlineAt = startedAt + this.requestTimeoutMs;
    this.activeRequests.set(requestId, startedAt);
    try {
      return await runWithDeadline({
        timeoutMs: this.requestTimeoutMs,
        signal: options.signal,
        timeoutError: () =>
          new EmbeddingError(`Embedding request exceeded ${this.requestTimeoutMs}ms`, {
            code: "EMBEDDING_REQUEST_TIMEOUT",
          }),
        run: async (signal) => {
          let reloadRetries = 0;
          let transportRetries = 0;
          let rateLimitRetries = 0;
          while (true) {
            signal.throwIfAborted();
            const retryAt = this.requestState().retryAt;
            if (retryAt !== null) {
              if (retryAt >= deadlineAt) throw new EmbeddingBackoffError(retryAt);
              await sleep(Math.max(0, retryAt - this.clock.now()), signal);
              continue;
            }
            try {
              // This request loop owns HTTP retries, including injected SDK clients.
              return await this.client.embeddings.create(params, {
                signal,
                maxRetries: 0,
                timeout: this.requestTimeoutMs,
              });
            } catch (error) {
              const http = embeddingHttpError(error);
              // Retain provider backoff even when this error cannot be retried.
              // The next caller, and every retry path below, share this floor.
              const providerRetryAt = embeddingRetryAt(http?.headers, this.clock.now());
              if (providerRetryAt !== null) {
                this.retryAt = Math.max(this.retryAt ?? 0, providerRetryAt);
              }
              signal.throwIfAborted();
              if (this.retryAt !== null && this.retryAt >= deadlineAt) {
                throw new EmbeddingBackoffError(this.retryAt, error);
              }
              if (http?.status === 429) {
                const fallback = this.rateLimitBaseDelayMs * 2 ** Math.min(rateLimitRetries, 16);
                const at = providerRetryAt ?? this.clock.now() + fallback;
                this.retryAt = Math.max(this.retryAt ?? 0, at);
                if (rateLimitRetries++ >= this.rateLimitMaxRetries || at >= deadlineAt) {
                  throw new EmbeddingBackoffError(this.retryAt, error);
                }
                continue;
              }
              const reloadDelay = this.modelReloadRetryDelaysMs[reloadRetries];
              if (isModelNotLoadedError(error) && reloadDelay !== undefined) {
                reloadRetries += 1;
                await sleep(reloadDelay, signal);
                continue;
              }
              const connectionFailure = findInErrorCauseChain(
                error,
                (cause): cause is InstanceType<typeof OpenAI.APIConnectionError> =>
                  cause instanceof OpenAI.APIConnectionError,
              );
              // The SDK's typed connection failures have no HTTP status. Keep
              // their recovery in this same bounded owner, never a second loop.
              if (
                (connectionFailure !== undefined ||
                  (http !== undefined &&
                    (http.status === 408 || http.status === 409 || http.status >= 500))) &&
                transportRetries++ < this.rateLimitMaxRetries
              ) {
                if (providerRetryAt === null) {
                  await sleep(
                    this.rateLimitBaseDelayMs * 2 ** Math.min(transportRetries - 1, 16),
                    signal,
                  );
                }
                continue;
              }
              const retainedRetryAt = this.requestState().retryAt;
              if (retainedRetryAt !== null) {
                throw new EmbeddingBackoffError(retainedRetryAt, error);
              }
              throw error;
            }
          }
        },
      });
    } finally {
      this.activeRequests.delete(requestId);
    }
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
  constructor(
    private readonly dims = 32,
    private readonly model = "fake-embed",
  ) {
    if (!Number.isInteger(dims) || dims <= 0) {
      throw new ConfigError("FakeEmbeddingClient dims must be a positive integer");
    }
  }

  get profile(): EmbeddingProfile {
    return { model: this.model, dimensions: this.dims };
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
