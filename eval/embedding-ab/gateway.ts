import { performance } from "node:perf_hooks";

import OpenAI from "openai";

import {
  OpenAICompatibleEmbeddingClient,
  type EmbeddingClient,
} from "../../src/embeddings/index.js";
import { OpenAICompatibleLLMClient, type LLMClient } from "../../src/llm/index.js";
import { sleep } from "../../src/util/clock.js";

import type {
  EmbeddingCallRecord,
  EmbeddingPurpose,
  ErrorSummary,
  LatencySummary,
} from "./types.js";

export const EMBEDDING_TIMEOUT_MS = 15_000;
const TRANSIENT_RETRY_DELAYS_MS = [500, 1_500] as const;
const LLM_TIMEOUT_MS = 120_000;
const DIMENSION_PROBE_TEXT = "Test wymiaru wektora osadzenia.";

const KNOWN_MODEL_DIMENSIONS: Readonly<Record<string, number>> = {
  "generative-apis/qwen3-embedding-8b": 4096,
  "scw/bge-m3": 1024,
  "generative-apis/bge-multilingual-gemma2": 3584,
};

const PREFERRED_GOLD_MODEL = "generative-apis/qwen3-235b-a22b-instruct-2507";

type EmbeddingParams = {
  input: string | string[];
  model: string;
  encoding_format?: "float" | "base64";
};

type EmbeddingResponse = {
  data: Array<{
    embedding: number[];
    index: number;
  }>;
};

export type GatewayModel = {
  id: string;
  metadata: Record<string, unknown>;
};

export type ModelEmbeddingRuntime = {
  model: string;
  dimensions: number;
  client: EmbeddingClient;
  transport: InstrumentedEmbeddingTransport;
};

export class ModelEmbeddingInitializationError extends Error {
  readonly calls: EmbeddingCallRecord[];

  constructor(
    readonly model: string,
    cause: unknown,
    calls: readonly EmbeddingCallRecord[],
  ) {
    const detail = cause instanceof Error ? cause.message : String(cause);
    super(`Failed to initialize embedding model ${model}: ${detail}`, { cause });
    this.name = "ModelEmbeddingInitializationError";
    this.calls = [...calls];
  }
}

function causeChain(error: unknown): unknown[] {
  const chain: unknown[] = [];
  let current = error;

  for (let depth = 0; depth < 8 && current !== undefined && current !== null; depth += 1) {
    chain.push(current);
    current =
      typeof current === "object" && "cause" in current
        ? (current as { cause?: unknown }).cause
        : undefined;
  }
  return chain;
}

function errorField(error: unknown, field: "name" | "message" | "code" | "status"): unknown {
  for (const current of causeChain(error)) {
    if (typeof current === "object" && current !== null && field in current) {
      const value = (current as Record<string, unknown>)[field];
      if (value !== undefined && value !== null) {
        return value;
      }
    }
  }
  return undefined;
}

export function summarizeError(error: unknown): ErrorSummary {
  const rawName = errorField(error, "name");
  const rawMessage = errorField(error, "message");
  const rawStatus = errorField(error, "status");
  const rawCode = errorField(error, "code");

  return {
    name: typeof rawName === "string" ? rawName : "Error",
    message:
      typeof rawMessage === "string"
        ? rawMessage.slice(0, 1_000)
        : error instanceof Error
          ? error.message.slice(0, 1_000)
          : String(error).slice(0, 1_000),
    ...(typeof rawStatus === "number" ? { status: rawStatus } : {}),
    ...(typeof rawCode === "string" ? { code: rawCode } : {}),
  };
}

export function isTimeoutError(error: unknown): boolean {
  if (errorField(error, "status") === 408) {
    return true;
  }

  return causeChain(error).some((current) => {
    if (typeof current !== "object" || current === null) {
      return false;
    }
    const candidate = current as { name?: unknown; code?: unknown };
    return (
      candidate.name === "APIConnectionTimeoutError" ||
      candidate.name === "TimeoutError" ||
      candidate.code === "ETIMEDOUT"
    );
  });
}

function isRetryableTransportError(error: unknown): boolean {
  if (isTimeoutError(error)) {
    return false;
  }

  const status = errorField(error, "status");
  if (typeof status === "number") {
    return status === 409 || status === 429 || status >= 500;
  }

  return causeChain(error).some((current) => {
    if (typeof current !== "object" || current === null) {
      return false;
    }
    const candidate = current as { name?: unknown; code?: unknown };
    return (
      candidate.name === "APIConnectionError" ||
      candidate.code === "ECONNRESET" ||
      candidate.code === "ECONNREFUSED" ||
      candidate.code === "EPIPE" ||
      candidate.code === "ENETUNREACH"
    );
  });
}

function percentile(values: readonly number[], fraction: number): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((left, right) => left - right);
  const index = Math.max(0, Math.ceil(fraction * sorted.length) - 1);
  return sorted[index] ?? null;
}

export function summarizeLatency(calls: readonly EmbeddingCallRecord[]): LatencySummary {
  const durations = calls.map((call) => call.latency_ms);
  return {
    call_count: calls.length,
    successful_call_count: calls.filter((call) => call.outcome === "success").length,
    error_count: calls.filter((call) => call.outcome === "error").length,
    timeout_count: calls.filter((call) => call.outcome === "timeout").length,
    retry_attempt_count: calls.filter((call) => call.attempt > 1).length,
    p50_ms: percentile(durations, 0.5),
    p95_ms: percentile(durations, 0.95),
    max_ms: durations.length === 0 ? null : Math.max(...durations),
    calls: [...calls],
  };
}

export class InstrumentedEmbeddingTransport {
  readonly calls: EmbeddingCallRecord[] = [];
  readonly adapter: {
    embeddings: {
      create: (params: EmbeddingParams) => Promise<EmbeddingResponse>;
    };
  };
  private purpose: EmbeddingPurpose = "dimension_probe";

  constructor(private readonly openai: OpenAI) {
    this.adapter = {
      embeddings: {
        create: (params) => this.create(params),
      },
    };
  }

  async withPurpose<T>(purpose: EmbeddingPurpose, run: () => Promise<T>): Promise<T> {
    const previous = this.purpose;
    this.purpose = purpose;
    try {
      return await run();
    } finally {
      this.purpose = previous;
    }
  }

  async probeDimensions(model: string): Promise<number> {
    const response = await this.withPurpose("dimension_probe", () =>
      this.create({
        input: DIMENSION_PROBE_TEXT,
        model,
        encoding_format: "float",
      }),
    );
    const dimensions = response.data[0]?.embedding.length;
    if (dimensions === undefined || !Number.isInteger(dimensions) || dimensions <= 0) {
      throw new Error(`Dimension probe for ${model} returned no valid embedding`);
    }
    return dimensions;
  }

  private async create(params: EmbeddingParams): Promise<EmbeddingResponse> {
    const batchSize = Array.isArray(params.input) ? params.input.length : 1;
    const attempts = TRANSIENT_RETRY_DELAYS_MS.length + 1;
    let lastError: unknown;

    for (let attempt = 1; attempt <= attempts; attempt += 1) {
      const startedAt = new Date();
      const started = performance.now();
      try {
        const response = (await this.openai.embeddings.create(params, {
          timeout: EMBEDDING_TIMEOUT_MS,
        })) as unknown as EmbeddingResponse;
        this.calls.push({
          sequence: this.calls.length + 1,
          purpose: this.purpose,
          batch_size: batchSize,
          attempt,
          started_at: startedAt.toISOString(),
          latency_ms: Math.round((performance.now() - started) * 100) / 100,
          outcome: "success",
        });
        return response;
      } catch (error) {
        lastError = error;
        const timeout = isTimeoutError(error);
        this.calls.push({
          sequence: this.calls.length + 1,
          purpose: this.purpose,
          batch_size: batchSize,
          attempt,
          started_at: startedAt.toISOString(),
          latency_ms: Math.round((performance.now() - started) * 100) / 100,
          outcome: timeout ? "timeout" : "error",
          error: summarizeError(error),
        });

        // A timed-out request is deliberately never retried: stalls must remain visible.
        if (timeout || !isRetryableTransportError(error) || attempt === attempts) {
          throw error;
        }

        const delayMs = TRANSIENT_RETRY_DELAYS_MS[attempt - 1] ?? 0;
        if (delayMs > 0) {
          await sleep(delayMs);
        }
      }
    }

    throw lastError;
  }
}

export function normalizeGatewayBaseUrl(raw: string): string {
  const value = raw.trim().replace(/\/+$/g, "");
  if (value.length === 0) {
    throw new Error("KRATOS_BASE_URL is required");
  }
  return /\/v1$/i.test(value) ? value : `${value}/v1`;
}

export function createOpenAIClient(baseUrl: string, apiKey: string): OpenAI {
  return new OpenAI({
    apiKey,
    baseURL: baseUrl,
    maxRetries: 0,
    timeout: EMBEDDING_TIMEOUT_MS,
  });
}

export function createGatewayLlmClient(baseUrl: string, apiKey: string): LLMClient {
  return new OpenAICompatibleLLMClient({
    baseUrl,
    apiKey,
    requestTimeoutMs: LLM_TIMEOUT_MS,
  });
}

export async function discoverGatewayModels(openai: OpenAI): Promise<GatewayModel[]> {
  const page = await openai.models.list({ timeout: EMBEDDING_TIMEOUT_MS });
  return page.data
    .map((model) => {
      const metadata = { ...(model as unknown as Record<string, unknown>) };
      return { id: model.id, metadata };
    })
    .sort((left, right) => left.id.localeCompare(right.id));
}

function metadataDimensions(model: GatewayModel | undefined): number | undefined {
  if (model === undefined) {
    return undefined;
  }
  for (const key of [
    "dimensions",
    "embedding_dimensions",
    "embedding_dimension",
    "max_embedding_dimension",
  ]) {
    const value = model.metadata[key];
    const numeric = typeof value === "string" ? Number(value) : value;
    if (typeof numeric === "number" && Number.isInteger(numeric) && numeric > 0) {
      return numeric;
    }
  }
  return undefined;
}

export async function createModelEmbeddingRuntime(input: {
  model: string;
  models: readonly GatewayModel[];
  openai: OpenAI;
  batchSize: number;
}): Promise<ModelEmbeddingRuntime> {
  const transport = new InstrumentedEmbeddingTransport(input.openai);

  try {
    const dimensions =
      KNOWN_MODEL_DIMENSIONS[input.model] ??
      metadataDimensions(input.models.find((model) => model.id === input.model)) ??
      (await transport.probeDimensions(input.model));
    const client = new OpenAICompatibleEmbeddingClient({
      model: input.model,
      dims: dimensions,
      client: transport.adapter,
      modelReloadRetryDelaysMs: [],
      maxBatchSize: input.batchSize,
    });

    return {
      model: input.model,
      dimensions,
      client,
      transport,
    };
  } catch (error) {
    throw new ModelEmbeddingInitializationError(input.model, error, transport.calls);
  }
}

export function selectStrongModel(models: readonly GatewayModel[]): string {
  const ids = models.map((model) => model.id);
  if (ids.includes(PREFERRED_GOLD_MODEL)) {
    return PREFERRED_GOLD_MODEL;
  }

  const glm = ids
    .filter((id) => id.toLocaleLowerCase("en-US").includes("glm"))
    .sort((left, right) => right.localeCompare(left))[0];
  if (glm !== undefined) {
    return glm;
  }

  const mistralSmall = ids
    .filter((id) => id.toLocaleLowerCase("en-US").includes("mistral-small"))
    .sort((left, right) => right.localeCompare(left))[0];
  if (mistralSmall !== undefined) {
    return mistralSmall;
  }

  throw new Error(
    `No preferred gold model is available. Expected ${PREFERRED_GOLD_MODEL}, a GLM model, or mistral-small. Gateway returned: ${ids.join(", ")}`,
  );
}
