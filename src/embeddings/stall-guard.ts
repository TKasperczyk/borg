/* Stall guard for embedding calls: short per-attempt timeout + immediate retry.
 *
 * The inference gateway's embedding backend intermittently hangs a request
 * indefinitely while healthy calls complete in ~0.2-0.35s. Without a guard,
 * a stalled call inherits the SDK's default timeout (minutes) and any caller
 * with a latency budget — most critically the sidecar recall path and its
 * client's hard recall cap — times out instead. A stalled call retried on a fresh
 * request almost always completes at healthy latency, so the guard caps each
 * attempt and retries immediately. Non-timeout errors are NOT retried here:
 * the wrapped client already handles model-reload retries and the SDK retries
 * retryable HTTP failures.
 */
import { ConfigError, EmbeddingError } from "../util/errors.js";

import type { EmbeddingClient } from "./index.js";

export type EmbeddingStallGuardOptions = {
  // Per-attempt cap for single-input calls (embed / one-element batches).
  timeoutMs?: number;
  // Per-attempt cap for multi-input batches (ingestion/maintenance paths),
  // which legitimately take longer than a single query embedding.
  batchTimeoutMs?: number;
  // Additional attempts after a stalled one.
  maxRetries?: number;
};

const DEFAULT_TIMEOUT_MS = 1000;
const DEFAULT_BATCH_TIMEOUT_MS = 20_000;
const DEFAULT_MAX_RETRIES = 1;
// Large maintenance batches (tag re-embeds, procedural evidence) legitimately
// take longer than the base batch cap; scale the deadline with input count so
// the guard only ever fires on genuine stalls, and bound the total so a stall
// can never approach the shared client's 120s request timeout.
const BATCH_PER_INPUT_MS = 100;
const MAX_BATCH_TIMEOUT_MS = 60_000;

class EmbeddingStallTimeout extends Error {
  constructor(timeoutMs: number) {
    super(`embedding call exceeded ${timeoutMs}ms`);
    this.name = "EmbeddingStallTimeout";
  }
}

function requirePositiveInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new ConfigError(`${label} must be a positive integer`);
  }

  return value;
}

function requireNonNegativeInteger(value: number, label: string): number {
  if (!Number.isInteger(value) || value < 0) {
    throw new ConfigError(`${label} must be a non-negative integer`);
  }

  return value;
}

async function raceWithStallTimeout<T>(operation: Promise<T>, timeoutMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;

  // The abandoned attempt keeps running after a timeout loss; swallow its
  // eventual rejection so it cannot surface as an unhandled rejection.
  operation.catch(() => undefined);

  try {
    return await Promise.race([
      operation,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new EmbeddingStallTimeout(timeoutMs)), timeoutMs);
        timer.unref?.();
      }),
    ]);
  } finally {
    if (timer !== undefined) {
      clearTimeout(timer);
    }
  }
}

export class StallGuardEmbeddingClient implements EmbeddingClient {
  private readonly inner: EmbeddingClient;
  private readonly timeoutMs: number;
  private readonly batchTimeoutMs: number;
  private readonly maxRetries: number;

  constructor(inner: EmbeddingClient, options: EmbeddingStallGuardOptions = {}) {
    this.inner = inner;
    this.timeoutMs = requirePositiveInteger(
      options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
      "Embedding stall-guard timeoutMs",
    );
    this.batchTimeoutMs = requirePositiveInteger(
      options.batchTimeoutMs ?? DEFAULT_BATCH_TIMEOUT_MS,
      "Embedding stall-guard batchTimeoutMs",
    );
    this.maxRetries = requireNonNegativeInteger(
      options.maxRetries ?? DEFAULT_MAX_RETRIES,
      "Embedding stall-guard maxRetries",
    );
  }

  async embed(text: string): Promise<Float32Array> {
    return this.runGuarded(() => this.inner.embed(text), this.timeoutMs);
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    if (texts.length === 0) {
      return this.inner.embedBatch(texts);
    }

    const timeoutMs =
      texts.length === 1
        ? this.timeoutMs
        : Math.min(this.batchTimeoutMs + BATCH_PER_INPUT_MS * texts.length, MAX_BATCH_TIMEOUT_MS);

    return this.runGuarded(() => this.inner.embedBatch(texts), timeoutMs);
  }

  private async runGuarded<T>(run: () => Promise<T>, timeoutMs: number): Promise<T> {
    const attempts = this.maxRetries + 1;
    let lastTimeout: EmbeddingStallTimeout | undefined;

    for (let attempt = 1; attempt <= attempts; attempt += 1) {
      try {
        return await raceWithStallTimeout(run(), timeoutMs);
      } catch (error) {
        if (!(error instanceof EmbeddingStallTimeout)) {
          throw error;
        }

        lastTimeout = error;

        if (attempt < attempts) {
          // Rescued stalls are otherwise invisible; keep them countable in logs.
          console.warn(
            `embedding stall-guard: attempt ${attempt} exceeded ${timeoutMs}ms, retrying`,
          );
        }
      }
    }

    throw new EmbeddingError(
      `Embedding call stalled: ${attempts} attempt(s) exceeded ${timeoutMs}ms each`,
      { cause: lastTimeout },
    );
  }
}
