import { EmbeddingError, findInErrorCauseChain } from "../util/errors.js";

export type EmbeddingRequestState = {
  retryAt: number | null;
  inFlight: number;
  oldestRequestStartedAt: number | null;
};

export class EmbeddingBackoffError extends EmbeddingError {
  constructor(
    readonly retryAt: number,
    cause?: unknown,
  ) {
    super(`Embedding provider backoff until ${new Date(retryAt).toISOString()}`, {
      code: "EMBEDDING_BACKOFF",
      cause,
    });
  }
}

export function embeddingHttpError(
  error: unknown,
): { status: number; headers?: unknown } | undefined {
  return findInErrorCauseChain(
    error,
    (value): value is { status: number; headers?: unknown } =>
      typeof value === "object" &&
      value !== null &&
      "status" in value &&
      typeof value.status === "number",
  );
}

// Provider protocol metadata only. No request or model-output text is inspected.
export function embeddingRetryAt(headers: unknown, now: number): number | null {
  if (headers === null || typeof headers !== "object") return null;
  const read = (name: string): string | undefined => {
    if ("get" in headers && typeof headers.get === "function") {
      const value: unknown = headers.get(name);
      return typeof value === "string" ? value : undefined;
    }
    const entry = Object.entries(headers).find(([key]) => key.toLowerCase() === name);
    return typeof entry?.[1] === "string" ? entry[1] : undefined;
  };
  const milliseconds = read("retry-after-ms");
  const seconds = read("retry-after");
  const numericDelay = (value: string | undefined, scale: number): number | null => {
    if (value === undefined || !/^\d+(?:\.\d+)?$/.test(value.trim())) return null;
    const delay = Number(value) * scale;
    const at = now + Math.ceil(delay);
    return Number.isSafeInteger(at) && at <= 8.64e15 ? at : null;
  };
  const numeric = numericDelay(milliseconds, 1) ?? numericDelay(seconds, 1000);
  if (numeric !== null) return numeric;
  if (seconds === undefined || /^[-+\d.]+$/.test(seconds.trim())) return null;
  const date = Date.parse(seconds);
  return Number.isFinite(date) ? Math.max(now, date) : null;
}
