import { StorageError, type BorgTypedErrorOptions } from "../util/errors.js";

type CodecErrorFactory = (message: string, options: BorgTypedErrorOptions) => Error;

export type JsonArrayCodecOptions = {
  errorCode: string;
  errorMessage: (label: string) => string;
  createError?: CodecErrorFactory;
};

export type Float32ArrayCodecOptions = {
  arrayLikeErrorMessage: string;
  nonFiniteErrorMessage: string;
  errorCode: string;
  createError?: CodecErrorFactory;
};

function createStorageError(message: string, options: BorgTypedErrorOptions): Error {
  return new StorageError(message, options);
}

export function parseJsonArray<T>(
  value: string,
  label: string,
  options: JsonArrayCodecOptions,
): T[] {
  try {
    const parsed = JSON.parse(value) as unknown;

    if (!Array.isArray(parsed)) {
      throw new TypeError(`${label} must be an array`);
    }

    return parsed as T[];
  } catch (error) {
    throw (options.createError ?? createStorageError)(options.errorMessage(label), {
      cause: error,
      code: options.errorCode,
    });
  }
}

export function quoteSqlString(value: string): string {
  return `'${value.replaceAll("'", "''")}'`;
}

export function toFloat32Array(
  vector: unknown,
  options: Float32ArrayCodecOptions,
): Float32Array {
  if (vector instanceof Float32Array) {
    return vector;
  }

  const candidate: unknown[] | null = Array.isArray(vector)
    ? vector
    : ArrayBuffer.isView(vector)
      ? Array.from(vector as unknown as ArrayLike<unknown>)
      : vector !== null &&
          typeof vector === "object" &&
          "length" in vector &&
          typeof vector.length === "number"
        ? Array.from(vector as ArrayLike<unknown>)
        : null;

  if (candidate === null) {
    throw (options.createError ?? createStorageError)(options.arrayLikeErrorMessage, {
      code: options.errorCode,
    });
  }

  const values = candidate.map((value) => {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw (options.createError ?? createStorageError)(options.nonFiniteErrorMessage, {
        code: options.errorCode,
      });
    }

    return value;
  });

  return Float32Array.from(values);
}
