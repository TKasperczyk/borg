import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

import { writeJsonFileAtomic } from "../../src/util/atomic-write.js";
import { appendDurableJsonl } from "../../src/util/durable-jsonl.js";

const VECTOR_CACHE_VERSION = 1;
const VALUE_CACHE_VERSION = 1;

type VectorDiskEntry = {
  text_sha256: string;
  vector_base64_le: string;
};

type VectorDiskRecord = {
  version: 1;
  model: string;
  dimensions: number;
  entries: VectorDiskEntry[];
};

type ValueDiskRecord<T> = {
  version: 1;
  key: string;
  value: T;
};

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function safeModelStem(model: string): string {
  const stem = model.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  return (stem || "model").slice(0, 80);
}

function parseJsonLines(path: string): unknown[] {
  if (!existsSync(path)) {
    return [];
  }

  const records: unknown[] = [];
  const contents = readFileSync(path, "utf8");
  const lines = contents.split("\n");

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    if (line === undefined || line.trim().length === 0) {
      continue;
    }

    try {
      records.push(JSON.parse(line) as unknown);
    } catch (error) {
      if (index === lines.length - 1 && !contents.endsWith("\n")) {
        // appendDurableJsonl repairs an interrupted final record before the next append.
        continue;
      }
      throw new Error(
        `Invalid JSONL cache record at ${path}:${index + 1}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  return records;
}

function encodeVector(vector: Float32Array): string {
  const bytes = Buffer.allocUnsafe(vector.length * Float32Array.BYTES_PER_ELEMENT);
  for (let index = 0; index < vector.length; index += 1) {
    bytes.writeFloatLE(vector[index] ?? 0, index * Float32Array.BYTES_PER_ELEMENT);
  }
  return bytes.toString("base64");
}

function decodeVector(encoded: string, dimensions: number): Float32Array {
  const bytes = Buffer.from(encoded, "base64");
  const expectedBytes = dimensions * Float32Array.BYTES_PER_ELEMENT;
  if (bytes.length !== expectedBytes) {
    throw new Error(
      `Cached embedding byte length mismatch: expected ${expectedBytes}, received ${bytes.length}`,
    );
  }

  const vector = new Float32Array(dimensions);
  for (let index = 0; index < dimensions; index += 1) {
    const value = bytes.readFloatLE(index * Float32Array.BYTES_PER_ELEMENT);
    if (!Number.isFinite(value)) {
      throw new Error(`Cached embedding contains a non-finite value at index ${index}`);
    }
    vector[index] = value;
  }
  return vector;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

export class VectorCache {
  readonly path: string;
  private readonly vectorsByTextHash = new Map<string, Float32Array>();
  private readonly privateDirectory: string;

  constructor(
    outDir: string,
    readonly model: string,
    readonly dimensions: number,
  ) {
    const modelHash = sha256(`${model}\0${dimensions}`).slice(0, 16);
    this.privateDirectory = join(outDir, "cache");
    this.path = join(
      outDir,
      "cache",
      "vectors",
      `${safeModelStem(model)}-${modelHash}-${dimensions}d.jsonl`,
    );
    this.load();
  }

  textHash(text: string): string {
    return sha256(text);
  }

  get(text: string): Float32Array | undefined {
    const vector = this.vectorsByTextHash.get(this.textHash(text));
    return vector === undefined ? undefined : new Float32Array(vector);
  }

  async putMany(entries: readonly { text: string; vector: Float32Array }[]): Promise<void> {
    if (entries.length === 0) {
      return;
    }

    const diskEntries = entries.map(({ text, vector }) => {
      if (vector.length !== this.dimensions) {
        throw new Error(
          `Cannot cache ${vector.length}-dimension vector for ${this.dimensions}-dimension model ${this.model}`,
        );
      }
      return {
        text_sha256: this.textHash(text),
        vector_base64_le: encodeVector(vector),
      } satisfies VectorDiskEntry;
    });
    const record: VectorDiskRecord = {
      version: VECTOR_CACHE_VERSION,
      model: this.model,
      dimensions: this.dimensions,
      entries: diskEntries,
    };

    await appendDurableJsonl(this.path, record, {
      privateDirectory: this.privateDirectory,
    });

    for (let index = 0; index < entries.length; index += 1) {
      const entry = entries[index];
      const diskEntry = diskEntries[index];
      if (entry !== undefined && diskEntry !== undefined) {
        this.vectorsByTextHash.set(diskEntry.text_sha256, new Float32Array(entry.vector));
      }
    }
  }

  private load(): void {
    for (const raw of parseJsonLines(this.path)) {
      if (!isRecord(raw)) {
        throw new Error(`Invalid vector cache record in ${this.path}`);
      }
      if (
        raw.version !== VECTOR_CACHE_VERSION ||
        raw.model !== this.model ||
        raw.dimensions !== this.dimensions ||
        !Array.isArray(raw.entries)
      ) {
        throw new Error(`Vector cache metadata mismatch in ${this.path}`);
      }

      for (const entry of raw.entries) {
        if (
          !isRecord(entry) ||
          typeof entry.text_sha256 !== "string" ||
          typeof entry.vector_base64_le !== "string"
        ) {
          throw new Error(`Invalid vector cache entry in ${this.path}`);
        }
        this.vectorsByTextHash.set(
          entry.text_sha256,
          decodeVector(entry.vector_base64_le, this.dimensions),
        );
      }
    }
  }
}

export class JsonlValueCache<T> {
  private readonly values = new Map<string, T>();

  constructor(
    readonly path: string,
    private readonly parseValue: (value: unknown) => T,
    private readonly privateDirectory: string,
  ) {
    for (const raw of parseJsonLines(path)) {
      if (
        !isRecord(raw) ||
        raw.version !== VALUE_CACHE_VERSION ||
        typeof raw.key !== "string" ||
        !("value" in raw)
      ) {
        throw new Error(`Invalid value cache record in ${path}`);
      }
      this.values.set(raw.key, this.parseValue(raw.value));
    }
  }

  get(key: string): T | undefined {
    return this.values.get(key);
  }

  async put(key: string, value: T): Promise<void> {
    const parsed = this.parseValue(value);
    const record: ValueDiskRecord<T> = {
      version: VALUE_CACHE_VERSION,
      key,
      value: parsed,
    };
    await appendDurableJsonl(this.path, record, {
      privateDirectory: this.privateDirectory,
    });
    this.values.set(key, parsed);
  }
}

export function writePrivateJson(path: string, value: unknown): void {
  writeJsonFileAtomic(path, value, { mode: 0o600, space: 2 });
}
