import type { EmbeddingClient } from "../../src/embeddings/index.js";

const DEFAULT_DIMS = 64;

function hashString(input: string): number {
  let hash = 2_166_136_261;

  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
    hash = Math.imul(hash, 16_777_619);
  }

  return hash >>> 0;
}

function nextRandom(seed: number): number {
  let value = seed >>> 0;
  value ^= value << 13;
  value ^= value >>> 17;
  value ^= value << 5;
  return value >>> 0;
}

function normalizeVector(vector: Float32Array): Float32Array {
  let norm = 0;

  for (const value of vector) {
    norm += value * value;
  }

  if (norm === 0) {
    return vector;
  }

  const scale = 1 / Math.sqrt(norm);

  for (let index = 0; index < vector.length; index += 1) {
    vector[index] = (vector[index] ?? 0) * scale;
  }

  return vector;
}

function tokenizeEvalText(text: string): Set<string> {
  return new Set(
    text
      .normalize("NFKC")
      .toLowerCase()
      .split(/[^\p{L}\p{N}_-]+/u)
      .map((token) => token.trim())
      .filter((token) => token.length >= 2),
  );
}

/**
 * Deterministic token-hash embeddings for eval fixtures. This is only for
 * stable substrate benchmarks and should not be used in production.
 */
export class DeterministicEmbeddingClient implements EmbeddingClient {
  private readonly tokenCache = new Map<string, Float32Array>();

  constructor(private readonly dims = DEFAULT_DIMS) {}

  async embed(text: string): Promise<Float32Array> {
    const [embedding] = await this.embedBatch([text]);

    if (embedding === undefined) {
      throw new Error("DeterministicEmbeddingClient returned no embedding");
    }

    return embedding;
  }

  async embedBatch(texts: readonly string[]): Promise<Float32Array[]> {
    return texts.map((text) => this.vectorize(text));
  }

  private vectorize(text: string): Float32Array {
    const unigramTokens = [...tokenizeEvalText(text)].sort();
    const bigramTokens: string[] = [];

    for (let index = 0; index < unigramTokens.length - 1; index += 1) {
      const left = unigramTokens[index];
      const right = unigramTokens[index + 1];

      if (left !== undefined && right !== undefined) {
        bigramTokens.push(`${left}_${right}`);
      }
    }

    const tokens = [...unigramTokens, ...bigramTokens];
    const vector = new Float32Array(this.dims);

    if (tokens.length === 0) {
      return vector;
    }

    for (const token of tokens) {
      const tokenVector = this.getTokenVector(token);

      for (let index = 0; index < vector.length; index += 1) {
        vector[index] = (vector[index] ?? 0) + (tokenVector[index] ?? 0);
      }
    }

    return normalizeVector(vector);
  }

  private getTokenVector(token: string): Float32Array {
    const cached = this.tokenCache.get(token);

    if (cached !== undefined) {
      return cached;
    }

    const vector = new Float32Array(this.dims);
    let seed = hashString(token);

    for (let index = 0; index < vector.length; index += 1) {
      seed = nextRandom(seed);
      vector[index] = (seed / 0xffffffff) * 2 - 1;
    }

    const normalized = normalizeVector(vector);
    this.tokenCache.set(token, normalized);
    return normalized;
  }
}
