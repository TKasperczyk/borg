import { z } from "zod";
import { isPlainRecord } from "../util/guards.js";
import {
  buildEpisodeEmbeddingText,
  consolidationEmbeddingInputSchema,
} from "../memory/episodic/protected-lines.js";
import { buildNodeEmbeddingText } from "../memory/semantic/embedding-text.js";
import { semanticObservationMetadataSchema } from "../memory/semantic/types.js";
import { EmbeddingBankError } from "./bank-profile.js";
import type { EmbeddingClient } from "./index.js";

export type SerializedEmbeddingPreparer = (payload: unknown) => Promise<unknown>;

/** Structural JSON inspection only: these are serialized storage fields. */
export function serializedEmbeddingPaths(value: unknown, path = "$"): string[] {
  if (Array.isArray(value)) {
    return value.flatMap((item, index) => serializedEmbeddingPaths(item, `${path}[${index}]`));
  }
  if (!isPlainRecord(value)) return [];
  return Object.entries(value).flatMap(([key, item]) =>
    key === "embedding" && (Array.isArray(item) || ArrayBuffer.isView(item))
      ? [`${path}.${key}`]
      : serializedEmbeddingPaths(item, `${path}.${key}`),
  );
}

export function assertUsableEmbedding(vector: ArrayLike<number>, dimensions: number): void {
  let norm = 0;
  for (let index = 0; index < vector.length; index += 1) {
    const value = vector[index];
    if (value === undefined || !Number.isFinite(value)) {
      throw new EmbeddingBankError("Embedding contains non-finite values", {
        code: "EMBEDDING_VECTOR_INVALID",
      });
    }
    norm += value * value;
  }
  if (vector.length !== dimensions || norm === 0 || !Number.isFinite(norm)) {
    throw new EmbeddingBankError("Embedding has invalid dimensions or zero norm", {
      code: "EMBEDDING_VECTOR_INVALID",
    });
  }
}

/**
 * Historical payloads have no trustworthy model label. Recompute their vectors
 * before any writes, including same-dimension model changes. Never mutate the
 * saved plan/audit itself, or trust a vector merely because its length fits.
 */
export async function refreshSerializedEmbeddings(
  payload: unknown,
  client: EmbeddingClient,
  resolveNode?: (id: string) => Promise<unknown>,
): Promise<unknown> {
  async function visit(
    value: unknown,
    parent: Record<string, unknown>,
    path: string,
  ): Promise<unknown> {
    if (Array.isArray(value)) {
      const result: unknown[] = [];
      for (const [index, item] of value.entries())
        result.push(await visit(item, parent, `${path}[${index}]`));
      return result;
    }
    if (!isPlainRecord(value)) return value;
    const output: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value)) {
      if (key !== "embedding") output[key] = await visit(item, value, `${path}.${key}`);
    }
    if (!("embedding" in value)) return output;
    let body = value;
    if (typeof value.label !== "string" && typeof parent.node_id === "string" && resolveNode) {
      const existing = await resolveNode(parent.node_id);
      if (isPlainRecord(existing)) body = { ...existing, ...value };
    }
    let text: string | undefined;
    if (typeof body.label === "string" && typeof body.description === "string") {
      text = buildNodeEmbeddingText({
        label: body.label,
        description: body.description,
        aliases: z.array(z.string()).parse(body.aliases ?? []),
        observationMetadata: semanticObservationMetadataSchema
          .nullable()
          .parse(body.observation_metadata ?? null),
      });
    } else if (typeof body.title === "string" && typeof body.narrative === "string") {
      text = buildEpisodeEmbeddingText({
        title: body.title,
        narrative: body.narrative,
        tags: z.array(z.string()).parse(body.tags ?? []),
        participants: z.array(z.string()).parse(body.participants ?? []),
        episode_kind: z.string().nullable().optional().parse(body.episode_kind),
        consolidation_embedding_input: consolidationEmbeddingInputSchema
          .nullable()
          .optional()
          .parse(body.consolidation_embedding_input),
      });
    }
    if (text === undefined || text.trim().length === 0) {
      throw new EmbeddingBankError(
        `Serialized vector at ${path}.embedding has no recoverable embedding text; regenerate or reject this item`,
        {
          code: "SERIALIZED_EMBEDDING_INCOMPATIBLE",
        },
      );
    }
    const embedding = await client.embed(text);
    assertUsableEmbedding(embedding, client.profile?.dimensions ?? embedding.length);
    output.embedding = Array.from(embedding);
    return output;
  }
  return visit(payload, {}, "$");
}
