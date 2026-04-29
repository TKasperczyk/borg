import type { EmbeddingClient } from "../embeddings/index.js";
import type { ValueRecord } from "../memory/self/index.js";

export type RetrievalScoringFeatures = {
  goalVectors: readonly Float32Array[];
  primaryGoalVector?: Float32Array;
  valueVectors: readonly Float32Array[];
};

export async function buildRetrievalScoringFeatures(input: {
  embeddingClient: EmbeddingClient;
  goalDescriptions: readonly string[];
  primaryGoalDescription?: string;
  activeValues: readonly Pick<ValueRecord, "label" | "description">[];
}): Promise<RetrievalScoringFeatures> {
  const texts: string[] = [];
  const goalIndexes: number[] = [];
  let primaryGoalIndex: number | null = null;
  const valueIndexes: number[] = [];

  for (const description of input.goalDescriptions) {
    const text = description.trim();

    if (text.length === 0) {
      continue;
    }

    goalIndexes.push(texts.length);
    texts.push(text);
  }

  if (input.primaryGoalDescription !== undefined) {
    const text = input.primaryGoalDescription.trim();

    if (text.length > 0) {
      primaryGoalIndex = texts.length;
      texts.push(text);
    }
  }

  for (const value of input.activeValues) {
    const text = `${value.label}\n${value.description}`.trim();

    if (text.length === 0) {
      continue;
    }

    valueIndexes.push(texts.length);
    texts.push(text);
  }

  const embeddings = await input.embeddingClient.embedBatch(texts);

  return {
    goalVectors: goalIndexes.flatMap((index) => {
      const vector = embeddings[index];
      return vector === undefined ? [] : [vector];
    }),
    ...(primaryGoalIndex === null || embeddings[primaryGoalIndex] === undefined
      ? {}
      : { primaryGoalVector: embeddings[primaryGoalIndex] }),
    valueVectors: valueIndexes.flatMap((index) => {
      const vector = embeddings[index];
      return vector === undefined ? [] : [vector];
    }),
  };
}
