import type { EmbeddingClient } from "../embeddings/index.js";
import type { GoalRecord } from "../memory/self/index.js";
import { bestVectorSimilarity } from "../retrieval/embedding-similarity.js";
import type { GoalScoringVector } from "../retrieval/scoring-features.js";

export type ExecutiveContextFitByGoalId = ReadonlyMap<GoalRecord["id"], number>;

export async function computeExecutiveContextFits(input: {
  embeddingClient: EmbeddingClient;
  goalVectors: readonly GoalScoringVector[];
  contextText: string;
}): Promise<ExecutiveContextFitByGoalId> {
  const contextText = input.contextText.trim();

  if (input.goalVectors.length === 0 || contextText.length === 0) {
    return new Map();
  }

  const contextVector = await input.embeddingClient.embed(contextText);

  return new Map(
    input.goalVectors.map((goal) => [
      goal.goalId,
      bestVectorSimilarity(contextVector, [goal.vector]),
    ]),
  );
}
