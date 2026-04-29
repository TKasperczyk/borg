import type { EmbeddingClient } from "../embeddings/index.js";
import type { GoalRecord } from "../memory/self/index.js";
import { bestVectorSimilarity } from "../retrieval/embedding-similarity.js";

export type ExecutiveContextFitByGoalId = ReadonlyMap<GoalRecord["id"], number>;

export async function computeExecutiveContextFits(input: {
  embeddingClient: EmbeddingClient;
  goals: readonly GoalRecord[];
  contextText: string;
}): Promise<ExecutiveContextFitByGoalId> {
  const activeGoals = input.goals.filter((goal) => goal.status === "active");
  const contextText = input.contextText.trim();

  if (activeGoals.length === 0 || contextText.length === 0) {
    return new Map();
  }

  const embeddings = await input.embeddingClient.embedBatch([
    contextText,
    ...activeGoals.map((goal) => `${goal.description}\n${goal.progress_notes ?? ""}`.trim()),
  ]);
  const contextVector = embeddings[0];

  if (contextVector === undefined) {
    return new Map();
  }

  return new Map(
    activeGoals.flatMap((goal, index) => {
      const goalVector = embeddings[index + 1];

      if (goalVector === undefined) {
        return [];
      }

      return [[goal.id, bestVectorSimilarity(contextVector, [goalVector])] as const];
    }),
  );
}
