import type { EmbeddingClient } from "../embeddings/index.js";
import type { GoalRecord, ValueRecord } from "../memory/self/index.js";

export type RetrievalScoringFeatures = {
  goalVectors: readonly Float32Array[];
  primaryGoalVector?: Float32Array;
  valueVectors: readonly Float32Array[];
};

export type GoalScoringVector = {
  goalId: GoalRecord["id"];
  vector: Float32Array;
};

export type SelfScoringFeatureSet = {
  goalVectors: readonly GoalScoringVector[];
  valueVectors: readonly Float32Array[];
};

export function selectActiveScoringValues(
  values: readonly ValueRecord[],
  candidateLimit = 2,
): ValueRecord[] {
  const established = values.filter((value) => value.state === "established");
  const candidates = values
    .filter((value) => value.state !== "established")
    .sort((left, right) => right.priority - left.priority || left.created_at - right.created_at)
    .slice(0, candidateLimit);

  return [...established, ...candidates];
}

function goalScoringText(goal: Pick<GoalRecord, "description" | "progress_notes">): string {
  return `${goal.description}\n${goal.progress_notes ?? ""}`.trim();
}

function valueScoringText(value: Pick<ValueRecord, "label" | "description">): string {
  return `${value.label}\n${value.description}`.trim();
}

export async function buildSelfScoringFeatureSet(input: {
  embeddingClient: EmbeddingClient;
  goals: readonly GoalRecord[];
  activeValues: readonly Pick<ValueRecord, "label" | "description">[];
}): Promise<SelfScoringFeatureSet> {
  const activeGoals = input.goals.filter((goal) => goal.status === "active");
  const texts: string[] = [];
  const goalIndexes: Array<{ goalId: GoalRecord["id"]; index: number }> = [];
  const valueIndexes: number[] = [];

  for (const goal of activeGoals) {
    const text = goalScoringText(goal);

    if (text.length === 0) {
      continue;
    }

    goalIndexes.push({ goalId: goal.id, index: texts.length });
    texts.push(text);
  }

  for (const value of input.activeValues) {
    const text = valueScoringText(value);

    if (text.length === 0) {
      continue;
    }

    valueIndexes.push(texts.length);
    texts.push(text);
  }

  const embeddings = texts.length === 0 ? [] : await input.embeddingClient.embedBatch(texts);

  return {
    goalVectors: goalIndexes.flatMap(({ goalId, index }) => {
      const vector = embeddings[index];
      return vector === undefined ? [] : [{ goalId, vector }];
    }),
    valueVectors: valueIndexes.flatMap((index) => {
      const vector = embeddings[index];
      return vector === undefined ? [] : [vector];
    }),
  };
}

export function toRetrievalScoringFeatures(input: {
  selfFeatures: SelfScoringFeatureSet;
  primaryGoalId?: GoalRecord["id"] | null;
}): RetrievalScoringFeatures {
  const primaryGoalVector =
    input.primaryGoalId === null || input.primaryGoalId === undefined
      ? undefined
      : input.selfFeatures.goalVectors.find((item) => item.goalId === input.primaryGoalId)?.vector;

  return {
    goalVectors: input.selfFeatures.goalVectors.map((item) => item.vector),
    ...(primaryGoalVector === undefined ? {} : { primaryGoalVector }),
    valueVectors: input.selfFeatures.valueVectors,
  };
}

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
    const text = valueScoringText(value);

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
