import { z } from "zod";

export const QWEN_SIMILARITY_MODEL = "generative-apis/qwen3-embedding-8b";
export const BGE_SIMILARITY_MODEL = "scw/bge-m3";
const cosine = z.number().min(0).max(1);
export const similarityValuesSchema = z
  .object({
    actionPersistenceDuplicate: cosine,
    openQuestionRecall: cosine,
    commitmentEvidence: cosine,
    actionThread: cosine,
    skillSelection: cosine,
    consolidationSimilarity: z.number().positive(),
    consolidationDiameter: z.number().min(0).max(2),
    consolidationTemporalBypass: cosine,
    reflectionGoalAndTagGrouping: cosine,
    skillSynthesisDuplicate: cosine,
    ruminatorDuplicate: cosine,
    semanticRecall: cosine,
    semanticRevision: cosine,
    generationRepeatedInput: cosine,
    goalPromotionDuplicate: cosine,
    openQuestionDuplicateBackstop: cosine,
    observedEventTopic: cosine,
    semanticDuplicateReview: cosine,
    proceduralEvidenceCluster: cosine,
    semanticExtractionDuplicate: cosine,
    pendingActionMerge: cosine,
    reflectionInsightDuplicate: cosine,
    // Fused rawScore, not cosine; nonpositive values disable abstention.
    recallAbstain: z.number().finite(),
  })
  .strict();
export type SimilarityThresholds = Readonly<z.infer<typeof similarityValuesSchema>>;

const qwen: SimilarityThresholds = Object.freeze({
  actionPersistenceDuplicate: 0.85,
  openQuestionRecall: 0.01,
  commitmentEvidence: 0.3,
  actionThread: 0.85,
  skillSelection: 0.5,
  consolidationSimilarity: 0.82,
  consolidationDiameter: 0.18,
  consolidationTemporalBypass: 0.97,
  reflectionGoalAndTagGrouping: 0.82,
  skillSynthesisDuplicate: 0.88,
  ruminatorDuplicate: 0.9,
  semanticRecall: 0.01,
  semanticRevision: 0.01,
  generationRepeatedInput: 0.96,
  goalPromotionDuplicate: 0.9,
  openQuestionDuplicateBackstop: 0.9,
  observedEventTopic: 0.45,
  semanticDuplicateReview: 0.9,
  proceduralEvidenceCluster: 0.85,
  semanticExtractionDuplicate: 0.88,
  pendingActionMerge: 0.85,
  reflectionInsightDuplicate: 0.88,
  recallAbstain: 0,
});

export const DEFAULT_SIMILARITY_PROFILES: Readonly<Record<string, SimilarityThresholds>> =
  Object.freeze({
    [QWEN_SIMILARITY_MODEL]: qwen,
    [BGE_SIMILARITY_MODEL]: Object.freeze({
      ...qwen,
      // Phase A: team-agent-ai, 1321 episodes / 2527 semantic nodes.
      // Weak/empty populations and the 0.01 floors retain Qwen settings.
      consolidationSimilarity: 0.76,
      consolidationDiameter: 0.24,
      consolidationTemporalBypass: 0.94,
      reflectionGoalAndTagGrouping: 0.76,
      semanticDuplicateReview: 0.87,
      semanticExtractionDuplicate: 0.84,
      reflectionInsightDuplicate: 0.84,
    }),
  });

export const similarityConfigSchema = z
  .object({
    profiles: z
      .record(z.string().min(1), similarityValuesSchema)
      .default({})
      .transform((profiles) => ({ ...DEFAULT_SIMILARITY_PROFILES, ...profiles })),
    overrides: similarityValuesSchema.partial().default({}),
  })
  .strict()
  .prefault({});
export type SimilarityConfig = z.infer<typeof similarityConfigSchema>;

// Structural so lower-level components can use a client profile without importing
// the full configuration graph. Library open replaces embedding.model with the
// effective client identity validated by the bank guard before passing config on.
export type SimilarityConfigSource = {
  embedding?: { model: string };
  similarity?: {
    profiles?: Readonly<Record<string, SimilarityThresholds>>;
    overrides?: Partial<SimilarityThresholds>;
  };
  generation?: { evidenceLedger?: { actionThreadSimilarityThreshold?: number } };
  procedural?: { skillSelectionMinSimilarity?: number };
  offline?: {
    consolidator?: {
      similarityThreshold?: number;
      maxClusterDiameter?: number;
      highSimilarityTemporalBypassThreshold?: number;
    };
    reflector?: { goalSimilarityThreshold?: number };
    proceduralSynthesizer?: { dedupThreshold?: number };
    ruminator?: { duplicateSimilarityThreshold?: number };
  };
};

function resolveSimilarity(config: SimilarityConfigSource) {
  const model = config.embedding?.model ?? QWEN_SIMILARITY_MODEL;
  const profiles = { ...DEFAULT_SIMILARITY_PROFILES, ...config.similarity?.profiles };
  const fallback = !Object.hasOwn(profiles, model);
  const profileModel = fallback ? QWEN_SIMILARITY_MODEL : model;
  const legacy: Partial<SimilarityThresholds> = {
    actionThread: config.generation?.evidenceLedger?.actionThreadSimilarityThreshold,
    skillSelection: config.procedural?.skillSelectionMinSimilarity,
    consolidationSimilarity: config.offline?.consolidator?.similarityThreshold,
    consolidationDiameter: config.offline?.consolidator?.maxClusterDiameter,
    consolidationTemporalBypass:
      config.offline?.consolidator?.highSimilarityTemporalBypassThreshold,
    reflectionGoalAndTagGrouping: config.offline?.reflector?.goalSimilarityThreshold,
    skillSynthesisDuplicate: config.offline?.proceduralSynthesizer?.dedupThreshold,
    ruminatorDuplicate: config.offline?.ruminator?.duplicateSimilarityThreshold,
  };
  const overrides: Partial<z.infer<typeof similarityValuesSchema>> = {};
  for (const source of [legacy, config.similarity?.overrides ?? {}]) {
    for (const key of Object.keys(source) as (keyof SimilarityThresholds)[]) {
      if (source[key] !== undefined) overrides[key] = source[key];
    }
  }
  return {
    model,
    profileModel,
    fallback,
    overrides,
    values: Object.freeze({ ...profiles[profileModel]!, ...overrides }),
  };
}

const warnedModels = new Set<string>();
function profileLine(resolution: ReturnType<typeof resolveSimilarity>, context: string): string {
  return `${context}: similarity model=${JSON.stringify(resolution.model)} profile=${JSON.stringify(resolution.profileModel)} fallback=${resolution.fallback} overrides=${JSON.stringify(resolution.overrides)}`;
}

/** The single value accessor. No lexical model aliases or dimension guesses. */
export function similarityThresholds(config: SimilarityConfigSource = {}): SimilarityThresholds {
  const resolved = resolveSimilarity(config);
  if (resolved.fallback && !warnedModels.has(resolved.model)) {
    warnedModels.add(resolved.model);
    console.warn(profileLine(resolved, "borg (unknown embedding model)"));
  }
  return resolved.values;
}

/** One startup line, including a named warning when the Qwen fallback is used. */
export function logSimilarityProfile(config: SimilarityConfigSource, context: string): void {
  const resolved = resolveSimilarity(config);
  if (resolved.fallback) {
    warnedModels.add(resolved.model);
    console.warn(profileLine(resolved, context));
  } else console.info(profileLine(resolved, context));
}
