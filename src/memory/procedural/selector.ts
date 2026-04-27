import { sampleBeta } from "./bayes.js";
import { proceduralContextSchema, type ProceduralContext } from "./context.js";
import type {
  SkillRecord,
  SkillSelectionCandidate,
  SkillSelectionResult,
  SkillSearchCandidate,
  SkillStats,
} from "./types.js";
import type { ProceduralContextStatsRepository, SkillRepository } from "./repository.js";

function compareCandidates(left: SkillSelectionCandidate, right: SkillSelectionCandidate): number {
  if (left.sampledValue !== right.sampledValue) {
    return right.sampledValue - left.sampledValue;
  }

  if (left.skill.attempts !== right.skill.attempts) {
    return left.skill.attempts - right.skill.attempts;
  }

  return right.similarity - left.similarity;
}

export type SkillSelectorOptions = {
  repository: SkillRepository;
  contextStatsRepository?: Pick<ProceduralContextStatsRepository, "batchGetContextStats">;
  rng?: () => number;
  sampler?: (alpha: number, beta: number, rng: () => number) => number;
  minSimilarity?: number;
};

function computeContextualPosterior(
  skill: SkillRecord,
  contextStats: NonNullable<SkillSelectionCandidate["contextStats"]>,
): { alpha: number; beta: number } {
  const priorAlpha = Math.max(1, skill.alpha - skill.successes);
  const priorBeta = Math.max(1, skill.beta - skill.failures);
  const globalOtherSuccesses = Math.max(0, skill.successes - contextStats.successes);
  const globalOtherFailures = Math.max(0, skill.failures - contextStats.failures);

  return {
    alpha: priorAlpha + contextStats.successes + 0.25 * globalOtherSuccesses,
    beta: priorBeta + contextStats.failures + 0.25 * globalOtherFailures,
  };
}

export class SkillSelector {
  private readonly rng: () => number;
  private readonly sampler: (alpha: number, beta: number, rng: () => number) => number;

  constructor(private readonly options: SkillSelectorOptions) {
    this.rng = options.rng ?? Math.random;
    this.sampler = options.sampler ?? sampleBeta;
  }

  async select(
    text: string,
    options: {
      k?: number;
      exploreFraction?: number;
      minSimilarity?: number;
      proceduralContext?: ProceduralContext | null;
    } = {},
  ): Promise<SkillSelectionResult | null> {
    const limit = Math.max(1, options.k ?? 10);
    const exploreFraction = Math.max(0, Math.min(1, options.exploreFraction ?? 0));
    const minSimilarity = Math.max(
      0,
      Math.min(1, options.minSimilarity ?? this.options.minSimilarity ?? 0.5),
    );
    const candidates = await this.options.repository.searchByContext(text, limit);
    const eligibleCandidates = candidates.filter(
      (candidate) => candidate.similarity >= minSimilarity,
    );
    const proceduralContext =
      options.proceduralContext === null || options.proceduralContext === undefined
        ? null
        : proceduralContextSchema.parse(options.proceduralContext);

    if (eligibleCandidates.length === 0) {
      return null;
    }

    const contextStatsBySkill: ReadonlyMap<
      SkillRecord["id"],
      NonNullable<SkillSelectionCandidate["contextStats"]>
    > =
      proceduralContext === null || this.options.contextStatsRepository === undefined
        ? new Map<SkillRecord["id"], NonNullable<SkillSelectionCandidate["contextStats"]>>()
        : this.options.contextStatsRepository.batchGetContextStats(
            proceduralContext.context_key,
            eligibleCandidates.map((candidate) => candidate.skill.id),
          );

    const evaluatedCandidates = eligibleCandidates
      .map((candidate) => {
        const stats = this.options.repository.getStats(candidate.skill.id);
        const contextStats = contextStatsBySkill.get(candidate.skill.id) ?? null;
        const sampledPosterior =
          contextStats === null
            ? { alpha: candidate.skill.alpha, beta: candidate.skill.beta }
            : computeContextualPosterior(candidate.skill, contextStats);
        return {
          ...candidate,
          stats,
          contextStats,
          sampledAlpha: sampledPosterior.alpha,
          sampledBeta: sampledPosterior.beta,
          sampledValue: this.sampler(sampledPosterior.alpha, sampledPosterior.beta, this.rng),
        } satisfies SkillSelectionCandidate;
      })
      .sort(compareCandidates);

    const selected =
      evaluatedCandidates.length > 1 && exploreFraction > 0 && this.rng() < exploreFraction
        ? evaluatedCandidates[1]!
        : evaluatedCandidates[0]!;

    return {
      skill: selected.skill,
      sampledValue: selected.sampledValue,
      evaluatedCandidates,
      ...(proceduralContext === null ? {} : { proceduralContext }),
    };
  }
}

export type { SkillRecord, SkillSearchCandidate, SkillSelectionCandidate, SkillStats };
