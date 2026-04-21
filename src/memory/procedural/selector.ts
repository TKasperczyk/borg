import { sampleBeta } from "./bayes.js";
import type {
  SkillRecord,
  SkillSelectionCandidate,
  SkillSelectionResult,
  SkillSearchCandidate,
  SkillStats,
} from "./types.js";
import type { SkillRepository } from "./repository.js";

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
  rng?: () => number;
  sampler?: (alpha: number, beta: number, rng: () => number) => number;
};

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
    } = {},
  ): Promise<SkillSelectionResult | null> {
    const limit = Math.max(1, options.k ?? 10);
    const exploreFraction = Math.max(0, Math.min(1, options.exploreFraction ?? 0));
    const candidates = await this.options.repository.searchByContext(text, limit);

    if (candidates.length === 0) {
      return null;
    }

    const evaluatedCandidates = candidates
      .map((candidate) => {
        const stats = this.options.repository.getStats(candidate.skill.id);
        return {
          ...candidate,
          stats,
          sampledValue: this.sampler(candidate.skill.alpha, candidate.skill.beta, this.rng),
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
    };
  }
}

export type { SkillRecord, SkillSearchCandidate, SkillSelectionCandidate, SkillStats };
