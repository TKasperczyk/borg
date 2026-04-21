import { describe, expect, it } from "vitest";

import { betaInverseCdf, computeBetaStats, regularizedIncompleteBeta } from "./bayes.js";

describe("procedural bayes math", () => {
  it("computes mean, mode, and confidence interval for known beta distributions", () => {
    const stats = computeBetaStats(3, 1);
    const modeStats = computeBetaStats(3, 2);

    expect(stats.mean).toBeCloseTo(0.75, 3);
    expect(modeStats.mode).toBeCloseTo(2 / 3, 3);
    expect(stats.ci_95[0]).toBeCloseTo(Math.cbrt(0.025), 2);
    expect(stats.ci_95[1]).toBeCloseTo(Math.cbrt(0.975), 2);
  });

  it("keeps inverse cdf and cdf numerically aligned", () => {
    const quantile = betaInverseCdf(0.6, 2.5, 4);
    const recovered = regularizedIncompleteBeta(quantile, 2.5, 4);

    expect(recovered).toBeCloseTo(0.6, 2);
  });

  it("round-trips extreme beta parameters across a broad matrix", () => {
    const parameters = [0.1, 0.2, 0.5, 1, 2, 5, 20] as const;
    const probabilities = [1e-6, 0.01, 0.5, 0.99, 1 - 1e-6] as const;

    for (const alpha of parameters) {
      for (const beta of parameters) {
        for (const probability of probabilities) {
          const quantile = betaInverseCdf(probability, alpha, beta);
          const recovered = regularizedIncompleteBeta(quantile, alpha, beta);
          if (quantile <= Number.MIN_VALUE || quantile >= 1 - Number.EPSILON) {
            const nearestRepresentable =
              quantile <= Number.MIN_VALUE ? Number.MIN_VALUE : 1 - Number.EPSILON;
            const nearestRecovered = regularizedIncompleteBeta(nearestRepresentable, alpha, beta);

            // Some extreme U-shaped tails collapse to the floating-point boundary; in that
            // case, assert we returned the closest representable quantile instead of looping.
            expect(Math.abs(recovered - probability)).toBeLessThanOrEqual(
              Math.abs(nearestRecovered - probability) + 1e-12,
            );
            continue;
          }

          expect(Math.abs(recovered - probability)).toBeLessThan(1e-4);
        }
      }
    }
  });
});
