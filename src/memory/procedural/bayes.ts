const LANCZOS_COEFFICIENTS = [
  0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406,
  12.507343278686905, -0.13857109526572012, 0.000009984369578019572, 0.00000015056327351493116,
] as const;

const LANCZOS_G = 7;

function logGamma(value: number): number {
  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }

  const shifted = value - 1;
  let sum = LANCZOS_COEFFICIENTS[0];

  for (let index = 1; index < LANCZOS_COEFFICIENTS.length; index += 1) {
    sum += LANCZOS_COEFFICIENTS[index]! / (shifted + index);
  }

  const temp = shifted + LANCZOS_G + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(temp) - temp + Math.log(sum);
}

function logBeta(alpha: number, beta: number): number {
  return logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
}

function betaContinuedFraction(alpha: number, beta: number, x: number): number {
  const MAX_ITERATIONS = 200;
  const EPSILON = 1e-12;
  const MIN_VALUE = 1e-30;
  let qab = alpha + beta;
  let qap = alpha + 1;
  let qam = alpha - 1;
  let currentC = 1;
  let currentD = 1 - (qab * x) / qap;

  if (Math.abs(currentD) < MIN_VALUE) {
    currentD = MIN_VALUE;
  }

  currentD = 1 / currentD;
  let fraction = currentD;

  for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration += 1) {
    const evenFactor = iteration * (beta - iteration) * x;
    let numerator = evenFactor / ((qam + 2 * iteration) * (alpha + 2 * iteration));
    currentD = 1 + numerator * currentD;

    if (Math.abs(currentD) < MIN_VALUE) {
      currentD = MIN_VALUE;
    }

    currentC = 1 + numerator / currentC;

    if (Math.abs(currentC) < MIN_VALUE) {
      currentC = MIN_VALUE;
    }

    currentD = 1 / currentD;
    fraction *= currentD * currentC;

    const oddFactor = -(alpha + iteration) * (qab + iteration) * x;
    numerator = oddFactor / ((alpha + 2 * iteration) * (qap + 2 * iteration));
    currentD = 1 + numerator * currentD;

    if (Math.abs(currentD) < MIN_VALUE) {
      currentD = MIN_VALUE;
    }

    currentC = 1 + numerator / currentC;

    if (Math.abs(currentC) < MIN_VALUE) {
      currentC = MIN_VALUE;
    }

    currentD = 1 / currentD;
    const delta = currentD * currentC;
    fraction *= delta;

    if (Math.abs(delta - 1) < EPSILON) {
      break;
    }
  }

  return fraction;
}

export function regularizedIncompleteBeta(x: number, alpha: number, beta: number): number {
  if (!Number.isFinite(x) || !Number.isFinite(alpha) || !Number.isFinite(beta)) {
    throw new TypeError("Beta inputs must be finite");
  }

  if (alpha <= 0 || beta <= 0) {
    throw new TypeError("Beta parameters must be positive");
  }

  if (x <= 0) {
    return 0;
  }

  if (x >= 1) {
    return 1;
  }

  const logFront = alpha * Math.log(x) + beta * Math.log(1 - x) - logBeta(alpha, beta);
  const front = Math.exp(logFront);

  if (x < (alpha + 1) / (alpha + beta + 2)) {
    return (front * betaContinuedFraction(alpha, beta, x)) / alpha;
  }

  return 1 - (front * betaContinuedFraction(beta, alpha, 1 - x)) / beta;
}

export function betaInverseCdf(
  probability: number,
  alpha: number,
  beta: number,
  tolerance = 1e-6,
): number {
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    throw new TypeError("Probability must be between 0 and 1");
  }

  if (probability === 0) {
    return 0;
  }

  if (probability === 1) {
    return 1;
  }

  if (probability > 0.5) {
    const mirrored = betaInverseCdf(1 - probability, beta, alpha, tolerance);
    if (mirrored <= Number.EPSILON) {
      return 1 - Number.EPSILON;
    }

    const candidate = 1 - mirrored;
    return candidate >= 1 ? 1 - Number.EPSILON : candidate;
  }

  let low = 0;
  let high = 1;
  let best = 0.5;
  let bestError = Number.POSITIVE_INFINITY;
  const MAX_ITERATIONS = 200;

  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration += 1) {
    const midpoint = (low + high) / 2;
    const cdf = regularizedIncompleteBeta(midpoint, alpha, beta);
    const error = Math.abs(cdf - probability);

    if (error < bestError) {
      best = midpoint;
      bestError = error;
    }

    if (error < tolerance) {
      return midpoint;
    }

    if (cdf < probability) {
      low = midpoint;
    } else {
      high = midpoint;
    }
  }

  if (best <= 0) {
    return Number.MIN_VALUE;
  }

  if (best >= 1) {
    return 1 - Number.EPSILON;
  }

  return best;
}

export function sampleGamma(shape: number, rng: () => number = Math.random): number {
  if (!Number.isFinite(shape) || shape <= 0) {
    throw new TypeError("Gamma shape must be positive");
  }

  if (shape < 1) {
    return sampleGamma(shape + 1, rng) * Math.pow(rng(), 1 / shape);
  }

  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);

  while (true) {
    let gaussian = 0;
    let norm = 0;

    while (norm === 0 || norm >= 1) {
      const left = rng() * 2 - 1;
      const right = rng() * 2 - 1;
      norm = left * left + right * right;
      gaussian = left * Math.sqrt((-2 * Math.log(norm)) / norm);
    }

    const factor = 1 + c * gaussian;

    if (factor <= 0) {
      continue;
    }

    const cube = factor * factor * factor;
    const uniform = rng();

    if (uniform < 1 - 0.0331 * gaussian * gaussian * gaussian * gaussian) {
      return d * cube;
    }

    if (Math.log(uniform) < 0.5 * gaussian * gaussian + d * (1 - cube + Math.log(cube))) {
      return d * cube;
    }
  }
}

export function sampleBeta(alpha: number, beta: number, rng: () => number = Math.random): number {
  const left = sampleGamma(alpha, rng);
  const right = sampleGamma(beta, rng);
  return left / (left + right);
}

export type BetaStats = {
  mean: number;
  mode: number | undefined;
  ci_95: [number, number];
};

export function computeBetaStats(alpha: number, beta: number): BetaStats {
  if (!Number.isFinite(alpha) || !Number.isFinite(beta) || alpha <= 0 || beta <= 0) {
    throw new TypeError("Beta parameters must be positive");
  }

  const mean = alpha / (alpha + beta);
  const mode = alpha > 1 && beta > 1 ? (alpha - 1) / (alpha + beta - 2) : undefined;

  return {
    mean,
    mode,
    ci_95: [betaInverseCdf(0.025, alpha, beta), betaInverseCdf(0.975, alpha, beta)],
  };
}
