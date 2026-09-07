import { createHash } from "node:crypto";
import { setImmediate } from "node:timers/promises";
import { MeasurementError, type Families, type VectorRow } from "./bank.js";

export const CUTOFFS = [0.8, 0.85, 0.9, 0.95, 0.97] as const;

// Linear sample quantiles: h=(n-1)*p. Input is sorted ascending and finite.
export function quantile(sorted: readonly number[], p: number): number | null {
  if (!Number.isFinite(p) || p < 0 || p > 1)
    throw new MeasurementError("Percentile must be in [0,1]");
  if (sorted.length === 0) return null;
  const h = (sorted.length - 1) * p;
  const lo = Math.floor(h);
  const a = sorted[lo]!;
  return a + (sorted[Math.ceil(h)]! - a) * (h - lo);
}

// Inverse of the linear quantile curve. Ties use the middle rank; outside
// observed support clips to an endpoint (reported explicitly by the caller).
export function percentileRank(sorted: readonly number[], value: number): number | null {
  if (sorted.length === 0) return null;
  if (value < sorted[0]!) return 0;
  if (value > sorted[sorted.length - 1]!) return 1;
  if (sorted.length === 1) return 0.5;
  const bound = (inclusive: boolean) => {
    let lo = 0;
    let hi = sorted.length;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (sorted[mid]! < value || (inclusive && sorted[mid] === value)) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  const lower = bound(false);
  const upper = bound(true);
  if (upper > lower) return (lower + upper - 1) / 2 / (sorted.length - 1);
  const lowValue = sorted[lower - 1]!;
  return (lower - 1 + (value - lowValue) / (sorted[lower]! - lowValue)) / (sorted.length - 1);
}

export function distribution(values: readonly number[], population = values.length) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    population,
    count: sorted.length,
    sampled: sorted.length < population,
    percentiles: {
      p50: quantile(sorted, 0.5),
      p90: quantile(sorted, 0.9),
      p95: quantile(sorted, 0.95),
      p99: quantile(sorted, 0.99),
      max: quantile(sorted, 1),
      min: quantile(sorted, 0),
    },
    // Keep samples, not just five summary quantiles: proposals need the actual CDF.
    sorted_values: sorted,
  };
}
export type Distribution = ReturnType<typeof distribution>;

function seededRandom(seed: string): () => number {
  let state = createHash("sha256").update(seed).digest().readUInt32LE(0);
  return () => {
    // Mulberry32, independent of host random state and input scan order.
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 0x100000000;
  };
}

export function samplePairOrdinals(population: number, sample: number, seed: string): number[] {
  if (
    !Number.isSafeInteger(population) ||
    population < 0 ||
    !Number.isSafeInteger(sample) ||
    sample < 1
  ) {
    throw new MeasurementError(
      "Pair population and sample must be safe integers; sample must be positive",
    );
  }
  const random = seededRandom(seed);
  const selected = new Set<number>();
  // Floyd sampling: exactly min(sample, population) unique, unordered pairs.
  for (let j = population - Math.min(sample, population); j < population; j += 1) {
    const draw = Math.floor(random() * (j + 1));
    selected.add(selected.has(draw) ? j : draw);
  }
  return [...selected].sort((a, b) => a - b);
}

function reservoir(cap: number, seed: string) {
  let count = 0;
  const values: number[] = [];
  const random = seededRandom(seed);
  return {
    add(value: number) {
      count += 1;
      if (values.length < cap) values.push(value);
      else {
        const index = Math.floor(random() * count);
        if (index < cap) values[index] = value;
      }
    },
    finish: () => distribution(values, count),
  };
}

type PairExample = {
  left_id: string;
  right_id: string;
  left_title: string;
  right_title: string;
  cosine: number;
};
type DuplicateSweep = { cutoff: number; count: number; examples: PairExample[] };

export async function measureRows(input: {
  rows: readonly VectorRow[];
  sample: number;
  seed: string;
  families?: Families;
  progress?: (message: string) => void;
}) {
  const { rows, sample, seed } = input;
  const total = (rows.length * (rows.length - 1)) / 2;
  const ordinals = samplePairOrdinals(total, sample, `${seed}:pairs`);
  const randomValues: number[] = [];
  const randomPairHash = createHash("sha256");
  const within = reservoir(sample, `${seed}:within`);
  const across = reservoir(sample, `${seed}:across`);
  const nearestValues = new Float64Array(rows.length).fill(-Infinity);
  const nearestIds: (string | null)[] = rows.map(() => null);
  // Validated finite, nonzero vectors; precompute norms once. This is the same
  // cosine as src/retrieval/embedding-similarity.ts, avoiding O(n²*d) norm work.
  const norms = rows.map((row) =>
    Math.sqrt(row.vector.reduce((sum, value) => sum + value * value, 0)),
  );
  const nearDuplicates: DuplicateSweep[] = CUTOFFS.map((cutoff) => ({
    cutoff,
    count: 0,
    examples: [] as PairExample[],
  }));
  const familyMembers = new Map<string, string[]>();
  for (const row of rows) {
    const family = input.families?.byId.get(row.id);
    if (family !== undefined) {
      const members = familyMembers.get(family) ?? [];
      members.push(row.id);
      familyMembers.set(family, members);
    }
  }
  let ordinal = 0;
  let sampledIndex = 0;
  let lastProgress = Date.now();
  input.progress?.(`${rows.length} rows; ${total} exact unordered pairs`);
  for (let i = 0; i < rows.length; i += 1) {
    const left = rows[i]!;
    const leftFamily = input.families?.byId.get(left.id);
    for (let j = i + 1; j < rows.length; j += 1) {
      const right = rows[j]!;
      let dot = 0;
      for (let d = 0; d < left.vector.length; d += 1) dot += left.vector[d]! * right.vector[d]!;
      const cosine = Math.max(-1, Math.min(1, dot / (norms[i]! * norms[j]!)));
      if (cosine > nearestValues[i]!) {
        nearestValues[i] = cosine;
        nearestIds[i] = right.id;
      }
      if (cosine > nearestValues[j]!) {
        nearestValues[j] = cosine;
        nearestIds[j] = left.id;
      }
      if (ordinal === ordinals[sampledIndex]) {
        randomValues.push(cosine);
        randomPairHash.update(JSON.stringify([left.id, right.id]));
        sampledIndex += 1;
      }
      for (const cutoff of nearDuplicates) {
        if (cosine >= cutoff.cutoff) {
          cutoff.count += 1;
          // First five qualifying pairs in canonical ID order, not only the
          // highest-scoring pairs (which would hide borderline candidates).
          if (cutoff.examples.length < 5)
            cutoff.examples.push({
              left_id: left.id,
              right_id: right.id,
              left_title: left.title,
              right_title: right.title,
              cosine,
            });
        }
      }
      const rightFamily = input.families?.byId.get(right.id);
      if (leftFamily !== undefined && rightFamily !== undefined) {
        if (leftFamily === rightFamily) within.add(cosine);
        else across.add(cosine);
      }
      ordinal += 1;
      if (ordinal % 4096 === 0 && Date.now() - lastProgress >= 5_000) {
        input.progress?.(`${ordinal}/${total} pairs (${((100 * ordinal) / total).toFixed(1)}%)`);
        lastProgress = Date.now();
        await setImmediate();
      }
    }
  }
  const nearest = rows.map((row, index) => ({
    id: row.id,
    neighbor_id: nearestIds[index] ?? null,
    cosine: nearestIds[index] === null ? null : nearestValues[index]!,
  }));
  return {
    row_count: rows.length,
    pair_count: total,
    nearest_neighbor: distribution(
      nearest.flatMap((row) => (row.cosine === null ? [] : [row.cosine])),
    ),
    nearest_neighbor_rows: nearest,
    random_pair: distribution(randomValues, total),
    random_pair_ids_sha256: randomPairHash.digest("hex"),
    near_duplicates: nearDuplicates,
    families:
      input.families === undefined
        ? null
        : {
            source: input.families.source,
            unavailable_reason: input.families.reason,
            labeled_row_count: [...familyMembers.values()].reduce(
              (sum, ids) => sum + ids.length,
              0,
            ),
            groups: [...familyMembers]
              .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
              .map(([family_id, member_ids]) => ({ family_id, member_ids })),
            within_family: within.finish(),
            across_family: across.finish(),
          },
  };
}
export type Measurement = Awaited<ReturnType<typeof measureRows>>;
