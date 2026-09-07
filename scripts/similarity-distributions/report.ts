import { join } from "node:path";
import { z } from "zod";
import { readJsonFile, writeFileAtomic, writeJsonFileAtomic } from "../../src/util/atomic-write.js";
import { utf16SafePrefixEnd } from "../../src/util/utf16-boundary.js";
import { MeasurementError, type LoadedTable, type pairTables } from "./bank.js";
import { THRESHOLDS, type Threshold } from "./inventory.js";
import { percentileRank, quantile, type Distribution, type Measurement } from "./statistics.js";

export type TableReport = {
  dimensions: LoadedTable["dimensions"];
  present: boolean;
  comparison_sha256: string;
  pairing: ReturnType<typeof pairTables>["audit"];
  measurement: Measurement;
};
export type RunReport = {
  version: 1;
  bank: string;
  vectors: "current" | "prev";
  vector_directory: string;
  model: string;
  model_attribution: string;
  expected_dimensions: number;
  seed: string;
  sample: number;
  warnings: string[];
  tables: Record<string, TableReport>;
};

export function matchProposal(
  threshold: Threshold,
  qwen: readonly number[],
  bge: readonly number[],
) {
  const cosineThreshold =
    threshold.scale === "cosine_distance" ? 1 - threshold.value : threshold.value;
  const percentile = percentileRank(qwen, cosineThreshold);
  const mapped = percentile === null ? null : quantile(bge, percentile);
  return {
    qwen_cosine: cosineThreshold,
    qwen_percentile: percentile === null ? null : percentile * 100,
    proposed_bge_cosine: mapped,
    proposed_value:
      mapped === null ? null : threshold.scale === "cosine_distance" ? 1 - mapped : mapped,
    support:
      qwen.length === 0 || bge.length === 0
        ? "empty"
        : qwen.length < 2 || bge.length < 2
          ? "single_observation"
          : cosineThreshold < qwen[0]!
            ? "below_observed_range"
            : cosineThreshold > qwen[qwen.length - 1]!
              ? "above_observed_range"
              : "within_observed_range",
    qwen_count: qwen.length,
    bge_count: bge.length,
  };
}

export function buildProposals(current: RunReport | undefined, prev: RunReport | undefined) {
  return THRESHOLDS.map((threshold) => {
    const bge = current?.tables[threshold.table];
    const qwen = prev?.tables[threshold.table];
    let unavailable: string | null = null;
    if (threshold.scale === "fused_score") unavailable = "not_a_cosine_threshold";
    else if (current === undefined || prev === undefined) unavailable = "run_both_current_and_prev";
    else if (bge === undefined || qwen === undefined || !bge.present || !qwen.present)
      unavailable = "table_not_measured_in_both_models";
    else if (
      current.bank !== prev.bank ||
      current.seed !== prev.seed ||
      current.sample !== prev.sample ||
      bge.comparison_sha256 !== qwen.comparison_sha256 ||
      bge.measurement.random_pair_ids_sha256 !== qwen.measurement.random_pair_ids_sha256 ||
      !bge.pairing.paired ||
      !qwen.pairing.paired
    )
      unavailable = "incompatible_runs_rerun_both_on_the_same_copy_seed_and_sample";
    else if (
      bge.dimensions !== 1024 ||
      qwen.dimensions !== 4096 ||
      !current.model.toLowerCase().includes("bge-m3") ||
      !prev.model.toLowerCase().includes("qwen3-embedding-8b")
    ) {
      unavailable = "expected_bge_m3_1024_and_qwen3_embedding_8b_4096";
    }
    const matched = matchProposal(
      threshold,
      unavailable === null ? qwen!.measurement[threshold.distribution].sorted_values : [],
      unavailable === null ? bge!.measurement[threshold.distribution].sorted_values : [],
    );
    if (unavailable === null && matched.proposed_value === null) unavailable = "empty_distribution";
    return {
      ...threshold,
      ...matched,
      status: unavailable ?? "proxy_proposal",
      unavailable_reason: unavailable,
    };
  });
}

// Validate previously generated artifacts before combining independent model runs.
function readRun(path: string): RunReport | undefined {
  const value = readJsonFile<unknown>(path);
  if (value === undefined) return undefined;
  const scores = z
    .array(z.number().finite().min(-1).max(1))
    .refine(
      (values) => values.every((value, index) => index === 0 || value >= values[index - 1]!),
      "Expected sorted cosine values",
    );
  const nullableCosine = z.number().finite().min(-1).max(1).nullable();
  const count = z.number().int().nonnegative();
  const distributionSchema = z.object({
    sorted_values: scores,
    count,
    population: count,
    sampled: z.boolean(),
    percentiles: z.object({
      p50: nullableCosine,
      p90: nullableCosine,
      p95: nullableCosine,
      p99: nullableCosine,
      min: nullableCosine,
      max: nullableCosine,
    }),
  });
  const invalidRows = z.array(z.object({ id: z.string(), reason: z.string() }));
  const schema: z.ZodType<RunReport> = z.object({
    version: z.literal(1),
    bank: z.string(),
    vectors: z.enum(["current", "prev"]),
    vector_directory: z.string(),
    model: z.string(),
    model_attribution: z.string(),
    expected_dimensions: z.number().int().positive(),
    seed: z.string(),
    sample: z.number().int().positive(),
    warnings: z.array(z.string()),
    tables: z.record(
      z.string(),
      z
        .object({
          dimensions: z.number().int().positive().nullable(),
          present: z.boolean(),
          comparison_sha256: z.string(),
          pairing: z.object({
            paired: z.boolean(),
            common_count: count.nullable(),
            current_count: count,
            prev_count: count.nullable(),
            current_only_ids: z.array(z.string()),
            prev_only_ids: z.array(z.string()),
            current_invalid: invalidRows,
            prev_invalid: invalidRows,
          }),
          measurement: z
            .object({
              random_pair_ids_sha256: z.string(),
              random_pair: distributionSchema,
              nearest_neighbor: distributionSchema,
              row_count: count,
              pair_count: count,
              nearest_neighbor_rows: z.array(
                z.object({
                  id: z.string(),
                  neighbor_id: z.string().nullable(),
                  cosine: nullableCosine,
                }),
              ),
              near_duplicates: z.array(
                z.object({
                  cutoff: z.number(),
                  count,
                  examples: z.array(
                    z.object({
                      left_id: z.string(),
                      right_id: z.string(),
                      left_title: z.string(),
                      right_title: z.string(),
                      cosine: z.number().finite().min(-1).max(1),
                    }),
                  ),
                }),
              ),
              families: z
                .object({
                  source: z.string().nullable(),
                  unavailable_reason: z.string().nullable(),
                  labeled_row_count: count,
                  groups: z.array(
                    z.object({ family_id: z.string(), member_ids: z.array(z.string()) }),
                  ),
                  within_family: distributionSchema,
                  across_family: distributionSchema,
                })
                .nullable(),
            })
            .passthrough(),
        })
        .passthrough(),
    ),
  });
  return schema.parse(value);
}

const number = (value: number | null | undefined) =>
  value === null || value === undefined ? "—" : value.toFixed(5);
function cell(value: string): string {
  const excerpt = value.slice(0, utf16SafePrefixEnd(value, 240));
  return (
    excerpt
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll("|", "&#124;")
      .replaceAll("\n", " ")
      .replaceAll("\r", " ")
      .replaceAll("`", "&#96;")
      .replaceAll("[", "&#91;")
      .replaceAll("]", "&#93;") + (excerpt.length < value.length ? "…" : "")
  );
}

export function renderSummary(current: RunReport | undefined, prev: RunReport | undefined) {
  const run = current ?? prev!;
  const lines = [
    `# Similarity distributions: ${cell(run.bank)}`,
    "",
    "Phase A measurement only. No runtime thresholds changed. All stored rows (including inactive/archived rows) are measured on the valid ID intersection when previous vectors exist.",
    "",
    `Seed: ${cell(run.seed)}. Random-pair sample: ${run.sample} per table, without replacement.`,
    "",
    "Nearest neighbors and cutoff counts are exhaustive; self pairs and reversed duplicates are excluded. Family strata contain only rows with non-null SQLite family IDs, with independent reservoir samples capped at --sample per stratum. Missing family IDs are not treated as separate families.",
    "",
    "Percentiles use linear sample quantiles h=(n−1)p; proposals invert that curve with midpoint ranks for ties. Proposals outside observed support clip to an endpoint and need more tail data. Every numeric proposal is a proxy for review, not an approved setting. Stored survivors omit previously rejected duplicates and do not reproduce live queries or eligibility filters.",
    "",
    "## Distributions",
    "",
    "| Model / table | Distribution | Samples / population | p50 | p90 | p95 | p99 | max |",
    "|---|---|---:|---:|---:|---:|---:|---:|",
  ];
  for (const model of [prev, current]) {
    if (model === undefined) continue;
    for (const [table, report] of Object.entries(model.tables)) {
      const m = report.measurement;
      const distributions: [string, Distribution][] = [
        ["nearest_neighbor", m.nearest_neighbor],
        ["random_pair", m.random_pair],
      ];
      if (m.families !== null)
        distributions.push(
          ["within_family", m.families.within_family],
          ["across_family", m.families.across_family],
        );
      for (const [name, d] of distributions) {
        const p = d.percentiles;
        lines.push(
          `| ${cell(model.model)} / ${table} (${report.dimensions ?? "?"}d) | ${name} | ${d.count} / ${d.population} | ${number(p.p50)} | ${number(p.p90)} | ${number(p.p95)} | ${number(p.p99)} | ${number(p.max)} |`,
        );
      }
    }
  }
  lines.push("", "## Pairing and family coverage", "");
  for (const model of [prev, current]) {
    if (model === undefined) continue;
    lines.push(
      `- ${model.vectors}: ${cell(model.vector_directory)}; ${cell(model.model_attribution)}.`,
    );
    for (const warning of model.warnings) lines.push(`- ${cell(warning)}`);
    for (const [name, table] of Object.entries(model.tables)) {
      const p = table.pairing;
      lines.push(
        `- ${model.vectors}/${name}: ${table.present ? "present" : "missing"}; ${table.measurement.row_count} measured; ${p.current_only_ids.length} current-only, ${p.prev_only_ids.length} prev-only, ${p.current_invalid.length}/${p.prev_invalid.length} invalid current/prev. Full IDs/reasons are in JSON.`,
      );
      const family = table.measurement.families;
      if (family !== null)
        lines.push(
          `  Families: ${family.groups.length}, labeled rows: ${family.labeled_row_count}. ${cell(family.unavailable_reason ?? family.source ?? "")}`,
        );
    }
  }
  lines.push(
    "",
    "## Near-duplicate candidates",
    "",
    "Counts are exact for cosine >= cutoff. Examples are the first five qualifying pairs in sorted ID order; a human must judge duplication.",
    "",
  );
  for (const model of [prev, current]) {
    if (model === undefined) continue;
    for (const [table, report] of Object.entries(model.tables)) {
      lines.push(
        `### ${model.vectors} / ${table}`,
        "",
        "| Cutoff | Count | Example title pairs (cosine) |",
        "|---:|---:|---|",
      );
      for (const sweep of report.measurement.near_duplicates) {
        lines.push(
          `| ${sweep.cutoff.toFixed(2)} | ${sweep.count} | ${sweep.examples.map((pair) => `${cell(pair.left_title)} ↔ ${cell(pair.right_title)} (${number(pair.cosine)})`).join("<br>")} |`,
        );
      }
      lines.push("");
    }
  }
  lines.push(
    "## Percentile-matched proposals",
    "",
    "| Threshold | Raw value | Distribution proxy | Qwen percentile | Proposed BGE value | Support / status |",
    "|---|---:|---|---:|---:|---|",
  );
  for (const proposal of buildProposals(current, prev)) {
    lines.push(
      `| ${proposal.id} | ${proposal.value} | ${proposal.table}/${proposal.distribution} | ${number(proposal.qwen_percentile)} | ${number(proposal.proposed_value)} | ${proposal.unavailable_reason ?? proposal.support} |`,
    );
  }
  lines.push(
    "",
    "## Threshold inventory (fb25a5c8)",
    "",
    "| Threshold / file:line | Value | Compared items / gate | Vector store | Proposal limitation |",
    "|---|---:|---|---|---|",
  );
  for (const t of THRESHOLDS)
    lines.push(
      `| ${t.id}<br>${t.locations.join("<br>")} | ${t.value} | ${cell(t.compares)} | ${cell(t.store)} | ${cell(t.caveat)} |`,
    );
  lines.push(
    "",
    "BORG_RECALL_ABSTAIN_THRESHOLD defaults to 0 (disabled) and gates a fused rawScore, not cosine. It deliberately has no numeric cosine proposal; phase B needs query replay for this gate. This tool does not read runtime threshold overrides.",
    "",
  );
  return lines.join("\n");
}

export function writeReports(directory: string, run: RunReport): void {
  const otherSource = run.vectors === "current" ? "prev" : "current";
  const other = readRun(join(directory, `${otherSource}.json`));
  if (other !== undefined && (other.bank !== run.bank || other.vectors !== otherSource)) {
    throw new MeasurementError(
      "Existing report belongs to a different bank/vector source; choose a separate --out directory",
    );
  }
  const current = run.vectors === "current" ? run : other;
  const prev = run.vectors === "prev" ? run : other;
  const summary = renderSummary(current, prev);
  writeJsonFileAtomic(join(directory, `${run.vectors}.json`), run, { mode: 0o600 });
  writeJsonFileAtomic(
    join(directory, "report.json"),
    {
      version: 1,
      bank: run.bank,
      current: current ?? null,
      prev: prev ?? null,
      proposals: buildProposals(current, prev),
    },
    { mode: 0o600 },
  );
  writeFileAtomic(join(directory, "summary.md"), summary, { mode: 0o600 });
}
