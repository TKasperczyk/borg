import commitmentComplianceMetric from "./metrics/commitment-compliance.js";
import contradictionDetectionMetric from "./metrics/contradiction-detection.js";
import crossAudienceLeakageMetric from "./metrics/cross-audience-leakage.js";
import dedupCorrectnessMetric from "./metrics/dedup-correctness.js";
import episodicExtractionQualityMetric from "./metrics/episodic-extraction-quality.js";
import falseMemoryRateMetric from "./metrics/false-memory-rate.js";
import goalProgressAttributionMetric from "./metrics/goal-progress-attribution.js";
import retrievalPrecisionRecallMetric from "./metrics/retrieval-precision-recall.js";
import { buildScorecard, formatHumanScorecard, type EvalMetricModule } from "./support/scorecard.js";

const REGISTRY: readonly EvalMetricModule[] = [
  retrievalPrecisionRecallMetric,
  episodicExtractionQualityMetric,
  dedupCorrectnessMetric,
  crossAudienceLeakageMetric,
  commitmentComplianceMetric,
  contradictionDetectionMetric,
  falseMemoryRateMetric,
  goalProgressAttributionMetric,
] as const;

type ParsedArgs = {
  json: boolean;
  includeTiming: boolean;
  list: boolean;
  metrics: string[];
};

function parseArgs(argv: readonly string[]): ParsedArgs {
  const parsed: ParsedArgs = {
    json: false,
    includeTiming: false,
    list: false,
    metrics: [],
  };

  for (let index = 2; index < argv.length; index += 1) {
    const arg = argv[index];

    if (arg === "--json") {
      parsed.json = true;
      continue;
    }

    if (arg === "--include-timing") {
      parsed.includeTiming = true;
      continue;
    }

    if (arg === "--list") {
      parsed.list = true;
      continue;
    }

    if (arg === "--metric") {
      const value = argv[index + 1];

      if (value === undefined || value.startsWith("--")) {
        throw new Error("--metric requires a metric name");
      }

      parsed.metrics.push(value);
      index += 1;
      continue;
    }

    throw new Error(`Unknown argument: ${arg}`);
  }

  return parsed;
}

function selectMetrics(names: readonly string[]): EvalMetricModule[] {
  if (names.length === 0) {
    return [...REGISTRY];
  }

  const byName = new Map(REGISTRY.map((metric) => [metric.name, metric]));
  const selected: EvalMetricModule[] = [];

  for (const name of names) {
    const metric = byName.get(name);

    if (metric === undefined) {
      throw new Error(`Unknown metric: ${name}`);
    }

    if (!selected.some((entry) => entry.name === metric.name)) {
      selected.push(metric);
    }
  }

  return selected;
}

async function main(): Promise<number> {
  const args = parseArgs(process.argv);

  if (args.list) {
    for (const metric of REGISTRY) {
      process.stdout.write(`${metric.name}\t${metric.description}\n`);
    }
    return 0;
  }

  const selectedMetrics = selectMetrics(args.metrics);
  const results = [];

  for (const metric of selectedMetrics) {
    results.push(await metric.run());
  }

  const scorecard = buildScorecard(results, {
    includeTiming: args.json ? args.includeTiming : true,
  });

  if (args.json) {
    process.stdout.write(`${JSON.stringify(scorecard, null, 2)}\n`);
  } else {
    process.stdout.write(formatHumanScorecard(scorecard));
  }

  return scorecard.summary.failed > 0 ? 1 : 0;
}

try {
  const exitCode = await main();

  if (exitCode !== 0) {
    process.exitCode = exitCode;
  }
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`eval runner failed: ${message}\n`);
  process.exitCode = 1;
}
