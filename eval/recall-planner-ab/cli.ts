import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { writeFileAtomic } from "../../src/util/atomic-write.js";
import {
  MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS,
  MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS,
} from "../../src/retrieval/recall-expansion.js";

import { assertScratchOutsideBank } from "../embedding-ab/cli.js";
import { writePrivateJson } from "../embedding-ab/cache.js";
import { loadRecallPlannerCases } from "./cases.js";
import { runRecallPlannerAbEvaluation } from "./evaluate.js";
import { renderRecallPlannerAbReport } from "./report.js";

type CliArgs =
  | { help: true }
  | {
      help: false;
      dataDir: string;
      casesPath: string;
      outDir: string;
      variantCounts: number[];
      embeddingModel?: string;
      plannerTimeZone?: string;
      plannerTimeoutMs?: number;
      judgeRequested: boolean;
      judgeModel?: string;
      baseline: boolean;
      generateCases: number;
    };

export function usage(): string {
  return [
    "Usage: npx tsx eval/recall-planner-ab/cli.ts --data-dir <bank-copy> --cases <json> --out <scratch-dir> --variant-counts <n,n> [options]",
    "",
    "Options:",
    "  --embedding-model <id>      Embedding model (default: copied bank config)",
    "  --planner-time-zone <iana>  Zone the planner resolves relative time in (default: bank config)",
    "  --planner-timeout-ms <n>    Planner deadline in ms (default: bank config)",
    "  --judge-model [id]          Judge top-5 relevance; omit id to auto-select a strong model",
    "  --baseline                  Also run raw FOCUS-blob-only retrieval with no LLM expansion",
    "  --generate-cases <n>        Generate and evaluate N additional referential Polish cases",
    "  --help                      Show this help",
    "",
    "Environment:",
    "  KRATOS_BASE_URL             OpenAI-compatible gateway; /v1 is appended when absent",
    "  LLM_API_KEY                 Gateway API key",
    "  BORG_MODEL_RECALL_EXPANSION Planner model override (otherwise copied bank config)",
    "  NODE_EXTRA_CA_CERTS         Optional corporate CA bundle handled by Node",
  ].join("\n");
}

function requiredValue(argv: readonly string[], index: number, option: string): string {
  const value = argv[index + 1];
  if (value === undefined || value.startsWith("--")) {
    throw new Error(`${option} requires a value`);
  }
  return value;
}

function parseNonNegativeInteger(raw: string, option: string): number {
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 0) {
    throw new Error(`${option} must be a non-negative integer, received ${JSON.stringify(raw)}`);
  }
  return value;
}

function parseVariantCounts(raw: string): number[] {
  const counts = [...new Set(raw.split(",").map((part) => Number(part.trim())))];

  if (
    counts.length === 0 ||
    counts.some(
      (count) =>
        !Number.isInteger(count) ||
        count < MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS ||
        count > MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS,
    )
  ) {
    throw new Error(
      `--variant-counts must be comma-separated integers from ${MIN_RECALL_EXPANSION_SEMANTIC_VARIANTS} to ${MAX_RECALL_EXPANSION_SEMANTIC_VARIANTS}`,
    );
  }

  return counts;
}

export function parseCliArgs(argv: readonly string[]): CliArgs {
  let dataDir: string | undefined;
  let casesPath: string | undefined;
  let outDir: string | undefined;
  let variantCounts: number[] | undefined;
  let embeddingModel: string | undefined;
  let plannerTimeZone: string | undefined;
  let plannerTimeoutMs: number | undefined;
  let judgeRequested = false;
  let judgeModel: string | undefined;
  let baseline = false;
  let generateCases = 0;

  for (let index = 0; index < argv.length; index += 1) {
    const argument = argv[index];
    if (argument === "--help" || argument === "-h") {
      return { help: true };
    }
    if (argument === "--data-dir") {
      dataDir = requiredValue(argv, index, argument);
      index += 1;
      continue;
    }
    if (argument === "--cases") {
      casesPath = requiredValue(argv, index, argument);
      index += 1;
      continue;
    }
    if (argument === "--out") {
      outDir = requiredValue(argv, index, argument);
      index += 1;
      continue;
    }
    if (argument === "--variant-counts") {
      variantCounts = parseVariantCounts(requiredValue(argv, index, argument));
      index += 1;
      continue;
    }
    if (argument === "--planner-time-zone") {
      plannerTimeZone = requiredValue(argv, index, argument);
      index += 1;
      continue;
    }
    if (argument === "--planner-timeout-ms") {
      plannerTimeoutMs = Number(requiredValue(argv, index, argument));
      if (!Number.isInteger(plannerTimeoutMs) || plannerTimeoutMs < 0) {
        throw new Error("--planner-timeout-ms must be a non-negative integer");
      }
      index += 1;
      continue;
    }
    if (argument === "--embedding-model") {
      embeddingModel = requiredValue(argv, index, argument);
      index += 1;
      continue;
    }
    if (argument === "--judge-model") {
      judgeRequested = true;
      const possibleModel = argv[index + 1];
      if (possibleModel !== undefined && !possibleModel.startsWith("--")) {
        judgeModel = possibleModel;
        index += 1;
      }
      continue;
    }
    if (argument === "--baseline") {
      baseline = true;
      continue;
    }
    if (argument === "--generate-cases") {
      generateCases = parseNonNegativeInteger(requiredValue(argv, index, argument), argument);
      index += 1;
      continue;
    }
    throw new Error(`Unknown argument: ${argument ?? ""}`);
  }

  if (dataDir === undefined || dataDir.trim().length === 0) {
    throw new Error("--data-dir is required");
  }
  if (casesPath === undefined || casesPath.trim().length === 0) {
    throw new Error("--cases is required");
  }
  if (outDir === undefined || outDir.trim().length === 0) {
    throw new Error("--out is required");
  }
  if (variantCounts === undefined) {
    throw new Error("--variant-counts is required");
  }

  return {
    help: false,
    dataDir: resolve(dataDir),
    casesPath: resolve(casesPath),
    outDir: resolve(outDir),
    variantCounts,
    ...(embeddingModel === undefined ? {} : { embeddingModel }),
    ...(plannerTimeZone === undefined ? {} : { plannerTimeZone }),
    ...(plannerTimeoutMs === undefined ? {} : { plannerTimeoutMs }),
    judgeRequested,
    ...(judgeModel === undefined ? {} : { judgeModel }),
    baseline,
    generateCases,
  };
}

function requiredEnv(env: NodeJS.ProcessEnv, name: "KRATOS_BASE_URL" | "LLM_API_KEY"): string {
  const value = env[name]?.trim();
  if (value === undefined || value.length === 0) {
    throw new Error(`${name} is required`);
  }
  return value;
}

export async function main(
  argv: readonly string[] = process.argv.slice(2),
  env: NodeJS.ProcessEnv = process.env,
): Promise<number> {
  const args = parseCliArgs(argv);
  if (args.help) {
    process.stdout.write(`${usage()}\n`);
    return 0;
  }

  assertScratchOutsideBank(args.dataDir, args.outDir);
  const loadedCases = loadRecallPlannerCases(args.casesPath);
  const plannerModel = env.BORG_MODEL_RECALL_EXPANSION?.trim();
  const results = await runRecallPlannerAbEvaluation({
    dataDir: args.dataDir,
    casesPath: loadedCases.path,
    cases: loadedCases.cases,
    outDir: args.outDir,
    variantCounts: args.variantCounts,
    baseline: args.baseline,
    ...(args.embeddingModel === undefined ? {} : { embeddingModel: args.embeddingModel }),
    ...(args.plannerTimeZone === undefined ? {} : { plannerTimeZone: args.plannerTimeZone }),
    ...(args.plannerTimeoutMs === undefined ? {} : { plannerTimeoutMs: args.plannerTimeoutMs }),
    judgeRequested: args.judgeRequested,
    ...(args.judgeModel === undefined ? {} : { judgeModel: args.judgeModel }),
    generateCases: args.generateCases,
    baseUrl: requiredEnv(env, "KRATOS_BASE_URL"),
    apiKey: requiredEnv(env, "LLM_API_KEY"),
    ...(plannerModel === undefined || plannerModel.length === 0 ? {} : { plannerModel }),
    log: (message) => process.stderr.write(`${message}\n`),
  });
  const resultsPath = join(args.outDir, "results.json");
  const reportPath = join(args.outDir, "report.md");
  writePrivateJson(resultsPath, results);
  writeFileAtomic(reportPath, renderRecallPlannerAbReport(results), { mode: 0o600 });

  if (args.generateCases > 0) {
    const generatedCasesPath = join(args.outDir, "generated-cases.json");
    writePrivateJson(
      generatedCasesPath,
      results.generation.records.flatMap((record) => (record.case === null ? [] : [record.case])),
    );
    process.stdout.write(`Wrote ${generatedCasesPath}\n`);
  }

  process.stdout.write(`Wrote ${resultsPath}\nWrote ${reportPath}\n`);
  return 0;
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().then(
    (exitCode) => {
      process.exitCode = exitCode;
    },
    (error: unknown) => {
      process.stderr.write(
        `${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`,
      );
      process.exitCode = 1;
    },
  );
}
