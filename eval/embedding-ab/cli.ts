import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { writeFileAtomic } from "../../src/util/atomic-write.js";
import { isPathWithin, resolveRealPathForCreation } from "../../src/util/path.js";

import { writePrivateJson } from "./cache.js";
import { runEmbeddingAbEvaluation } from "./evaluate.js";
import { renderEmbeddingAbReport } from "./report.js";

const DEFAULT_GOLD_SIZE = 60;
const DEFAULT_BATCH_SIZE = 8;

type CliArgs =
  | { help: true }
  | {
      help: false;
      dataDir: string;
      models: string[];
      outDir: string;
      queries: string[];
      queriesSource: string | null;
      goldSize: number;
      judgeRequested: boolean;
      judgeModel?: string;
      batchSize: number;
    };

function usage(): string {
  return [
    "Usage: npm run embedding:ab -- --data-dir <bank-copy> --models <id,id> --out <scratch-dir> [options]",
    "",
    "Options:",
    "  --queries <path-or-json>     File containing a JSON string array, or an inline JSON array",
    `  --gold-size <n>              Seeded synthetic-gold sample size (default ${DEFAULT_GOLD_SIZE}; 0 disables)`,
    "  --judge-model [model-id]     Judge model top-5 relevance; omit id to use auto-selected strong model",
    `  --batch-size <n>             Embedding inputs per gateway call (default ${DEFAULT_BATCH_SIZE})`,
    "  --help                       Show this help",
    "",
    "Environment:",
    "  KRATOS_BASE_URL              Gateway URL; /v1 is appended when absent",
    "  LLM_API_KEY                  Gateway API key",
    "  NODE_EXTRA_CA_CERTS          Corporate CA bundle (handled by Node)",
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

function parsePositiveInteger(raw: string, option: string): number {
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error(`${option} must be a positive integer, received ${JSON.stringify(raw)}`);
  }
  return value;
}

function parseQueries(raw: string): { queries: string[]; source: string } {
  const possiblePath = resolve(raw);
  const source = existsSync(possiblePath) ? possiblePath : "inline-json";
  const json = source === "inline-json" ? raw : readFileSync(possiblePath, "utf8");
  let parsed: unknown;

  try {
    parsed = JSON.parse(json) as unknown;
  } catch (error) {
    throw new Error(
      `--queries must be a readable JSON file or inline JSON array: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  if (
    !Array.isArray(parsed) ||
    parsed.some((query) => typeof query !== "string" || query.length === 0)
  ) {
    throw new Error("--queries JSON must be an array of non-empty strings");
  }

  // Return the strings without trimming or wrapper parsing: replay inputs are verbatim.
  return { queries: parsed as string[], source };
}

export function parseCliArgs(argv: readonly string[]): CliArgs {
  let dataDir: string | undefined;
  let rawModels: string | undefined;
  let outDir: string | undefined;
  let queries: string[] = [];
  let queriesSource: string | null = null;
  let goldSize = DEFAULT_GOLD_SIZE;
  let judgeRequested = false;
  let judgeModel: string | undefined;
  let batchSize = DEFAULT_BATCH_SIZE;

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
    if (argument === "--models") {
      rawModels = requiredValue(argv, index, argument);
      index += 1;
      continue;
    }
    if (argument === "--out") {
      outDir = requiredValue(argv, index, argument);
      index += 1;
      continue;
    }
    if (argument === "--queries") {
      const parsed = parseQueries(requiredValue(argv, index, argument));
      queries = parsed.queries;
      queriesSource = parsed.source;
      index += 1;
      continue;
    }
    if (argument === "--gold-size") {
      goldSize = parseNonNegativeInteger(requiredValue(argv, index, argument), argument);
      index += 1;
      continue;
    }
    if (argument === "--batch-size") {
      batchSize = parsePositiveInteger(requiredValue(argv, index, argument), argument);
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
    throw new Error(`Unknown argument: ${argument ?? ""}`);
  }

  if (dataDir === undefined || dataDir.trim().length === 0) {
    throw new Error("--data-dir is required");
  }
  if (rawModels === undefined || rawModels.trim().length === 0) {
    throw new Error("--models is required");
  }
  if (outDir === undefined || outDir.trim().length === 0) {
    throw new Error("--out is required");
  }

  const models = [...new Set(rawModels.split(",").map((model) => model.trim()))].filter(
    (model) => model.length > 0,
  );
  if (models.length === 0) {
    throw new Error("--models must contain at least one model id");
  }

  return {
    help: false,
    dataDir: resolve(dataDir),
    models,
    outDir: resolve(outDir),
    queries,
    queriesSource,
    goldSize,
    judgeRequested,
    ...(judgeModel === undefined ? {} : { judgeModel }),
    batchSize,
  };
}

function requiredEnv(env: NodeJS.ProcessEnv, name: "KRATOS_BASE_URL" | "LLM_API_KEY"): string {
  const value = env[name]?.trim();
  if (value === undefined || value.length === 0) {
    throw new Error(`${name} is required`);
  }
  return value;
}

export function assertScratchOutsideBank(dataDir: string, outDir: string): void {
  const canonicalData = resolveRealPathForCreation(dataDir);
  const canonicalOut = resolveRealPathForCreation(outDir);

  if (isPathWithin(canonicalData, canonicalOut)) {
    throw new Error("--out must be outside --data-dir so the source bank remains read-only");
  }
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
  const results = await runEmbeddingAbEvaluation({
    dataDir: args.dataDir,
    models: args.models,
    outDir: args.outDir,
    queries: args.queries,
    queriesSource: args.queriesSource,
    goldSize: args.goldSize,
    judgeRequested: args.judgeRequested,
    ...(args.judgeModel === undefined ? {} : { judgeModel: args.judgeModel }),
    batchSize: args.batchSize,
    baseUrl: requiredEnv(env, "KRATOS_BASE_URL"),
    apiKey: requiredEnv(env, "LLM_API_KEY"),
    log: (message) => process.stderr.write(`${message}\n`),
  });
  const resultsPath = join(args.outDir, "results.json");
  const reportPath = join(args.outDir, "report.md");
  writePrivateJson(resultsPath, results);
  writeFileAtomic(reportPath, renderEmbeddingAbReport(results), { mode: 0o600 });
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
