/**
 * Blind offline judge for live planner A/B replay rows. This script opens no
 * Borg substrate, repository, stream writer, retrieval service, or working
 * memory. It reads immutable eval files and performs unary structured calls.
 * Judgments are never fed back into cognition or any live turn.
 */
import { realpathSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { loadConfig } from "../src/config/index.ts";
import {
  aggregatePlannerAbJudgments,
  judgePlannerAbPair,
  parsePlannerAbReplayResultForJudge,
  plannerAbStdoutSummary,
  plannerAbJudgeExclusionReason,
  type PlannerAbJudgmentRecord,
} from "../src/cognition/deliberation/planner-ab-judge.ts";
import {
  parsePlannerContextCaptureRecord,
  plannerContextCapturePath,
  type PlannerContextCaptureRecord,
} from "../src/cognition/deliberation/planner-context-capture.ts";
import {
  resolveContextCaptureStoragePath,
  writePrivateContextCaptureJson,
} from "../src/cognition/deliberation/context-capture-storage.ts";
import { appendDurableJsonl } from "../src/util/durable-jsonl.ts";
import { isPathWithin, resolveRealPathForCreation } from "../src/util/path.ts";
import {
  assertReplayOAuthCredentialsOutsideDataDir,
  createLiveLlmClient,
  openPlannerCaptureSnapshot,
} from "./planner-ab-replay.ts";

type Args = {
  dataDir?: string;
  inputPath?: string;
  capturesPath?: string;
  outputPath?: string;
  summaryPath?: string;
  limit?: number;
};

function usage(message?: string): never {
  if (message !== undefined) {
    console.error(message);
    console.error("");
  }
  console.error(
    [
      "Usage: pnpm planner:ab-judge -- [--data-dir DIR] [--input FILE] [--captures FILE] [--output FILE] [--summary FILE] [--limit N]",
      "",
      "The cognition model judges eligible live replay pairs through the unary LLM transport.",
      "Outputs must be direct children of <dataDir>/captures.",
    ].join("\n"),
  );
  process.exit(1);
}

function requiredValue(argv: readonly string[], index: number, flag: string): string {
  const value = argv[index + 1];
  if (value === undefined || value.length === 0) usage(`${flag} requires a value`);
  return value;
}

export function parsePlannerAbJudgeArgs(argv: readonly string[]): Args {
  const args: Args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case "--data-dir":
        args.dataDir = requiredValue(argv, index, arg);
        index += 1;
        break;
      case "--input":
        args.inputPath = requiredValue(argv, index, arg);
        index += 1;
        break;
      case "--captures":
        args.capturesPath = requiredValue(argv, index, arg);
        index += 1;
        break;
      case "--output":
        args.outputPath = requiredValue(argv, index, arg);
        index += 1;
        break;
      case "--summary":
        args.summaryPath = requiredValue(argv, index, arg);
        index += 1;
        break;
      case "--limit": {
        const limit = Number(requiredValue(argv, index, arg));
        if (!Number.isInteger(limit) || limit <= 0) usage("--limit must be a positive integer");
        args.limit = limit;
        index += 1;
        break;
      }
      case "--help":
      case "-h":
        usage();
      default:
        usage(`Unknown argument: ${arg}`);
    }
  }
  return args;
}

export type ResolvedPlannerAbJudgePaths = {
  dataDirectory: string;
  capturesDirectory: string;
  inputPath: string;
  sourceCapturesPath: string;
  outputPath: string;
  summaryPath: string;
};

function directPrivateOutput(
  requested: string | undefined,
  fallback: string,
  capturesDirectory: string,
  label: string,
): string {
  const output = resolveRealPathForCreation(resolve(requested ?? fallback));
  if (!isPathWithin(capturesDirectory, output) || dirname(output) !== capturesDirectory) {
    throw new Error(`${label} must be a direct child of dataDir/captures`);
  }
  return output;
}

export function resolvePlannerAbJudgePaths(input: {
  dataDir: string;
  inputPath?: string;
  capturesPath?: string;
  outputPath?: string;
  summaryPath?: string;
}): ResolvedPlannerAbJudgePaths {
  const dataDirectory = realpathSync(input.dataDir);
  const defaultOutput = resolveContextCaptureStoragePath(
    dataDirectory,
    "planner-ab-judgments.jsonl",
  );
  const capturesDirectory = defaultOutput.captureDirectory;
  const inputPath = realpathSync(
    resolve(input.inputPath ?? join(capturesDirectory, "planner-ab-results.jsonl")),
  );
  const sourceCapturesPath = realpathSync(
    resolve(input.capturesPath ?? plannerContextCapturePath(dataDirectory)),
  );
  const outputPath = directPrivateOutput(
    input.outputPath,
    defaultOutput.path,
    capturesDirectory,
    "Planner A/B judgment output",
  );
  const summaryPath = directPrivateOutput(
    input.summaryPath,
    join(capturesDirectory, "planner-ab-judgment-summary.json"),
    capturesDirectory,
    "Planner A/B judgment summary",
  );
  const paths = [inputPath, sourceCapturesPath, outputPath, summaryPath];
  if (new Set(paths).size !== paths.length) {
    throw new Error("Planner A/B judge input, capture, judgment, and summary paths must differ");
  }
  return {
    dataDirectory,
    capturesDirectory,
    inputPath,
    sourceCapturesPath,
    outputPath,
    summaryPath,
  };
}

function parseJsonLine(path: string, lineNumber: number, line: string): unknown {
  try {
    return JSON.parse(line) as unknown;
  } catch (error) {
    throw new Error(
      `Invalid JSON at ${path}:${lineNumber}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

async function readReplayCohort(
  path: string,
  lines: ReturnType<typeof openPlannerCaptureSnapshot>["lines"],
  limit: number | undefined,
) {
  const records: ReturnType<typeof parsePlannerAbReplayResultForJudge>[] = [];
  let lineNumber = 0;
  for await (const line of lines) {
    lineNumber += 1;
    if (line.trim().length === 0) continue;
    records.push(parsePlannerAbReplayResultForJudge(parseJsonLine(path, lineNumber, line)));
    if (limit !== undefined && records.length >= limit) break;
  }
  return records;
}

async function readRequiredCaptures(
  path: string,
  lines: ReturnType<typeof openPlannerCaptureSnapshot>["lines"],
  captureIds: ReadonlySet<string>,
): Promise<Map<string, PlannerContextCaptureRecord>> {
  const captures = new Map<string, PlannerContextCaptureRecord>();
  if (captureIds.size === 0) return captures;
  let lineNumber = 0;
  for await (const line of lines) {
    lineNumber += 1;
    if (line.trim().length === 0) continue;
    const parsed = parseJsonLine(path, lineNumber, line) as { capture_id?: unknown };
    if (typeof parsed.capture_id !== "string" || !captureIds.has(parsed.capture_id)) continue;
    const capture = parsePlannerContextCaptureRecord(parsed);
    captures.set(capture.capture_id, capture);
    if (captures.size === captureIds.size) break;
  }
  return captures;
}

export async function runPlannerAbJudgeCli(
  argv: readonly string[],
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  const args = parsePlannerAbJudgeArgs(argv);
  const config = loadConfig({
    env,
    ...(args.dataDir === undefined ? {} : { dataDir: args.dataDir }),
  });
  const paths = resolvePlannerAbJudgePaths({
    dataDir: config.dataDir,
    ...(args.inputPath === undefined ? {} : { inputPath: args.inputPath }),
    ...(args.capturesPath === undefined ? {} : { capturesPath: args.capturesPath }),
    ...(args.outputPath === undefined ? {} : { outputPath: args.outputPath }),
    ...(args.summaryPath === undefined ? {} : { summaryPath: args.summaryPath }),
  });
  assertReplayOAuthCredentialsOutsideDataDir({ dataDirectory: paths.dataDirectory, env });
  const llmClient = createLiveLlmClient(config, env);
  // Freeze both inputs before reading either one. Rows appended after this
  // boundary belong to the next cohort, matching capture/replay semantics.
  const replaySnapshot = openPlannerCaptureSnapshot(paths.inputPath);
  const captureSnapshot = openPlannerCaptureSnapshot(paths.sourceCapturesPath);
  const replayRows = await readReplayCohort(paths.inputPath, replaySnapshot.lines, args.limit);
  const neededCaptureIds = new Set(
    replayRows
      .filter((row) => plannerAbJudgeExclusionReason(row, null) === "missing_capture")
      .map((row) => row.capture_id),
  );
  const captures = await readRequiredCaptures(
    paths.sourceCapturesPath,
    captureSnapshot.lines,
    neededCaptureIds,
  );
  const judgments: PlannerAbJudgmentRecord[] = [];

  for (const replay of replayRows) {
    const judgment = await judgePlannerAbPair(replay, captures.get(replay.capture_id) ?? null, {
      llmClient,
      model: config.anthropic.models.cognition,
    });
    judgments.push(judgment);
    await appendDurableJsonl(paths.outputPath, judgment, {
      privateDirectory: paths.capturesDirectory,
    });
    console.log(
      `${replay.capture_id} status=${judgment.status}${judgment.status === "completed" ? ` winner=${judgment.deblinded.overall.winner}` : judgment.status === "excluded" ? ` reason=${judgment.reason}` : ""}`,
    );
  }

  const summary = aggregatePlannerAbJudgments(judgments, {
    generatedAt: Date.now(),
    inputRecords: replayRows.length,
  });
  writePrivateContextCaptureJson({
    dataDir: paths.dataDirectory,
    fileName: basename(paths.summaryPath),
    value: summary,
  });
  console.log(JSON.stringify(plannerAbStdoutSummary(summary), null, 2));
  console.log(
    `planner A/B judging complete: records=${replayRows.length} judgments=${paths.outputPath} summary=${paths.summaryPath}`,
  );
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  runPlannerAbJudgeCli(process.argv.slice(2)).catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
