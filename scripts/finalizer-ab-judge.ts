/**
 * Blind offline judge for finalizer A/B replay rows. This process opens no
 * Borg repository, stream, retrieval, dispatcher, or working-memory service.
 */
import { basename, join } from "node:path";

import { loadConfig } from "../src/config/index.ts";
import {
  aggregateFinalizerAbJudgments,
  finalizerAbJudgeExclusionReason,
  finalizerAbStdoutSummary,
  judgeFinalizerAbPair,
  parseFinalizerAbReplayResultForJudge,
  type FinalizerAbJudgmentRecord,
} from "../src/cognition/deliberation/finalizer-ab-judge.ts";
import {
  parseFinalizerContextCaptureRecord,
  type FinalizerContextCaptureRecord,
} from "../src/cognition/deliberation/finalizer-context-capture.ts";
import { writePrivateContextCaptureJson } from "../src/cognition/deliberation/context-capture-storage.ts";
import { appendDurableJsonl } from "../src/util/durable-jsonl.ts";
import {
  assertReplayOAuthCredentialsOutsideDataDir,
  createLiveLlmClient,
  openPlannerCaptureSnapshot,
} from "./planner-ab-replay.ts";
import { resolvePlannerAbJudgePaths } from "./planner-ab-judge.ts";
import { parseAbJudgeCliArgs, runAbCliEntrypoint } from "./ab-cli.ts";

export function resolveFinalizerAbJudgePaths(input: {
  dataDir: string;
  inputPath?: string;
  capturesPath?: string;
  outputPath?: string;
  summaryPath?: string;
}) {
  const capturesDirectory = join(input.dataDir, "captures");
  return resolvePlannerAbJudgePaths({
    dataDir: input.dataDir,
    inputPath: input.inputPath ?? join(capturesDirectory, "finalizer-ab-results.jsonl"),
    capturesPath: input.capturesPath ?? join(capturesDirectory, "finalizer-contexts.jsonl"),
    outputPath: input.outputPath ?? join(capturesDirectory, "finalizer-ab-judgments.jsonl"),
    summaryPath: input.summaryPath ?? join(capturesDirectory, "finalizer-ab-judgment-summary.json"),
  });
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

export async function runFinalizerAbJudgeCli(
  argv: readonly string[],
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  const args = parseAbJudgeCliArgs(argv, "finalizer:ab-judge");
  const config = loadConfig({
    env,
    ...(args.dataDir === undefined ? {} : { dataDir: args.dataDir }),
  });
  const paths = resolveFinalizerAbJudgePaths({
    dataDir: config.dataDir,
    ...(args.inputPath === undefined ? {} : { inputPath: args.inputPath }),
    ...(args.capturesPath === undefined ? {} : { capturesPath: args.capturesPath }),
    ...(args.outputPath === undefined ? {} : { outputPath: args.outputPath }),
    ...(args.summaryPath === undefined ? {} : { summaryPath: args.summaryPath }),
  });
  assertReplayOAuthCredentialsOutsideDataDir({ dataDirectory: paths.dataDirectory, env });
  const replaySnapshot = openPlannerCaptureSnapshot(paths.inputPath);
  const captureSnapshot = openPlannerCaptureSnapshot(paths.sourceCapturesPath);
  const replayRows = [];
  let replayLine = 0;
  for await (const line of replaySnapshot.lines) {
    replayLine += 1;
    if (line.trim().length === 0) continue;
    replayRows.push(
      parseFinalizerAbReplayResultForJudge(parseJsonLine(paths.inputPath, replayLine, line)),
    );
    if (args.limit !== undefined && replayRows.length >= args.limit) break;
  }
  const needed = new Set(
    replayRows
      .filter((row) => finalizerAbJudgeExclusionReason(row, null) === "missing_capture")
      .map((row) => row.capture_id),
  );
  const captures = new Map<string, FinalizerContextCaptureRecord>();
  let captureLine = 0;
  for await (const line of captureSnapshot.lines) {
    captureLine += 1;
    if (line.trim().length === 0) continue;
    const value = parseJsonLine(paths.sourceCapturesPath, captureLine, line) as {
      capture_id?: unknown;
    };
    if (typeof value.capture_id !== "string" || !needed.has(value.capture_id)) continue;
    const capture = parseFinalizerContextCaptureRecord(value);
    captures.set(capture.capture_id, capture);
    if (captures.size === needed.size) break;
  }
  const llmClient = createLiveLlmClient(config, env);
  const judgments: FinalizerAbJudgmentRecord[] = [];
  for (const replay of replayRows) {
    const judgment = await judgeFinalizerAbPair(replay, captures.get(replay.capture_id) ?? null, {
      llmClient,
      model: config.anthropic.models.cognition,
    });
    judgments.push(judgment);
    await appendDurableJsonl(paths.outputPath, judgment, {
      privateDirectory: paths.capturesDirectory,
    });
  }
  const summary = aggregateFinalizerAbJudgments(judgments, {
    generatedAt: Date.now(),
    inputRecords: replayRows.length,
  });
  writePrivateContextCaptureJson({
    dataDir: paths.dataDirectory,
    fileName: basename(paths.summaryPath),
    value: summary,
  });
  console.log(JSON.stringify(finalizerAbStdoutSummary(summary), null, 2));
}

runAbCliEntrypoint(import.meta.url, runFinalizerAbJudgeCli);
