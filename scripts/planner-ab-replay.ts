/**
 * Paired compact/legacy planner replay. Dry mode is the default. Live mode
 * opens no Borg substrate, repository, StreamWriter, retrieval service, or
 * working-memory service; it only uses the ordinary unary planner transport.
 */
import { createReadStream, realpathSync, statSync } from "node:fs";
import { join, resolve } from "node:path";
import { createInterface } from "node:readline";

import { resolveClaudeOAuthCredentialsPath } from "../src/auth/claude-oauth.ts";
import { loadConfig } from "../src/config/index.ts";
import { AnthropicLLMClient, type LLMClient } from "../src/llm/index.ts";
import {
  parsePlannerContextCaptureRecord,
  plannerContextCapturePath,
} from "../src/cognition/deliberation/planner-context-capture.ts";
import { replayPlannerContextCapture } from "../src/cognition/deliberation/planner-ab-replay.ts";
import { appendDurableJsonl } from "../src/util/durable-jsonl.ts";
import { isPathWithin, resolveRealPathForCreation } from "../src/util/path.ts";
import { parseAbReplayCliArgs, runAbCliEntrypoint, type AbReplayCliArgs } from "./ab-cli.ts";

export function parsePlannerAbReplayArgs(argv: readonly string[]): AbReplayCliArgs {
  return parseAbReplayCliArgs(argv, {
    command: "planner:ab-replay",
    includeNonCompleted: true,
  });
}

export type ResolvedPlannerAbReplayPaths = {
  dataDirectory: string;
  capturesDirectory: string;
  inputPath: string;
  outputPath: string;
};

export function resolvePlannerAbReplayPaths(input: {
  dataDir: string;
  inputPath?: string;
  outputPath?: string;
}): ResolvedPlannerAbReplayPaths {
  const dataDirectory = realpathSync(input.dataDir);
  const capturesDirectory = resolveRealPathForCreation(join(dataDirectory, "captures"));
  if (!isPathWithin(dataDirectory, capturesDirectory) || capturesDirectory === dataDirectory) {
    throw new Error("The captures directory must resolve below the Borg data dir");
  }
  const inputPath = realpathSync(
    resolve(input.inputPath ?? plannerContextCapturePath(dataDirectory)),
  );
  const outputPath = resolveRealPathForCreation(
    resolve(input.outputPath ?? join(capturesDirectory, "planner-ab-results.jsonl")),
  );
  if (inputPath === outputPath) {
    throw new Error("Planner A/B input and output paths must differ after resolving symlinks");
  }
  if (isPathWithin(dataDirectory, outputPath) && !isPathWithin(capturesDirectory, outputPath)) {
    throw new Error("Planner A/B output inside dataDir must stay within dataDir/captures");
  }
  return { dataDirectory, capturesDirectory, inputPath, outputPath };
}

export function assertReplayOAuthCredentialsOutsideDataDir(input: {
  dataDirectory: string;
  env: NodeJS.ProcessEnv;
}): void {
  const credentialsPath = resolveRealPathForCreation(
    resolveClaudeOAuthCredentialsPath({ env: input.env }),
  );
  if (isPathWithin(input.dataDirectory, credentialsPath)) {
    throw new Error("Live planner replay OAuth credentials must resolve outside the Borg data dir");
  }
}

export function createLiveLlmClient(
  config: ReturnType<typeof loadConfig>,
  env: NodeJS.ProcessEnv,
  attachmentResolver?: ConstructorParameters<typeof AnthropicLLMClient>[0]["attachmentResolver"],
): LLMClient {
  return new AnthropicLLMClient({
    authMode: config.anthropic.auth,
    ...(config.anthropic.apiKey === undefined ? {} : { apiKey: config.anthropic.apiKey }),
    env,
    oauthSseInactivityTimeoutMs: config.anthropic.oauthSseInactivityTimeoutMs,
    oauthSseFirstMessageEventTimeoutMs: config.anthropic.oauthSseFirstMessageEventTimeoutMs,
    oauthSseMessageEventGapTimeoutMs: config.anthropic.oauthSseMessageEventGapTimeoutMs,
    oauthFetchHeadersTimeoutMs: config.anthropic.oauthFetchHeadersTimeoutMs,
    oauthUnaryBodyTimeoutMs: config.anthropic.oauthUnaryBodyTimeoutMs,
    unaryCallTimeoutMs: config.anthropic.unaryCallTimeoutMs,
    streamingCallTimeoutMs: config.anthropic.streamingCallTimeoutMs,
    transportStallMaxRetries: config.anthropic.transportStallMaxRetries,
    ...(attachmentResolver === undefined ? {} : { attachmentResolver }),
  });
}

function printDrySummary(result: Awaited<ReturnType<typeof replayPlannerContextCapture>>): void {
  const compact = result.surfaces.compact;
  const legacy = result.surfaces.legacy;
  console.log(
    [
      result.capture_id,
      `status=${result.pairing_status}`,
      `source=${result.source_outcome.status}`,
      `compact=${compact.fingerprint.systemChars} chars/${compact.traceSummary.totalEstimatedTokens} est`,
      `legacy=${legacy.fingerprint.systemChars} chars/${legacy.traceSummary.totalEstimatedTokens} est`,
      `delta=${result.size_delta.compact_minus_legacy_chars} chars`,
      `fidelity=${compact.byteFaithfulToCapture && legacy.byteFaithfulToCapture && result.fidelity.currentSourceRequestMatchesCapture ? "match" : "drift"}`,
    ].join(" "),
  );
}

export function openPlannerCaptureSnapshot(path: string): {
  snapshotBytes: number;
  lines: ReturnType<typeof createInterface>;
} {
  const snapshotBytes = statSync(path).size;
  const input = createReadStream(path, {
    encoding: "utf8",
    start: 0,
    ...(snapshotBytes === 0 ? { end: 0 } : { end: snapshotBytes - 1 }),
  });
  return {
    snapshotBytes,
    lines: createInterface({ input, crlfDelay: Infinity }),
  };
}

export async function runPlannerAbReplayCli(
  argv: readonly string[],
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  const args = parsePlannerAbReplayArgs(argv);
  const config = loadConfig({
    env,
    ...(args.dataDir === undefined ? {} : { dataDir: args.dataDir }),
  });
  const paths = resolvePlannerAbReplayPaths({
    dataDir: config.dataDir,
    ...(args.inputPath === undefined ? {} : { inputPath: args.inputPath }),
    ...(args.outputPath === undefined ? {} : { outputPath: args.outputPath }),
  });
  if (args.mode === "live") {
    assertReplayOAuthCredentialsOutsideDataDir({
      dataDirectory: paths.dataDirectory,
      env,
    });
  }
  const llmClient = args.mode === "live" ? createLiveLlmClient(config, env) : undefined;
  // Freeze the cohort at startup. Captures appended during this replay are
  // intentionally left for the next run.
  const { snapshotBytes, lines } = openPlannerCaptureSnapshot(paths.inputPath);
  let processed = 0;
  let lineNumber = 0;

  for await (const line of lines) {
    lineNumber += 1;
    if (snapshotBytes === 0 || line.trim().length === 0) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line) as unknown;
    } catch (error) {
      throw new Error(
        `Invalid capture JSON at ${paths.inputPath}:${lineNumber}: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
    const capture = parsePlannerContextCaptureRecord(parsed);
    const common = {
      pairIndex: processed,
      includeNonCompleted: args.includeNonCompleted,
    };
    const result =
      args.mode === "live"
        ? await replayPlannerContextCapture(capture, {
            mode: "live",
            llmClient: llmClient!,
            ...common,
          })
        : await replayPlannerContextCapture(capture, { mode: "dry", ...common });
    await appendDurableJsonl(paths.outputPath, result, {
      ...(isPathWithin(paths.capturesDirectory, paths.outputPath)
        ? { privateDirectory: paths.capturesDirectory }
        : {}),
    });
    if (args.mode === "dry") printDrySummary(result);
    processed += 1;
    if (args.limit !== undefined && processed >= args.limit) break;
  }

  console.log(
    `planner A/B replay complete: mode=${args.mode} records=${processed} results=${paths.outputPath}`,
  );
}

runAbCliEntrypoint(import.meta.url, runPlannerAbReplayCli);
