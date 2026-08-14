/** Paired compact/legacy finalizer replay. Dry by default; never opens Borg repositories. */
import { lstatSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { loadConfig } from "../src/config/index.ts";
import type { AttachmentId } from "../src/util/ids.ts";
import { appendDurableJsonl } from "../src/util/durable-jsonl.ts";
import { isPathWithin } from "../src/util/path.ts";
import { resolveContextCaptureSubdirectory } from "../src/cognition/deliberation/context-capture-storage.ts";
import { sha256Bytes } from "../src/cognition/deliberation/request-fingerprint.ts";
import {
  parseFinalizerContextCaptureRecord,
  type FinalizerContextCaptureRecord,
} from "../src/cognition/deliberation/finalizer-context-capture.ts";
import { replayFinalizerContextCapture } from "../src/cognition/deliberation/finalizer-ab-replay.ts";
import {
  assertReplayOAuthCredentialsOutsideDataDir,
  createLiveLlmClient,
  openPlannerCaptureSnapshot,
  resolvePlannerAbReplayPaths,
} from "./planner-ab-replay.ts";

type Args = {
  mode: "dry" | "live";
  dataDir?: string;
  inputPath?: string;
  outputPath?: string;
  limit?: number;
};

function usage(message?: string): never {
  if (message !== undefined) console.error(message);
  console.error(
    "Usage: pnpm finalizer:ab-replay -- [--dry|--live] [--data-dir DIR] [--input FILE] [--output FILE] [--limit N]",
  );
  process.exit(1);
}

function parseArgs(argv: readonly string[]): Args {
  const args: Args = { mode: "dry" };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index]!;
    if (arg === "--dry" || arg === "--live") {
      args.mode = arg === "--live" ? "live" : "dry";
      continue;
    }
    if (arg === "--data-dir" || arg === "--input" || arg === "--output" || arg === "--limit") {
      const value = argv[++index];
      if (value === undefined) usage(`${arg} requires a value`);
      if (arg === "--data-dir") args.dataDir = value;
      else if (arg === "--input") args.inputPath = value;
      else if (arg === "--output") args.outputPath = value;
      else {
        const limit = Number(value);
        if (!Number.isInteger(limit) || limit <= 0) usage("--limit must be positive");
        args.limit = limit;
      }
      continue;
    }
    if (arg === "--help" || arg === "-h") usage();
    usage(`Unknown argument: ${arg}`);
  }
  return args;
}

function attachmentResolver(
  dataDir: string,
  record: FinalizerContextCaptureRecord,
): (attachmentId: AttachmentId) => { mediaType: string; bytes: Buffer } {
  const byId = new Map(record.image_sidecars.map((sidecar) => [sidecar.attachment_id, sidecar]));
  const sidecarDirectory = resolveContextCaptureSubdirectory(dataDir, "finalizer-images");
  return (attachmentId) => {
    const sidecar = byId.get(attachmentId);
    if (sidecar === undefined) throw new Error(`Missing finalizer replay image ${attachmentId}`);
    const path = join(sidecarDirectory, sidecar.sha256);
    const stats = lstatSync(path);
    if (stats.isSymbolicLink() || !stats.isFile()) {
      throw new Error(`Finalizer replay image sidecar is not a regular file: ${attachmentId}`);
    }
    const bytes = readFileSync(path);
    if (sha256Bytes(bytes) !== sidecar.sha256) {
      throw new Error(`Finalizer replay image sidecar hash mismatch for ${attachmentId}`);
    }
    return {
      mediaType: sidecar.media_type,
      bytes,
    };
  };
}

export async function runFinalizerAbReplayCli(
  argv: readonly string[],
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  const args = parseArgs(argv);
  const config = loadConfig({
    env,
    ...(args.dataDir === undefined ? {} : { dataDir: args.dataDir }),
  });
  const paths = resolvePlannerAbReplayPaths({
    dataDir: config.dataDir,
    inputPath: args.inputPath ?? join(config.dataDir, "captures", "finalizer-contexts.jsonl"),
    outputPath: args.outputPath ?? join(config.dataDir, "captures", "finalizer-ab-results.jsonl"),
  });
  if (args.mode === "live") {
    assertReplayOAuthCredentialsOutsideDataDir({ dataDirectory: paths.dataDirectory, env });
  }
  const { lines } = openPlannerCaptureSnapshot(paths.inputPath);
  let processed = 0;
  for await (const line of lines) {
    if (line.trim().length === 0) continue;
    const record = parseFinalizerContextCaptureRecord(JSON.parse(line) as unknown);
    const result =
      args.mode === "live"
        ? await replayFinalizerContextCapture(record, {
            mode: "live",
            pairIndex: processed,
            llmClient: createLiveLlmClient(
              config,
              env,
              attachmentResolver(paths.dataDirectory, record),
            ),
          })
        : await replayFinalizerContextCapture(record, { mode: "dry", pairIndex: processed });
    await appendDurableJsonl(paths.outputPath, result, {
      ...(isPathWithin(paths.capturesDirectory, paths.outputPath)
        ? { privateDirectory: paths.capturesDirectory }
        : {}),
    });
    processed += 1;
    if (args.limit !== undefined && processed >= args.limit) break;
  }
  console.log(`finalizer A/B replay complete: mode=${args.mode} records=${processed}`);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  runFinalizerAbReplayCli(process.argv.slice(2)).catch((error: unknown) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
