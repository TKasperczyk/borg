/** Paired compact/legacy finalizer replay. Dry by default; never opens Borg repositories. */
import { lstatSync, readFileSync } from "node:fs";
import { join } from "node:path";

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
import { parseAbReplayCliArgs, runAbCliEntrypoint } from "./ab-cli.ts";

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
  const args = parseAbReplayCliArgs(argv, {
    command: "finalizer:ab-replay",
    includeNonCompleted: false,
  });
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

runAbCliEntrypoint(import.meta.url, runFinalizerAbReplayCli);
