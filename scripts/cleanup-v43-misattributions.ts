import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { Borg, type ReviewQueueItem } from "../src/index.ts";

/**
 * One-shot cleanup for v43 simulator data investigated in thread 019e1182
 * (eb95e822). The dismissal list is intentionally curated and fixed:
 * six audience-name false positives now covered by Sprint 9.6 grounding, plus
 * one stale item whose proposed patch would overwrite newer target text.
 *
 * Items 4, 33, and 44 are genuine/manual-review cases and are only surfaced in
 * stdout. This script is deliberately not a reusable aging policy or CLI
 * command.
 */

const DEFAULT_DATA_DIR = "/tmp/borg-assessor-be355a0b-simulator-tom/";
const SOURCE_PROCESS = "cleanup-v43-misattributions";
const TRACE_FILE_NAME = "cleanup-v43-misattributions.trace.jsonl";
const AUDIENCE_NAME_REASON = "suppressed-by-9.6-audience-name-grounding";
const STALE_REASON = "stale -- target text updated since enqueue";
const MANUAL_REVIEW_ITEM_IDS = [4, 33, 44] as const;

const DISMISSALS = [
  { id: 5, reason: STALE_REASON },
  { id: 6, reason: AUDIENCE_NAME_REASON },
  { id: 11, reason: AUDIENCE_NAME_REASON },
  { id: 12, reason: AUDIENCE_NAME_REASON },
  { id: 23, reason: AUDIENCE_NAME_REASON },
  { id: 30, reason: AUDIENCE_NAME_REASON },
  { id: 37, reason: AUDIENCE_NAME_REASON },
] as const;

function log(line: string): void {
  process.stdout.write(`${line}\n`);
}

function formatItemIds(ids: readonly number[]): string {
  return ids.join(", ");
}

function indexReviewItems(items: readonly ReviewQueueItem[]): Map<number, ReviewQueueItem> {
  return new Map(items.map((item) => [item.id, item]));
}

async function dismissOpenItem(borg: Borg, itemId: number, reason: string): Promise<void> {
  const resolved = await borg.review.resolve(
    itemId,
    {
      decision: "dismiss",
      reason,
    },
    {
      source: "manual",
      sourceProcess: SOURCE_PROCESS,
      traceTurnId: SOURCE_PROCESS,
    },
  );

  if (resolved === null) {
    log(`item ${itemId}: missing; skipped`);
    return;
  }

  log(`item ${itemId}: dismissed resolution=${resolved.resolution} reason="${reason}"`);
}

export async function cleanupV43Misattributions(
  env: NodeJS.ProcessEnv = process.env,
): Promise<void> {
  const dataDir = env.BORG_DATA_DIR?.trim() || DEFAULT_DATA_DIR;
  const borg = await Borg.open({
    dataDir,
    liveExtraction: false,
    tracerPath: join(dataDir, TRACE_FILE_NAME),
  });

  try {
    const itemsById = indexReviewItems(borg.review.list());

    for (const { id, reason } of DISMISSALS) {
      const item = itemsById.get(id);

      if (item === undefined) {
        log(`item ${id}: missing; skipped`);
        continue;
      }

      if (item.kind !== "misattribution") {
        log(`item ${id}: kind=${item.kind}; skipped`);
        continue;
      }

      if (item.resolved_at !== null) {
        log(`item ${id}: already resolved resolution=${item.resolution}; skipped`);
        continue;
      }

      await dismissOpenItem(borg, id, reason);
    }

    log(`Manual review required for items: ${formatItemIds(MANUAL_REVIEW_ITEM_IDS)}.`);
    log("Dismissed 7 historical misattributions. 3 items remain for manual review: 4, 33, 44.");
  } finally {
    await borg.close();
  }
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(process.argv[1]).href) {
  cleanupV43Misattributions().catch((error: unknown) => {
    process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`);
    process.exitCode = 1;
  });
}
