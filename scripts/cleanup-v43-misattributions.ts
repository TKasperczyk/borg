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

const MANUAL_REVIEW_ITEMS = [
  { id: 4, targetId: "semn_h4gfwa44mequ3zjn" },
  { id: 33, targetId: "ep_mq3q0j9w4bbcwsn7" },
  { id: 44, targetId: "ep_9uc5vju8xu7meah4" },
] as const;

const DISMISSALS = [
  { id: 5, targetId: "semn_xbhzpsl0lnjivsjb", reason: STALE_REASON },
  { id: 6, targetId: "semn_xr9in43hkkpvjzq0", reason: AUDIENCE_NAME_REASON },
  { id: 11, targetId: "semn_o49wmd9uz7a7odpv", reason: AUDIENCE_NAME_REASON },
  { id: 12, targetId: "semn_ygfnyui5uohufxka", reason: AUDIENCE_NAME_REASON },
  { id: 23, targetId: "semn_r7at1a724czzdvu9", reason: AUDIENCE_NAME_REASON },
  { id: 30, targetId: "semn_igtbdpc3fcnfoq9k", reason: AUDIENCE_NAME_REASON },
  { id: 37, targetId: "semn_3ttb13zxtbvl3o09", reason: AUDIENCE_NAME_REASON },
] as const;

function log(line: string): void {
  process.stdout.write(`${line}\n`);
}

function formatItemIds(ids: readonly number[]): string {
  return ids.length === 0 ? "none" : ids.join(", ");
}

function indexReviewItems(items: readonly ReviewQueueItem[]): Map<number, ReviewQueueItem> {
  return new Map(items.map((item) => [item.id, item]));
}

function reviewTargetId(item: ReviewQueueItem): string | null {
  const targetId = item.refs.target_id;

  return typeof targetId === "string" ? targetId : null;
}

function fingerprintMatches(item: ReviewQueueItem, expectedTargetId: string): boolean {
  return reviewTargetId(item) === expectedTargetId;
}

function isExpectedManualReviewItem(item: ReviewQueueItem): boolean {
  const expected = MANUAL_REVIEW_ITEMS.find((candidate) => candidate.id === item.id);

  return expected !== undefined && fingerprintMatches(item, expected.targetId);
}

async function dismissOpenItem(borg: Borg, itemId: number, reason: string): Promise<boolean> {
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
    return false;
  }

  log(`item ${itemId}: dismissed resolution=${resolved.resolution} reason="${reason}"`);
  return true;
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
    let dismissedCount = 0;
    let skippedCount = 0;

    for (const { id, targetId, reason } of DISMISSALS) {
      const item = itemsById.get(id);

      if (item === undefined) {
        log(`item ${id}: missing; skipped`);
        skippedCount += 1;
        continue;
      }

      if (item.kind !== "misattribution") {
        log(`item ${id}: kind=${item.kind}; skipped`);
        skippedCount += 1;
        continue;
      }

      if (!fingerprintMatches(item, targetId)) {
        log(
          `item ${id}: fingerprint mismatch expected target_id=${targetId} actual_target_id=${reviewTargetId(item) ?? "missing"}; skipped`,
        );
        skippedCount += 1;
        continue;
      }

      if (item.resolved_at !== null) {
        log(`item ${id}: already resolved resolution=${item.resolution}; skipped`);
        skippedCount += 1;
        continue;
      }

      if (await dismissOpenItem(borg, id, reason)) {
        dismissedCount += 1;
      } else {
        skippedCount += 1;
      }
    }

    const openManualReviewIds = borg.review
      .list({
        kind: "misattribution",
        openOnly: true,
      })
      .filter(isExpectedManualReviewItem)
      .map((item) => item.id)
      .sort((left, right) => left - right);

    log(`Manual review required for open items: ${formatItemIds(openManualReviewIds)}.`);
    log(
      `Dismissed ${dismissedCount} historical misattributions; skipped ${skippedCount}. ${openManualReviewIds.length} items remain for manual review: ${formatItemIds(openManualReviewIds)}.`,
    );
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
