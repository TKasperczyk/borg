/*
 * One-off backfill for Memory restructure Phase 1 / 1.1 (SelfDecisionEvent).
 *
 * Reconstructs historical autonomous decisions from the durable stream log so
 * the operator can introspect them immediately. STRUCTURAL derive (no LLM):
 *   - each `autonomous_action` internal_event -> one decision row
 *   - trigger_type from the paired preceding `autonomous_wake`
 *   - fire_event_id = the autonomous_action entry id (the fire-unique key)
 *   - decision_summary = the response summary if any, else the structural
 *     suppression reason from the turn's `agent_suppressed` entry (Phase 1.1:
 *     "stayed silent (deliberate silence): low value echo" etc.)
 * Delete-then-reinsert per fire_event_id so re-runs CORRECT existing rows.
 *
 * Usage: pnpm tsx scripts/backfill-self-decisions.ts <data-dir>
 */
import { readFileSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";

import { classifySuppressionReason } from "../src/cognition/index.js";
import { openDatabase } from "../src/storage/sqlite/index.js";
import { SelfDecisionRepository } from "../src/memory/self-decisions/index.js";
import type { SessionId, StreamEntryId } from "../src/util/ids.js";

const dataArg = process.argv[2]?.trim();
if (dataArg === undefined || dataArg.length === 0) {
  process.stderr.write("usage: pnpm tsx scripts/backfill-self-decisions.ts <data-dir>\n");
  process.exit(1);
}
const dataDir = resolve(dataArg);
const streamDir = join(dataDir, "stream");

type StreamEntry = {
  kind?: string;
  id?: string;
  timestamp?: number;
  session_id?: string;
  content?: Record<string, unknown>;
};
type WakeContext = { triggerType: "trigger" | "condition"; sourceName: string; wakeEntryId: string };
type Suppression = { reason: string; primary?: string };

function summarize(text: string): string {
  const c = text.replace(/\s+/g, " ").trim();
  return c.length <= 240 ? c : `${c.slice(0, 239)}…`;
}

function decisionSummaryFor(outcomeSummary: string, sup: Suppression | null): string {
  const emitted = summarize(outcomeSummary);
  if (emitted.length > 0) return emitted;
  if (sup !== null) {
    const cls = classifySuppressionReason(sup.reason).replaceAll("-", " ");
    const detail = (sup.primary ?? sup.reason).replaceAll("_", " ");
    return summarize(`Stayed silent (${cls}): ${detail}`);
  }
  return "";
}

const db = openDatabase(join(dataDir, "borg.db"));
const repo = new SelfDecisionRepository({ db });
const deleteByFire = db.prepare(`DELETE FROM self_decision_events WHERE fire_event_id = ?`);

let recorded = 0;
let scanned = 0;

for (const file of readdirSync(streamDir).filter((f) => f.endsWith(".jsonl"))) {
  const lines = readFileSync(join(streamDir, file), "utf8")
    .split("\n")
    .filter((l) => l.trim().length > 0);
  let lastWake: WakeContext | null = null;
  let lastSup: Suppression | null = null;

  for (const line of lines) {
    let entry: StreamEntry;
    try {
      entry = JSON.parse(line) as StreamEntry;
    } catch {
      continue;
    }

    if (entry.kind === "agent_suppressed") {
      const c = entry.content ?? {};
      if (typeof c.reason === "string") {
        lastSup = {
          reason: c.reason,
          primary: typeof c.primary_no_output_reason === "string" ? c.primary_no_output_reason : undefined,
        };
      }
      continue;
    }

    if (entry.kind !== "internal_event") continue;
    const content = entry.content ?? {};

    if (content.kind === "autonomous_wake") {
      const triggerType = content.trigger_type;
      if (triggerType === "trigger" || triggerType === "condition") {
        lastWake = {
          triggerType,
          sourceName: String(content.source_name ?? ""),
          wakeEntryId: String(entry.id ?? ""),
        };
      }
      lastSup = null; // new fire; reset until this turn's suppression (if any) is seen
      continue;
    }

    if (content.kind !== "autonomous_action") continue;
    scanned += 1;

    if (lastWake === null || entry.id === undefined || entry.session_id === undefined) {
      process.stderr.write(`skip action ${entry.id ?? "?"} (no paired wake)\n`);
      lastSup = null;
      continue;
    }

    const fireEventId = entry.id;
    deleteByFire.run(fireEventId);
    repo.record({
      occurredAt: Number(entry.timestamp ?? 0),
      sessionId: entry.session_id as SessionId,
      triggerName: String(content.trigger ?? lastWake.sourceName),
      triggerType: lastWake.triggerType,
      sourceEventId: lastWake.wakeEntryId,
      fireEventId: fireEventId as StreamEntryId,
      decisionSummary: decisionSummaryFor(String(content.outcome_summary ?? ""), lastSup),
      turnResultId: content.turn_result_id === undefined ? null : (content.turn_result_id as string | null),
      sourceStreamEntryIds: [lastWake.wakeEntryId as StreamEntryId, fireEventId as StreamEntryId],
    });
    recorded += 1;
    lastSup = null;
  }
}

process.stdout.write(`backfill complete: scanned=${scanned} recorded=${recorded}\n`);
