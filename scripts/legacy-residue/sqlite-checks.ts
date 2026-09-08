import type { SqliteDatabase } from "../../src/storage/sqlite/index.js";
import { z } from "zod";
import { consolidationFamilyIdSchema, episodeIdSchema } from "../../src/memory/episodic/types.js";
import { semanticNodeIdSchema } from "../../src/memory/semantic/types.js";
import { overseerFlagAuditPayloadSchema } from "../../src/offline/overseer/source-grounding.js";
import {
  counter,
  objectValue,
  parseStoredJson,
  ReportChecks,
  ResidueReportError,
  type CheckCount,
} from "./report.js";

export function columns(db: SqliteDatabase, table: string): Set<string> {
  const rows = db.prepare(`PRAGMA table_info("${table}")`).all();
  if (rows.length === 0) throw new ResidueReportError(`Missing table: ${table}`);
  return new Set(rows.map((row) => String(row.name)));
}

function projectedColumns(db: SqliteDatabase, table: string, wanted: string[]): string {
  const present = columns(db, table);
  return wanted.map((name) => (present.has(name) ? `"${name}"` : `NULL AS "${name}"`)).join(", ");
}

export function countSql(db: SqliteDatabase, sql: string): CheckCount {
  const matches = counter();
  for (const row of db.prepare(sql).iterate()) matches.add(row.sample_id);
  return matches.result;
}

export function totalSql(db: SqliteDatabase, table: string, where = "1"): number {
  columns(db, table);
  return Number(db.prepare(`SELECT COUNT(*) AS n FROM "${table}" WHERE ${where}`).get()?.n);
}

// SQL columns may be absent in historical schemas. JSON snapshots can omit keys.
// A nullable critical_domain is valid for advisory commitments and for critical
// kinds whose default domain is null. Do not count every SQL NULL as legacy.
export function needsCommitmentNormalization(value: Record<string, unknown>): boolean {
  if (value.kind == null || value.enforcement_class == null || value.updated_at == null)
    return true;
  if (!Object.hasOwn(value, "critical_domain")) return true;
  if (value.enforcement_class !== "critical") return value.critical_domain !== null;
  return (
    value.critical_domain == null && (value.kind === "boundary" || value.kind === "audience_rule")
  );
}

const COMMITMENT_LOGIC =
  "needsCommitmentNormalization: kind/enforcement_class/updated_at absent or null; " +
  "critical_domain key/column absent; critical boundary/audience_rule with null critical_domain; " +
  "or non-critical enforcement_class with non-null critical_domain. Advisory null domains are valid. " +
  "Matches the four-field defaults in src/memory/commitments/types.ts normalizeLegacyCommitmentValue; no normalization is applied.";

function legacyMisattribution(refs: unknown): boolean {
  if (!objectValue(refs) || !Object.hasOwn(refs, "overseer_flag")) return true;
  const parsed = overseerFlagAuditPayloadSchema.safeParse(refs.overseer_flag);
  if (!parsed.success) return true;
  const payload = parsed.data;
  if (
    payload.quoted_span === undefined ||
    payload.cited_stream_ids === undefined ||
    payload.source_assessment === undefined
  )
    return true;
  if (refs.target_type === "episode") return !episodeIdSchema.safeParse(refs.target_id).success;
  if (refs.target_type === "semantic_node")
    return !semanticNodeIdSchema.safeParse(refs.target_id).success;
  return true;
}

const MISATTRIBUTION_LOGIC =
  "JSON.parse(refs); overseer_flag absent or rejected by overseerFlagAuditPayloadSchema, " +
  "or parsed overseer_flag lacks quoted_span/cited_stream_ids/source_assessment, " +
  "or refs lacks a valid episode/semantic_node target_type and target_id. " +
  "The structural skipped_legacy branches of src/offline/overseer/revalidate.ts; no age filter, source hydration, or revalidation writes.";

// The current reverser schema is private to src/offline/consolidator/index.ts.
const currentConsolidationReversal = z.object({
  familyId: consolidationFamilyIdSchema,
  versionEpisodeId: episodeIdSchema,
  previousCurrentVersionEpisodeId: episodeIdSchema.nullable(),
  previousCoverageHash: z.string().min(1).nullable(),
  previousPolicyVersion: z.number().int().positive().nullable(),
});

export async function sqliteChecks(db: SqliteDatabase, report: ReportChecks): Promise<void> {
  const sqlCheck = async (
    id: string,
    description: string,
    table: string,
    where: string,
    key = "id",
  ) => {
    const sql = `SELECT "${key}" AS sample_id FROM "${table}" WHERE ${where}`;
    await report.run(id, description, `${sql}; total: SELECT COUNT(*) FROM "${table}"`, () => ({
      ...countSql(db, sql),
      total: totalSql(db, table),
    }));
  };

  await sqlCheck(
    "L2",
    "Stream index rows without kind",
    "stream_entry_index",
    "kind IS NULL",
    "entry_id",
  );

  await report.run(
    "L6.commitments",
    "Commitments needing legacy field normalization",
    `SELECT id, kind, enforcement_class, critical_domain, updated_at FROM commitments (absent columns projected as NULL); ${COMMITMENT_LOGIC}`,
    () => {
      const present = columns(db, "commitments");
      const sql = `SELECT ${projectedColumns(db, "commitments", ["id", "kind", "enforcement_class", "critical_domain", "updated_at"])} FROM commitments`;
      const matches = counter();
      let total = 0;
      for (const row of db.prepare(sql).iterate()) {
        total += 1;
        if (!present.has("critical_domain") || needsCommitmentNormalization(row))
          matches.add(row.id);
      }
      return { ...matches.result, total };
    },
  );

  await report.run(
    "L6.identity_events",
    "Identity event rows containing legacy commitment snapshots",
    `SELECT id, old_value_json, new_value_json FROM identity_events WHERE record_type = 'commitment'; count each row once if either non-null root snapshot needs normalization. SQL/JSON null snapshots are tombstones. ${COMMITMENT_LOGIC}`,
    () => {
      const matches = counter();
      let total = 0;
      for (const row of db
        .prepare(
          "SELECT id, old_value_json, new_value_json FROM identity_events WHERE record_type = 'commitment'",
        )
        .iterate()) {
        total += 1;
        const snapshots = [row.old_value_json, row.new_value_json].map((value) =>
          value == null ? null : parseStoredJson(value),
        );
        if (snapshots.some((value) => value !== null && !objectValue(value))) {
          throw new ResidueReportError("Non-object commitment snapshot; count unavailable");
        }
        if (snapshots.some((value) => objectValue(value) && needsCommitmentNormalization(value)))
          matches.add(row.id);
      }
      return { ...matches.result, total };
    },
  );

  for (const unreverted of [false, true]) {
    const sql = `SELECT id, reversal, reverted_at FROM maintenance_audit${unreverted ? " WHERE reverted_at IS NULL" : ""}`;
    await report.run(
      unreverted ? "L7.unreverted" : "L7",
      unreverted
        ? "Unreverted obsolete consolidation reversal payloads"
        : "Obsolete consolidation reversal payloads",
      `${sql}; JSON.parse(reversal) is an object with a string newEpisodeId and fails the current consolidation reverser schema: valid familyId/versionEpisodeId, nullable valid previousCurrentVersionEpisodeId, nullable nonempty previousCoverageHash, nullable positive-integer previousPolicyVersion. All processes/actions included.`,
      () => {
        const matches = counter();
        let total = 0;
        for (const row of db.prepare(sql).iterate()) {
          total += 1;
          const reversal = parseStoredJson(row.reversal);
          if (!objectValue(reversal))
            throw new ResidueReportError("Non-object audit reversal; count unavailable");
          if (
            typeof reversal.newEpisodeId === "string" &&
            !currentConsolidationReversal.safeParse(reversal).success
          )
            matches.add(row.id);
        }
        return { ...matches.result, total };
      },
    );
  }

  await sqlCheck(
    "L8",
    "Legacy relationship reviews, all statuses",
    "review_queue",
    "kind = 'relationship_claim_ungrounded'",
  );
  for (const [status, filter] of [
    ["open", "resolved_at IS NULL"],
    ["resolved", "resolved_at IS NOT NULL"],
  ]) {
    await sqlCheck(
      `L8.${status}`,
      `Legacy relationship reviews, ${status}`,
      "review_queue",
      `kind = 'relationship_claim_ungrounded' AND ${filter}`,
    );
  }

  for (const open of [false, true]) {
    const sql = `SELECT id, refs FROM review_queue WHERE kind = 'misattribution'${open ? " AND resolved_at IS NULL" : ""}`;
    await report.run(
      open ? "L9.open" : "L9",
      `Misattribution reviews with legacy persisted inputs, ${open ? "open" : "all statuses"}`,
      `${sql}; ${MISATTRIBUTION_LOGIC}`,
      () => {
        const matches = counter();
        let total = 0;
        for (const row of db.prepare(sql).iterate()) {
          total += 1;
          if (legacyMisattribution(parseStoredJson(row.refs))) matches.add(row.id);
        }
        return { ...matches.result, total };
      },
    );
  }

  await report.run(
    "L10",
    "Resolver diagnostic stamps without a usable attempts counter",
    "SELECT id, refs FROM review_queue; JSON.parse(refs).__borg_review_resolver_diagnostic is non-null/present and its attempts is not a positive integer (src/offline/review-resolver/index.ts needsManualAttempts fallback). Total is all stamped rows.",
    () => {
      const matches = counter();
      let total = 0;
      for (const row of db.prepare("SELECT id, refs FROM review_queue").iterate()) {
        const refs = parseStoredJson(row.refs);
        if (!objectValue(refs))
          throw new ResidueReportError("Non-object review refs; count unavailable");
        const diagnostic = refs.__borg_review_resolver_diagnostic;
        if (diagnostic == null) continue;
        total += 1;
        const attempts = objectValue(diagnostic) ? diagnostic.attempts : undefined;
        if (typeof attempts !== "number" || !Number.isInteger(attempts) || attempts <= 0)
          matches.add(row.id);
      }
      return { ...matches.result, total };
    },
  );

  await sqlCheck(
    "L11",
    "Goal follow-up watermarks without deadline/stale suffix",
    "stream_watermarks",
    "process_name GLOB 'autonomy:goal-followup-due:*' AND process_name NOT GLOB '*:deadline' AND process_name NOT GLOB '*:stale'",
    "process_name",
  );
  await sqlCheck(
    "L14",
    "Shared state entries without state_key",
    "shared_state_entries",
    "state_key IS NULL",
  );

  await report.run(
    "S4.metadata",
    "Inbox sessions lacking complete source metadata",
    "SELECT session_id, source_external_id, audience_entity_id, label, audience_label, conversation_kind FROM sessions WHERE source_type = 'teams_inbox' (absent metadata columns projected as NULL); require nonempty source_external_id/audience_entity_id/label/audience_label and conversation_kind in dm/channel/thread/demo. source_url and last_turn_id are nullable by contract.",
    () => {
      const sql = `SELECT ${projectedColumns(db, "sessions", ["session_id", "source_external_id", "audience_entity_id", "label", "audience_label", "conversation_kind"])} FROM sessions WHERE source_type = 'teams_inbox'`;
      const matches = counter();
      let total = 0;
      for (const row of db.prepare(sql).iterate()) {
        total += 1;
        if (
          [row.source_external_id, row.audience_entity_id, row.label, row.audience_label].some(
            (value) => typeof value !== "string" || value.length === 0,
          ) ||
          !["dm", "channel", "thread", "demo"].includes(String(row.conversation_kind))
        )
          matches.add(row.session_id);
      }
      return { ...matches.result, total };
    },
  );
  await sqlCheck(
    "S5",
    "Sessions with empty labels or empty non-null optional strings",
    "sessions",
    "label IS NULL OR label = '' OR audience_label IS NULL OR audience_label = '' OR source_external_id = '' OR source_url = '' OR last_turn_id = ''",
    "session_id",
  );

  await backlogChecks(db, report);
}

async function backlogChecks(db: SqliteDatabase, report: ReportChecks): Promise<void> {
  const base = `FROM sessions AS s
    JOIN stream_entry_index AS e ON e.session_id = s.session_id
    LEFT JOIN stream_watermarks AS w ON w.session_id = s.session_id AND w.process_name = 'chat-response'
    LEFT JOIN stream_entry_index AS cursor ON cursor.entry_id = w.last_entry_id
    WHERE s.source_type = 'teams_inbox' AND e.kind = 'user_msg' AND e.turn_id IS NULL
      AND e.source_message_key_source_type IS NULL`;
  const invalid = `SELECT e.entry_id AS sample_id ${base}
    AND (e.entry_index IS NULL OR (w.process_name IS NOT NULL AND
      (w.last_entry_id IS NULL OR w.last_ts IS NULL OR cursor.entry_id IS NULL OR cursor.entry_index IS NULL OR cursor.session_id <> s.session_id OR cursor.timestamp <> w.last_ts)))`;
  await report.run(
    "S4.backlog_unknown_order",
    "Pre-inbox backlog rows with missing or mismatched durable cursor facts",
    invalid,
    () => countSql(db, invalid),
  );
  for (const before of [false, true]) {
    const where = before
      ? "w.last_entry_id IS NOT NULL AND e.entry_index <= cursor.entry_index"
      : "w.last_entry_id IS NULL OR e.entry_index > cursor.entry_index";
    const sql = `SELECT DISTINCT s.session_id AS sample_id ${base} AND (${where})`;
    await report.run(
      before ? "S4.backlog_before_watermark" : "S4.pending_pre_inbox_backlog",
      before
        ? "Inbox sessions with unassigned legacy entries at/before the watermark (covered prefix)"
        : "Inbox sessions with pending pre-inbox backlog after the watermark, or no watermark",
      `${sql}; first reject missing/mismatched order using: ${invalid}. Pre-inbox means no source-message key source type. No stream files are read. No receipt_pending or active filter: this reports unconsumed backlog, including a blocked prefix.`,
      () => {
        if (countSql(db, invalid).count !== 0)
          throw new ResidueReportError("Backlog order is incomplete; see S4.backlog_unknown_order");
        return {
          ...countSql(db, sql),
          total: totalSql(db, "sessions", "source_type = 'teams_inbox'"),
        };
      },
    );
  }
}
