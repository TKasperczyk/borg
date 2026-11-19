import type { SqliteDatabase } from "../storage/sqlite/index.js";
import { ToolError } from "../util/errors.js";
import * as ids from "../util/ids.js";

export const MIN_RECORD_ID_PREFIX_CHARS = 4;
export const MAX_ID_PREFIX_CANDIDATES = 10;

// Only substrate-owned identifiers are inspected. These are storage addresses,
// not semantic labels. Never add lifecycle, audience, session, or role predicates.
export const RECORD_ID_STORES = {
  episode: { helper: ids.episodeIdHelpers, table: "episode_stats", column: "episode_id" },
  stream_entry: {
    helper: ids.streamEntryIdHelpers,
    table: "stream_entry_index",
    column: "entry_id",
    timestamp: "timestamp",
  },
  semantic_node: {
    helper: ids.semanticNodeIdHelpers,
    table: "semantic_nodes",
    column: "id",
    timestamp: "created_at",
  },
  semantic_edge: {
    helper: ids.semanticEdgeIdHelpers,
    table: "semantic_edges",
    column: "id",
    timestamp: "created_at",
  },
  goal: { helper: ids.goalIdHelpers, table: "goals", column: "id", timestamp: "created_at" },
  open_question: {
    helper: ids.openQuestionIdHelpers,
    table: "open_questions",
    column: "id",
    timestamp: "created_at",
  },
  action: {
    helper: ids.actionIdHelpers,
    table: "action_records",
    column: "id",
    timestamp: "created_at",
  },
  shared_state_entry: {
    helper: ids.sharedStateEntryIdHelpers,
    table: "shared_state_entries",
    column: "id",
    timestamp: "created_at",
  },
  value: { helper: ids.valueIdHelpers, table: "values", column: "id", timestamp: "created_at" },
  trait: { helper: ids.traitIdHelpers, table: "traits", column: "id" },
  commitment: {
    helper: ids.commitmentIdHelpers,
    table: "commitments",
    column: "id",
    timestamp: "created_at",
  },
  autobiographical_period: {
    helper: ids.autobiographicalPeriodIdHelpers,
    table: "autobiographical_periods",
    column: "id",
    timestamp: "created_at",
  },
  growth_marker: {
    helper: ids.growthMarkerIdHelpers,
    table: "growth_markers",
    column: "id",
    timestamp: "ts",
  },
  entity: { helper: ids.entityIdHelpers, table: "entities", column: "id", timestamp: "created_at" },
  session: {
    helper: ids.sessionIdHelpers,
    table: "sessions",
    column: "session_id",
    timestamp: "created_at",
  },
  scheduled_wake: {
    helper: ids.scheduledWakeIdHelpers,
    table: "scheduled_wakes",
    column: "id",
    timestamp: "created_at",
  },
  executive_step: {
    helper: ids.executiveStepIdHelpers,
    table: "executive_steps",
    column: "id",
    timestamp: "created_at",
  },
} as const;

export type RecordIdKind = keyof typeof RECORD_ID_STORES;
export type RecordIdResolution =
  | { status: "unchanged"; id: string }
  | { status: "resolved"; id: string }
  | { status: "not_found"; id: string };

export type RecordIdResolver = {
  resolve(
    input: string,
    kinds: readonly RecordIdKind[],
    requireExisting?: boolean,
  ): RecordIdResolution;
};

export function createRecordIdResolver(db: SqliteDatabase): RecordIdResolver {
  return {
    resolve(input, kinds, requireExisting = false) {
      const store = kinds
        .map((kind) => RECORD_ID_STORES[kind])
        .find(({ helper }) => input.startsWith(helper.prefix));
      // Preserve exact full-id inputs (including unknown full ids), and let the
      // existing tool schema handle other identifier grammars, e.g. "default".
      if (store === undefined) return { status: "unchanged", id: input };
      if (store.helper.is(input)) {
        // Reference-only create inputs have no native not-found check. Require
        // existence for both forms before a write; other full IDs keep the fast path.
        if (
          requireExisting &&
          db.prepare(`SELECT 1 FROM "${store.table}" WHERE "${store.column}" = ?`).get(input) ===
            undefined
        )
          return { status: "not_found", id: input };
        return { status: "unchanged", id: input };
      }
      const suffix = input.slice(store.helper.prefix.length);
      if (!/^[a-z0-9]*$/.test(suffix) || suffix.length >= ids.ID_LENGTH) {
        return { status: "unchanged", id: input };
      }
      if (suffix.length < MIN_RECORD_ID_PREFIX_CHARS) {
        throw new ToolError(
          JSON.stringify({
            code: "invalid_id_prefix",
            input,
            minimum_suffix_chars: MIN_RECORD_ID_PREFIX_CHARS,
          }),
          { code: "invalid_id_prefix" },
        );
      }
      // The identifier alphabet is ASCII. Increment its last code unit for an
      // exclusive upper bound; unlike LIKE, '_' is never a wildcard here.
      const upper =
        input.slice(0, -1) + String.fromCharCode(input.charCodeAt(input.length - 1) + 1);
      const from = `FROM "${store.table}" WHERE "${store.column}" >= ? AND "${store.column}" < ?`;
      // Keep the candidate list and omitted count on one read snapshot.
      return db
        .transaction((): RecordIdResolution => {
          const candidates = db
            .prepare(`SELECT "${store.column}" AS id ${from} ORDER BY "${store.column}" LIMIT ?`)
            .all(input, upper, MAX_ID_PREFIX_CANDIDATES + 1)
            .map((row) => row.id as string);
          // Exact stored IDs win even in stores containing legacy shorter IDs.
          if (candidates[0] === input) return { status: "unchanged", id: input };
          if (candidates.length === 0) return { status: "not_found", id: input };
          if (candidates.length === 1) return { status: "resolved", id: candidates[0]! };
          const total =
            candidates.length <= MAX_ID_PREFIX_CANDIDATES
              ? candidates.length
              : Number(db.prepare(`SELECT COUNT(*) AS total ${from}`).get(input, upper)!.total);
          throw new ToolError(
            JSON.stringify({
              code: "ambiguous_id_prefix",
              input,
              candidates: candidates.slice(0, MAX_ID_PREFIX_CANDIDATES).map((id) => {
                if (!("timestamp" in store)) return { id };
                // Bounded point lookups retain the covering index for prefix
                // resolution and fetch context only when the model must choose.
                const row = db
                  .prepare(
                    `SELECT "${store.timestamp}" AS timestamp FROM "${store.table}" WHERE "${store.column}" = ?`,
                  )
                  .get(id)!;
                return {
                  id,
                  ...(row.timestamp == null
                    ? {}
                    : { timestamp: new Date(Number(row.timestamp)).toISOString() }),
                };
              }),
              omitted_count: total - Math.min(total, MAX_ID_PREFIX_CANDIDATES),
              message:
                "Retry with a longer prefix or inspect a candidate with the matching read tool.",
            }),
            { code: "ambiguous_id_prefix" },
          );
        })
        .deferred();
    },
  };
}
