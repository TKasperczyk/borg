import type { ActivityRepository } from "../../memory/activity/index.js";
import {
  buildInboxReplyActivityProjection,
  type InboxReplyActivityProjectionInput,
  type InboxReplyActivitySkipReason,
} from "../../memory/activity/inbox-reply-projection.js";
import type { EntityRepository } from "../../memory/commitments/index.js";
import type { SessionsRepository } from "../../sessions/index.js";
import type { StreamEntryIndexRecord, StreamEntryIndexRepository } from "../../stream/index.js";
import type { EntityId, StreamEntryId } from "../../util/ids.js";
import { parsedSourceEntryIds } from "./chat-response-watermark.js";

// Idempotent repair for Teams inbox sessions whose agent_msg reply terminals never received a
// borg_replied activity event (the inbox path skipped the projection before 2026-09-05, and a
// crash between the terminal commit and the projection leaves the same gap). Only terminals
// stamped response_to.kind = "stream_backlog" are inbox replies; unstamped agent_msg entries are
// left alone. Kind + source dedupe in the activity repository makes re-running safe; dry runs
// count without writing. The pass runs under the tenant's exclusive lease, so it is bounded by an
// insert limit and a scan cap and reports whether it completed.

export const DEFAULT_INBOX_REPLY_ACTIVITY_RECONCILE_LIMIT = 500;
export const MAX_INBOX_REPLY_ACTIVITY_RECONCILE_LIMIT = 5_000;
export const INBOX_REPLY_ACTIVITY_RECONCILE_SCAN_CAP = 20_000;
export const INBOX_REPLY_ACTIVITY_RECONCILE_FAILED_ID_CAP = 50;
const INBOX_REPLY_ACTIVITY_SOURCE_TYPE = "teams_inbox";
// SessionsRepository.list clamps to this many rows; more sessions than that mark the pass partial.
const SESSION_PAGE_LIMIT = 1_000;

export type InboxReplyActivityReconcileInput = {
  dryRun: boolean;
  sinceMs?: number;
  untilMs?: number;
  limit?: number;
};

export type InboxReplyActivityReconcileResult = {
  dry_run: boolean;
  source_type: string;
  since_ms: number | null;
  until_ms: number | null;
  limit: number;
  sessions_scanned: number;
  sessions_truncated: boolean;
  terminals_scanned: number;
  inactive_skipped: number;
  already_recorded: number;
  inserted: number;
  skipped: Record<InboxReplyActivitySkipReason | "malformed_stamp" | "projection_failed", number>;
  failed_terminal_ids: string[];
  truncated: boolean;
  complete: boolean;
};

export type InboxReplyActivityReconcileDependencies = {
  entryIndex: Pick<
    StreamEntryIndexRepository,
    "lookupSessionStreamBacklogResponseStamps" | "lookupMany"
  >;
  sessionsRepository: Pick<SessionsRepository, "list">;
  entityRepository: Pick<EntityRepository, "getSelf">;
  activityRepository: Pick<ActivityRepository, "getByKindAndSource">;
  projectRepliedTurn: (input: InboxReplyActivityProjectionInput) => unknown;
};

export function reconcileInboxReplyActivity(
  deps: InboxReplyActivityReconcileDependencies,
  input: InboxReplyActivityReconcileInput,
): InboxReplyActivityReconcileResult {
  const limit = Math.max(
    1,
    Math.min(
      MAX_INBOX_REPLY_ACTIVITY_RECONCILE_LIMIT,
      Math.floor(input.limit ?? DEFAULT_INBOX_REPLY_ACTIVITY_RECONCILE_LIMIT),
    ),
  );
  const result: InboxReplyActivityReconcileResult = {
    dry_run: input.dryRun,
    source_type: INBOX_REPLY_ACTIVITY_SOURCE_TYPE,
    since_ms: input.sinceMs ?? null,
    until_ms: input.untilMs ?? null,
    limit,
    sessions_scanned: 0,
    sessions_truncated: false,
    terminals_scanned: 0,
    inactive_skipped: 0,
    already_recorded: 0,
    inserted: 0,
    skipped: {
      session_missing: 0,
      self_missing: 0,
      audience_missing: 0,
      session_record_incomplete: 0,
      malformed_stamp: 0,
      projection_failed: 0,
    },
    failed_terminal_ids: [],
    truncated: false,
    complete: false,
  };
  const selfEntityId = deps.entityRepository.getSelf()?.id ?? null;
  const sessions = deps.sessionsRepository.list({
    sourceType: INBOX_REPLY_ACTIVITY_SOURCE_TYPE,
    limit: SESSION_PAGE_LIMIT,
  });
  result.sessions_truncated = sessions.length >= SESSION_PAGE_LIMIT;
  let examined = 0;

  scan: for (const session of sessions) {
    result.sessions_scanned += 1;
    const terminals = deps.entryIndex
      .lookupSessionStreamBacklogResponseStamps({
        sessionId: session.session_id,
        terminalKinds: ["agent_msg"],
      })
      .sort((a, b) => a.timestamp - b.timestamp);

    for (const terminal of terminals) {
      if (input.sinceMs !== undefined && terminal.timestamp < input.sinceMs) {
        continue;
      }
      if (input.untilMs !== undefined && terminal.timestamp > input.untilMs) {
        continue;
      }
      if (examined >= INBOX_REPLY_ACTIVITY_RECONCILE_SCAN_CAP) {
        result.truncated = true;
        break scan;
      }
      examined += 1;
      if (!terminal.active) {
        result.inactive_skipped += 1;
        continue;
      }
      result.terminals_scanned += 1;
      const terminalId = terminal.entry_id as StreamEntryId;
      if (deps.activityRepository.getByKindAndSource("borg_replied", [terminalId]) !== null) {
        result.already_recorded += 1;
        continue;
      }
      if (result.inserted >= limit) {
        result.truncated = true;
        break scan;
      }
      const senderEntityIds = senderEntityIdsInStampOrder(deps.entryIndex, terminal);
      if (senderEntityIds === null) {
        result.skipped.malformed_stamp += 1;
        continue;
      }
      const projection = buildInboxReplyActivityProjection({
        session,
        selfEntityId,
        terminal: { id: terminalId, sessionId: session.session_id, timestamp: terminal.timestamp },
        senderEntityIds,
      });
      if (projection.kind === "skip") {
        result.skipped[projection.reason] += 1;
        continue;
      }
      if (input.dryRun) {
        result.inserted += 1;
        continue;
      }
      try {
        deps.projectRepliedTurn(projection.input);
        result.inserted += 1;
      } catch {
        result.skipped.projection_failed += 1;
        if (result.failed_terminal_ids.length < INBOX_REPLY_ACTIVITY_RECONCILE_FAILED_ID_CAP) {
          result.failed_terminal_ids.push(terminalId);
        }
      }
    }
  }

  result.complete =
    !result.truncated && !result.sessions_truncated && result.skipped.projection_failed === 0;
  return result;
}

// Senders of the batch the terminal answered, in the stamp's own order. A stamp whose source id
// list is missing, malformed, or inconsistent with its count is reported instead of projected
// with a partial participant list that later runs would treat as reconciled.
function senderEntityIdsInStampOrder(
  entryIndex: Pick<StreamEntryIndexRepository, "lookupMany">,
  terminal: StreamEntryIndexRecord,
): EntityId[] | null {
  const sourceEntryIds = parsedSourceEntryIds(terminal);
  if (sourceEntryIds === null) {
    return null;
  }
  if (terminal.response_to_count !== null && terminal.response_to_count !== sourceEntryIds.length) {
    return null;
  }
  const records = entryIndex.lookupMany(sourceEntryIds);
  const senders: EntityId[] = [];
  for (const sourceEntryId of sourceEntryIds) {
    const senderEntityId = records.get(sourceEntryId)?.sender_entity_id ?? null;
    if (senderEntityId !== null) {
      senders.push(senderEntityId);
    }
  }
  return senders;
}
