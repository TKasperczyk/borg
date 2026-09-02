import { z } from "zod";

import { DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET } from "../../cognition/deliberation/constants.js";
import {
  memoryDisclosureLabelMetadataSchema,
  selfPrivateMemoryDisclosureLabel,
  unknownMemoryDisclosureLabel,
} from "../../memory/common/disclosure-label.js";
import { memoryDisclosurePayloadFields } from "../../memory/common/disclosure-serializers.js";
import type {
  TrainOfThoughtJournalEntry,
  TrainOfThoughtRangeCursor,
} from "../../memory/train-of-thought/index.js";
import type { StreamEntry, StreamEntryIndexRecord } from "../../stream/index.js";
import { SystemClock, type Clock } from "../../util/clock.js";
import { ToolError } from "../../util/errors.js";
import { sessionIdSchema, streamEntryIdSchema } from "../../util/id-schemas.js";
import type { SessionId, StreamEntryId } from "../../util/ids.js";
import { formatRelativeAge } from "../../util/relative-time.js";
import { estimatePromptTokens, stringifyPromptContent } from "../../util/token-estimate.js";
import { OWN_RECORDS_PAGE_END_CLAIM } from "./own-records-page-end-claim.js";
import type { ToolDefinition } from "../dispatcher.js";

const OWN_RECORD_KINDS = ["thought", "journal"] as const;
const OWN_RECORD_PAYLOAD_STATUSES = [
  "exact",
  "check_not_completed_budget",
  "check_not_completed_retrieval_unavailable",
] as const;
const OWN_RECORD_ORIGIN_TIME_BASES = ["stream_timestamp", "journal_created_at"] as const;
const OWN_RECORDS_PAGE_END_REASONS = [
  "range_exhausted",
  "limit_reached",
  "context_budget",
] as const;
const DEFAULT_OWN_RECORDS_LIMIT = 20;
const MAX_OWN_RECORDS_LIMIT = 50;

const ownRecordKindSchema = z.enum(OWN_RECORD_KINDS);

const ownRecordsListInputSchema = z
  .object({
    since: z.iso.datetime({ offset: true }),
    until: z.iso.datetime({ offset: true }),
    kinds: z.array(ownRecordKindSchema).min(1).max(OWN_RECORD_KINDS.length).optional(),
    session_id: sessionIdSchema.optional(),
    limit: z.number().int().positive().max(MAX_OWN_RECORDS_LIMIT).optional(),
    cursor: z.string().min(1).optional(),
  })
  .strict()
  .superRefine((input, context) => {
    if (Date.parse(input.since) > Date.parse(input.until)) {
      context.addIssue({
        code: "custom",
        message: "since must be earlier than or equal to until",
        path: ["since"],
      });
    }
  });

const ownRecordCursorPayloadSchema = z.discriminatedUnion("kind", [
  z
    .object({
      version: z.literal(1),
      occurred_at: z.number().int().finite(),
      kind: z.literal("thought"),
      source_id: streamEntryIdSchema,
    })
    .strict(),
  z
    .object({
      version: z.literal(1),
      occurred_at: z.number().int().finite(),
      kind: z.literal("journal"),
      source_id: z.number().int().positive(),
    })
    .strict(),
]);

const ownRecordForCognitionSchema = z
  .object({
    handle: z.string().min(1),
    kind: ownRecordKindSchema,
    content: z.string().nullable(),
    payload_status: z.enum(OWN_RECORD_PAYLOAD_STATUSES),
    payload_included_chars: z.number().int().nonnegative(),
    payload_total_chars: z.number().int().nonnegative(),
    occurred_at: z.number().int().finite(),
    occurred_at_iso: z.iso.datetime({ offset: true }),
    origin_time_basis: z.enum(OWN_RECORD_ORIGIN_TIME_BASES),
    relative_age: z.string().min(1),
    session_id: sessionIdSchema.nullable(),
    turn_id: z.string().min(1).nullable(),
    stream_entry_id: streamEntryIdSchema.nullable(),
    journal_entry_id: z.number().int().positive().nullable(),
    marker_stream_entry_id: streamEntryIdSchema.nullable(),
    self_entity_id: z.string().min(1).nullable(),
    oversized_anchors_omitted: z.boolean(),
    disclosure: z.string().min(1),
    disclosure_label: memoryDisclosureLabelMetadataSchema,
  })
  .strict();

const ownRecordsListOutputSchema = z
  .object({
    records: z.array(ownRecordForCognitionSchema),
    has_more: z.boolean(),
    page_end_reason: z.enum(OWN_RECORDS_PAGE_END_REASONS),
    next_cursor: z.string().min(1).nullable(),
  })
  .strict();

export type OwnRecordKind = z.infer<typeof ownRecordKindSchema>;

export type OwnRecordsListRange = {
  sinceMs: number;
  untilMs: number;
  limit: number;
  sessionId?: SessionId;
};

export type OwnRecordsListToolOptions = {
  listThoughtRecords: (
    input: OwnRecordsListRange & {
      cursor?: {
        timestamp: number;
        entryId: StreamEntryId;
      };
    },
  ) => StreamEntryIndexRecord[] | Promise<StreamEntryIndexRecord[]>;
  readThoughtRecord: (
    record: StreamEntryIndexRecord,
  ) => StreamEntry | null | Promise<StreamEntry | null>;
  listJournalRecords: (
    input: OwnRecordsListRange & { cursor?: TrainOfThoughtRangeCursor },
  ) => TrainOfThoughtJournalEntry[] | Promise<TrainOfThoughtJournalEntry[]>;
  clock?: Clock;
};

type OwnRecordCursor = z.infer<typeof ownRecordCursorPayloadSchema>;

type OwnRecordCandidate =
  | {
      kind: "thought";
      occurredAt: number;
      sourceId: StreamEntryId;
      record: StreamEntryIndexRecord;
    }
  | {
      kind: "journal";
      occurredAt: number;
      sourceId: number;
      record: TrainOfThoughtJournalEntry;
    };

type OwnRecordForCognition = z.infer<typeof ownRecordForCognitionSchema>;
type OwnRecordsListOutput = z.infer<typeof ownRecordsListOutputSchema>;
type OwnRecordsPageEndReason = OwnRecordsListOutput["page_end_reason"];

// has_more says only that the range holds more; it never says how much, and it
// fires identically whether the requested limit or the result's own token
// budget ended the page. The reason names which, so a page shorter than the
// requested limit is not read as evidence the range is shallow.
function pageEndReason(input: {
  hasMore: boolean;
  endedOnBudget: boolean;
}): OwnRecordsPageEndReason {
  if (!input.hasMore) {
    return "range_exhausted";
  }

  return input.endedOnBudget ? "context_budget" : "limit_reached";
}

const SOURCE_ORDER = {
  thought: 0,
  journal: 1,
} as const satisfies Record<OwnRecordKind, number>;

function encodeOwnRecordCursor(candidate: OwnRecordCandidate): string {
  const payload: OwnRecordCursor = {
    version: 1,
    occurred_at: candidate.occurredAt,
    kind: candidate.kind,
    source_id: candidate.sourceId,
  } as OwnRecordCursor;

  return Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
}

function decodeOwnRecordCursor(cursor: string): OwnRecordCursor {
  try {
    const decoded = Buffer.from(cursor, "base64url").toString("utf8");
    return ownRecordCursorPayloadSchema.parse(JSON.parse(decoded));
  } catch (error) {
    throw new ToolError("Invalid own-records cursor", {
      cause: error,
      code: "OWN_RECORDS_CURSOR_INVALID",
    });
  }
}

function sourceUntilForCursor(
  sourceKind: OwnRecordKind,
  requestedUntilMs: number,
  cursor: OwnRecordCursor | undefined,
): number {
  if (cursor === undefined) {
    return requestedUntilMs;
  }

  const sourceOrder = SOURCE_ORDER[sourceKind];
  const cursorOrder = SOURCE_ORDER[cursor.kind];

  return Math.min(
    requestedUntilMs,
    sourceOrder < cursorOrder ? cursor.occurred_at - 1 : cursor.occurred_at,
  );
}

function compareOwnRecordCandidates(left: OwnRecordCandidate, right: OwnRecordCandidate): number {
  if (left.occurredAt !== right.occurredAt) {
    return right.occurredAt - left.occurredAt;
  }

  if (left.kind !== right.kind) {
    return SOURCE_ORDER[left.kind] - SOURCE_ORDER[right.kind];
  }

  if (left.sourceId === right.sourceId) {
    return 0;
  }

  return left.sourceId > right.sourceId ? -1 : 1;
}

function resultFitsBudget(output: OwnRecordsListOutput, maxTokens: number): boolean {
  return estimatePromptTokens(JSON.stringify(output)) <= maxTokens;
}

function withoutPayload(
  row: OwnRecordForCognition,
  status: "check_not_completed_budget" | "check_not_completed_retrieval_unavailable",
): OwnRecordForCognition {
  return {
    ...row,
    content: null,
    payload_status: status,
    payload_included_chars: 0,
  };
}

function withoutOversizedAnchors(row: OwnRecordForCognition): OwnRecordForCognition {
  return {
    handle: row.handle,
    kind: row.kind,
    content: null,
    payload_status: "check_not_completed_budget",
    payload_included_chars: 0,
    payload_total_chars: row.payload_total_chars,
    occurred_at: row.occurred_at,
    occurred_at_iso: row.occurred_at_iso,
    origin_time_basis: row.origin_time_basis,
    relative_age: row.relative_age,
    session_id: null,
    turn_id: null,
    stream_entry_id: null,
    journal_entry_id: null,
    marker_stream_entry_id: null,
    self_entity_id: null,
    oversized_anchors_omitted: true,
    disclosure: row.disclosure,
    disclosure_label: row.disclosure_label,
  };
}

export function createOwnRecordsListTool(
  options: OwnRecordsListToolOptions,
): ToolDefinition<
  z.infer<typeof ownRecordsListInputSchema>,
  z.infer<typeof ownRecordsListOutputSchema>
> {
  const clock = options.clock ?? new SystemClock();

  return {
    name: "tool.ownRecords.list",
    description:
      "Browse my own durable thought stream and train-of-thought journal by inclusive origin-time range. This is global unless I explicitly pass session_id. It has no text query: I choose dates, kinds, and pages, then inspect exact content. " +
      OWN_RECORDS_PAGE_END_CLAIM,
    menuSummary:
      "Browse my own thoughts and journal globally by origin-time range, with an optional explicit session filter. " +
      OWN_RECORDS_PAGE_END_CLAIM,
    allowedOrigins: ["autonomous", "deliberator"],
    writeScope: "read",
    inputSchema: ownRecordsListInputSchema,
    outputSchema: ownRecordsListOutputSchema,
    async invoke(input) {
      const sinceMs = Date.parse(input.since);
      const untilMs = Date.parse(input.until);
      const limit = input.limit ?? DEFAULT_OWN_RECORDS_LIMIT;
      const selectedKinds = new Set<OwnRecordKind>(input.kinds ?? OWN_RECORD_KINDS);
      const cursor = input.cursor === undefined ? undefined : decodeOwnRecordCursor(input.cursor);
      const baseRange = {
        sinceMs,
        limit: limit + 1,
        ...(input.session_id === undefined ? {} : { sessionId: input.session_id }),
      };
      const thoughtRecords = selectedKinds.has("thought")
        ? await options.listThoughtRecords({
            ...baseRange,
            untilMs: sourceUntilForCursor("thought", untilMs, cursor),
            ...(cursor?.kind === "thought"
              ? {
                  cursor: {
                    timestamp: cursor.occurred_at,
                    entryId: cursor.source_id,
                  },
                }
              : {}),
          })
        : [];
      const journalRecords = selectedKinds.has("journal")
        ? await options.listJournalRecords({
            ...baseRange,
            untilMs: sourceUntilForCursor("journal", untilMs, cursor),
            ...(cursor?.kind === "journal"
              ? {
                  cursor: {
                    createdAt: cursor.occurred_at,
                    id: cursor.source_id,
                  },
                }
              : {}),
          })
        : [];
      const merged = [
        ...thoughtRecords.map((record): OwnRecordCandidate => ({
          kind: "thought",
          occurredAt: record.timestamp,
          sourceId: record.entry_id as StreamEntryId,
          record,
        })),
        ...journalRecords.map((record): OwnRecordCandidate => ({
          kind: "journal",
          occurredAt: record.created_at,
          sourceId: record.id,
          record,
        })),
      ].sort(compareOwnRecordCandidates);
      const pageCandidates = merged.slice(0, limit);
      const moreInStore = merged.length > pageCandidates.length;
      const nowMs = clock.now();
      const records: OwnRecordForCognition[] = [];
      let lastConsumed: OwnRecordCandidate | null = null;

      for (let index = 0; index < pageCandidates.length; index += 1) {
        const candidate = pageCandidates[index]!;
        const disclosureLabel = selfPrivateMemoryDisclosureLabel();
        let row: OwnRecordForCognition;

        if (candidate.kind === "thought") {
          let entry: StreamEntry | null;

          try {
            entry = await options.readThoughtRecord(candidate.record);
          } catch {
            entry = null;
          }

          const availableEntry =
            entry?.id === candidate.sourceId && entry.kind === "thought" ? entry : null;
          const available = availableEntry !== null;
          const content =
            availableEntry === null ? null : stringifyPromptContent(availableEntry.content);
          const label = available ? disclosureLabel : unknownMemoryDisclosureLabel();

          row = {
            handle: `thought:${candidate.sourceId}`,
            kind: "thought",
            content,
            payload_status: available ? "exact" : "check_not_completed_retrieval_unavailable",
            payload_included_chars: content?.length ?? 0,
            payload_total_chars: content?.length ?? 0,
            occurred_at: candidate.occurredAt,
            occurred_at_iso: new Date(candidate.occurredAt).toISOString(),
            origin_time_basis: "stream_timestamp",
            relative_age: formatRelativeAge(candidate.occurredAt, nowMs),
            session_id: candidate.record.session_id,
            turn_id: candidate.record.turn_id,
            stream_entry_id: candidate.sourceId,
            journal_entry_id: null,
            marker_stream_entry_id: null,
            self_entity_id: null,
            oversized_anchors_omitted: false,
            ...memoryDisclosurePayloadFields(label),
          };
        } else {
          row = {
            handle: `journal:${candidate.sourceId}`,
            kind: "journal",
            content: candidate.record.text,
            payload_status: "exact",
            payload_included_chars: candidate.record.text.length,
            payload_total_chars: candidate.record.text.length,
            occurred_at: candidate.occurredAt,
            occurred_at_iso: new Date(candidate.occurredAt).toISOString(),
            origin_time_basis: "journal_created_at",
            relative_age: formatRelativeAge(candidate.occurredAt, nowMs),
            session_id: null,
            turn_id: candidate.record.source_turn_id,
            stream_entry_id: null,
            journal_entry_id: candidate.sourceId,
            marker_stream_entry_id: candidate.record.marker_stream_entry_id as StreamEntryId | null,
            self_entity_id: candidate.record.self_entity_id,
            oversized_anchors_omitted: false,
            ...memoryDisclosurePayloadFields(disclosureLabel),
          };
        }

        const hasMoreAfterCandidate = index + 1 < pageCandidates.length || moreInStore;
        const budgetCutReason = pageEndReason({
          hasMore: hasMoreAfterCandidate,
          endedOnBudget: true,
        });
        const exactOutput = {
          records: [...records, row],
          has_more: hasMoreAfterCandidate,
          page_end_reason: budgetCutReason,
          next_cursor: hasMoreAfterCandidate ? encodeOwnRecordCursor(candidate) : null,
        } satisfies OwnRecordsListOutput;

        if (resultFitsBudget(exactOutput, DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET)) {
          records.push(row);
          lastConsumed = candidate;
          continue;
        }

        if (records.length > 0) {
          return {
            records,
            has_more: true,
            page_end_reason: pageEndReason({ hasMore: true, endedOnBudget: true }),
            next_cursor: encodeOwnRecordCursor(lastConsumed!),
          };
        }

        const payloadlessRow = withoutPayload(row, "check_not_completed_budget");
        const payloadlessOutput = {
          records: [payloadlessRow],
          has_more: hasMoreAfterCandidate,
          page_end_reason: budgetCutReason,
          next_cursor: hasMoreAfterCandidate ? encodeOwnRecordCursor(candidate) : null,
        } satisfies OwnRecordsListOutput;

        if (resultFitsBudget(payloadlessOutput, DEFAULT_RETRIEVAL_CONTEXT_TOKEN_BUDGET)) {
          return payloadlessOutput;
        }

        return {
          records: [withoutOversizedAnchors(row)],
          has_more: hasMoreAfterCandidate,
          page_end_reason: budgetCutReason,
          next_cursor: hasMoreAfterCandidate ? encodeOwnRecordCursor(candidate) : null,
        };
      }

      const consumedAllCandidates = records.length === pageCandidates.length;
      const hasMore = lastConsumed !== null && (!consumedAllCandidates || moreInStore);

      return {
        records,
        has_more: hasMore,
        page_end_reason: pageEndReason({ hasMore, endedOnBudget: !consumedAllCandidates }),
        next_cursor: hasMore && lastConsumed !== null ? encodeOwnRecordCursor(lastConsumed) : null,
      };
    },
  };
}
