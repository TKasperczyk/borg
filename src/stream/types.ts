import { z } from "zod";

import { sessionIdSchema, streamEntryIdSchema } from "../util/id-schemas.js";
import { DEFAULT_SESSION_ID, entityIdHelpers, type EntityId, type SessionId } from "../util/ids.js";
import { jsonValueSchema, type JsonValue } from "../util/json-value.js";

export { sessionIdSchema, streamEntryIdSchema };

export const STREAM_ENTRY_KINDS = [
  "user_msg",
  "user_image_attachment",
  "agent_msg",
  "agent_suppressed",
  "agent_observed",
  "thought",
  "tool_call",
  "tool_result",
  "perception",
  "internal_event",
  "dream_report",
] as const;

export const NARRATIVE_STREAM_ENTRY_KINDS = ["user_msg", "agent_msg"] as const;
export const STREAM_ENTRY_PERSISTENCE_CLASSES = ["assistant_self_report"] as const;
export const STREAM_CONVERSATION_TYPES = ["personal", "groupChat", "channel"] as const;

export const streamEntryKindSchema = z.enum(STREAM_ENTRY_KINDS);
export const streamEntryPersistenceClassSchema = z.enum(STREAM_ENTRY_PERSISTENCE_CLASSES);
export const streamTurnStatusSchema = z.enum(["active", "aborted"]);
export const streamConversationSchema = z.object({
  type: z.enum(STREAM_CONVERSATION_TYPES),
  name: z.string().transform((value) => value.trim()),
});

export const streamCursorSchema = z.object({
  ts: z.number().finite(),
  entryId: streamEntryIdSchema,
});

export const streamSourceMessageKeySchema = z.object({
  source_type: z.string().min(1),
  source_external_id: z.string().min(1),
  external_message_id: z.string().min(1),
});

// Stamped on the emitted entry only, so an inbound entry never carries it. `source_entry_ids`
// is the authoritative record of which entries a reply answered; index adjacency is not a
// substitute for it, because a session can interleave inbound and outbound entries and the
// entry immediately preceding a reply need not be the one it answered.
export const streamBacklogResponseToSchema = z.object({
  kind: z.literal("stream_backlog"),
  from_cursor_exclusive: streamCursorSchema.nullable(),
  through_cursor_inclusive: streamCursorSchema,
  source_entry_ids: z.array(streamEntryIdSchema),
  count: z.number().int().nonnegative(),
});

export const taskEventSchema = z
  .object({
    schema_version: z.literal(1),
    event_id: z.string().min(1),
    task_id: z.string().min(1),
    task_version: z.number().int(),
    kind: z.enum(["task_completed", "task_failed"]),
    occurred_at: z.iso.datetime({ offset: true }),
    outcome: z
      .object({
        status: z.enum(["succeeded", "failed", "timed_out", "cancelled"]),
        summary: z.string().max(8000),
        artifacts: z.array(z.object({ label: z.string(), url: z.url() }).strict()).optional(),
        error: z.string().optional(),
      })
      .strict(),
    origin: z.object({ source_entry_ids: z.array(z.string().min(1)) }).strict(),
  })
  .strict();

export const taskEventResponseToSchema = z
  .object({
    kind: z.literal("task_event"),
    event_id: z.string().min(1),
    event_entry_id: streamEntryIdSchema,
    task_id: z.string().min(1),
    task_version: z.number().int(),
  })
  .strict();

export const streamResponseToSchema = z.discriminatedUnion("kind", [
  streamBacklogResponseToSchema,
  taskEventResponseToSchema,
]);

export const streamEntryEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid stream entry entity id",
  })
  .transform((value) => value as EntityId);

export const streamEntrySchema = z.object({
  id: streamEntryIdSchema,
  timestamp: z.number().finite(),
  observed_at: z.number().int().nonnegative().optional(),
  entry_index: z.number().int().nonnegative().optional(),
  kind: streamEntryKindSchema,
  content: z.unknown(),
  turn_id: z.string().min(1).optional(),
  turn_status: streamTurnStatusSchema.default("active"),
  token_estimate: z.number().int().nonnegative().optional(),
  tool_calls: z.array(z.unknown()).optional(),
  audience: z.string().min(1).optional(),
  conversation: streamConversationSchema.optional(),
  metadata: z.record(z.string(), jsonValueSchema).optional(),
  sender_entity_id: streamEntryEntityIdSchema.nullable().default(null),
  reply_target_entity_id: streamEntryEntityIdSchema.nullable().default(null),
  source_message_key: streamSourceMessageKeySchema.optional(),
  response_to: streamResponseToSchema.optional(),
  persistence_class: streamEntryPersistenceClassSchema.optional(),
  receipt_pending: z.boolean().optional(),
  session_id: sessionIdSchema,
  compressed: z.boolean().default(false),
});

export const streamEntryInputSchema = streamEntrySchema
  .omit({
    id: true,
    timestamp: true,
    entry_index: true,
    session_id: true,
  })
  .extend({
    compressed: z.boolean().optional(),
  });

export type StreamEntryKind = z.infer<typeof streamEntryKindSchema>;
export type StreamEntryPersistenceClass = z.infer<typeof streamEntryPersistenceClassSchema>;
export type StreamConversation = z.infer<typeof streamConversationSchema>;
export type StreamEntryMetadata = Record<string, JsonValue>;
export type NarrativeStreamEntryKind = (typeof NARRATIVE_STREAM_ENTRY_KINDS)[number];
export type StreamTurnStatus = z.infer<typeof streamTurnStatusSchema>;
export type StreamEntry = Omit<z.infer<typeof streamEntrySchema>, "turn_status"> & {
  turn_status?: StreamTurnStatus;
  /** Read-only preservation of a future stamp; never interpreted as an answered window. */
  opaque_response_to?: Record<string, JsonValue> & { kind: string };
};

const opaqueResponseToSchema = z
  .object({ kind: z.string().min(1) })
  .catchall(jsonValueSchema)
  .refine((stamp) => stamp.kind !== "stream_backlog" && stamp.kind !== "task_event");

// Reading is forward tolerant; streamEntrySchema/InputSchema remain strict about
// stamps on writes. Malformed *known* stamps still fail validation.
export const streamEntryReadSchema = streamEntrySchema
  .extend({
    response_to: z.union([streamResponseToSchema, opaqueResponseToSchema]).optional(),
  })
  .transform(({ response_to, ...entry }): StreamEntry => {
    if (response_to === undefined) return entry;
    const known = streamResponseToSchema.safeParse(response_to);
    return known.success
      ? { ...entry, response_to: known.data }
      : { ...entry, opaque_response_to: opaqueResponseToSchema.parse(response_to) };
  });
export type StreamEntryInput = z.input<typeof streamEntryInputSchema>;
export type StreamCursor = z.infer<typeof streamCursorSchema>;
export type StreamSourceMessageKey = z.infer<typeof streamSourceMessageKeySchema>;
export type StreamResponseTo = z.infer<typeof streamResponseToSchema>;
export type StreamBacklogResponseTo = z.infer<typeof streamBacklogResponseToSchema>;
export type TaskEventResponseTo = z.infer<typeof taskEventResponseToSchema>;
export type TaskEvent = z.infer<typeof taskEventSchema>;

export type StreamIterateOptions = {
  sinceTs?: number;
  sinceCursor?: StreamCursor;
  untilTs?: number;
  untilCursor?: StreamCursor;
  kinds?: readonly StreamEntryKind[];
  limit?: number;
};

export { DEFAULT_SESSION_ID };
export type { SessionId };

export function isNarrativeStreamEntry(
  entry: Pick<StreamEntry, "kind">,
): entry is StreamEntry & { kind: NarrativeStreamEntryKind } {
  return (NARRATIVE_STREAM_ENTRY_KINDS as readonly StreamEntryKind[]).includes(entry.kind);
}

export function isEpisodicSourceEntry(entry: StreamEntry): boolean {
  return isNarrativeStreamEntry(entry);
}
