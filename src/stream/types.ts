import { z } from "zod";

import {
  DEFAULT_SESSION_ID,
  entityIdHelpers,
  type EntityId,
  type SessionId,
  type StreamEntryId,
  isSessionId,
  parseSessionId,
  streamEntryIdHelpers,
} from "../util/ids.js";

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

export const streamEntryKindSchema = z.enum(STREAM_ENTRY_KINDS);
export const streamEntryPersistenceClassSchema = z.enum(STREAM_ENTRY_PERSISTENCE_CLASSES);
export const streamTurnStatusSchema = z.enum(["active", "aborted"]);

export const streamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const streamCursorSchema = z.object({
  ts: z.number().finite(),
  entryId: streamEntryIdSchema,
});

export const streamSourceMessageKeySchema = z.object({
  source_type: z.string().min(1),
  source_external_id: z.string().min(1),
  external_message_id: z.string().min(1),
});

export const streamResponseToSchema = z.object({
  kind: z.literal("stream_backlog"),
  from_cursor_exclusive: streamCursorSchema.nullable(),
  through_cursor_inclusive: streamCursorSchema,
  source_entry_ids: z.array(streamEntryIdSchema),
  count: z.number().int().nonnegative(),
});

export const sessionIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid session id",
  })
  .transform((value) => parseSessionId(value));

export const streamEntryEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid stream entry entity id",
  })
  .transform((value) => value as EntityId);

export const streamEntrySchema = z.object({
  id: streamEntryIdSchema,
  timestamp: z.number().finite(),
  entry_index: z.number().int().nonnegative().optional(),
  kind: streamEntryKindSchema,
  content: z.unknown(),
  turn_id: z.string().min(1).optional(),
  turn_status: streamTurnStatusSchema.default("active"),
  token_estimate: z.number().int().nonnegative().optional(),
  tool_calls: z.array(z.unknown()).optional(),
  audience: z.string().min(1).optional(),
  sender_entity_id: streamEntryEntityIdSchema.nullable().default(null),
  reply_target_entity_id: streamEntryEntityIdSchema.nullable().default(null),
  source_message_key: streamSourceMessageKeySchema.optional(),
  response_to: streamResponseToSchema.optional(),
  persistence_class: streamEntryPersistenceClassSchema.optional(),
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
export type NarrativeStreamEntryKind = (typeof NARRATIVE_STREAM_ENTRY_KINDS)[number];
export type StreamTurnStatus = z.infer<typeof streamTurnStatusSchema>;
export type StreamEntry = Omit<z.infer<typeof streamEntrySchema>, "turn_status"> & {
  turn_status?: StreamTurnStatus;
};
export type StreamEntryInput = z.input<typeof streamEntryInputSchema>;
export type StreamCursor = z.infer<typeof streamCursorSchema>;
export type StreamSourceMessageKey = z.infer<typeof streamSourceMessageKeySchema>;
export type StreamResponseTo = z.infer<typeof streamResponseToSchema>;

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
