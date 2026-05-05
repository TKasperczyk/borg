import { z } from "zod";

import {
  DEFAULT_SESSION_ID,
  type SessionId,
  type StreamEntryId,
  isSessionId,
  parseSessionId,
  streamEntryIdHelpers,
} from "../util/ids.js";

export const STREAM_ENTRY_KINDS = [
  "user_msg",
  "agent_msg",
  "agent_suppressed",
  "thought",
  "tool_call",
  "tool_result",
  "perception",
  "internal_event",
  "dream_report",
] as const;

export const NARRATIVE_STREAM_ENTRY_KINDS = ["user_msg", "agent_msg"] as const;

export const streamEntryKindSchema = z.enum(STREAM_ENTRY_KINDS);
export const streamTurnStatusSchema = z.enum(["active", "aborted"]);

export const streamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const sessionIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid session id",
  })
  .transform((value) => parseSessionId(value));

export const streamEntrySchema = z.object({
  id: streamEntryIdSchema,
  timestamp: z.number().finite(),
  kind: streamEntryKindSchema,
  content: z.unknown(),
  turn_id: z.string().min(1).optional(),
  turn_status: streamTurnStatusSchema.default("active"),
  token_estimate: z.number().int().nonnegative().optional(),
  tool_calls: z.array(z.unknown()).optional(),
  audience: z.string().min(1).optional(),
  session_id: sessionIdSchema,
  compressed: z.boolean().default(false),
});

export const streamEntryInputSchema = streamEntrySchema
  .omit({
    id: true,
    timestamp: true,
    session_id: true,
  })
  .extend({
    compressed: z.boolean().optional(),
  });

export type StreamEntryKind = z.infer<typeof streamEntryKindSchema>;
export type NarrativeStreamEntryKind = (typeof NARRATIVE_STREAM_ENTRY_KINDS)[number];
export type StreamTurnStatus = z.infer<typeof streamTurnStatusSchema>;
export type StreamEntry = Omit<z.infer<typeof streamEntrySchema>, "turn_status"> & {
  turn_status?: StreamTurnStatus;
};
export type StreamEntryInput = z.input<typeof streamEntryInputSchema>;

export type StreamCursor = {
  ts: number;
  entryId: StreamEntryId;
};

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
