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
  "thought",
  "tool_call",
  "tool_result",
  "perception",
  "internal_event",
  "dream_report",
] as const;

export const streamEntryKindSchema = z.enum(STREAM_ENTRY_KINDS);

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
export type StreamEntry = z.infer<typeof streamEntrySchema>;
export type StreamEntryInput = z.infer<typeof streamEntryInputSchema>;

export type StreamIterateOptions = {
  sinceTs?: number;
  untilTs?: number;
  kinds?: readonly StreamEntryKind[];
  limit?: number;
};

export { DEFAULT_SESSION_ID };
export type { SessionId };
