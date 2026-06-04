import { z } from "zod";

import {
  isSessionId,
  selfDecisionEventIdHelpers,
  streamEntryIdHelpers,
  type SelfDecisionEventId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";
import type { AutonomyWakeSourceType } from "../../autonomy/types.js";

export const SELF_DECISION_ORIGINS = ["autonomous"] as const;
export const SELF_DECISION_DISCLOSURE_CLASSES = ["self_private"] as const;

export const selfDecisionEventIdSchema = z
  .string()
  .refine((value) => selfDecisionEventIdHelpers.is(value), {
    message: "Invalid self decision event id",
  })
  .transform((value) => value as SelfDecisionEventId);

export const selfDecisionSessionIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid self decision session id",
  })
  .transform((value) => value as SessionId);

export const selfDecisionStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid self decision stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const selfDecisionOriginSchema = z.enum(SELF_DECISION_ORIGINS);
export const selfDecisionDisclosureClassSchema = z.enum(SELF_DECISION_DISCLOSURE_CLASSES);
export const selfDecisionTriggerTypeSchema = z.enum(["trigger", "condition"]);

export const selfDecisionEventSchema = z
  .object({
    id: selfDecisionEventIdSchema,
    occurred_at: z.number().int().finite(),
    session_id: selfDecisionSessionIdSchema,
    trigger_name: z.string().min(1),
    trigger_type: selfDecisionTriggerTypeSchema,
    source_event_id: z.string().min(1),
    fire_event_id: selfDecisionStreamEntryIdSchema,
    origin: selfDecisionOriginSchema,
    decision_summary: z.string(),
    decision_rationale: z.string().nullable(),
    turn_result_id: z.string().min(1).nullable(),
    source_stream_entry_ids: z.array(selfDecisionStreamEntryIdSchema).min(1),
    disclosure_class: selfDecisionDisclosureClassSchema,
    created_at: z.number().int().finite(),
    updated_at: z.number().int().finite(),
  })
  .strict();

export type SelfDecisionOrigin = z.infer<typeof selfDecisionOriginSchema>;
export type SelfDecisionDisclosureClass = z.infer<typeof selfDecisionDisclosureClassSchema>;
export type SelfDecisionEvent = z.infer<typeof selfDecisionEventSchema>;
export type SelfDecisionTriggerType = AutonomyWakeSourceType;
