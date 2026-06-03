import { z } from "zod";

import {
  entityIdHelpers,
  isSessionId,
  observedEventIdHelpers,
  streamEntryIdHelpers,
  type EntityId,
  type ObservedEventId,
  type SessionId,
  type StreamEntryId,
} from "../../util/ids.js";

export const OBSERVED_EVENT_STANCES = ["rejected_frame", "accepted_frame", "noted_frame"] as const;
export const OBSERVED_EVENT_TAINTS = ["none", "quarantined"] as const;
export const OBSERVED_EVENT_BELIEF_EFFECTS = ["unchanged", "updated", "reinforced"] as const;
export const OBSERVED_EVENT_DISCLOSURE_CLASSES = ["social_observed", "self_private"] as const;

const SLUG = /^[a-z][a-z0-9_]*$/;

export const observedEventIdSchema = z
  .string()
  .refine((value) => observedEventIdHelpers.is(value), {
    message: "Invalid observed event id",
  })
  .transform((value) => value as ObservedEventId);

export const observedEventSessionIdSchema = z
  .string()
  .refine((value) => isSessionId(value), {
    message: "Invalid observed event session id",
  })
  .transform((value) => value as SessionId);

export const observedEventStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid observed event stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const observedEventEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid observed event entity id",
  })
  .transform((value) => value as EntityId);

export const observedEventStanceSchema = z.string().regex(SLUG, {
  message: "stance must be a lowercase slug matching /^[a-z][a-z0-9_]*$/",
});
export const observedEventTaintSchema = z.string().regex(SLUG, {
  message: "taint must be a lowercase slug matching /^[a-z][a-z0-9_]*$/",
});
export const observedEventBeliefEffectSchema = z.string().regex(SLUG, {
  message: "belief_effect must be a lowercase slug matching /^[a-z][a-z0-9_]*$/",
});
export const observedEventClassificationKindSchema = z.string().regex(SLUG, {
  message: "classification_kind must be a lowercase slug matching /^[a-z][a-z0-9_]*$/",
});
export const observedEventDisclosureClassSchema = z.enum(OBSERVED_EVENT_DISCLOSURE_CLASSES);

export const observedEventSchema = z
  .object({
    id: observedEventIdSchema,
    occurred_at: z.number().int().finite(),
    session_id: observedEventSessionIdSchema,
    stance: observedEventStanceSchema,
    taint: observedEventTaintSchema,
    belief_effect: observedEventBeliefEffectSchema,
    classification_kind: observedEventClassificationKindSchema,
    disclosure_class: observedEventDisclosureClassSchema,
    interaction_text: z.string().min(1),
    recurrence_key: z.string().min(1),
    recurrence_count: z.number().int().positive(),
    last_seen_at: z.number().int().finite(),
    speaker_entity_id: observedEventEntityIdSchema.nullable(),
    audience_entity_id: observedEventEntityIdSchema.nullable(),
    source_entity_id: observedEventEntityIdSchema.nullable(),
    source_stream_entry_ids: z.array(observedEventStreamEntryIdSchema).min(1),
    created_at: z.number().int().finite(),
    updated_at: z.number().int().finite(),
  })
  .strict();

export type ObservedEventStance = z.infer<typeof observedEventStanceSchema>;
export type ObservedEventTaint = z.infer<typeof observedEventTaintSchema>;
export type ObservedEventBeliefEffect = z.infer<typeof observedEventBeliefEffectSchema>;
export type ObservedEventDisclosureClass = z.infer<typeof observedEventDisclosureClassSchema>;
export type ObservedEvent = z.infer<typeof observedEventSchema>;
