import { z } from "zod";

import {
  entityIdHelpers,
  relationalSlotIdHelpers,
  streamEntryIdHelpers,
  type EntityId,
  type RelationalSlotId,
  type StreamEntryId,
} from "../../util/ids.js";

export const RELATIONAL_SLOT_STATES = [
  "established",
  "contested",
  "quarantined",
  "revoked",
] as const;

export const relationalSlotIdSchema = z
  .string()
  .refine((value) => relationalSlotIdHelpers.is(value), {
    message: "Invalid relational slot id",
  })
  .transform((value) => value as RelationalSlotId);

export const relationalSlotEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid relational slot entity id",
  })
  .transform((value) => value as EntityId);

export const relationalSlotStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid relational slot stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const relationalSlotStateSchema = z.enum(RELATIONAL_SLOT_STATES);
export const relationalSlotAssertionConfirmationSchema = z
  .enum(["direct", "explicit", "assistant_seeded"])
  .default("direct");

export const relationalSlotAlternateValueSchema = z
  .object({
    value: z.string().min(1),
    evidence_stream_entry_ids: z.array(relationalSlotStreamEntryIdSchema).min(1),
  })
  .strict();

export const relationalSlotSchema = z
  .object({
    id: relationalSlotIdSchema,
    subject_entity_id: relationalSlotEntityIdSchema,
    slot_key: z.string().min(1),
    value: z.string().min(1),
    state: relationalSlotStateSchema,
    evidence_stream_entry_ids: z.array(relationalSlotStreamEntryIdSchema),
    contradicted_by_stream_entry_ids: z.array(relationalSlotStreamEntryIdSchema),
    alternate_values: z.array(relationalSlotAlternateValueSchema),
    created_at: z.number().finite(),
    updated_at: z.number().finite(),
  })
  .strict();

export const relationalSlotAssertionSchema = z
  .object({
    subject_entity_id: relationalSlotEntityIdSchema,
    slot_key: z.string().min(1),
    asserted_value: z.string().min(1),
    source_stream_entry_ids: z.array(relationalSlotStreamEntryIdSchema).min(1),
    confirmation: relationalSlotAssertionConfirmationSchema,
  })
  .strict();

export const relationalSlotNegationSchema = z
  .object({
    subject_entity_id: relationalSlotEntityIdSchema,
    slot_key: z.string().min(1),
    rejected_value: z.string().min(1).nullable(),
    source_stream_entry_ids: z.array(relationalSlotStreamEntryIdSchema).min(1),
  })
  .strict();

export type RelationalSlotState = z.infer<typeof relationalSlotStateSchema>;
export type RelationalSlotAlternateValue = z.infer<typeof relationalSlotAlternateValueSchema>;
export type RelationalSlot = z.infer<typeof relationalSlotSchema>;
export type RelationalSlotAssertion = z.input<typeof relationalSlotAssertionSchema>;
export type RelationalSlotAssertionConfirmation = z.infer<
  typeof relationalSlotAssertionConfirmationSchema
>;
export type RelationalSlotNegation = z.infer<typeof relationalSlotNegationSchema>;
