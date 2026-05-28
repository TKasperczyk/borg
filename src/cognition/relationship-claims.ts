import { z } from "zod";

import {
  entityIdHelpers,
  relationalSlotIdHelpers,
  streamEntryIdHelpers,
  type EntityId,
  type RelationalSlotId,
  type StreamEntryId,
} from "../util/ids.js";

export const RELATIONSHIP_LABEL_FAMILIES = [
  "kinship",
  "caregiver",
  "intimate_partner",
  "household",
  "other_sensitive",
] as const;

export const relationshipLabelFamilySchema = z.enum(RELATIONSHIP_LABEL_FAMILIES);

const relationshipClaimEntityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid relationship claim entity id",
  })
  .transform((value) => value as EntityId)
  .nullable()
  .default(null);

const relationshipClaimRelationalSlotIdSchema = z
  .string()
  .refine((value) => relationalSlotIdHelpers.is(value), {
    message: "Invalid relationship claim relational slot id",
  })
  .transform((value) => value as RelationalSlotId);

const relationshipClaimStreamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid relationship claim stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const relationshipClaimSchema = z
  .object({
    label_family: relationshipLabelFamilySchema.describe(
      "Language-agnostic sensitive relationship family selected by meaning.",
    ),
    subject_entity_id: relationshipClaimEntityIdSchema.describe(
      "Known subject entity id when supplied context identifies one, otherwise null.",
    ),
    object_entity_id: relationshipClaimEntityIdSchema.describe(
      "Known object entity id when supplied context identifies one, otherwise null.",
    ),
    object_text: z
      .string()
      .trim()
      .min(1)
      .nullable()
      .default(null)
      .describe("Object text from the claim when no object entity id is available."),
    requires_grounding: z
      .boolean()
      .describe("True when this relationship assertion must be backed by accepted evidence."),
    evidence_relational_slot_ids: z
      .array(relationshipClaimRelationalSlotIdSchema)
      .default([])
      .describe("Relational slot ids supplied as evidence for this relationship claim."),
    evidence_stream_entry_ids: z
      .array(relationshipClaimStreamEntryIdSchema)
      .default([])
      .describe("User stream entry ids supplied as evidence for this relationship claim."),
  })
  .strict();

export type RelationshipLabelFamily = z.infer<typeof relationshipLabelFamilySchema>;
export type RelationshipClaim = z.infer<typeof relationshipClaimSchema>;
