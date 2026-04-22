import { z } from "zod";

import { provenanceSchema } from "../common/provenance.js";

export const IDENTITY_RECORD_TYPES = [
  "episode",
  "semantic_node",
  "value",
  "goal",
  "trait",
  "commitment",
  "open_question",
] as const;

export const identityRecordTypeSchema = z.enum(IDENTITY_RECORD_TYPES);

export const identityEventSchema = z.object({
  id: z.number().int().positive(),
  record_type: identityRecordTypeSchema,
  record_id: z.string().min(1),
  action: z.string().min(1),
  old_value: z.unknown().nullable(),
  new_value: z.unknown().nullable(),
  reason: z.string().nullable(),
  provenance: provenanceSchema,
  review_item_id: z.number().int().positive().nullable(),
  overwrite_without_review: z.boolean(),
  ts: z.number().finite(),
});

export type IdentityRecordType = z.infer<typeof identityRecordTypeSchema>;
export type IdentityEvent = z.infer<typeof identityEventSchema>;
