import { z } from "zod";

import { entityIdHelpers, type EntityId } from "../../util/ids.js";
import type { MemoryDisclosureLabel } from "../common/disclosure-label.js";

export const OPERATOR_ATTENTION_SUBJECT_MAX_CHARS = 240;
export const OPERATOR_ATTENTION_RECENT_LIMIT = 20;

// This is an envelope, not a memory of the filing's contents. Strictness at the
// HTTP/import boundary prevents a caller from passing a body through this API.
export const operatorAttentionRecordSchema = z.strictObject({
  record_key: z.string().min(1).max(200),
  filed_at: z.number().int().nonnegative().max(8_640_000_000_000_000),
  filer_entity_id: z
    .string()
    .refine((value) => entityIdHelpers.is(value), { message: "Invalid filer entity id" })
    .transform((value) => value as EntityId),
  // Line delimiters are protocol structure; the subject's language is never interpreted.
  subject: z
    .string()
    .trim()
    .min(1)
    .max(OPERATOR_ATTENTION_SUBJECT_MAX_CHARS)
    .regex(/^[^\r\n\u2028\u2029]*$/u, "Subject must be one line")
    .nullable(),
});

export type OperatorAttentionRecord = z.infer<typeof operatorAttentionRecordSchema>;

export type OperatorAttentionIndexRow = OperatorAttentionRecord & {
  disclosure_label: MemoryDisclosureLabel;
};

export type OperatorAttentionIndex = {
  total: number;
  records: OperatorAttentionIndexRow[];
};

export type BorgOperatorAttentionFacade = {
  record(input: OperatorAttentionRecord): { inserted: boolean };
  snapshot(): OperatorAttentionIndex;
};
