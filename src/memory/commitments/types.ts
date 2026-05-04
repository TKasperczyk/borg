import { z } from "zod";

import { provenanceSchema, type Provenance } from "../common/provenance.js";
import {
  commitmentIdHelpers,
  entityIdHelpers,
  streamEntryIdHelpers,
  type CommitmentId,
  type EntityId,
  type StreamEntryId,
} from "../../util/ids.js";

export const COMMITMENT_TYPES = ["promise", "boundary", "rule", "preference"] as const;

export function normalizeDirectiveFamily(value: string): string {
  const lower = value.trim().toLowerCase();
  const chars: string[] = [];
  let previousWasSeparator = false;

  for (const char of lower) {
    const code = char.codePointAt(0) ?? 0;
    const isAsciiLetter = code >= 97 && code <= 122;
    const isDigit = code >= 48 && code <= 57;

    if (isAsciiLetter || isDigit) {
      chars.push(char);
      previousWasSeparator = false;
      continue;
    }

    if (!previousWasSeparator && chars.length > 0) {
      chars.push("_");
      previousWasSeparator = true;
    }
  }

  while (chars.at(-1) === "_") {
    chars.pop();
  }

  return chars.join("");
}

export const entityIdSchema = z
  .string()
  .refine((value) => entityIdHelpers.is(value), {
    message: "Invalid entity id",
  })
  .transform((value) => value as EntityId);

export const commitmentIdSchema = z
  .string()
  .refine((value) => commitmentIdHelpers.is(value), {
    message: "Invalid commitment id",
  })
  .transform((value) => value as CommitmentId);

export const streamEntryIdSchema = z
  .string()
  .refine((value) => streamEntryIdHelpers.is(value), {
    message: "Invalid stream entry id",
  })
  .transform((value) => value as StreamEntryId);

export const commitmentTypeSchema = z.enum(COMMITMENT_TYPES);

export const directiveFamilySchema = z
  .string()
  .min(1)
  .transform((value) => normalizeDirectiveFamily(value))
  .pipe(z.string().min(1).max(64));

export const entityRecordSchema = z.object({
  id: entityIdSchema,
  canonical_name: z.string().min(1),
  aliases: z.array(z.string().min(1)),
  created_at: z.number().finite(),
});

export const commitmentSchema = z.object({
  id: commitmentIdSchema,
  type: commitmentTypeSchema,
  directive_family: directiveFamilySchema,
  directive: z.string().min(1),
  priority: z.number().int(),
  made_to_entity: entityIdSchema.nullable(),
  restricted_audience: entityIdSchema.nullable(),
  about_entity: entityIdSchema.nullable(),
  provenance: provenanceSchema,
  source_stream_entry_ids: z.array(streamEntryIdSchema).min(1).optional(),
  created_at: z.number().finite(),
  expires_at: z.number().finite().nullable(),
  expired_at: z.number().finite().nullable(),
  revoked_at: z.number().finite().nullable(),
  revoked_reason: z.string().nullable(),
  revoke_provenance: provenanceSchema.nullable(),
  superseded_by: commitmentIdSchema.nullable(),
  last_reinforced_at: z.number().finite(),
});

export const commitmentPatchSchema = commitmentSchema
  .omit({
    id: true,
    created_at: true,
  })
  .partial()
  .strict();

export type EntityRecord = z.infer<typeof entityRecordSchema>;
export type CommitmentRecord = z.infer<typeof commitmentSchema>;
export type CommitmentPatch = z.infer<typeof commitmentPatchSchema>;
export type CommitmentType = z.infer<typeof commitmentTypeSchema>;
export type CommitmentProvenance = Provenance;

export type CommitmentListOptions = {
  activeOnly?: boolean;
  audience?: EntityId | null;
  aboutEntity?: EntityId | null;
  nowMs?: number;
};

export type CommitmentApplicableOptions = {
  audience?: EntityId | null;
  aboutEntity?: EntityId | null;
  nowMs?: number;
};
