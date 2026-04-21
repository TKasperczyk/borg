import { z } from "zod";

import { episodeIdSchema } from "../episodic/types.js";
import {
  commitmentIdHelpers,
  entityIdHelpers,
  type EpisodeId,
  type CommitmentId,
  type EntityId,
} from "../../util/ids.js";

export const COMMITMENT_TYPES = ["promise", "boundary", "rule", "preference"] as const;

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

export const commitmentTypeSchema = z.enum(COMMITMENT_TYPES);

export const entityRecordSchema = z.object({
  id: entityIdSchema,
  canonical_name: z.string().min(1),
  aliases: z.array(z.string().min(1)),
  created_at: z.number().finite(),
});

export const commitmentSchema = z.object({
  id: commitmentIdSchema,
  type: commitmentTypeSchema,
  directive: z.string().min(1),
  priority: z.number().int(),
  made_to_entity: entityIdSchema.nullable(),
  restricted_audience: entityIdSchema.nullable(),
  about_entity: entityIdSchema.nullable(),
  source_episode_ids: z.array(episodeIdSchema),
  created_at: z.number().finite(),
  expires_at: z.number().finite().nullable(),
  revoked_at: z.number().finite().nullable(),
  superseded_by: commitmentIdSchema.nullable(),
});

export const commitmentPatchSchema = commitmentSchema
  .omit({
    id: true,
    created_at: true,
  })
  .partial();

export type EntityRecord = z.infer<typeof entityRecordSchema>;
export type CommitmentRecord = z.infer<typeof commitmentSchema>;
export type CommitmentPatch = z.infer<typeof commitmentPatchSchema>;
export type CommitmentType = z.infer<typeof commitmentTypeSchema>;

export type CommitmentListOptions = {
  activeOnly?: boolean;
  audience?: EntityId | null;
  aboutEntity?: EntityId | null;
};

export type CommitmentApplicableOptions = {
  audience?: EntityId | null;
  aboutEntity?: EntityId | null;
  nowMs?: number;
};
