import { z } from "zod";

import {
  episodeIdHelpers,
  goalIdHelpers,
  traitIdHelpers,
  valueIdHelpers,
  type EpisodeId,
  type GoalId,
  type TraitId,
  type ValueId,
} from "../../util/ids.js";
import { provenanceSchema, type Provenance } from "../common/provenance.js";

export const goalStatusSchema = z.enum(["active", "done", "abandoned", "blocked"]);
export const identityStateSchema = z.enum(["candidate", "established"]);

export const valueIdSchema = z
  .string()
  .refine((value) => valueIdHelpers.is(value), {
    message: "Invalid value id",
  })
  .transform((value) => value as ValueId);

export const goalIdSchema = z
  .string()
  .refine((value) => goalIdHelpers.is(value), {
    message: "Invalid goal id",
  })
  .transform((value) => value as GoalId);

export const traitIdSchema = z
  .string()
  .refine((value) => traitIdHelpers.is(value), {
    message: "Invalid trait id",
  })
  .transform((value) => value as TraitId);

export const valueSourceEpisodeIdSchema = z
  .string()
  .refine((value) => episodeIdHelpers.is(value), {
    message: "Invalid episode id",
  })
  .transform((value) => value as EpisodeId);

export const valueSchema = z.object({
  id: valueIdSchema,
  label: z.string().min(1),
  description: z.string().min(1),
  priority: z.number().finite(),
  created_at: z.number().finite(),
  last_affirmed: z.number().finite().nullable(),
  state: identityStateSchema,
  established_at: z.number().finite().nullable(),
  provenance: provenanceSchema,
});

export const valuePatchSchema = valueSchema
  .omit({
    id: true,
    created_at: true,
  })
  .partial();

export const goalSchema = z.object({
  id: goalIdSchema,
  description: z.string().min(1),
  priority: z.number().finite(),
  parent_goal_id: goalIdSchema.nullable(),
  status: goalStatusSchema,
  progress_notes: z.string().nullable(),
  created_at: z.number().finite(),
  target_at: z.number().finite().nullable(),
  provenance: provenanceSchema,
});

export const traitSchema = z.object({
  id: traitIdSchema,
  label: z.string().min(1),
  strength: z.number().min(0).max(1),
  last_reinforced: z.number().finite(),
  last_decayed: z.number().finite().nullable(),
  state: identityStateSchema,
  established_at: z.number().finite().nullable(),
  provenance: provenanceSchema,
});

export const traitPatchSchema = traitSchema
  .omit({
    id: true,
  })
  .partial();

export const goalPatchSchema = goalSchema
  .omit({
    id: true,
    created_at: true,
  })
  .partial();

export type ValueRecord = z.infer<typeof valueSchema>;
export type GoalRecord = z.infer<typeof goalSchema>;
export type GoalStatus = z.infer<typeof goalStatusSchema>;
export type TraitRecord = z.infer<typeof traitSchema>;
export type SelfProvenance = Provenance;
export type IdentityState = z.infer<typeof identityStateSchema>;

export type GoalTreeNode = GoalRecord & {
  children: GoalTreeNode[];
};
